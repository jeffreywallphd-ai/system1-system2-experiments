"""Sequential governed runner. Labels are loaded only by the separate scorer.

Completed trial records are immutable. Journals are durable before calls
and actions, and interrupted attempts are retained as failed outcomes on
resume. An interrupted attempt is never invisibly replaced with a lucky retry.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from .config import validate
from .domain import Failure, Request, digest, view_from_dict
from .storage import (read_json, write_json, read_jsonl, write_jsonl, checksum_tree,
                      verify_tree, file_hash)
from .governor import Governor
from .policies import Policy
from .models.fake import ScriptedModel
from .models.prompts import PROMPTS
from .calibration import validate_gate
from .benchmarks.alfworld import FakeEnvironment, AlfworldEnvironment, ObservationMemory


def software_fingerprint():
    package = Path(__file__).parent
    sources = {p.relative_to(package).as_posix(): file_hash(p) for p in sorted(package.rglob("*.py"))}
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        git = None
    return dict(git_revision=git, source_hash=digest(sources), source_files=sources,
                prompt_hash=digest(PROMPTS), python=platform.python_version(),
                offline_lock_hash=file_hash(package.parent / "requirements-offline.lock") if (package.parent / "requirements-offline.lock").exists() else None)


def freeze_configuration(config, protocol_path, config_path):
    """Freeze permitted inputs/settings without inspecting test labels or running models."""
    if Path(protocol_path).exists() or Path(config_path).exists():
        raise ValueError("Freeze outputs must be new files")
    c = validate(config | {"stage": "confirmatory", "frozen_protocol": str(Path(protocol_path).resolve())})
    dataset = Path(c["dataset"])
    verify_tree(dataset)
    locks = {}
    for model in c["models"].values():
        for key in ("asset_lock", "runtime_lock"):
            locks[model[key]] = file_hash(model[key])
    for key in ("routing_artifact", "baseline_artifact"):
        if c.get(key):
            locks[c[key]] = file_hash(c[key])
    write_json(protocol_path, dict(schema_version=1, config_hash=digest(c),
        source_hash=software_fingerprint()["source_hash"], data_hash=digest(read_json(dataset / "checksums.json")),
        locks=locks, frozen_at=datetime.now(timezone.utc).isoformat(),
        scope="settings_freeze_only; not capability_validation_or_external_preregistration"))
    write_json(config_path, c)
    return Path(config_path)


def doctor(config, load_models=False):
    c = validate(config)
    report = {"execution_mode": c["execution_mode"], "python": platform.python_version(),
              "platform": platform.platform(), "cpu_count": os.cpu_count(), "models": {},
              "gpu_measurement": "not_probed_without_explicit_model_loading"}
    if c["execution_mode"] == "mock":
        report["models"] = {r: {"status": "READY", "identity": ScriptedModel(r).identity} for r in ("s1", "s2")}
    else:
        from importlib.util import find_spec
        for role, package in (("s1", "laya"), ("s2", "transformers")):
            model = c["models"][role]
            reasons = []
            if find_spec(package) is None:
                reasons.append(f"Missing optional package {package}")
            if not Path(model.get("local_path", "__missing__")).is_dir():
                reasons.append("Local model snapshot is missing")
            if not model.get("asset_lock") or not Path(model["asset_lock"]).is_file():
                reasons.append("Asset lock missing")
            report["models"][role] = {"status": "BLOCKED" if reasons else "UNVERIFIED", "reasons": reasons,
                                      "encoding": "not_measured", "identity": model}
    if c.get("dataset"):
        report["dataset"] = {"path": c["dataset"], "available": Path(c["dataset"], "manifest.json").is_file()}
    if load_models:
        models, readiness = build_models(c)
        report["models"] = readiness
        for model in models.values():
            model.close()
    return report


def build_models(config):
    models, readiness = {}, {}
    needed = set()
    for policy in config["policies"]:
        if policy != "B0_simple" and policy != "B2_s2" and policy != "C1_s2_workers":
            needed.add("s1")
        if policy not in {"B0_simple", "B1_laya"}:
            needed.add("s2")
    if config["context_track"] == "common_fit":
        needed.update(("s1", "s2"))
    for role in sorted(needed):
        try:
            if config["execution_mode"] == "mock":
                models[role] = ScriptedModel(role)
            else:
                from .models.process import ProcessModel
                models[role] = ProcessModel(role, config["models"][role], config["budgets"]["model_seconds"])
            readiness[role] = {"status": "READY", "identity": models[role].identity}
        except Exception as exc:
            readiness[role] = {"status": "BLOCKED", "reason": str(exc)}
    return models, readiness


def _interactive(view, game, policy, governor, execution_mode):
    env = None
    try:
        try:
            if execution_mode == "mock":
                env = FakeEnvironment(game, governor.seed, governor.budget.limits["steps"])
            else:
                from .benchmarks.process import IsolatedAlfworld
                env = IsolatedAlfworld(game, governor.seed, governor.budget.limits["steps"],
                    min(governor.budget.limits["model_seconds"], governor.budget.remaining_seconds()))
            observation, commands = env.reset()
        except Exception as exc:
            raise Failure("ENVIRONMENT_SETUP_FAILURE", str(exc)) from exc
        memory = ObservationMemory(view.case_id, view.family_id, view.question)
        current = memory.append(observation, commands)
        governor.event("observation", state=current.payload())
        visited, repeated, trigger = set(), 0, None
        while True:
            choice, version, action_hash = policy.action(current, trigger)
            command = governor.authorize_action(current, choice, version, action_hash)
            previous = observation
            if execution_mode != "mock":
                env.timeout = min(governor.budget.limits["model_seconds"], governor.budget.remaining_seconds())
            observation, commands, done, success = env.step(command)
            signature = digest([previous, command, observation])
            repeated += int(signature in visited)
            trigger = {"kind": "repeated_transition", "signature": signature} if signature in visited else None
            visited.add(signature)
            if "nothing happens" in observation.casefold() or "can't" in observation.casefold():
                trigger = {"kind": "explicit_action_failure", "observation": observation}
            if trigger:
                trigger["event_id"] = governor.event("observable_trigger", trigger=trigger.copy())
            # Termination is environment-owned; won/reward do not enter memory.
            if done or success:
                governor.event("environment_terminal", success=success, observation=observation)
                return None, "COMPLETED" if success else "ENVIRONMENT_FAILURE", success, repeated
            current = memory.append(observation, commands, command)
            governor.event("observation", state=current.payload())
    finally:
        if env is not None:
            env.close()


def _trial(view, policy_id, seed, config, models, journal, routing, baseline, game):
    g = Governor(models, config["budgets"], seed, journal)
    policy = Policy(policy_id, g, routing, baseline)
    result = dict(case_id=view.case_id, family_id=view.family_id, benchmark=view.benchmark,
                  policy_id=policy_id, seed=seed, execution_mode=config["execution_mode"],
                  prediction=None, success=None, repeated_actions=0, probabilities=None, status="COMPLETED")
    started = time.perf_counter()
    try:
        if view.benchmark == "alfworld":
            _, result["status"], result["success"], result["repeated_actions"] = _interactive(view, game, policy, g, config["execution_mode"])
        else:
            choice = policy.static(view)
            result["prediction"] = view.meaning(choice)
            if policy_id in {"B1_laya", "B2_s2"} and policy.last_probabilities:
                result["probabilities"] = {view.meaning(key): value for key, value in policy.last_probabilities.items()}
        g.event("prediction", prediction=result["prediction"], status=result["status"])
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "CANCELLED" if isinstance(exc, KeyboardInterrupt) else getattr(exc, "status", "ERROR")
        result["error"] = str(exc)
        g.event("terminal_failure", status=result["status"], error=str(exc))
    result["usage"] = g.budget.used
    result["wall_seconds"] = time.perf_counter() - started
    return result, g.events


def _interrupted_result(identity, pending, config):
    events = _recover_journal(pending)
    g = Governor({}, config["budgets"], identity["seed"])
    for event in events:
        if event["kind"] == "model_request":
            role = event["role"]
            g.budget.used[role + "_calls"] += 1
            g.budget.used["s1_questions"] += int(role == "s1")
            g.budget.used["generated_tokens"] += event["reserved_generated_tokens"]
            g.budget.used["unknown_usage_calls"] += 1
        elif event["kind"] == "action":
            g.budget.used["actions"] += 1
    return identity | dict(execution_mode=config["execution_mode"], prediction=None, success=None,
        status="CANCELLED", error="Interrupted attempt retained; usage conservatively reserved, not retried",
        usage=g.budget.used, wall_seconds=None, repeated_actions=0, probabilities=None)


def _recover_journal(pending):
    path = pending / "events.jsonl"
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8")
    lines, events = raw.splitlines(), []
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError("Interrupted journal is corrupt before its final record")
            # Retain the exact truncated bytes as evidence, then salvage complete records.
            (pending / "interrupted_journal.txt").write_text(raw, encoding="utf-8")
    return events


def run(config, destination, models=None):
    c = validate(config)
    root, dataset = Path(destination), Path(c["dataset"])
    verify_tree(dataset)
    if c["stage"] == "confirmatory":
        protocol = read_json(c["frozen_protocol"])
        if (protocol["config_hash"] != digest(c) or protocol["source_hash"] != software_fingerprint()["source_hash"]
                or protocol["data_hash"] != digest(read_json(dataset / "checksums.json"))):
            raise ValueError("Frozen protocol no longer matches configuration, code, or data")
        if any(file_hash(path) != expected for path, expected in protocol["locks"].items()):
            raise ValueError("Frozen fitting/model/runtime artifact changed")
    public = {r["case_id"]: view_from_dict(r) for r in read_jsonl(dataset / "public.jsonl")}
    cases = [r for r in read_jsonl(dataset / "cases.jsonl") if r["partition"] == c["partition"]]
    if not cases:
        raise ValueError("Selected partition has no cases")
    if any(r["benchmark"] == "alfworld" for r in cases):
        if c["budgets"]["s2_calls"] < c["budgets"]["steps"] * (1 + c["budgets"]["repairs"]):
            raise ValueError("Interactive S2 call ceiling must allow every action plus format repairs")
    routing = read_json(c["routing_artifact"]) if c.get("routing_artifact") else None
    baseline = read_json(c["baseline_artifact"]) if c.get("baseline_artifact") else None
    stable_s1_identity = ScriptedModel("s1").identity if c["execution_mode"] == "mock" else c["models"]["s1"]
    if routing:
        validate_gate(routing, stable_s1_identity, cases, c["partition"])
    if baseline and c["partition"] in {"test", "ood", "seen", "unseen"}:
        if any(r["case_id"] in baseline["training_ids"] or r["family_id"] in baseline["training_families"] for r in cases):
            raise ValueError("Majority fitting data overlaps held-out evaluation")
    assets = {}
    for model in c.get("models", {}).values():
        for key in ("asset_lock", "runtime_lock"):
            if model.get(key) and Path(model[key]).is_file():
                assets[model[key]] = file_hash(model[key])
    if c.get("frozen_protocol"):
        assets[c["frozen_protocol"]] = file_hash(c["frozen_protocol"])
    binding = dict(config=c, dataset_checksums=read_json(dataset / "checksums.json"),
                   dataset_manifest=read_json(dataset / "manifest.json"), software=software_fingerprint(),
                   routing=routing, baseline=baseline, asset_locks=assets)
    fingerprint = digest(binding)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "runner.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("Run is locked. If the process crashed, verify it stopped before removing runner.lock.") from exc
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    owned_models = models is None
    try:
        if (root / "manifest.json").exists():
            saved_manifest = read_json(root / "manifest.json")
            if saved_manifest["fingerprint"] != fingerprint or digest(saved_manifest["binding"]) != fingerprint:
                raise ValueError("Resume refused: configuration/data/software/artifact mismatch")
        else:
            write_json(root / "manifest.json", dict(schema_version=1, fingerprint=fingerprint, binding=binding,
                started_at=datetime.now(timezone.utc).isoformat(), execution_mode=c["execution_mode"],
                hardware={"platform": platform.platform(), "cpu_count": os.cpu_count()},
                command=sys.argv, concurrency=1, cache=False, energy_measurement=None,
                deviations=["Sequential logical workers", "No automatic model downloads", "No warmup; cold loading recorded separately",
                            "Cold model loading and shared encoding preflight are outside trial timing"]))
            write_json(root / "resolved_config.json", c)
        already_complete = (root / "intended_trials.jsonl").exists() and all(
            (root / "trials" / digest(row)[:24] / "checksums.json").exists()
            for row in read_jsonl(root / "intended_trials.jsonl"))
        if models is None and already_complete:
            models, readiness = {}, read_json(root / "readiness.json")
        elif models is None:
            models, readiness = build_models(c)
        else:
            readiness = {role: {"status": "READY", "identity": model.identity} for role, model in models.items()}
        if any((model.identity.get("execution_mode") == "mock") != (c["execution_mode"] == "mock") for model in models.values()):
            raise ValueError("Adapter execution mode disagrees with run mode; model substitution refused")
        if not (root / "readiness.json").exists():
            write_json(root / "readiness.json", readiness)
        elif not already_complete:
            write_json(root / f"resume_readiness_{time.time_ns()}.json", readiness)
        if c["stage"] == "confirmatory":
            required_roles = set()
            if any(p not in {"B0_simple", "B2_s2", "C1_s2_workers"} for p in c["policies"]):
                required_roles.add("s1")
            if any(p not in {"B0_simple", "B1_laya"} for p in c["policies"]):
                required_roles.add("s2")
            if c["context_track"] == "common_fit":
                required_roles.update(("s1", "s2"))
            if any(readiness.get(role, {}).get("status") != "READY" for role in required_roles):
                raise ValueError("Requested confirmatory condition is not ready")
        exclusions = []
        if (root / "exclusions.jsonl").exists():
            exclusions = read_jsonl(root / "exclusions.jsonl")
        elif c["context_track"] == "common_fit":
            if set(models) != {"s1", "s2"}:
                raise ValueError("Common-fit eligibility requires both encoders")
            for row in cases:
                view = public[row["case_id"]]
                if view.benchmark == "alfworld":
                    raise ValueError("Interactive dynamic context limits are outcomes; use full_coverage")
                for role, model in models.items():
                    try:
                        request = Request("decision", view.question, view, view.options, {"encoding_only": True}, 0)
                        model.infer(request, c["budgets"]["tokens_per_call"] if role == "s2" else 0, c["budgets"]["model_seconds"])
                    except Failure as exc:
                        if exc.status != "CONTEXT_LIMIT":
                            raise
                        exclusions.append({"case_id": view.case_id, "family_id": view.family_id, "role": role, "reason": str(exc)})
                        break
        write_jsonl(root / "exclusions.jsonl", exclusions)
        excluded = {r["case_id"] for r in exclusions}
        eligible = [r for r in cases if r["case_id"] not in excluded]
        if c["context_track"] == "common_fit":
            # Exclude whole ContextRules families, preserving the all-variants endpoint.
            excluded_families = {r["family_id"] for r in cases if r["case_id"] in excluded and r["benchmark"] == "context_rules"}
            for row in eligible[:]:
                if row["family_id"] in excluded_families:
                    exclusions.append(row | {"reason": "Another variant in this family did not fit"})
                    eligible.remove(row)
            write_jsonl(root / "exclusions.jsonl", exclusions)
        write_jsonl(root / "case_manifest.jsonl", eligible)
        intended = [dict(case_id=r["case_id"], family_id=r["family_id"], benchmark=r["benchmark"], policy_id=p, seed=s)
                    for r in eligible for s in c["seeds"] for p in c["policies"]]
        write_jsonl(root / "intended_trials.jsonl", intended)
        games = read_json(dataset / "games.json") if (dataset / "games.json").exists() else {}
        if c["execution_mode"] == "real":
            for game in games.values():
                if file_hash(game["game_path"]) != game["sha256"]:
                    raise ValueError("ALFWorld game changed after preparation")
        predictions, events = [], []
        for identity in intended:
            trial_id = digest(identity)[:24]
            complete = root / "trials" / trial_id
            pending = complete
            cancelled_now = False
            if (complete / "checksums.json").exists():
                verify_tree(complete)
                result = read_json(complete / "prediction.json")
                trial_events = read_jsonl(complete / "events.jsonl")
            else:
                if pending.exists():
                    result = _interrupted_result(identity, pending, c)
                    trial_events = _recover_journal(pending)
                else:
                    pending.mkdir(parents=True)
                    result, trial_events = _trial(public[identity["case_id"]], identity["policy_id"], identity["seed"],
                        c, models, pending / "events.jsonl", routing, baseline, games.get(identity["case_id"]))
                    cancelled_now = result["status"] == "CANCELLED"
                write_json(pending / "prediction.json", result)
                write_jsonl(pending / "events.jsonl", trial_events)
                write_json(pending / "checksums.json", checksum_tree(pending))
            result = result | {"trial_id": trial_id, "events_ref": f"trials/{trial_id}/events.jsonl"}
            predictions.append(result)
            events.extend(e | {"trial_id": trial_id} for e in trial_events)
            if cancelled_now:
                # Save aggregates before honoring user cancellation/interruption.
                break
        write_jsonl(root / "predictions.jsonl", predictions)
        write_jsonl(root / "events.jsonl", events)
        write_jsonl(root / "usage.jsonl", [{"trial_id": p["trial_id"], **p["usage"]} for p in predictions])
        write_jsonl(root / "encoding_audits.jsonl", [{"trial_id": e["trial_id"], "request_event": e["request_event"], **e["encoding"]}
            for e in events if e["kind"] in {"model_response", "model_failure"} and e.get("encoding")])
        write_jsonl(root / "errors.jsonl", [p for p in predictions if p["status"] != "COMPLETED"])
        write_json(root / "execution.json", {"finished_at": datetime.now(timezone.utc).isoformat(),
            "intended": len(intended), "recorded": len(predictions), "status": "COMPLETE" if len(predictions) == len(intended) else "INCOMPLETE"})
    finally:
        if owned_models and models:
            for model in models.values():
                model.close()
        lock_path.unlink(missing_ok=True)
    write_json(root / "checksums.json", checksum_tree(root))
    return root
