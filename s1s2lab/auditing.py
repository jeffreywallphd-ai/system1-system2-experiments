"""Decision audits and matched-plan replay: separate costs, immutable decisions.

Auditors receive public evidence and permitted execution records, never gold
labels or oracle ASTs. Evaluator-only scoring happens after all audit calls.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import random
import re
from .domain import Failure, LABELS, Source, decision_options, digest, view_from_dict
from .governor import Governor
from .policies import Policy
from .runner import build_models
from .storage import read_json, read_jsonl, write_json, write_jsonl, checksum_tree, verify_tree

CONDITIONS = ("E0_posthoc", "E1_trace", "E2_probe")


def validate_audit(content, view, events):
    required = {"decision_assessment", "supporting_source_ids", "supporting_event_ids", "claims",
                "alternative_explanations", "limitations", "probe_predictions", "explanation_kind"}
    if set(content) != required or content["decision_assessment"] not in {"SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"}:
        raise ValueError("Invalid audit contract")
    if content["explanation_kind"] != "EVIDENCE_AND_BEHAVIOR_NOT_INTERNAL_THOUGHT":
        raise ValueError("Unsupported explanation claim")
    for key in ("supporting_source_ids", "supporting_event_ids", "claims", "alternative_explanations", "limitations"):
        if not isinstance(content[key], list):
            raise ValueError("Audit collections must be lists")
    if any(not isinstance(c, dict) or set(c) != {"source_id", "quote", "claim"}
           or any(not isinstance(v, str) for v in c.values()) for c in content["claims"]):
        raise ValueError("Malformed claim")
    # Invalid citations are measured as audit errors, rather than repaired away.
    if not isinstance(content["probe_predictions"], dict) or any(v not in LABELS for v in content["probe_predictions"].values()):
        raise ValueError("Probe predictions must be semantic labels")
    return content


def safe_probes(view):
    """A fixed public-text transformation registry. No arbitrary model edits."""
    candidates = [s for s in view.sources if re.fullmatch(r"Requirement \d+ is (?:not )?met\.", s.text)]
    probes = {}
    if candidates:
        selected = candidates[0]
        probes["remove_fact"] = replace(view, sources=tuple(s for s in view.sources if s != selected), version=view.version + 1)
        changed = selected.text.replace("is not met", "is met") if "is not met" in selected.text else selected.text.replace("is met", "is not met")
        probes["change_fact"] = replace(view, sources=tuple(replace(s, text=changed) if s == selected else s for s in view.sources), version=view.version + 1)
    next_id = "probe_source"
    probes["add_distractor"] = replace(view, sources=view.sources + (Source(next_id, "The applicant's notebook is purple; color is unrelated to eligibility."),), version=view.version + 1)
    probes["reorder_options"] = replace(view, options=decision_options(("MORE", "NO", "YES", "IRRELEVANT")), version=view.version + 1)
    return probes


def score_audit(content, view, original_events, prediction, private):
    """Reference validity and decision support are deliberately separate.

    Bounded ContextRules citation support is checked against the private AST
    using only facts in the cited public blocks. Unrestricted prose claims are
    explicitly left unverified; exact quotation does not establish entailment.
    """
    from .benchmarks.context_rules import label, variables
    sources = {s.source_id: s.text for s in view.sources}
    event_ids = {e["event_id"] for e in original_events}
    refs = content["supporting_source_ids"]
    valid = [s in sources for s in refs] + [e in event_ids for e in content["supporting_event_ids"]]
    quotes = [bool(c["quote"]) and c["source_id"] in sources and c["quote"] in sources[c["source_id"]] for c in content["claims"]]
    facts = {name: None for name in variables(private["expression"])}
    for source in view.sources:
        if source.source_id not in refs:
            continue
        match = re.fullmatch(r"Requirement (\d+) is (not )?met\.", source.text)
        correction = re.search(r"Correction: replace the previous statement about requirement (\d+)\. That requirement is now (not )?met\.", source.text)
        if match or correction:
            parsed = correction or match
            facts[f"p{int(parsed.group(1)) - 1}"] = not bool(parsed.group(2))
    bounded_support = bool(refs) and all(valid) and "s0" in refs and label(private["expression"], facts, private["relevant"]) == prediction
    correct_decision = prediction == private["gold"]
    expected = "SUPPORTED" if correct_decision else "UNSUPPORTED"
    return dict(assessment_correct=content["decision_assessment"] == expected,
                decision_correct=correct_decision, valid_reference_rate=sum(valid) / len(valid) if valid else None,
                exact_quote_rate=sum(quotes) / len(quotes) if quotes else None,
                decision_supported_by_cited_evidence=bounded_support,
                incorrect_decision_detected=content["decision_assessment"] == "UNSUPPORTED" if not correct_decision else None,
                false_alarm=content["decision_assessment"] == "UNSUPPORTED" if correct_decision else None,
                prose_claim_entailment="UNVERIFIED", internal_causal_claims="not_mechanistically_verified",
                execution_consistency="references_checked_only; semantic_trace_review_required")


def audit_run(run_path, destination, limit=6, seed=821, challenge=False):
    root, output = Path(run_path), Path(destination)
    verify_tree(root)
    if output.exists():
        raise ValueError("Audit bundles are immutable; use a new destination")
    config = read_json(root / "resolved_config.json")
    public = {r["case_id"]: view_from_dict(r) for r in read_jsonl(Path(config["dataset"]) / "public.jsonl")}
    candidates = [r for r in read_jsonl(root / "predictions.jsonl") if r["benchmark"] == "context_rules" and r["prediction"] in LABELS]
    random.Random(seed).shuffle(candidates)
    sample = candidates[:limit]
    models, readiness = build_models(config | {"policies": ["B1_laya", "B2_s2"]})
    results, audit_events = [], []
    output.mkdir(parents=True)
    write_json(output / "manifest.json", {"source_run": read_json(root / "manifest.json")["fingerprint"],
        "execution_mode": config["execution_mode"], "seed": seed, "sample": [r["trial_id"] for r in sample],
        "sample_kind": "challenge_corrupted_decisions" if challenge else "random_recorded_decisions", "readiness": readiness})
    try:
        for item in sample:
            view = public[item["case_id"]]
            prediction = item["prediction"]
            if challenge:
                prediction = LABELS[(LABELS.index(prediction) + 1) % len(LABELS)]
            trace = [e for e in read_jsonl(root / item["events_ref"]) if e["kind"] in {
                "plan", "task_result", "aggregation", "route", "model_failure", "validation_failure"}]
            for condition in CONDITIONS:
                g = Governor(models, config["budgets"], seed)
                visible_trace = trace if condition != "E0_posthoc" else []
                probes = safe_probes(view) if condition == "E2_probe" else {}
                details = {"selected_decision": prediction, "trace": visible_trace,
                           "probes": {name: probe.payload() for name, probe in probes.items()}}
                row = dict(trial_id=item["trial_id"], case_id=view.case_id, condition=condition,
                           execution_mode=config["execution_mode"], selected_decision=prediction,
                           sample_kind="challenge" if challenge else "random", status="COMPLETED")
                try:
                    first, _ = g.request("s2", "audit", "Audit the recorded decision; disagreement is permitted.",
                        view, (), details, lambda content: validate_audit(content, view, visible_trace))
                    g.event("frozen_probe_predictions", predictions=first["probe_predictions"])
                    outcomes = {}
                    for name, probe in probes.items():
                        decision = Policy("B1_laya", g).static(probe)
                        outcomes[name] = probe.meaning(decision)
                    final = first
                    if probes:
                        final, _ = g.request("s2", "audit", "Reassess with recorded probe outcomes.", view, (),
                            details | {"probe_outcomes": outcomes, "initial_audit": first},
                            lambda content: validate_audit(content, view, visible_trace))
                    row.update(audit=final, initial_audit=first, probe_outcomes=outcomes,
                        predicted_probe_accuracy=sum(first["probe_predictions"].get(k) == v for k, v in outcomes.items()) / len(outcomes) if outcomes else None)
                except Failure as exc:
                    row.update(status=exc.status, error=str(exc))
                row["usage"] = g.budget.used
                results.append(row)
                audit_events.extend(e | {"trial_id": item["trial_id"], "condition": condition} for e in g.events)
    finally:
        for model in models.values():
            model.close()
    # Private labels become available only AFTER all audit inference has finished.
    private = {r["case_id"]: r for r in read_jsonl(Path(config["dataset"]) / "private" / "labels.jsonl")}
    for row in results:
        if row["status"] == "COMPLETED":
            item = next(x for x in sample if x["trial_id"] == row["trial_id"])
            visible_events = [] if row["condition"] == "E0_posthoc" else [e for e in read_jsonl(root / item["events_ref"]) if e["kind"] in {
                "plan", "task_result", "aggregation", "route", "model_failure", "validation_failure"}]
            row["scores"] = score_audit(row["audit"], public[row["case_id"]], visible_events,
                                         row["selected_decision"], private[row["case_id"]])
    write_jsonl(output / "audit_results.jsonl", results)
    write_jsonl(output / "events.jsonl", audit_events)
    write_json(output / "checksums.json", checksum_tree(output))
    return output


def matched_replay(run_path, destination, limit=6):
    root, output = Path(run_path), Path(destination)
    verify_tree(root)
    if output.exists():
        raise ValueError("Replay bundles are immutable")
    config = read_json(root / "resolved_config.json")
    public = {r["case_id"]: view_from_dict(r) for r in read_jsonl(Path(config["dataset"]) / "public.jsonl")}
    candidates = sorted([r for r in read_jsonl(root / "predictions.jsonl") if r["policy_id"] == "H2_mixed" and r["benchmark"] != "alfworld"], key=lambda r: r["trial_id"])
    models, readiness = build_models(config | {"policies": ["H2_mixed", "C1_s2_workers"]})
    results, events = [], []
    output.mkdir(parents=True)
    try:
        for original in candidates[:limit]:
            trace = read_jsonl(root / original["events_ref"])
            plans = [e for e in trace if e["kind"] == "plan"]
            if not plans:
                continue
            frozen = plans[0]["plan"]
            prefix = trace[:trace.index(plans[0])]
            plan_calls = sum(e["kind"] == "model_request" for e in prefix)
            plan_tokens = sum(e.get("output_tokens", 0) or 0 for e in prefix if e["kind"] == "model_response")
            for policy_id in ("H2_mixed", "C1_s2_workers"):
                g = Governor(models, config["budgets"], original["seed"])
                row = dict(case_id=original["case_id"], policy_id=policy_id, plan_hash=digest(frozen),
                           execution_mode=config["execution_mode"], timing_kind="cached_plan_worker_only",
                           allocated_initial_plan_calls=plan_calls, allocated_initial_plan_tokens=plan_tokens)
                try:
                    choice, _ = Policy(policy_id, g).execute_plan(public[original["case_id"]], frozen)
                    row.update(prediction=public[original["case_id"]].meaning(choice), status="COMPLETED")
                except Failure as exc:
                    row.update(prediction=None, status=exc.status, error=str(exc))
                row["worker_usage"] = g.budget.used
                row["hypothetical_total_s2_calls"] = plan_calls + g.budget.used["s2_calls"]
                results.append(row)
                events.extend(e | {"case_id": original["case_id"], "policy_id": policy_id} for e in g.events)
    finally:
        for model in models.values():
            model.close()
    write_json(output / "manifest.json", {"source_run": read_json(root / "manifest.json")["fingerprint"], "readiness": readiness,
               "execution_mode": config["execution_mode"], "not_live_end_to_end_timing": True})
    write_jsonl(output / "replay_results.jsonl", results)
    write_jsonl(output / "events.jsonl", events)
    write_json(output / "checksums.json", checksum_tree(output))
    return output
