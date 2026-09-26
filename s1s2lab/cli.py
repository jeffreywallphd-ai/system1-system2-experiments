"""Repository-native CLI. `demo` and ordinary tests need only Python 3.11+."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from .config import load, validate, POLICIES
from .storage import read_json, write_json, read_jsonl


def mock_config(dataset, benchmark, root):
    return validate(dict(schema_version=1, experiment_id=f"mock_{benchmark}", execution_mode="mock", stage="smoke",
        policies=list(POLICIES), seeds=[17], dataset=str(Path(dataset).resolve()), partition="development",
        context_track="full_coverage", budgets={"s2_calls": 110, "generated_tokens": 450560} if benchmark == "alfworld" else {},
        routing_artifact=str((root / "gate.json").resolve()), baseline_artifact=str((root / "majority.json").resolve()),
        analysis={"resamples": 200, "seed": 812}))


def demo(destination):
    from .benchmarks.prepare import prepare_mock
    from .calibration import fit_gate, majority
    from .models.fake import ScriptedModel
    from .runner import run
    from .evaluation import score_run, compare_study
    from .reporting import report_run
    from .auditing import matched_replay, audit_run
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_root = root / "datasets"
    paths = {name: data_root / name for name in ("context_rules", "sharc", "alfworld")}
    if not data_root.exists():
        paths = prepare_mock(data_root)
    if not (root / "gate.json").exists():
        calibration, development = [], []
        for i, label in enumerate(("YES", "NO", "IRRELEVANT", "MORE")):
            row = dict(case_id=f"fit-cal-{i}", family_id=f"cal-family-{i}", partition="calibration", gold=label,
                       fast_prediction="YES", probabilities={x: 0.7 if x == "YES" else 0.1 for x in ("YES", "NO", "IRRELEVANT", "MORE")})
            calibration.append(row)
            development.append(row | dict(case_id=f"fit-dev-{i}", family_id=f"dev-family-{i}", partition="development", slow_prediction=label))
        write_json(root / "gate.json", fit_gate(calibration, development, ScriptedModel("s1").identity))
        write_json(root / "majority.json", majority([dict(case_id="mock-training-example", family_id="mock-training-family", partition="train", gold="YES")]))
    runs = []
    for benchmark, dataset in paths.items():
        target = root / benchmark
        config = mock_config(dataset, benchmark, root)
        run(config, target)
        score_run(target)
        report_run(target)
        runs.append(target)
    compare_study(runs, root / "study_comparisons.json")
    if not (root / "matched_replay").exists():
        matched_replay(root / "context_rules", root / "matched_replay", limit=2)
    if not (root / "audit_random").exists():
        audit_run(root / "context_rules", root / "audit_random", limit=2)
    if not (root / "audit_challenge").exists():
        audit_run(root / "context_rules", root / "audit_challenge", limit=2, challenge=True)
    lines = ["# SYNTHETIC STUDY — NO REAL MODEL INFERENCE", "",
             "This reproducible smoke study exercises all eight policies, three benchmark interfaces, paired comparisons, matched-plan replay, and three audit conditions. Its scripted outputs are not model capability evidence.", "",
             "| Benchmark | Report | Decisions and traces |", "|---|---|---|"]
    for benchmark in paths:
        lines.append(f"| {benchmark} | [Report]({benchmark}/report.md) | [{benchmark}/]({benchmark}/) |")
    lines += ["", "The Holm-adjusted three-endpoint comparison is in `study_comparisons.json`. Audit and replay outputs have separate manifests and usage. Run the same demo command again to verify/resume without duplicating completed trials.", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return root / "README.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="Complete offline synthetic study")
    p.add_argument("--out", default="artifacts/mock-study-v2")
    p = sub.add_parser("freeze", help="Freeze explicit real settings, data, code, and fitting/model locks")
    p.add_argument("--config", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--out", required=True)
    for name in ("validate", "plan-run", "doctor", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "doctor":
            p.add_argument("--load-models", action="store_true")
        if name == "run":
            p.add_argument("--out", required=True)
            p.add_argument("--real", action="store_true", help="Explicitly opt into local model execution")
    p = sub.add_parser("prepare")
    p.add_argument("--benchmark", choices=("context_rules", "sharc", "alfworld"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--families", type=int, default=60)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--release-manifest")
    p.add_argument("--family-disjoint", action="store_true", help="Separately labelled ShARC generalization track")
    for name in ("score", "report"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    for name in ("audit", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--limit", type=int, default=6)
        p.add_argument("--real", action="store_true")
        if name == "audit":
            p.add_argument("--challenge", action="store_true")
    for name in ("calibrate", "calibrate-actions"):
        p = sub.add_parser(name)
        p.add_argument("--calibration-rows", required=True)
        p.add_argument("--development-rows", required=True)
        p.add_argument("--model-identity", required=True)
        p.add_argument("--out", required=True)
    p = sub.add_parser("fit-majority")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("lock-runtime")
    p.add_argument("--out", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--role", choices=("s1", "s2"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--subfolder")
    p.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            print(demo(args.out))
        elif args.command == "freeze":
            from .runner import freeze_configuration
            print(freeze_configuration(load(args.config), args.protocol, args.out))
        elif args.command in {"validate", "plan-run", "doctor", "run"}:
            from .runner import doctor, run
            config = load(args.config)
            if args.command == "validate":
                print("Configuration schema is valid; capability readiness is checked separately.")
            elif args.command == "doctor":
                print_json(doctor(config, args.load_models))
            elif args.command == "plan-run":
                cases = read_jsonl(Path(config["dataset"]) / "cases.jsonl")
                count = sum(r["partition"] == config["partition"] for r in cases) * len(config["seeds"]) * len(config["policies"])
                print_json({"intended_trials": count, "s1_call_ceiling": count * config["budgets"]["s1_calls"],
                            "s2_call_ceiling": count * config["budgets"]["s2_calls"], "per_trial_budgets": config["budgets"]})
            else:
                if config["execution_mode"] == "real" and not args.real:
                    raise ValueError("Real inference requires the explicit --real flag")
                print(run(config, args.out))
        elif args.command == "prepare":
            from .benchmarks.prepare import prepare_context, prepare_sharc, prepare_alfworld
            if args.benchmark == "context_rules":
                print(prepare_context(args.out, args.families, args.seed))
            else:
                if not args.release_manifest:
                    raise ValueError("An enumerated, checksummed release manifest is required")
                if args.benchmark == "sharc":
                    print(prepare_sharc(args.out, args.release_manifest, args.family_disjoint))
                else:
                    print(prepare_alfworld(args.out, args.release_manifest))
        elif args.command == "score":
            from .evaluation import score_run
            score_run(args.run)
            print(Path(args.run) / "metrics.json")
        elif args.command == "report":
            from .reporting import report_run
            print(report_run(args.run))
        elif args.command == "compare":
            from .evaluation import compare_study
            compare_study(args.runs, args.out)
            print(args.out)
        elif args.command in {"audit", "replay"}:
            from .auditing import audit_run, matched_replay
            if read_json(Path(args.run) / "resolved_config.json")["execution_mode"] == "real" and not args.real:
                raise ValueError("Real audit/replay inference requires --real")
            kwargs = {"challenge": args.challenge} if args.command == "audit" else {}
            print((audit_run if args.command == "audit" else matched_replay)(args.run, args.out, args.limit, **kwargs))
        elif args.command in {"calibrate", "calibrate-actions"}:
            from .calibration import fit_gate, fit_action_gate
            fit = fit_gate if args.command == "calibrate" else fit_action_gate
            write_json(args.out, fit(read_jsonl(args.calibration_rows), read_jsonl(args.development_rows), read_json(args.model_identity)))
            print(args.out)
        elif args.command == "fit-majority":
            from .calibration import majority
            rows = read_jsonl(Path(args.dataset) / "private" / "labels.jsonl")
            write_json(args.out, majority([r for r in rows if r["partition"] == "train"]))
            print(args.out)
        elif args.command == "lock-runtime":
            from .assets import lock_runtime
            print_json(lock_runtime(args.out))
        elif args.command == "fetch":
            license_name = "Apache-2.0" if args.role == "s1" else "MIT"
            print(f"Model license: {license_name}. Download may require multiple GB (8B BF16 weights alone: about 16.4 GB). Review the model card and available disk first.")
            if args.download:
                from .assets import fetch_model
                lock = fetch_model(args.role, args.out, args.lock, args.revision, args.subfolder)
                print(f"Locked revision: {lock['revision']}")
            else:
                print("No download performed. Add --download to explicitly fetch and lock assets.")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


def print_json(value):
    import json
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
