"""Copy a small, reproducible review snapshot from a completed synthetic study.

Full traces remain in the ignored artifacts directory; one example of each
policy is included here so review does not require rerunning the harness.
"""
import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1s2lab.storage import read_json, read_jsonl, write_json, write_jsonl, file_hash, checksum_tree


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--out", default="reports/mock-study")
    args = p.parse_args()
    source, target = Path(args.study), Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    fingerprints = {}
    for benchmark in ("sharc", "context_rules", "alfworld"):
        run = source / benchmark
        manifest = read_json(run / "manifest.json")
        if manifest["execution_mode"] != "mock":
            raise ValueError("This promotion script is only for synthetic results")
        output = target / benchmark
        output.mkdir(exist_ok=True)
        for name in ("report.md", "metrics.json", "comparisons.csv", "per_case_results.csv", "readiness.json", "manifest.json", "resolved_config.json"):
            shutil.copyfile(run / name, output / name)
        predictions = read_jsonl(run / "predictions.jsonl")
        seen, examples, traces = set(), [], []
        for row in predictions:
            if row["policy_id"] in seen:
                continue
            seen.add(row["policy_id"])
            examples.append(row)
            traces.extend(e | {"trial_id": row["trial_id"], "policy_id": row["policy_id"]} for e in read_jsonl(run / row["events_ref"]))
        write_jsonl(output / "example_predictions.jsonl", examples)
        write_jsonl(output / "example_traces.jsonl", traces)
        fingerprints[benchmark] = manifest["fingerprint"]
    for name in ("study_comparisons.json",):
        shutil.copyfile(source / name, target / name)
    for name, result_file in (("matched_replay", "replay_results.jsonl"), ("audit_random", "audit_results.jsonl"), ("audit_challenge", "audit_results.jsonl")):
        directory = target / name
        directory.mkdir(exist_ok=True)
        for filename in ("manifest.json", result_file):
            shutil.copyfile(source / name / filename, directory / filename)
    write_json(target / "source_runs.json", fingerprints)
    (target / "README.md").write_text(
        "# SYNTHETIC RESULTS — REVIEW SNAPSHOT\n\n"
        "No real Laya, DeepSeek, or official benchmark execution is represented here. "
        "The scripted smoke study produced 176 trial records and exercised all eight policies.\n\n"
        "- [ShARC-schema fixture report](sharc/report.md)\n"
        "- [ContextRules report](context_rules/report.md)\n"
        "- [Synthetic interactive report](alfworld/report.md)\n"
        "- [Three-endpoint Holm comparison](study_comparisons.json)\n"
        "- [Matched-plan replay](matched_replay/replay_results.jsonl)\n"
        "- [Random audits](audit_random/audit_results.jsonl) and [challenge audits](audit_challenge/audit_results.jsonl)\n\n"
        "Each benchmark folder includes metrics, per-case CSV, comparisons, a manifest, and one saved request/response trace "
        "per policy in `example_traces.jsonl`. References to `trials/` and full aggregate files in the generated reports "
        "refer to the complete ignored run bundle. Reproduce it with:\n\n"
        "```console\npython -m s1s2lab.cli demo --out artifacts/mock-study-v2\n```\n\n"
        "See [verification.json](verification.json) for actual test/readiness evidence. "
        "Mock service and token measurements are synthetic plumbing checks, not performance claims.\n", encoding="utf-8")
    write_json(target / "checksums.json", checksum_tree(target))


if __name__ == "__main__":
    main()
