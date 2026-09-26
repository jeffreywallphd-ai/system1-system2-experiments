"""Human review views derived entirely from the saved, checksummed outputs."""
from pathlib import Path
from .storage import read_json, read_jsonl, write_json, checksum_tree, verify_tree


def report_run(run_path):
    root = Path(run_path)
    verify_tree(root)
    manifest = read_json(root / "manifest.json")
    config = manifest["binding"]["config"]
    scored = read_jsonl(root / "scored.jsonl")
    results = read_json(root / "metrics.json")["results"]
    mock = manifest["execution_mode"] == "mock"
    lines = ["# SYNTHETIC RESULTS — SCRIPTED MOCKS ONLY" if mock else "# Local model experiment results", "",
        "These numbers test software plumbing. They are **not empirical Laya or DeepSeek performance**." if mock else
        "These measurements apply only to the recorded checkpoints, inputs, interfaces, and budgets.", "",
        f"Run fingerprint: `{manifest['fingerprint']}`", "",
        f"Stage: `{config['stage']}` · partition: `{config['partition']}` · context: `{config['context_track']}`", "",
        f"Dataset provenance: `{manifest['binding']['dataset_manifest']['provenance']}`", "",
        "| Benchmark / policy | Primary endpoint | Correct / intended | S1 calls | S2 calls | Failures |",
        "|---|---:|---:|---:|---:|---:|"]
    for key, m in results.items():
        endpoint = f"{m['primary_endpoint']:.3f}" if m["primary_endpoint"] is not None else "undefined"
        correct = round(m["accuracy"] * m["intended"]) if m["accuracy"] is not None else 0
        lines.append(f"| {key} | {endpoint} | {correct} / {m['intended']} | {m['resource_totals'].get('s1_calls', 0)} | {m['resource_totals'].get('s2_calls', 0)} | {sum(m['failures'].values())} |")
    lines += ["", "Primary endpoints: ShARC four-class balanced accuracy; ContextRules all-variants-correct family rate; ALFWorld environment success.", "",
              "## Review a decision", "", "Open `per_case_results.csv` to find a case, then use `predictions.jsonl` to locate its immutable `trials/<id>/` directory. Each model request, response, validated plan, worker result, and deterministic aggregation is recorded in `events.jsonl`.", "",
              "## Comparisons", "", "Differences are paired within cases and seeds. Bootstrap and permutation units are whole families. A missing class can make a ShARC bootstrap interval undefined; it is not replaced with zero. The three-benchmark study file applies Holm correction. Secondary comparisons remain exploratory.", "",
              "| Comparison (right minus Laya) | Difference | 95% interval | Unadjusted p |",
              "|---|---:|---|---:|"]
    for c in read_json(root / "comparisons.json"):
        lines.append(f"| {c['benchmark']} / {c['right']} | {c['difference']} | {c['ci95']} | {c['p_value']} |")
    failures = sorted([r for r in scored if not r["correct"]], key=lambda r: (r["case_id"], r["policy_id"], r["seed"]))[:5]
    lines += ["", "## Reproducibly selected failure examples", "", "The first five incorrect records in case/policy/seed order (not selected for a favorable conclusion):", ""]
    for row in failures:
        lines.append(f"- `{row['case_id']}` / `{row['policy_id']}` / seed {row['seed']}: gold `{row.get('gold')}`, prediction `{row['prediction']}`, status `{row['status']}`.")
    exclusions = read_jsonl(root / "exclusions.jsonl")
    lines += ["", "## Conditions and limitations", "", f"- Shared input exclusions: {len(exclusions)}; see `exclusions.jsonl`.",
              "- Full raw observations are retained; long contexts fail explicitly. No windowing or hidden summaries.",
              "- Local inference is not free compute. Tokens and calls are different resource units, not matched FLOPs.",
              "- Energy and host peak RAM are not measured. GPU memory is recorded only when the backend supplies it.",
              "- Model loading is separate in readiness metadata; these runs do not include a warmup study.",
              "- ContextRules English templates have machine checks but await independent linguistic review.",
              "- Audit citations do not establish internal explanation faithfulness.",
              "- Small pilot samples and public-data contamination limit generalization.", "",
              "See `manifest.json`, `readiness.json`, `usage.jsonl`, `metrics.json`, and `checksums.json` for the recorded conditions.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(root / "checksums.json", checksum_tree(root))
    return root / "report.md"
