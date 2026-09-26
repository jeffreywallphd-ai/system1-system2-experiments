# SYNTHETIC RESULTS — REVIEW SNAPSHOT

No real Laya, DeepSeek, or official benchmark execution is represented here. The scripted smoke study produced 176 trial records and exercised all eight policies.

- [ShARC-schema fixture report](sharc/report.md)
- [ContextRules report](context_rules/report.md)
- [Synthetic interactive report](alfworld/report.md)
- [Three-endpoint Holm comparison](study_comparisons.json)
- [Matched-plan replay](matched_replay/replay_results.jsonl)
- [Random audits](audit_random/audit_results.jsonl) and [challenge audits](audit_challenge/audit_results.jsonl)

Each benchmark folder includes metrics, per-case CSV, comparisons, a manifest, and one saved request/response trace per policy in `example_traces.jsonl`. References to `trials/` and full aggregate files in the generated reports refer to the complete ignored run bundle. Reproduce it with:

```console
python -m s1s2lab.cli demo --out artifacts/mock-study-v2
```

See [verification.json](verification.json) for actual test/readiness evidence. Mock service and token measurements are synthetic plumbing checks, not performance claims.
