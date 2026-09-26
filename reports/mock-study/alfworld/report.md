# SYNTHETIC RESULTS — SCRIPTED MOCKS ONLY

These numbers test software plumbing. They are **not empirical Laya or DeepSeek performance**.

Run fingerprint: `b3f57459e3d639f76ffe99b5e23df538abbe4e3e330408ce484ae1443a82e8b0`

Stage: `smoke` · partition: `development` · context: `full_coverage`

Dataset provenance: `{'execution_mode': 'mock', 'interface': 'synthetic_admissible_commands'}`

| Benchmark / policy | Primary endpoint | Correct / intended | S1 calls | S2 calls | Failures |
|---|---:|---:|---:|---:|---:|
| alfworld/B0_simple | 1.000 | 2 / 2 | 0 | 0 | 0 |
| alfworld/B1_laya | 1.000 | 2 / 2 | 6 | 0 | 0 |
| alfworld/B2_s2 | 1.000 | 2 / 2 | 0 | 6 | 0 |
| alfworld/B3_cascade | 1.000 | 2 / 2 | 6 | 0 | 0 |
| alfworld/B4_triage | 1.000 | 2 / 2 | 12 | 0 | 0 |
| alfworld/H1_plan_once | 1.000 | 2 / 2 | 6 | 2 | 0 |
| alfworld/H2_mixed | 1.000 | 2 / 2 | 6 | 2 | 0 |
| alfworld/C1_s2_workers | 1.000 | 2 / 2 | 0 | 8 | 0 |

Primary endpoints: ShARC four-class balanced accuracy; ContextRules all-variants-correct family rate; ALFWorld environment success.

## Review a decision

Open `per_case_results.csv` to find a case, then use `predictions.jsonl` to locate its immutable `trials/<id>/` directory. Each model request, response, validated plan, worker result, and deterministic aggregation is recorded in `events.jsonl`.

## Comparisons

Differences are paired within cases and seeds. Bootstrap and permutation units are whole families. A missing class can make a ShARC bootstrap interval undefined; it is not replaced with zero. The three-benchmark study file applies Holm correction. Secondary comparisons remain exploratory.

| Comparison (right minus Laya) | Difference | 95% interval | Unadjusted p |
|---|---:|---|---:|
| alfworld / B0_simple | 0.0 | [0.0, 0.0] | 1.0 |
| alfworld / B2_s2 | 0.0 | [0.0, 0.0] | 1.0 |
| alfworld / B3_cascade | 0.0 | [0.0, 0.0] | 1.0 |
| alfworld / B4_triage | 0.0 | [0.0, 0.0] | 1.0 |
| alfworld / H1_plan_once | 0.0 | [0.0, 0.0] | 1.0 |
| alfworld / H2_mixed | 0.0 | [0.0, 0.0] | 1.0 |
| alfworld / C1_s2_workers | 0.0 | [0.0, 0.0] | 1.0 |

## Reproducibly selected failure examples

The first five incorrect records in case/policy/seed order (not selected for a favorable conclusion):


## Conditions and limitations

- Shared input exclusions: 0; see `exclusions.jsonl`.
- Full raw observations are retained; long contexts fail explicitly. No windowing or hidden summaries.
- Local inference is not free compute. Tokens and calls are different resource units, not matched FLOPs.
- Energy and host peak RAM are not measured. GPU memory is recorded only when the backend supplies it.
- Model loading is separate in readiness metadata; these runs do not include a warmup study.
- ContextRules English templates have machine checks but await independent linguistic review.
- Audit citations do not establish internal explanation faithfulness.
- Small pilot samples and public-data contamination limit generalization.

See `manifest.json`, `readiness.json`, `usage.jsonl`, `metrics.json`, and `checksums.json` for the recorded conditions.
