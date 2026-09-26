# SYNTHETIC RESULTS — SCRIPTED MOCKS ONLY

These numbers test software plumbing. They are **not empirical Laya or DeepSeek performance**.

Run fingerprint: `836d8f3e2b8803c7739a540df2e508956ac21d38ba64bc678ad5be557894bb49`

Stage: `smoke` · partition: `development` · context: `full_coverage`

Dataset provenance: `{'generator': 'context_rules_v1', 'seed': 17, 'requested_families': 60, 'achieved_families': 60, 'variants_per_family': 8, 'linguistic_review': 'not_yet_human_reviewed', 'execution_mode': 'mock', 'selection': 'first_two_development_family_hashes'}`

| Benchmark / policy | Primary endpoint | Correct / intended | S1 calls | S2 calls | Failures |
|---|---:|---:|---:|---:|---:|
| context_rules/B0_simple | 0.000 | 4 / 16 | 0 | 0 | 0 |
| context_rules/B1_laya | 0.000 | 4 / 16 | 16 | 0 | 0 |
| context_rules/B2_s2 | 0.000 | 4 / 16 | 0 | 16 | 0 |
| context_rules/B3_cascade | 0.000 | 4 / 16 | 16 | 16 | 0 |
| context_rules/B4_triage | 0.000 | 4 / 16 | 32 | 0 | 0 |
| context_rules/H1_plan_once | 0.000 | 4 / 16 | 16 | 16 | 0 |
| context_rules/H2_mixed | 0.000 | 4 / 16 | 16 | 16 | 0 |
| context_rules/C1_s2_workers | 0.000 | 4 / 16 | 0 | 32 | 0 |

Primary endpoints: ShARC four-class balanced accuracy; ContextRules all-variants-correct family rate; ALFWorld environment success.

## Review a decision

Open `per_case_results.csv` to find a case, then use `predictions.jsonl` to locate its immutable `trials/<id>/` directory. Each model request, response, validated plan, worker result, and deterministic aggregation is recorded in `events.jsonl`.

## Comparisons

Differences are paired within cases and seeds. Bootstrap and permutation units are whole families. A missing class can make a ShARC bootstrap interval undefined; it is not replaced with zero. The three-benchmark study file applies Holm correction. Secondary comparisons remain exploratory.

| Comparison (right minus Laya) | Difference | 95% interval | Unadjusted p |
|---|---:|---|---:|
| context_rules / B0_simple | 0.0 | [0.0, 0.0] | 1.0 |
| context_rules / B2_s2 | 0.0 | [0.0, 0.0] | 1.0 |
| context_rules / B3_cascade | 0.0 | [0.0, 0.0] | 1.0 |
| context_rules / B4_triage | 0.0 | [0.0, 0.0] | 1.0 |
| context_rules / H1_plan_once | 0.0 | [0.0, 0.0] | 1.0 |
| context_rules / H2_mixed | 0.0 | [0.0, 0.0] | 1.0 |
| context_rules / C1_s2_workers | 0.0 | [0.0, 0.0] | 1.0 |

## Reproducibly selected failure examples

The first five incorrect records in case/policy/seed order (not selected for a favorable conclusion):

- `107e033f50f690faa6df` / `B0_simple` / seed 17: gold `IRRELEVANT`, prediction `YES`, status `COMPLETED`.
- `107e033f50f690faa6df` / `B1_laya` / seed 17: gold `IRRELEVANT`, prediction `YES`, status `COMPLETED`.
- `107e033f50f690faa6df` / `B2_s2` / seed 17: gold `IRRELEVANT`, prediction `YES`, status `COMPLETED`.
- `107e033f50f690faa6df` / `B3_cascade` / seed 17: gold `IRRELEVANT`, prediction `YES`, status `COMPLETED`.
- `107e033f50f690faa6df` / `B4_triage` / seed 17: gold `IRRELEVANT`, prediction `YES`, status `COMPLETED`.

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
