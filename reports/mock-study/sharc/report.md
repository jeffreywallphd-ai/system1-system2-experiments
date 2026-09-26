# SYNTHETIC RESULTS — SCRIPTED MOCKS ONLY

These numbers test software plumbing. They are **not empirical Laya or DeepSeek performance**.

Run fingerprint: `98b0da2baf604e61cf2e4d0409bb3e23c16751c55240c98750e85ff56d1d3db8`

Stage: `smoke` · partition: `development` · context: `full_coverage`

Dataset provenance: `{'execution_mode': 'mock', 'benchmark_variant': 'synthetic_sharc_schema_fixtures'}`

| Benchmark / policy | Primary endpoint | Correct / intended | S1 calls | S2 calls | Failures |
|---|---:|---:|---:|---:|---:|
| sharc/B0_simple | 0.250 | 1 / 4 | 0 | 0 | 0 |
| sharc/B1_laya | 0.250 | 1 / 4 | 4 | 0 | 0 |
| sharc/B2_s2 | 0.250 | 1 / 4 | 0 | 4 | 0 |
| sharc/B3_cascade | 0.250 | 1 / 4 | 4 | 4 | 0 |
| sharc/B4_triage | 0.250 | 1 / 4 | 8 | 0 | 0 |
| sharc/H1_plan_once | 0.250 | 1 / 4 | 4 | 4 | 0 |
| sharc/H2_mixed | 0.250 | 1 / 4 | 4 | 4 | 0 |
| sharc/C1_s2_workers | 0.250 | 1 / 4 | 0 | 8 | 0 |

Primary endpoints: ShARC four-class balanced accuracy; ContextRules all-variants-correct family rate; ALFWorld environment success.

## Review a decision

Open `per_case_results.csv` to find a case, then use `predictions.jsonl` to locate its immutable `trials/<id>/` directory. Each model request, response, validated plan, worker result, and deterministic aggregation is recorded in `events.jsonl`.

## Comparisons

Differences are paired within cases and seeds. Bootstrap and permutation units are whole families. A missing class can make a ShARC bootstrap interval undefined; it is not replaced with zero. The three-benchmark study file applies Holm correction. Secondary comparisons remain exploratory.

| Comparison (right minus Laya) | Difference | 95% interval | Unadjusted p |
|---|---:|---|---:|
| sharc / B0_simple | 0.0 | None | 1.0 |
| sharc / B2_s2 | 0.0 | None | 1.0 |
| sharc / B3_cascade | 0.0 | None | 1.0 |
| sharc / B4_triage | 0.0 | None | 1.0 |
| sharc / H1_plan_once | 0.0 | None | 1.0 |
| sharc / H2_mixed | 0.0 | None | 1.0 |
| sharc / C1_s2_workers | 0.0 | None | 1.0 |

## Reproducibly selected failure examples

The first five incorrect records in case/policy/seed order (not selected for a favorable conclusion):

- `fixture-1` / `B0_simple` / seed 17: gold `NO`, prediction `YES`, status `COMPLETED`.
- `fixture-1` / `B1_laya` / seed 17: gold `NO`, prediction `YES`, status `COMPLETED`.
- `fixture-1` / `B2_s2` / seed 17: gold `NO`, prediction `YES`, status `COMPLETED`.
- `fixture-1` / `B3_cascade` / seed 17: gold `NO`, prediction `YES`, status `COMPLETED`.
- `fixture-1` / `B4_triage` / seed 17: gold `NO`, prediction `YES`, status `COMPLETED`.

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
