# Implementation and evidence

The deliverable is a working research harness with runnable synthetic evidence. **No empirical Laya/DeepSeek benchmark result is included.** Software checks and capability measurements are different forms of evidence.

| Area from the brief | Implemented | Evidence / remaining work |
|---|---|---|
| Repository fit, contracts, configuration, runner | Standard-library Python; public contracts; strict modes; durable trials | Offline end-to-end study; config, budget, state, resume tests |
| Laya adapter | Fixed SDK/checkpoint loading, typed choices, normalized probabilities, actual encoding checks | Synthetic SDK contract tests; real inference still unexecuted |
| Reasoner adapter | Local Transformers, checkpoint template, final JSON parsing, full output-token accounting | Parser/process tests; real model load/generation unexecuted |
| Assets and manifests | Explicit resumable fetch; concrete revisions/file hashes; observed runtime locks; settings freeze | No weights downloaded; optional runtime input constraints are not a validated installation lock |
| Static comparisons | All eight policies; bounded graph contracts; deterministic aggregation; adaptive replan | Worker substitution and worker-influence tests; ShARC and ContextRules smoke scores |
| ShARC | Allowlisted preparation, answer normalization, official overlap audit, separate disjoint track | Synthetic schema fixtures; actual release preparation and external results unexecuted |
| ContextRules | Eight variants per read-once canonical family; depth/OOD split; independent oracle | Generated property/contrast tests; independent English review and final power study pending |
| ALFWorld | Text adapter, complete admissible actions, versioned memory, independent reset, killable process | Fake episode tests; official environment check unexecuted; no downloaded game assets |
| Calibration | Static temperature scaling; separate interactive observed-rollout gate; disjoint calibration/development | Partition/model-identity tests; real fitting observations and costs not yet available |
| Statistical analysis | Explicit denominators; class/family/pair metrics; paired cluster bootstrap/permutation; Holm | Analytic fixtures and synthetic study; no real effect inference |
| Auditing | E0/E1/E2, fixed probes, frozen probe predictions, bounded evidence support, separate challenge set | Synthetic audits; unrestricted claim entailment and semantic trace review remain human work |
| Mechanism diagnostics | Matched recorded-plan replay, leaf/worker/supervisor counters, perturbation tests | Cached replay labelled separately; live end-to-end causal evidence unexecuted |
| Reproducibility | Config/source/prompt/data/artifact bindings, trial checksums, refused mismatched resume | Full mock study reruns without duplicated trials |

## Deliberate scope choices

- Sequential logical workers, one question per S1 call, no performance claims from parallelism or batching.
- Complete context or an explicit context limit. No window aggregation, generated summaries, action shortlist, or hierarchy is substituted.
- Interactive subgoals expire on their allocated step count. The plan does not claim automatic semantic subgoal verification.
- No warmup study, energy instrumentation, CPU peak-memory measurement, or monetary-cost assumptions.
- No weight adaptation, learned triage, optional ProofWriter/ToolSandbox, or unrestricted agent tools. These were optional/later phases in the brief.
- Process isolation limits hangs and removes hidden state from requests; it is not a hostile-code security sandbox.
- Automatic fitting assumes trusted prepared partition metadata. It protects accidental test-label fitting but cannot prove that an externally authored file was truthfully labelled.

## Verification

The commands and counts for the delivered snapshot are recorded in [verification.json](../reports/mock-study/verification.json). Ordinary tests block socket connections in each offline fixture and import no heavyweight model packages. Real tests are opt-in and skipped by default. CI configuration covers Windows/Linux and Python 3.11/3.12; those hosted CI jobs were added but not executed during this local implementation.

The next scientific step is a **development-only real readiness pilot**: set up pinned snapshots/runtime, verify short and long encoding behavior, test structured-plan reliability, install/enumerate official data, gather actual calibration observations, and record the result even if model quality is poor. Freeze sample sizes and confirmatory settings only after that pilot.
