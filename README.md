# System 1 / System 2 experiments

An inspectable experiment harness for a fixed **Laya** decision model and **DeepSeek-R1-0528-Qwen3-8B**, based on the supplied [experiment brief](docs/experiment-brief.md). It compares direct decisions, routing, decomposition, and adaptive supervision using the same public evidence and governed execution interface.

**Current evidence:** the offline study and correctness tests run successfully. The included results use **scripted fake models**, not Laya or DeepSeek inference. Real adapters are implemented but need pinned local assets and opt-in capability checks. No model weights or benchmark archives were downloaded for this implementation, and no paid API was used.

## Start here

From the repository root, with Python 3.11 or newer:

```console
python -m unittest discover -s tests -v
python -m s1s2lab.cli doctor --config configs/mock.json
python -m s1s2lab.cli demo --out artifacts/mock-study-v2
```

No installation, network, GPU, or third-party Python dependency is needed. The demo creates **176 trial records**: eight policies × 16 ContextRules variants, four synthetic ShARC-schema cases, and two synthetic interactive episodes, with one seed. It also runs matched-plan replay and all three audit conditions on small samples. The same demo command resumes completed work without new inference.

Open `artifacts/mock-study-v2/README.md` for the study index. The [checked-in review snapshot](reports/mock-study/README.md) provides a compact version of the generated results.

## Experimental design

The scientific question is whether bounded decisions by a small non-generative model and decomposition by an 8B reasoner offer complementary computation. “System 1” and “System 2” are operational role names, not a theory of human cognition. A hybrid is allowed to lose.

| Policy | Behavior | What it controls or tests |
|---|---|---|
| `B0_simple` | Training-majority static label; seeded random admissible action | Simple reference |
| `B1_laya` | Laya decisions; repeated action selection when interactive | Fast-model baseline |
| `B2_s2` | Reasoner direct decisions or next-action choices | Reasoner-only baseline |
| `B3_cascade` | Laya first; escalate below a fitted confidence gate | Value beyond simple escalation |
| `B4_triage` | Laya chooses FAST or REASON before execution | The proposed triage mechanism |
| `H1_plan_once` | One reasoning plan, then eligible Laya workers | Value of an initial decomposition |
| `H2_mixed` | Bounded reasoning plans and mixed workers; limited replanning | Adaptive delegation |
| `C1_s2_workers` | Same graph contract, every model leaf uses the reasoner | Decomposition without mixed model types |

Static plans contain propositions, evidence references, dependencies, and a small `AND`/`OR`/`NOT`/`IDENTITY` combination language. They cannot contain executable code or completed worker answers. Laya assesses leaves as supported, contradicted, or unknown; deterministic code combines their outputs. A relevance branch can produce `IRRELEVANT`. The supervisor can still propose a semantically wrong graph: this is an experimental failure, not something the evaluator repairs.

Interactive plans contain subgoals, worker assignments, and step allocations. Workers choose only from the current admissible commands. Observable failed actions, repeated transitions, or expired allocations can trigger bounded replanning. The environment owns success. Subgoal expiration is an allocation rule, not a verified completion judgment.

### Benchmarks and endpoints

| Benchmark | Public input | Primary endpoint |
|---|---|---|
| ShARC decision-only adaptation | Rule, question, scenario, allowed clarification history | Four-class balanced accuracy: YES / NO / IRRELEVANT / MORE |
| ContextRules, a new synthetic diagnostic | Rendered rules, facts, explicit corrections, question | Fraction of canonical rule families with **every variant correct** |
| ALFWorld text interaction | Goal, observations, accumulated public history, full admissible action list | Environment-defined episode success within the action budget |

ContextRules supports read-once, independent Boolean predicates, nesting depths 1–4, explicit negation, and unknown facts. Every generated family has a base, certified decisive flip, necessary-fact removal, distractor, paraphrase, explicit correction, option-order permutation, and irrelevant query. A second exhaustive-completion evaluator checks the oracle. Family splits use canonical expression structure; renaming variables does not create independent examples. Depth-four shapes form a separately labelled structural OOD partition.

ShARC uses a recursive allowlist and never passes source `answer` or `evidence` fields to policies. Official split overlap is reported; `prepare --family-disjoint` creates a separately labelled generalization track. ALFWorld uses the official text backend with admissible commands, so results are not directly comparable to unrestricted command-generation scores. Seen and unseen partitions are run and reported separately.

### Evidence and resource controls

- Private labels, ASTs, mutations, and administrative identifiers are separated from model-visible payloads. Policies have no evaluator or environment handle.
- Real models load only local, checksummed snapshots. There is no hidden checkpoint switch, automatic router, endpoint fallback, model download, or agent shell tool.
- Model calls reserve generation budget before execution. Failed calls and repairs count; unknown usage keeps its full reservation and receives a flag.
- Model and real-environment processes can be terminated on timeout. State versions and action-set hashes prevent stale selections from executing.
- Laya preflight compares its actual encoding with an untruncated encoding and checks the SDK's individual-option cap. The reasoner uses the selected snapshot's own chat template. No silent truncation or uncharged summaries.
- `common_fit` eligibility uses input encodings only. A failing ContextRules variant excludes its whole family. Dynamic interactive context failures remain outcomes in `full_coverage`.
- Every requested prediction, including invalid, failed, and missing results, stays in the scoring denominator. Calls/tokens are separate resource units, not matched FLOPs or zero local cost.

### Calibration, statistics, and mechanism studies

Calibration uses a fixed temperature grid over actual option probabilities on a calibration partition. A separate development partition selects a cascade threshold using accuracy minus a declared escalation penalty. Artifacts bind model identity, class order, fitting examples, and fitting families. Missing calibration disables the cascade with `NOT_READY` instead of inventing a gate. The four-way calibration implementation is for static decisions; **a real interactive cascade needs a separate task-appropriate gate study**. The mock interactive gate is solely an interface fixture.

Comparisons pair the same cases and seeds. Bootstrap and permutation units are whole families, with repeated seeds averaged within the family endpoint. The three primary mixed-versus-Laya comparisons receive Holm correction. Small samples that cannot support a ShARC bootstrap interval report it as undefined. Other comparisons remain exploratory. No non-inferiority margin or adequate sample size has been established yet.

Matched-plan replay freezes an actual recorded plan and runs it with mixed and reasoner workers. It reports worker-only measurements and allocates initial plan-generation calls/tokens to both hypothetical deployments. It is labelled as cached-plan replay, not live deployment timing.

The audit study compares `E0_posthoc`, `E1_trace`, and `E2_probe`. Probe predictions are frozen before additional Laya calls; outcomes can inform a separately recorded reassessment. Random samples and corrupted-decision challenge samples stay separate. Reference validity, exact quotation, and bounded ContextRules decision support are distinct. Unrestricted prose entailment and internal causal explanations remain unverified. Audits cannot change primary predictions.

## How to review the code

The package uses ordinary dataclasses, JSON, and explicit control flow. Start with these files:

| File | Responsibility |
|---|---|
| [domain.py](s1s2lab/domain.py) | Public contracts and the sole case-to-model serialization path |
| [policies.py](s1s2lab/policies.py) | All eight experimental policies in one place |
| [plans.py](s1s2lab/plans.py) | Graph validation and finite combination language |
| [governor.py](s1s2lab/governor.py) | Admission, reservations, repairs, events, stale-action rejection |
| [runner.py](s1s2lab/runner.py) | Trial lifecycle, independent resets, resumability, artifacts |
| [benchmarks/](s1s2lab/benchmarks/) | Preparation, public/private separation, oracle, environment boundary |
| [models/](s1s2lab/models/) | Scripted fixtures, real adapters, process isolation, prompts |
| [evaluation.py](s1s2lab/evaluation.py) | Metrics, family-clustered comparisons, Holm correction |
| [calibration.py](s1s2lab/calibration.py) | Fitting restrictions and routing provenance |
| [auditing.py](s1s2lab/auditing.py) | Evidence audits, safe probes, matched-plan replay |
| [tests/](tests/) | Offline invariants, adapter contracts, end-to-end study, opt-in real checks |

The [research protocol](docs/research-protocol.md), [architecture decisions](docs/architecture-decisions.md), and [implementation status](docs/implementation-status.md) explain design choices and remaining validation. The [runbook](docs/runbook.md) covers setup and individual commands.

## How to review the results

1. Read a run's `report.md`, then `manifest.json` and `readiness.json` to establish what actually ran.
2. Sort `per_case_results.csv` by policy, status, or correctness. `metrics.json` retains confusion matrices, pair outcomes, coverage, and resource totals.
3. Find the selected row in `predictions.jsonl`. Its `events_ref` points to an immutable `trials/<id>/events.jsonl` journal containing public requests, final responses, plans, leaf outputs, routes, and aggregation.
4. Inspect `comparisons.csv` and `study_comparisons.json`. Look at effect sizes and intervals alongside adjusted p-values.
5. Check `exclusions.jsonl`, `errors.jsonl`, `usage.jsonl`, and `encoding_audits.jsonl`. Absent predictions are not silently removed.

Manifests bind configuration, source and prompt hashes, data hashes, model/runtime locks, routing artifacts, seeds, and hardware metadata. Trial checksums are written atomically last. Changed code/configuration/data refuses resume. Interrupted attempts remain recorded and are not automatically retried. Large bundles are ignored by Git; the small [review snapshot](reports/mock-study/README.md) is included.

## Real execution and current limits

The [low-VRAM](configs/local_low_vram.json) and [reference](configs/reference.json) profiles are explicit **setup templates**, with unresolved revisions. Low-VRAM places Laya on CPU and requests NF4 reasoner inference; it does not promise that an 8 GB GPU is sufficient. Reference measurements use a separate precision and budget series.

Run `doctor` first and follow the [runbook](docs/runbook.md). Fetching requires `fetch --download`; real runs require `run --real`. Paid or remote endpoints are not implemented. Confirmatory configurations reject floating revisions, missing locks, and unresolved gates. Final sample/prompt/budget freezing and real capability review are still required before a scientific study.

Optional weight adaptation, ProofWriter, ToolSandbox, human claim adjudication, learned triage training, action hierarchies, and windowed Laya inference are deferred. Passing the offline tests does not demonstrate model quality.
