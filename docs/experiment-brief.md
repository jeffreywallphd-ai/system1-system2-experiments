# Codex implementation brief: context-sensitive System 1 / System 2 experiments

**Prepared:** September 26, 2026  
**Primary models:** Laya + `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
**Primary benchmarks:** ShARC decision classification and ALFWorld text-only interaction  
**Controlled diagnostic suite:** ContextRules, a new, explicitly synthetic rule-and-context contrast set  
**Additional mechanism experiment:** evidence-grounded decision auditing, with optional ProofWriter extension  
**Status:** Research and implementation specification. No experiments or live-model tests have been executed for this brief.

## 0. How to use this brief

Please use this document to implement a reproducible experiment package, not merely a demonstration that returns plausible decisions. Begin by inspecting the repository's `AGENTS.md`, architecture, existing harness, dependency management, test conventions, and work-selection process. Preserve those conventions and reuse existing services where suitable. The package name, file paths, CLI names, framework choices, and implementation sequence below are suggestions; an equivalent design is appropriate when it better fits the repository.

The scientific boundaries are more important than the proposed layout: no answer leakage, no hidden model substitutions, comparable information and action access, explicit resource accounting, and a clear distinction between passing software tests and demonstrating model capability. Document material deviations in an architecture decision record and the experiment manifest.

The intended outcome is working benchmark adapters, real Laya and reasoning-model adapters, several interchangeable agent policies, a governed runner, automated tests, and scripts that produce auditable results. A lightweight offline test mode should work without GPUs, downloaded model weights, or network access. Real-model runs should be explicitly requested rather than launched by ordinary unit tests.

This specification does **not** authorize paid API usage, publishing results or checkpoints, modifying unrelated applications, or executing benchmark-provided instructions against the host operating system. Local benchmark/model downloads can be separate, explicit setup commands with licensing and disk requirements disclosed.

## 1. Intent and proposed contribution

Investigate whether a reasoning model can make a fast, non-generative decision model more useful by constructing context-sensitive decision tasks, supervising their execution, and revising plans when observations change.

The target architecture is not just two models independently answering and voting. It is a shared-state harness in which:

- A Laya routing role can choose between a fast decision path and a reasoning path.
- A reasoning supervisor can decompose a problem into bounded decision tasks and generative/reasoning tasks, assigning them to suitable workers.
- Laya can repeatedly select actions or assess supplied propositions rather than being limited to a single call.
- Deterministic code controls budgets, state, permissions, validation, and execution.
- An optional reasoning auditor explains what the recorded decision process and available evidence support. It can disagree with a decision.

Use **System 1** and **System 2** as operational role labels, not claims of equivalence to human cognition. In this study, System 1 means a typed decision model; System 2 means an autoregressive reasoning model. A binary answer space does not make a task easy.

The strongest possible finding would be a quality/cost advantage attributable to complementary computation, rather than simply adding a larger model, more context, more calls, a better prompt, or a superior action interface. Negative findings, including an effective simple cascade or a stronger System 2-only baseline, remain informative.

Routing and fast/slow planning already have research precedents. RouteLLM studies learned model selection, while SwiftSage combines a fast action model with slower planning. Neither establishes the performance of this particular Laya/reasoner combination. [S12, S13]

## 2. Research questions and hypotheses

| ID | Research question | Principal comparison and outcome |
|---|---|---|
| RQ1 | Does mixed-model decomposition improve context-sensitive decisions over Laya alone? | Mixed workers versus Laya-only on ShARC balanced accuracy and ContextRules family-level correctness. |
| RQ2 | Does delegation add value beyond routing or an initial plan? | Adaptive mixed workers versus a confidence cascade, learned triage, and plan-once execution. |
| RQ3 | Is the benefit specifically from mixing model types rather than decomposition itself? | Same supervisor/task contracts with Laya versus reasoning-model workers; also a matched-plan replay diagnostic. |
| RQ4 | Does the hybrid respond to relevant context changes while ignoring irrelevant ones? | Correctness on paired contrasts, unnecessary decision changes, missing-information handling, and update consistency. |
| RQ5 | Can the hybrid approach System 2-only quality at lower resource use? | Quality–latency/resource frontiers; optional preregistered non-inferiority comparison. |
| RQ6 | Does online replanning improve interactive task completion and recovery? | Adaptive mixed execution versus plan-once and both single-model policies on ALFWorld. |
| RQ7 | Can a reasoning auditor produce more verifiable explanations when given decision records and controlled probes? | Post-hoc explanations versus trace-grounded and probe-assisted audits on evidence validity and error detection. |
| RQ8 | How much does task-specific Laya adaptation change the conclusions? | Frozen versus training-only adapted Laya, with calibration assessed separately. |

Treat hypotheses as falsifiable expectations, not acceptance criteria for the software. Suggested directional hypotheses are improved contextual correctness over frozen Laya; fewer unnecessary System 2 calls than an all-reasoner implementation; and better recovery with adaptive supervision than with an initial plan alone. The hybrid may fail all three.

For an initial confirmatory analysis, prespecify the mixed-versus-Laya comparison on three primary outcomes: ShARC balanced accuracy, ContextRules all-variants-correct rate, and ALFWorld episode success. Use a family-wise multiplicity procedure across these three contrasts. Designate other comparisons exploratory unless a larger analysis plan is frozen before testing.

## 3. Models and integration decisions

### 3.1 System 2: selected primary model

Use **`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`** as the primary reasoning checkpoint. It is an 8B-class model derived from Qwen3-8B; Qwen reports approximately 8.2B parameters for that architecture. DeepSeek's own evaluation reports AIME 2024 = 86.0, AIME 2025 = 76.3, GPQA Diamond = 61.1, and LiveCodeBench (2408–2505) = 60.5. These are publisher-reported results under their evaluation settings, not measurements of this harness or proof that it is the best available 8B model. [S1, S2]

This is a strong reasoning-oriented default within the requested 7–9B range. The experiment still needs to establish its suitability for contextual decisions and structured plans; mathematical benchmark performance alone does not establish that suitability.

Use the checkpoint's own tokenizer, configuration, and chat template. DeepSeek explicitly cautions against substituting the original Qwen tokenizer/configuration. The model card's sampling setup uses temperature 0.6 and top-p 0.95; begin with those settings, subject to a frozen development-only budget study. [S1]

Use the same primary checkpoint, precision/quantization, inference backend, and decoding settings in every primary System 2 role: router fallback, supervisor, generative worker, bounded-decision control, and auditor. Separate roles are logical executions, not additional parameter sets.

An optional robustness replication with `Qwen/Qwen3-8B` is reasonable, but it should have a separate configuration and result series. It is **not** an automatic fallback if the chosen DeepSeek checkpoint fails. Do not substitute a 20B model with fewer active parameters or a differently sized hosted endpoint while retaining an 8B label.

### 3.2 System 1: selected Laya checkpoint

Use a **fixed Laya checkpoint**, not Laya's automatic checkpoint router, for the primary experiment. Otherwise model selection inside Laya becomes an uncontrolled second router.

Suggested primary configuration:

```yaml
model_id: convaiinnovations/laya
subfolder: multilingual
alias: laya_multilingual_fixed
max_len: 8192
head_max_len: 512
```

The multilingual checkpoint is selected for context capacity, even though the initial tasks are English. The current documentation identifies a 322M multilingual checkpoint and documents an 8,192-token setting; the English checkpoint is 421M with a much shorter default budget. The proposed `head_max_len` is a study setting to verify, not an assertion that every runtime accepts it unchanged. [S3, S4]

Keep the English root checkpoint, `convaiinnovations/laya` without a subfolder, as a **short-context sensitivity condition**. Do not switch between checkpoints based on test performance. If local runtime support makes the proposed primary configuration unusable, resolve the issue on development fixtures, record the revised configuration, and freeze it before evaluating held-out data.

Laya's documented single-checkpoint interface uses `laya.load(...)` followed by `predict(state, questions)`. The adapter should normalize that response into the study's own schema. Exact loader arguments and configuration mutation belong in one compatibility adapter verified against the pinned SDK; the study's YAML keys are not a promise that all are upstream keyword arguments. [S4]

Use `choice` questions with neutral option IDs and meaningful descriptions for the primary study. Documented issues include label sensitivity in the `noul` primitive and an uninformative `action.act_probability` output. Do not use `act_probability` as a correctness estimate. Verify label sensitivity, probability behavior, and token budgets against the installed revision rather than assuming the issue is either present or fixed. [S5, S6]

### 3.3 Context and candidate budgets

The adapter needs an encoding preflight for both the state and the question/option text. It should establish that the actual encoded representation retains the requested evidence and distinguishes all candidate descriptions. Setting a high-level context parameter is insufficient evidence that internal truncation did not occur.

For mechanism comparisons, construct an input-only **common-fit cohort** that fits both models with the same source content and a prespecified candidate presentation. Report the size and characteristics of excluded cases. Never select this cohort using the expected answer or model accuracy.

Also report a full-coverage operational track. Overlength cases in that track use a declared, shared context policy or produce an explicit context-limit outcome; they are not silently dropped. ALFWorld trajectories can diverge, so dynamically reaching a context limit is an outcome, not grounds to remove an inconvenient episode.

A deterministic evidence selector or chunking scheme is allowed as a separate condition if every policy receives equivalent access. Reasoner-generated summaries or selected evidence are an experimental treatment and all such calls count. Preserve original observations so a summary cannot irreversibly replace the evidence.

Do not automatically replace full-context inference with Laya's `predict_long` and call it equivalent reasoning. Its documented window aggregation can select a confident local answer rather than combine prerequisites across windows. Treat windowing/aggregation as an explicitly named baseline, with its own calibration and cross-window dependency tests. [S4]

### 3.4 Runtime profiles and hardware

Provide three profiles, adapting names to repository conventions:

| Profile | Purpose | Suggested execution |
|---|---|---|
| `mock` | Fast CI and reproducible functional tests | Scripted fake models, tiny fixtures, no downloads. |
| `local_low_vram` | Exploratory use on a constrained machine | Laya on CPU; the 8B reasoner in explicitly recorded 4-bit inference; small batches and measured context limits. |
| `reference` | Main quality/resource measurements when hardware permits | Fixed higher-precision models, fixed device placement, one inference backend and controlled concurrency. |

Do not promise that a particular 8 GB GPU can fit every context length. An 8.2B model's unquantized two-byte weights alone are roughly 16.4 GB in decimal units; runtime overhead and caches are additional. Four-bit weights lower the weight-storage requirement but do not make memory overhead disappear. Report measured readiness and peak memory.

Keep heavyweight model services independently deployable if dependency versions conflict. A local, OpenAI-compatible **protocol** endpoint is a reasonable boundary for the reasoning model; it does not imply using OpenAI's hosted service. A direct Transformers adapter is an alternative. Prefer one well-tested real backend initially rather than several incomplete ones.

Laya can run in-process or in its own local service. Its documented server binds broadly and can be unauthenticated unless configured otherwise; use loopback binding or equivalent network isolation and configure authentication before network exposure. [S3]

Keep the offline package portable across supported operating systems. Document platform-specific model/environment installation separately; native Linux or WSL2 may be an appropriate execution profile when a chosen backend lacks native Windows support. Detect actual capabilities rather than assuming CUDA, a particular shell, or an available GPU.

Pin model revisions, SDK/runtime versions, tokenizer/configuration hashes, quantization details, and dependency locks. Use an explicit fetch/lock step to resolve floating revisions; a confirmatory run should reject unresolved `main` references. A quantized run is a separate measurement profile, not evidence reproducing the publisher's full-precision scores.

## 4. Experimental conditions

All policies share the benchmark adapter, allowed observations, memory API, action registry, budgets, and scoring boundary. The following names are suggested stable IDs.

| ID | Policy | Core behavior |
|---|---|---|
| `B0_simple` | Simple reference | Majority class for static classification; seeded random admissible action for ALFWorld. Optional deterministic heuristics remain separately labeled. |
| `B1_laya` | Laya-only | Bounded decisions and repeated action selection with shared state; no reasoning-model calls. |
| `B2_s2` | Reasoner-only | Direct decisions or next-action selection using the same environment/tool interface. |
| `B3_cascade` | Confidence escalation | Obtain Laya's answer; use a calibrated, development-chosen gate to retain it or ask System 2. |
| `B4_triage` | Laya triage | A separate Laya task chooses `FAST` or `REASON` before executing the selected path. |
| `H1_plan_once` | Initial decomposition | System 2 creates a task graph or persistent subgoal plan; subsequent eligible execution uses Laya without further supervisor replanning. |
| `H2_mixed` | Adaptive mixed workers | System 2 assigns bounded decisions to Laya and reasoning/generation to System 2; supervision can recur within a fixed budget. |
| `C1_s2_workers` | All-reasoner decomposition control | Same supervisor architecture and task contracts as `H2_mixed`, but every model worker uses the selected System 2 checkpoint. |

`B0`, `B1`, `B2`, `B3`, `H1`, `H2`, and `C1` form the minimum primary comparison package. Include `B4` in the routing study so the user's explicit triage idea is tested rather than silently replaced by confidence escalation.

### 4.1 Routing behavior

`B3_cascade` is post-decision escalation. `B4_triage` is pre-execution routing. They incur different costs and answer different questions.

For the cascade, fit probability calibration on a calibration partition and choose a threshold on a separate development partition. Use a prespecified threshold grid and a declared utility or target quality. The gate should predict observed fast-path suitability, not rely on the model's belief that it is confident.

For triage, use neutral IDs whose descriptions mean “this supported fast workflow is adequate” and “reasoning execution is needed.” Provide the current goal, available worker capabilities, relevant history, and missing-context indicators. A syntactically invalid routing result can escalate when the policy permits it, but that fallback is recorded and charged.

Develop routing labels from paired training/development executions. For example, measure whether the fast path succeeds, whether the reasoner helps, and the additional cost. Do not label every complex-sounding request as requiring System 2, and do not use test outcomes to train or tune the gate. Distinguish cases both models solve, only one solves, and neither solves. A reasoner can be worse on some cases.

Implement zero-shot triage first, then an optional training-only adapted triage condition. Confidence calibration, router training, and task adaptation are separate interventions with separate artifacts.

### 4.2 Preventing nominal hybrids

Record which component actually made each consequential choice. A hybrid in which System 2 solves every case and Laya only repeats the answer is not evidence of useful delegation.

Suggested diagnostics include delegated-decision count, fraction of leaf tasks assigned to Laya, fraction later overridden, supervisor calls, and downstream changes caused by worker results. Use controlled worker-output perturbations on **offline fixtures or development cases** to check that outputs genuinely influence the plan; do not contaminate the primary evaluation with those perturbations.

A supervisor prompt should request task contracts, not completed worker answers. That does not prove the supervisor has not internally solved the task, so use a **matched-plan replay** diagnostic: generate and freeze a plan from permitted inputs, run exactly that plan with Laya workers and with System 2 workers, and compare the outputs. Charge the initial plan generation once to each hypothetical deployment when reporting total cost. Also show worker-only measurements; label cached/replayed results separately from live end-to-end timing.

## 5. Benchmarks and data protocol

### 5.1 ShARC: primary static decision benchmark

ShARC supplies a rule passage, a user question, scenario context, and prior clarification dialogue. The input fields permitted by the authors are `snippet`, `question`, `scenario`, and `history`; `answer` and `evidence` are not prediction inputs. The original answer can be a terminal answer or a follow-up question. [S7, S8]

Implement an explicitly labeled **four-way decision-only adaptation**:

```text
YES         source answer is the terminal label "Yes"
NO          source answer is the terminal label "No"
IRRELEVANT  source answer is the terminal label "Irrelevant"
MORE        source answer is a nonterminal follow-up question
```

Use the pinned release's official normalization where available. Normalize whitespace/case for terminal labels. Validate source records so corrupt, empty, or unexpected terminal fields do not silently become `MORE`. The source labels stay evaluator-side; models see identical generic label definitions and permitted input fields only.

A correct `MORE` decision counts as success on this classification task. It is not the same thing as a model infrastructure abstention. In the initial study, no simulated user's answer is generated and no free-form question quality is scored.

The project states that the original test set is now public. Use the official train/development/test assignment as the replication track after verifying the downloaded artifacts. Do not assume the split sizes or a deprecated dataset-loading script still work. Direct release files with hashes are acceptable. [S8]

Use only training data for adaptation; carve calibration material from training by rule family, reserving development for model/prompt/gate selection. Audit `tree_id`, normalized rule text, and near-duplicate passages across splits. Where official overlap exists, report it and add a separately named family-disjoint generalization track; do not silently change the official split and retain the official benchmark label.

**Outputs:** four-way accuracy, balanced accuracy, macro-F1, confusion matrix, `MORE` precision/recall, unnecessary-clarification rate, inappropriate terminal-answer rate when `MORE` is correct, input coverage, and resource measurements. Treat invalid outputs as incorrect, not as `MORE`.

### 5.2 ContextRules: controlled contextual contrasts

Create a new synthetic diagnostic benchmark, explicitly named **ContextRules** rather than claiming it is an existing published benchmark. Its purpose is controlled testing of contextual sensitivity and missing-information behavior. It is not a substitute for external evaluation.

Generate cases from a restricted, executable rule language and render the rules and facts into readable text. Keep the ground-truth abstract syntax tree, assignments, labels, mutation descriptions, and solution certificates exclusively in the evaluator/generator process. Model-visible source IDs should be opaque and should not encode labels.

Suggested semantics:

- Rule expressions use `AND`, `OR`, and `NOT` over Boolean prerequisites with `TRUE`, `FALSE`, or `UNKNOWN` observations.
- `NOT UNKNOWN = UNKNOWN`; an absent fact is not false.
- `AND` is false when any child is false, true when all are true, and otherwise unknown.
- `OR` is true when any child is true, false when all are false, and otherwise unknown.
- For a relevant query, expression true maps to `YES`, false to `NO`, and unknown to `MORE`.
- `IRRELEVANT` is generated only when a clearly defined query topic falls outside the supplied policy's scope; it is not a catch-all for uncertainty.
- Updates resolve a previous fact only when the rendered text explicitly identifies a correction or replacement under the benchmark's declared update rule. Mere later mention does not automatically override earlier facts.
- Ambiguous contradictory cases are either given an explicit uncertainty convention or excluded from the initial suite and counted. Do not invent a hidden tie-breaking policy.

For version 1, use read-once expressions: each independent prerequisite appears at most once in an expression, although its evidence may be repeated or explicitly corrected. This makes the stated three-valued semantics agree with exhaustive completion semantics. Start with nesting depth 1–4, conjunctions, disjunctions, explicit negation, distractors, and correction events. The model receives only the rendered task, not the machine-readable answer expression. Expressions with repeated logical variables are a separately specified extension, not silently admitted into this version.

Each family should contain a base case and verified variants:

| Variant | Expected relation |
|---|---|
| Decisive fact flip | Recompute the oracle; retain as a flip case only when the correct decision actually changes. |
| Necessary fact removed | Retain as a missing-information case only when the oracle becomes `MORE`. |
| Irrelevant detail added | Correct decision remains unchanged. |
| Meaning-preserving paraphrase | Correct decision remains unchanged; use reviewed templates initially. |
| Explicit correction | Decision follows the resolved updated fact set. |
| Candidate-order permutation | Semantic label should remain unchanged after ID remapping. |

Keep every variant in the same partition as its base. Group by canonical rule/template family, not only by individual row. Changing names or random seeds does not create an independent semantic family. Use a separate structural out-of-distribution track for held-out expression shapes or rule depth.

Suggested pilot sizes are 50 held-out families with 4–6 variants each, plus separate training/calibration/development families. A larger profile can target 200 or more test families **only if** the generator actually produces that many distinct canonical families; report the achieved count, not the requested count. Final sample size should follow a development-based precision/power assessment, not an arbitrary claim of adequacy.

Validate the generator with two implementations: the main expression evaluator and an independent brute-force completion evaluator for small expressions. Define known output as a result that is identical across every assignment of unknown prerequisites; disagreement across completions maps to `MORE`. Test equivalence to the three-valued evaluator only for the supported expression class where that equivalence holds. Repeated or logically dependent predicates can violate equivalence under simple Kleene evaluation; either restrict the grammar to read-once independent predicates or use completion semantics consistently. State the chosen semantics explicitly.

A few hand-authored fixtures should exercise boundary cases not produced by the random generator. Natural-language renderers should have independent manual review before claims about linguistic understanding. Machine consistency alone does not establish unambiguous English.

### 5.3 ALFWorld: primary interactive benchmark

Use ALFWorld's **text-only** environment, not its vision/THOR variant. The official code exposes `admissible_commands` for the text setting. This supports evaluating selection among actions without requiring Laya to generate command strings. The original BUTLER setup did not use the same command-list interface, so published command-generation scores are not directly comparable. [S9, S10]

Every policy receives the same interface: user goal, current text observation, public history/memory, and the current admissible actions. The environment exposes only observations available to an agent. Hidden simulator state, walkthroughs, solution trajectories, reward shaping internals, and oracle plans are not model inputs.

Use training episodes for development/adaptation and hold out evaluation episodes. Discover the available seen/unseen partitions from the pinned release, record exact game IDs, and report seen/unseen results separately. The commonly quoted number of games is not an implementation constant; enumerate the release actually installed.

Reset the environment independently for each policy and episode seed. Permit divergence after different actions; identical subsequent observations cannot be demanded of different trajectories. Never replay another policy's favorable observations as if they arose from the current policy's actions.

Use environment-defined success as the primary endpoint. The agent declaring success is not enough. Initial proposed action limit: 50 environment steps, with a separately reported 100-step sensitivity profile. These are study budgets, not assertions about the benchmark's official limits.

For failure detection, use observable signals such as repeated action/observation states, explicit unsuccessful action responses, invalid selections, or exhaustion of a subgoal's step budget. These are triggers to consider replanning, not proof that a task is impossible. Actions such as inspection can be useful even when location does not change.

Long action sets need special care. Prefer the complete list. If it exceeds Laya's candidate encoding budget, use the same documented, deterministic hierarchical selection interface for all policies in that track, keeping every action reachable and charging all selections. Alternatively, report a common-fit action-list track. Do not use an uncharged System 2 shortlist for Laya-only or quietly remove hard candidate sets. Record option order and any hierarchy because they can affect decisions.

**Outputs:** episode success, number of executed steps, repeated unproductive actions, invalid model proposals, replans, worker dispatches, observed recovery events, wall time, CPU/GPU resource measurements, context-limit failures, and completion status. Report overall steps with the cap applied and successful-only steps separately, avoiding survivor bias.

### 5.4 Optional extensions, not blockers for the core package

**ProofWriter** can extend explanation and inference-depth evaluation using natural-language rules and proof supervision. Keep the proof/labels evaluator-side; explicitly choose open- or closed-world semantics and accept any valid derivation rather than exact string equality with one reference proof. A solver verifying non-entailment may be needed for unknown cases. [S11]

**ToolSandbox** can be added later for business-like stateful tool workflows. **Typed Decisions** is useful for adapter smoke testing, but its authors state that its targets measure agreement with a teacher model rather than correctness. It should not be the central evidence of good judgment. [S14, S15]

Do not reuse arbitrary ShARC text edits as confidently labeled counterfactuals. Human-validated ShARC contrast pairs are a separate extension, with reviewer agreement and adjudication recorded. ContextRules supplies the initial machine-checkable contrasts.

## 6. Proposed harness architecture

### 6.1 Component and data-flow sketch

```text
                         experiment configuration + frozen manifest
                                           |
                                           v
Benchmark adapter --> Model-visible case / observation --> Experiment runner
      |                                                    |
      |                                                    v
      |                                         Interchangeable policy
      |                                         /         |          \
      |                                  Laya-only    Routing      Supervisor
      |                                                             |
      |                                                     Validated task graph
      |                                                      /       |       \
      |                                                  Laya      S2      Deterministic
      |                                                 worker   worker       tool
      |                                                      \       |       /
      |                                                       Task results
      |                                                             |
      |                                                Versioned shared state
      |                                                             |
      |                                          Selected label / allowed action
      |                                                             |
      |                                               Harness validation + execute
      |                                                             |
      +---------------- next public observation --------------------+

Private labels / oracle / hidden environment state
      |
      v
Evaluator process <----- completed predictions + immutable event records
      |
      v
Metrics, paired comparisons, failure taxonomy, reproducibility report

Optional auditor: reads permitted evidence + frozen decision records;
its output cannot rewrite decisions already counted in the main experiment.
```

The runner, not an LLM, owns task creation limits, dispatch, timeouts, permissions, state versions, and final termination. The supervisor can propose work, but cannot create arbitrary executable commands, enlarge its own budget, or grant itself access to labels.

### 6.2 Modules with substantive responsibilities

Favor a few deep modules with stable interfaces over one class per trivial operation:

| Module | Responsibility | Important boundary |
|---|---|---|
| `domain` | Typed cases, requests, results, task graphs, evidence references, and statuses | No model libraries or benchmark-specific gold fields. |
| `models` | Inference, encoding audits, output normalization, usage and capability reporting | Does not decide benchmark correctness or choose experiment policies. |
| `benchmarks` | Loading, public-view construction, environment interaction, split provenance | Private labels are separated before policy execution. |
| `harness` | Budgets, state, task scheduling, action authorization, event recording | Does not silently solve tasks with benchmark-specific heuristics. |
| `policies` | Baselines, routing, decomposition, and replanning | Uses interfaces, not hidden simulator state or evaluator imports. |
| `evaluation` | Metrics, calibration fitting, statistical comparisons, reports | Predictions are evaluated after or outside policy inference. |
| `auditing` | Evidence-grounded explanations and controlled probes | Separate from primary decision quality; no retrospective score changes. |

A single local process with sequential logical workers is sufficient initially. Parallel execution is an optional performance condition, not a prerequisite for multi-agent roles. Keep the scheduling policy identical across comparable arms.

### 6.3 Core contracts

Suggested data structures are shown below. Their names and serialization library can follow the repository. Prefer immutable or validated objects, explicit enums, and machine-checkable schemas.

**`CaseView`**

```text
case_id, family_id, benchmark_id, public_task_text
source_blocks: [{source_id, text, source_type, public_timestamp?}]
allowed_decisions: [{option_id, description}]
observation_version, public_history, context_policy_id
```

`family_id` and other administrative IDs should not appear in the model prompt unless they are semantically necessary; they exist for grouping and provenance. The model should not see source URLs, split names, private rule ASTs, expected depth, reference proofs, or target answers merely because they are present in a source record.

**`DecisionRequest`**

```text
request_id, state_version, task_id, objective
source_block_ids, rendered_context, question_text
options: [{option_id, description}]
output_schema_id, context_encoding_policy, budget_reservation_id
```

**`DecisionResult`**

```text
request_id, task_id, state_version, selected_option_id?
probabilities_by_option?, raw_confidence?, calibrated_confidence?
status: OK | INVALID_OUTPUT | CONTEXT_LIMIT | TIMEOUT | MODEL_UNAVAILABLE | ERROR
model_revision, usage, timings, encoding_audit_ref, raw_output_ref
```

For probabilities, validate finite values, allowable range, key alignment, and normalization with a declared numerical tolerance. Preserve the raw response. Do not fabricate a probability distribution or use the reasoner's self-reported confidence as a calibrated probability. A response with unavailable probabilities can still be a valid label but cannot support probability-based metrics without a justified separate estimator.

**`TaskSpec`**

```text
task_id, parent_id?, dependencies, state_version
kind: DECIDE | REASON | COMBINE | ENVIRONMENT_ACTION
requested_worker: LAYA | S2 | DETERMINISTIC
objective, source_refs, options?, output_schema_id
acceptance_rule_id, escalation_policy_id, budgets
```

All acceptance and tool references identify pre-registered code or schemas. They are not arbitrary Python, shell fragments, or dynamically evaluated expressions. Distinguish a format-valid task result from a semantically correct one; most correctness checks remain evaluator-side.

**`Plan`**

```text
plan_id, version, goal, evidence_version, nodes: [TaskSpec]
observable_replan_triggers, termination_policy_id
```

Validate unique IDs, acyclicity, maximum depth, dependency types, source references, option IDs, worker capabilities, and total reservations. Allow a small finite instruction language for `COMBINE`, never `eval` or an LLM-produced function body.

**`EpisodeResult`**

```text
run_id, policy_id, benchmark_id, case_id, family_id, seed
prediction?, externally_scored_success?, terminal_status
events_ref, final_public_state_ref, usage_totals, quality_flags
```

Terminal statuses should distinguish completed prediction, budget exhaustion, model error, context limit, invalid output, environment setup failure, and cancellation. `MORE` is a predicted decision, not a terminal infrastructure status.

### 6.4 Static-task decomposition

For ShARC and ContextRules, a useful supervisor output is a graph of relevant conditions, bounded evidence checks, and combination operations. Laya leaves can answer `SUPPORTED`, `CONTRADICTED`, or `UNKNOWN` for a supplied proposition. A separate relevance leaf can determine whether the policy applies.

The supervisor infers this graph from **permitted natural-language inputs**, not from the evaluator's rule representation. A generated graph can be syntactically valid and semantically wrong; such mistakes are part of the measured architecture performance.

Example intended behavior, not a gold plan available to the models:

```text
S2 proposes conditions: training complete; no overdue items; equipment available
Laya checks each condition against the user's supplied circumstances
Harness combines returned condition values using the proposed conjunction
Result: YES, NO, or MORE; relevance is handled separately
```

Use deterministic combination when the proposed graph supports it. For more complex ShARC language, allow a final reasoning aggregation step under an explicitly named condition and charge it. Do not hide a mandatory final System 2 answer behind the term “Laya execution.” Report which aggregation method produced each final decision.

`H1_plan_once` uses one valid initial plan and a fixed execution sequence; an unsupported plan terminates with an explicit failure or a prespecified declared fallback. `H2_mixed` can revise the task graph after uncertainty, contradictory worker outputs, or failed format checks. `C1_s2_workers` uses the same graph protocol with reasoner decision leaves.

Suggested initial limit: eight leaf tasks per static problem, depth at most two, and no unrestricted recursive delegation. Treat these as development-adjustable budgets, then freeze them.

### 6.5 Interactive planning and replanning

For ALFWorld, the supervisor proposes an ordered or partially ordered set of subgoals and a dispatch policy. It should not invent actions absent from the environment registry. Laya chooses among allowed current commands and can report uncertainty or a need to reconsider a subgoal.

A typical control loop is:

```text
observe -> validate context -> consult current plan / worker assignment
        -> select allowed action -> execute -> append observation
        -> test observable progress / failure triggers
        -> continue locally OR request bounded replan
        -> terminate on environment success or an explicit resource/error status
```

Plan-once does not regain supervisor access after a failure. Adaptive mixed execution does, within its budget. Both retain the same raw observations and memory tools. The reasoner-only policy also has history and retry opportunities; it is not a deliberately weak one-shot baseline.

Record trigger IDs and their observed evidence. If no progress can be verified from public observations, record the uncertainty rather than consulting the hidden reward mechanism. A backend-returned official terminal success flag can terminate the episode without exposing private intermediate reward information to the policy.

### 6.6 Shared state, memory, and stale work

Keep an append-only observation log plus a versioned working state. Every derived claim should identify its origin as observed evidence, model hypothesis, selected action, or verified tool result. A model assertion does not become an observation merely because it was written to memory.

Bind task requests and action selections to state and action-set versions. Reject or explicitly reconsider stale results when relevant state changes. Changes in available actions invalidate old index-to-command mappings.

Memory selection must use only evidence actually observed in that episode. Public deterministic operations such as retrieving an observed block by ID can be shared by all policies. Reasoner-generated summaries belong to named experimental arms. Do not share working memory between independent cases, contrast variants, benchmark partitions, or policies.

### 6.7 Model output handling

Laya should return normalized typed choices. The reasoning model can generate internally and then emit a concise final JSON contract. Preserve model-supplied generation metadata, but do not require recording private reasoning text to produce a valid experiment. A compact task plan and cited evidence are the auditable artifacts.

Use the model's actual chat template and output delimiters. Do not transplant Qwen-specific thinking switches or token IDs into the DeepSeek-derived checkpoint. If a backend separates reasoning and final content, parse the final channel. If not, test a model-specific parser against real outputs and version it.

Avoid constraining the entire generation to JSON in a way that unintentionally removes the model's normal reasoning mode. Final-answer schema enforcement, when supported, should be tested on the chosen backend. Record whether guided decoding was used.

Suggested retry policy: at most one format-repair call, with original evidence and validation errors but no target answer; count both attempts. The identical rule applies to every generative condition. Reaching the output limit without a valid final answer is not a successful prediction. A fallback cannot silently promote an incomplete answer into a correct one.

## 7. Suggested folder structure

Integrate this under an existing experiments/package area if the repository already has one. The tree describes responsibilities, not a requirement to create a parallel application.

```text
s1s2-experiments/
├── README.md
├── pyproject.toml
├── <repository-standard dependency lock>
├── .gitignore
├── .env.example                         # variable names only; no secrets
├── configs/
│   ├── models/
│   │   ├── laya_multilingual.yaml
│   │   ├── laya_english_short.yaml
│   │   └── deepseek_r1_qwen3_8b.yaml
│   ├── profiles/
│   │   ├── mock.yaml
│   │   ├── local_low_vram.yaml
│   │   └── reference.yaml
│   ├── experiments/
│   │   ├── sharc_smoke.yaml
│   │   ├── sharc_pilot.yaml
│   │   ├── sharc_confirmatory.yaml
│   │   ├── context_rules.yaml
│   │   ├── alfworld_smoke.yaml
│   │   ├── alfworld_pilot.yaml
│   │   ├── alfworld_confirmatory.yaml
│   │   ├── matched_plan_replay.yaml
│   │   └── explanation_audit.yaml
│   └── schemas/                         # if not generated from typed models
├── prompts/
│   ├── s2_direct.md
│   ├── s1_triage.yaml
│   ├── s2_static_planner.md
│   ├── s2_interactive_planner.md
│   ├── s2_decision_worker.md
│   ├── s2_replan.md
│   └── s2_auditor.md
├── src/s1s2lab/
│   ├── domain.py                        # core contracts; split only as needed
│   ├── config.py
│   ├── cli.py
│   ├── models/
│   │   ├── protocols.py
│   │   ├── laya_adapter.py
│   │   ├── reasoning_adapter.py
│   │   ├── encoding_audit.py
│   │   └── fake_models.py
│   ├── benchmarks/
│   │   ├── protocols.py
│   │   ├── sharc.py
│   │   ├── context_rules.py
│   │   ├── alfworld.py
│   │   ├── splits.py
│   │   └── private_labels.py            # evaluator-side; not policy imports
│   ├── harness/
│   │   ├── runner.py
│   │   ├── scheduler.py
│   │   ├── state.py
│   │   ├── budgets.py
│   │   ├── validation.py
│   │   └── events.py
│   ├── policies/
│   │   ├── single_model.py
│   │   ├── routing.py
│   │   └── supervised.py
│   ├── evaluation/
│   │   ├── scoring.py
│   │   ├── calibration.py
│   │   ├── comparisons.py
│   │   ├── failure_analysis.py
│   │   └── reporting.py
│   └── auditing/
│       ├── explain.py
│       ├── probes.py
│       └── verification.py
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   ├── metamorphic/
│   ├── e2e/
│   ├── real_models/                     # opt-in; unavailable is not passed
│   └── fixtures/
│       ├── sharc_synthetic.jsonl
│       ├── context_rules_cases.jsonl
│       ├── fake_alfworld_episodes.json
│       └── model_responses/
├── scripts/
│   ├── bootstrap.*                      # adapt to supported platforms
│   ├── fetch_assets.py
│   ├── prepare_data.py
│   ├── run_suite.py
│   └── validate_manifest.py
├── docs/
│   ├── research_protocol.md
│   ├── architecture.md
│   ├── model_and_data_cards.md
│   ├── runbook.md
│   ├── testing.md
│   ├── limitations.md
│   └── decisions/
├── manifests/                           # small, shareable revision/split records
├── data/                                # downloaded/generated large data ignored
│   ├── raw/
│   ├── processed/
│   └── private_labels/
├── artifacts/                           # run outputs ignored by default
└── reports/                             # optional, explicitly promoted reports
```

Avoid making the central runner depend directly on Transformers, Laya internals, or ALFWorld implementation classes. Use narrow interfaces and lazy imports so offline tests can import the package without heavyweight dependencies.

## 8. Configuration and reproducible execution

### 8.1 Suggested normalized experiment configuration

This is an example of the **configuration to implement**, not a claim that this CLI or YAML schema already exists. Resolve placeholders through the asset-lock step. A confirmatory configuration with null revisions or unfrozen thresholds should fail validation.

```yaml
schema_version: 1
experiment_id: sharc_pilot_v1
stage: pilot
benchmark:
  name: sharc_decision
  source_manifest: manifests/sharc_assets.json
  split_manifest: manifests/sharc_pilot_splits.json
  evaluation_partition: development
  limit: 200
models:
  s1:
    adapter: laya_sdk
    model_id: convaiinnovations/laya
    subfolder: multilingual
    revision: null
    max_len: 8192
    head_max_len: 512
    device: cpu
  s2:
    adapter: local_compatible_endpoint
    model_id: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    revision: null
    endpoint_env: S2_BASE_URL
    tokenizer_from_same_revision: true
    quantization_manifest: manifests/s2_quantization.json
    temperature: 0.6
    top_p: 0.95
    max_new_tokens_per_call: 4096
policies:
  - B1_laya
  - B2_s2
  - B3_cascade
  - B4_triage
  - H1_plan_once
  - H2_mixed
  - C1_s2_workers
context:
  track: common_fit
  policy: complete_source_blocks_v1
  silent_truncation: false
  action_presentation: full_list
routing:
  calibration_artifact: null
  threshold_artifact: null
  unresolved_gates: disable_condition
budgets:
  max_s1_questions_per_case: 64
  max_s2_calls_per_case: 8
  max_s2_generated_tokens_per_case: 16384
  max_leaf_tasks: 8
  max_task_depth: 2
  max_replans: 2
  max_format_repairs_per_result: 1
  model_timeout_seconds: 600
  case_timeout_seconds: 1800
execution:
  seeds: [17, 29, 43]
  concurrency: 1
  warmup_cases: 2
  response_cache: false
  resume: true
  allow_network_during_inference: false
  allow_loopback_model_endpoint: true
  allow_paid_endpoints: false
output:
  root: artifacts
  save_public_inputs: true
  save_provider_reasoning_text: false
  save_usage_and_final_contracts: true
```

Here `unresolved_gates: disable_condition` means the runner explicitly marks routing conditions not ready until calibration/threshold artifacts exist. It must not invent a threshold or quietly execute a different policy. Confirmatory runs require every requested condition to be ready, unless the frozen study protocol explicitly excludes it.

The network restriction allows only the explicitly configured loopback model service; it blocks external requests by benchmark agents. A separately authorized remote deployment profile should declare its endpoint allowlist and exposure of benchmark data.

The timeouts are configurable operational safeguards, not predictions of runtime. Tune them on development hardware before freezing. Do not let a slower device silently cause more timeouts for one policy without reporting that confound.

### 8.2 Budget profiles and reasoning headroom

Suggested starting budgets are below; revise on development data and freeze before final testing.

| Profile | Static cases | ALFWorld episodes |
|---|---|---|
| Smoke | 8–12 synthetic cases, scripted models; one separately marked real-model sanity case | 2–3 tiny fake episodes; optional real environment check |
| Pilot | Up to 200 development cases; 4,096 output tokens/call; 16,384 total S2 output tokens/case | Up to 20 development games; 50 actions; 4,096 output tokens/call; 131,072 total S2 tokens/episode |
| Reference | Frozen held-out cohort; budget sweep selected on development, including a higher-headroom setting | Frozen seen/unseen cohorts; 50-step primary and optional 100-step sensitivity; explicit episode token/call ceilings |

For ALFWorld, the default System 2 call cap must allow the reasoner-only policy to choose an action on every permitted environment step, plus the declared repair allowance. A static-task cap of eight calls must not accidentally carry over to a 50-step interactive episode.

Use equal resource **ceilings** where appropriate and report actual consumption. Equal generated-token limits do not imply equal FLOPs, energy, or runtime across an encoder and an autoregressive model. Include a quality-first reasoner reference with sufficient headroom to reveal whether an apparent hybrid advantage is caused by starving the baseline of generation tokens. Publisher scores obtained with much larger budgets should not be claimed under smaller local caps.

Stop and record a budget outcome before admitting new work that cannot fit its reserved maximum. Reconcile reservations with actual usage, including repairs and failed calls. With concurrency, budget reservation should be atomic. Cancellation should not leave unknown charges or untracked background work.

### 8.3 Manifest and provenance

Each run should contain a resolved manifest with:

- Git revision, dirty-worktree patch hash or equivalent, config hash, prompt hashes, and software lock hash.
- Actual checkpoint revisions, tokenizer/chat-template hashes, parameter-count metadata, precision, quantization, and backend/version.
- Hardware, device placement, thread count, concurrency, memory limits, and model loading/warmup policy.
- Dataset version and file hashes, split/family manifests, eligibility rules, and all excluded IDs with reasons.
- Seeds for generation, data order, candidate order, and the environment; any known nondeterministic kernels.
- Calibration and routing artifact provenance, fitting partitions, thresholds, training costs when applicable, and analysis-plan version.
- Start/end timestamps, experiment status, executed command, resource accounting availability, and known deviations.

Repeated seeds estimate stochastic variability; they do not turn the same case into independent evidence. Do not average only successful runs or pick the best seed for each policy.

### 8.4 Suggested CLI behavior

Implement equivalent commands using the repository's preferred CLI framework. These names describe desired behavior, not commands that exist before implementation.

```bash
python -m s1s2lab.cli doctor --profile mock
python -m s1s2lab.cli fetch --asset-group primary --lock manifests/assets.lock.json
python -m s1s2lab.cli prepare --benchmark sharc --stage pilot
python -m s1s2lab.cli prepare --benchmark context_rules --stage pilot
python -m s1s2lab.cli validate --config configs/experiments/sharc_pilot.yaml
python -m s1s2lab.cli calibrate --config configs/experiments/sharc_pilot.yaml
python -m s1s2lab.cli plan-run --config configs/experiments/sharc_pilot.yaml
python -m s1s2lab.cli run --config configs/experiments/sharc_pilot.yaml
python -m s1s2lab.cli score --run artifacts/<run_id>
python -m s1s2lab.cli compare --study-manifest manifests/<study_id>.json
python -m s1s2lab.cli audit --run artifacts/<run_id> --sample-manifest manifests/<audit_sample>.json
python -m s1s2lab.cli report --study-manifest manifests/<study_id>.json
```

`doctor` reports package health, model availability, benchmark availability, encoding readiness, and hardware readiness separately. `plan-run` estimates case/call ceilings from configuration without claiming model outcomes. `run` produces immutable decisions and traces; `score` attaches evaluator-side results; `compare` refuses incompatible manifests rather than blending different checkpoints or action settings.

Asset downloads should be resumable, checksummed, licensed, and separate from inference. Use environment variables for credentials and endpoint addresses; never commit keys. Safe archive extraction rejects path traversal. Where a remote compatible endpoint cannot prove its revision, record that limitation and exclude it from a strict revision-pinned primary series.

## 9. Explanation and decision-audit experiment

This experiment addresses the idea of System 2 explaining a System 1 decision without claiming access to a uniquely recoverable internal thought process. Fluent rationales are not automatically faithful; prior work shows that generated explanations can omit experimentally influential factors. [S16]

Use three auditor conditions, all with the same primary reasoning model:

| ID | Available information | What it tests |
|---|---|---|
| `E0_posthoc` | Original public input and selected decision | Whether an explanation from the outcome alone is justified. |
| `E1_trace` | The same input/decision plus actual task, evidence, routing, and execution records | Whether recorded process evidence improves audit validity. |
| `E2_probe` | `E1_trace` plus a small budget of controlled new calls to Laya on input variants | Whether behavioral tests expose unsupported explanations or sensitivity. |

Run the initial audit on ContextRules, whose ground truth is mechanically checkable. Add a separately reported subset of ShARC with human validation if resources permit. ProofWriter is an optional proof-depth extension, not necessary to implement the basic audit.

The auditor should return a schema such as:

```json
{
  "decision_assessment": "INSUFFICIENT_EVIDENCE",
  "supporting_source_ids": [],
  "supporting_event_ids": [],
  "claims": [],
  "alternative_explanations": [],
  "probes_proposed": [],
  "limitations": [],
  "explanation_kind": "EVIDENCE_AND_BEHAVIOR_NOT_INTERNAL_THOUGHT"
}
```

The example is one possible output instance. The actual `decision_assessment` enum is `SUPPORTED`, `UNSUPPORTED`, or `INSUFFICIENT_EVIDENCE`; the empty lists are placeholders in this example, not a complete useful audit.

Require source references to exist and match the claim's cited text. Existence of a citation is not proof of entailment. On ContextRules, evaluate supporting facts/rules using the private evaluator; for unrestricted prose, report what remains unverified rather than accepting a second LLM's agreement as truth.

For `E2_probe`, begin with up to four prespecified transformations per case: removing a fact, changing a fact, adding a distractor, and reordering options. The auditor can propose a hypothesis and select among safe probe operations; the harness validates the transformation and tracks all additional calls. For tests of explanation prediction, freeze the auditor's predicted decision change **before** revealing the probe result. Do not describe an after-the-fact fit as successful prediction.

Use both randomly sampled decisions and an explicitly labeled challenge set containing incorrect decisions, irrelevant citations, and altered records. Do not pool oversampled failures into an overall prevalence estimate without reweighting. The auditor must not be told which cases were corrupted, and challenge fixtures must not be used to tune final-test explanations.

Suggested audit metrics:

- Accuracy of the supported/unsupported assessment against the private evaluator.
- Valid-reference rate and valid-support rate, reported separately.
- Detection of genuinely incorrect decisions and false alarms on correct decisions.
- Predicted versus observed response to probes, including inappropriate sensitivity to distractors.
- Unsupported causal-claim rate under a documented rubric, plus extra latency and model calls.

Keep two labels separate: **decision justified by available evidence** and **account consistent with the recorded execution**. A good decision can have a bad explanation; a trace-faithful explanation can describe a mistaken decision procedure. Neither demonstrates recovery of neural mechanisms.

Audit output should never retroactively replace the prediction used in the primary experiment. A future “audit then repair” arm would be a new policy, with repair costs and final outcomes reported separately.

## 10. Metrics, calibration, and statistical analysis

### 10.1 Quality metrics

| Benchmark | Primary endpoint | Additional diagnostics |
|---|---|---|
| ShARC decision adaptation | Balanced accuracy over the four labels | Accuracy, macro-F1, class recall, clarification behavior, invalid outputs. |
| ContextRules | Fraction of case families with every required variant correct | Pair accuracy, relevant-change accuracy, irrelevant-change errors, update consistency, depth breakdown. |
| ALFWorld | Fraction of episodes reaching environment-defined success within budget | Capped steps, repeats, failure/recovery categories, replans, intervention counts. |
| Audit experiment | Supported/unsupported assessment accuracy on the defined audit set | Source/claim validity, error detection, predicted probe behavior, unsupported claims. |

Define pair accuracy as:

```text
pair_correct = 1 when both the base prediction and variant prediction are correct;
               0 otherwise.
```

For an invariant pair, two identical wrong answers are not correct robustness. For a flip pair, changing the answer is not sufficient; both outputs must match the independently determined labels.

For ShARC, balanced accuracy is the mean per-class recall. Invalid or missing predictions count against the true class. Implement the metric so an invalid-output sentinel does not create a fifth class whose recall changes the denominator.

For clarification, distinguish `P(predicted MORE | gold is terminal)` from `P(predicted terminal | gold MORE)`. Also report how often the model produces an invalid result. Do not hide invalid answers by recoding them as a cautious request for information.

Do not construct one pooled score across unrelated benchmarks unless a weighting rule was defined before observing results. Always retain benchmark-specific outcomes and task-family breakdowns.

### 10.2 Calibration and selective decision-making

Use Laya's actual option probabilities where available. Calculate multiclass Brier score and log loss against benchmark labels, with a declared numerical clipping rule for log loss. Show reliability plots and an expected calibration error summary with fixed bins, recognizing that binning and sample size affect ECE.

Fit calibration only on the calibration partition. If using temperature scaling on probabilities, apply it as a documented transformation equivalent to rescaling log probabilities; handle zeros explicitly. Do not claim access to logits unless the pinned adapter actually exposes them.

Use a separate development partition to choose routing thresholds. Save fitted parameters, class order, source model revision, data hashes, and validation scores. Reject an artifact fitted on a different checkpoint, output schema, or label ordering.

Report risk–coverage behavior: as more uncertain Laya cases are escalated, what is the error rate among retained cases and the final error rate of the complete cascade? An uncertainty signal can be useful for routing without being a well-calibrated correctness probability; distinguish those claims.

### 10.3 Resource accounting

Collect, where measurable:

```text
S1 calls; S1 questions; actual S1 encoder tokens per question/batch;
S2 calls; S2 prompt tokens; all S2 generated tokens, including reasoning;
format repairs; supervisor calls; replay/probe calls; environment actions;
end-to-end wall time; per-component service time; model load time;
CPU/GPU placement; peak allocated/reserved device memory;
power/energy measurements only when the instrumentation supports them;
cache status, batching, and concurrency.
```

A single batched Laya call can represent several questions. Reporting one call as one decision would conceal work. Do not assume an encoder token costs the same as a generated reasoning token.

Separate cold-start from warmed measurements. Primary interactive timing should reflect the actual sequential service placement. Do not claim low latency using a cached plan while omitting the time needed to generate it. A replay diagnostic can isolate worker behavior but is not a deployment timing result.

When local monetary cost is not known, report compute/resource use rather than `$0`. A monetary estimate can use a user-supplied hardware-hour or electricity rate, with its assumptions and excluded costs disclosed. Hardware power samples are not necessarily attributable precisely to one process. Do not invent FLOPs or energy measurements.

Define cost per successful case as total measured cost for **all attempted cases**, including failures, divided by the number of successes. If there are no successes, report undefined/infinite as appropriate rather than zero.

Show quality/resource frontiers over a development-selected set of budgets. A proposed non-inferiority margin, such as two absolute percentage points, is only an example: justify and freeze the actual margin before testing. A hybrid below the margin is not “equivalent” simply because a superiority test is nonsignificant.

### 10.4 Comparison design

Use paired evaluations on the same cases and seeds. Average repeated stochastic outcomes within a case for the designated estimator; retain seed-specific outputs for diagnostics. Confidence intervals should resample independent families rather than treating every internal decision or repeated seed as a separate observation.

Suggested clustering:

- ShARC: rule/tree family, with the official item-weighted point estimate preserved.
- ContextRules: canonical base-rule family, keeping all variants together.
- ALFWorld: episode/game family; preserve seen/unseen strata and shared underlying game grouping where applicable.

Use a paired cluster bootstrap for uncertainty on quality differences and resource ratios. A paired cluster-level permutation test is a reasonable inferential option. State the estimator, weighting, confidence level, number of resamples, and seed. Test the implementation on analytically simple fixtures. Use an exact McNemar test only where its independent paired-binary assumptions fit the actual sampling design.

Apply Holm correction to the three prespecified primary mixed-versus-Laya endpoint tests. Keep secondary comparisons labeled exploratory or specify a separate multiplicity family in advance. Report effect sizes and confidence intervals, not only p-values.

Use the development pilot to estimate precision or minimum detectable effect; do not claim the pilot has sufficient power merely because several hundred decisions were made. Freeze the final case/sample manifest, budgets, prompts, and analysis plan before a confirmatory run. Repeatedly examining the final test and selecting a better prompt invalidates its held-out status.

### 10.5 Exclusions and failures

Perform common environment/data readiness checks before running policies. Any unusable asset excluded at that stage must have an ID, reason, and shared exclusion rule. Once evaluation begins, policy-specific invalid answers, timeouts, context failures, and budget exhaustion remain measured outcomes.

A machine crash may require resuming an interrupted run, but retain the interrupted attempt and the reason for retry. Do not select the successful attempt after several model failures without reporting the retry policy and total cost.

Do not compare only the subset where every policy returned a valid answer. Reports should show intended, attempted, completed, invalid, and missing counts for each policy, with a consistent quality denominator or an explicit limitation when a run is incomplete.

## 11. Automated test plan

Separate **software correctness tests** from **model-quality experiments**. Tests should protect contracts and scientific validity, not assert that a hybrid always outperforms a baseline. Real-model behavior can fail an evaluation without indicating a software bug; a benchmark leak can invalidate an excellent score despite every model call succeeding.

### 11.1 Offline unit tests

Expected tests include:

| Area | Representative checks |
|---|---|
| Configuration | Invalid budgets, unresolved revisions in confirmatory mode, incompatible calibration artifacts, and hidden fallback models are rejected. |
| Input schemas | Missing fields, unknown option IDs, malformed history, and unexpected gold-bearing fields are handled explicitly. |
| ShARC normalization | `Yes`, `No`, and `Irrelevant` normalize correctly; valid follow-ups map to `MORE`; malformed records fail validation. |
| Three-valued logic | Supported truth tables, explicit unknowns, negation, conjunction/disjunction, and declared correction semantics. |
| Plan validation | Duplicate IDs, cycles, excessive depth, missing dependencies, unknown source IDs, and arbitrary code expressions are rejected. |
| Probabilities | NaN, infinity, negative values, bad sums, wrong keys, absent distributions, and numerical clipping are handled without invented confidence. |
| Budgets | Calls, questions, tokens, repairs, probes, cancellations, and concurrent reservations are correctly charged. |
| State | Version progression, immutable observations, provenance, stale-result rejection, and per-case isolation. |
| Parsing | Valid final JSON, extra prose, reasoning delimiters, truncated generations, multiple conflicting JSON objects, and bounded repair behavior. |
| Metrics | Invalid-output denominators, class imbalance, family weighting, pair scoring, zero-success costs, and known bootstrap/permutation fixtures. |

For parsing, do not take the last convenient JSON-looking substring when multiple incompatible candidate outputs exist. Define a final-output convention and test ambiguous cases as invalid unless the model template unambiguously resolves them.

### 11.2 Model adapter contract tests

Use recorded **synthetic** response fixtures for offline tests and separately versioned real responses for opt-in compatibility checks. At minimum:

- Laya responses map to stable study option IDs; SDK field drift produces a clear incompatibility error.
- Question ordering and batching do not swap results between cases.
- Candidate descriptions retain distinct tokens after encoding; shared opaque IDs do not replace missing descriptions.
- Encoding audits expose truncation and input length. If actual post-encoding coverage cannot be inspected, the adapter reports an unverified audit state rather than asserting success.
- The reasoning adapter reports the served checkpoint identity, applies the correct chat template, and accounts for reasoning/output usage where available.
- Unsupported thinking switches, missing final content, absent usage fields, rate limits, timeouts, and backend disconnection are explicit outcomes.
- No adapter silently downloads assets or changes the model when the requested checkpoint is unavailable.

SDK hooks can assist telemetry, but they are not inherently read-only: Laya documents hooks that can mutate requests/results or skip inference. Keep observational hooks pure in primary runs and log any transformative hook as part of the experimental policy. [S17]

### 11.3 Information-boundary and leakage tests

These are especially important:

1. A ShARC fixture contains unique sentinel strings in `answer` and `evidence`; neither string appears in any serialized model request, log returned to an agent, prompt, plan, cache key text exposed to a model, or audit input.
2. Recursive allowlisting prevents nested private fields from leaking through `history`, generic metadata, or serialization of a full source object.
3. Ground-truth ContextRules ASTs, target assignments, mutation labels, and source template names stay evaluator-side. Natural-language rules remain model-visible as intended.
4. ALFWorld public observations omit walkthroughs, hidden object placements, private reward fields, and internal goal certificates. No policy imports the evaluator or hidden-state helpers.
5. Dataset IDs, family variants, aliases, and near-duplicate examples are checked for prohibited split crossover. Calibration artifacts cannot reference test case IDs.
6. Fit functions receive permitted fitting partitions only. A unit test intentionally requests test-label fitting and verifies refusal.
7. Source strings attempting to issue instructions cannot grant extra tools or filesystem access. Behavioral susceptibility is measured separately; the deterministic permission boundary must hold regardless of the text.

Physical process separation or separate file permissions for private labels is preferable for confirmatory runs where practical. At minimum use distinct data types, explicit allowlists, import-boundary tests, and no evaluator handle in any policy dependency graph.

### 11.4 Harness integration tests with fake models

Build a small scripted model pair and fake environment that exercise actual interfaces, not a parallel toy pipeline. Scenarios should include:

| Scenario | Expected invariant |
|---|---|
| Laya succeeds directly | No System 2 call in `B1`; correct usage and decision trace. |
| Cascade triggers | Laya call plus one allowed System 2 decision; both charged. |
| Triage chooses reasoning | Execution follows that route; triage cost remains visible. |
| Supervisor delegates | Task contracts reach the selected workers; dependencies and source views are correct. |
| Laya worker returns uncertainty | The configured policy continues, replans, or fails explicitly; it does not read a gold label. |
| Worker error changes plan | Replan is linked to the actual error event and new state version. |
| Budget exhausted | No further inference or environment action occurs; partial work and usage are saved. |
| Tool says failure | A fluent agent success claim cannot override the environment outcome. |
| Candidate list changes | An action selected from the old list cannot execute by stale index. |
| Corrupt model output | At most the configured repairs occur; no silent correct-answer substitution. |
| S2 unavailable | Laya-only can still run; mixed/S2 conditions report not ready or failed, not fake success. |
| Resume after interruption | Existing completed trials are not duplicated; mismatched manifests refuse resume. |
| Concurrent state updates | Reservations and evidence versions remain consistent; no cross-case contamination. |

Test `H2_mixed` and `C1_s2_workers` with exactly the same frozen plan to show that worker substitution is genuinely implemented. Test a case where changing a leaf result changes the final decision; otherwise the worker may be decorative.

### 11.5 Property-based and metamorphic tests

Use a property-testing library if already compatible with the repository. Useful properties include:

- Stable serialization and request fingerprints for semantically identical normalized objects.
- Permuting candidate IDs/order and remapping outputs preserves evaluator meaning. This is a software property; a real model's choice may change, which belongs in the robustness results.
- ContextRules invariant transformations preserve oracle labels; certified flip and missing-information transformations produce their intended changes.
- All variants remain in one partition, including after shuffling and resumable preparation.
- Budget consumption never becomes negative and never exceeds a reserved ceiling through accounting errors.
- Source IDs in derived claims resolve to the correct episode, not merely any globally existing ID.
- A deterministic selector never introduces unobserved facts.
- Adding invalid or missing predictions never improves the main accuracy score.
- Equivalent event replay reconstructs the same public state and decisions without new inference.

Test ContextRules with independent evaluators and explicit dependent-predicate counterexamples so the semantics are not accidentally overstated. The renderer and oracle should not share every implementation path; correlated bugs can otherwise pass self-checks.

### 11.6 End-to-end offline test

A single documented command should execute a tiny mixed-policy study using fake adapters, save a complete artifact bundle, score it, compare it, and render a report. Assertions should cover manifest integrity, row counts, expected known scores, usage totals, failure categories, and reproducible resumption.

Fake outputs and reports must carry `execution_mode=mock` and a prominent synthetic-results label. Report generation should refuse to present them as empirical model performance. Ordinary CI should block network access and fail if a test tries to fetch weights or benchmark archives.

### 11.7 Opt-in real-model tests

Use explicit markers such as `real_model`, `gpu`, `network`, and `slow`, adapted to existing conventions. A useful invocation would be equivalent to:

```bash
pytest -m "not real_model and not network and not slow"
pytest tests/real_models -m real_model --run-real-models
```

The first command should work offline. The opt-in flag should be registered and documented if implemented. Missing assets should produce `SKIPPED` or `BLOCKED` with a reason, not `PASSED`.

Real checks should establish that the pinned Laya model loads, a short typed decision is returned, the proposed long-context configuration actually preserves input, the selected reasoner generates a parsable contract, and one real ALFWorld episode can be reset/stepped/scored. Do not make CI depend on a brittle expectation that a stochastic model answers every sanity question correctly. Save capability observations for the evaluation report.

A tokenizer/encoding sentinel test can establish that text was presented to a model; it does not establish that the model used that text correctly. Both instrumentation and behavioral evaluation are needed.

## 12. Training and adaptation extension

Implement frozen-model experiments first. Task-specific adaptation is an optional second phase that strengthens the Laya baseline rather than a prerequisite for basic execution.

For static decisions, use training-only labels in Laya's typed-choice format, preferably with option-order randomization and preserved source provenance. Keep dedicated calibration and development partitions. Match checkpoint selection budgets and disclose training compute. Fine-tuning should use the pinned SDK's verified training interface, not an invented `.fit()` call.

A benchmark-specific `laya-typed-decisions` checkpoint is not a substitute for training on the study's domains. Its performance should not be assumed to transfer to ShARC or ALFWorld. Similarly, a gain after adaptation should not be described as a zero-shot architecture gain.

For ALFWorld, the environment does not automatically provide a gold next action in the policy's public view. Any behavioral cloning or teacher-generated action dataset is a separate training pipeline with training-only episode access and disclosed teacher/oracle provenance. No evaluation walkthrough is a source of worker training labels.

Keep calibration distinct from weight updates. Save the fitted model revision, training examples, split hashes, hyperparameters, stopping criteria, and cost. A fine-tuned checkpoint should receive a new identity and matched baselines in the comparison matrix.

## 13. Results and reporting artifacts

Suggested output bundle:

```text
artifacts/<run_id>/
├── manifest.json
├── resolved_config.yaml
├── readiness.json
├── case_manifest.jsonl
├── predictions.jsonl
├── events.jsonl
├── encoding_audits.jsonl
├── usage.jsonl
├── exclusions.jsonl
├── errors.jsonl
├── checksums.json
├── metrics.json
├── per_case_results.csv
├── comparisons.csv
├── audit_results.jsonl                 # only when the audit is run
└── report.md
```

Keep raw data large enough for replay in referenced sidecars where needed. Avoid plaintext secrets and unrelated user material. Persist atomically and record schema versions. A completed prediction should not be edited in place after scoring; attach a correction or new attempt record.

The report should contain the actual model/backend/hardware conditions, benchmark variant, data coverage, quality/resource results with uncertainty, failure counts, excluded cases, and limitations. Include a small failure analysis sampled using a reproducible rule rather than selecting only examples that favor the hybrid.

Useful figures are quality versus resource use, context-pair correctness by mutation type, ALFWorld success by task family, calibration/risk–coverage, and the distribution of supervisor interventions. No figure should imply superiority when the interval or design does not support it.

### Claims the reporting code should prevent

- “Laya alone” when any hidden S2 preprocessing, candidate generation, or repair occurred.
- “Equal compute” based only on equal calls or token limits.
- “No cost” because inference was local.
- “State of the art” based on unmatched action interfaces or a small local pilot.
- “Faithful internal explanation” based solely on a persuasive rationale or valid source citation.
- “No test leakage” merely because raw answer fields were removed while family variants crossed splits.
- “Experiments passed” when only fake adapters or unit tests ran.
- “The hybrid works” when it only beats an artificially truncated or under-budgeted baseline.

## 14. Suggested implementation sequence and acceptance evidence

### Milestone A: repository fit and offline foundation

Inspect existing conventions; record the implementation plan; add typed contracts, configuration validation, fake adapters, the core runner, event records, and synthetic fixtures.

**Evidence:** offline tests pass; a mock end-to-end report has known expected values and a synthetic label; no heavy imports or network calls occur during ordinary tests.

### Milestone B: real adapters and readiness checks

Implement one real Laya adapter and one real adapter for the selected DeepSeek checkpoint. Resolve version pinning, precision configuration, template handling, encoding audits, and usage reporting.

**Evidence:** dependency/model manifests; real short-input responses; context and candidate-encoding audits; explicit blocked reasons for unavailable hardware. A blocked capability is documented without preventing offline development.

### Milestone C: ShARC and ContextRules comparisons

Implement data provenance and splits, the common public-view boundary, `B0`–`B4`, `H1`, `H2`, and `C1`, calibration artifacts, and the static scoring pipeline. Run development pilots before freezing final configurations.

**Evidence:** leakage tests; grouped-split audit; mock score verification; actual pilot predictions where models are available; failure taxonomy; frozen candidate final-study configuration. A positive performance outcome is not an acceptance criterion.

### Milestone D: interactive ALFWorld

Add the text adapter, admissible-action interface, observation-only memory, fixed and adaptive subgoal execution, and independent episode resets. Reuse the existing policies and contracts rather than building a second unrelated runner.

**Evidence:** fake interactive tests; actual environment setup/step checks; episode manifests; action/version safety tests; development pilot traces and resource accounting.

### Milestone E: audit study and mechanism diagnostics

Add trace-grounded audits, safe probes, matched-plan replay, and selected ablations. Verify that the auditor can identify an unsupported decision rather than being prompted to defend every outcome.

**Evidence:** controlled explanation fixtures, valid-reference versus valid-support scoring, probe predictions recorded before outcomes, and separate audit costs.

### Milestone F: frozen evaluation and reproducibility package

Freeze the final study manifest, model and data versions, prompts, calibration, budgets, and analysis before held-out execution. Generate tables from saved outputs and retain independent reproduction instructions.

**Evidence:** immutable run manifests, complete denominators, paired/clustered comparisons, full failure accounting, and a report stating what was actually executed. Record remaining limitations rather than filling missing results with estimates.

When delivering implementation work, summarize completed components, actual verification commands and outcomes, skipped/blocked real-model checks, known deviations, and the next repository-native work item. Do not claim a benchmark experiment ran when only the surrounding code was tested.

## 15. Scope boundaries and interpretation

The initial project should not attempt weight merging, joint end-to-end training of both models, unrestricted recursive agents, public web browsing by benchmark agents, or mechanistic neural interpretation. Those could become separately scoped studies after the inference-time architecture is understood.

Likewise, no UI is necessary to establish the research result. Favor a reproducible CLI and machine-readable experiment records over a dashboard that conceals untested evaluation logic.

This study can support claims about the selected checkpoints, benchmark variants, context policies, and resource settings. It cannot by itself establish a general theory of human-like dual-process cognition, universal calibration, broad real-world reliability, or recovery of a model's internal reasons.

Public benchmarks may have appeared in model training. New ContextRules families and structural holdouts reduce some memorization risks but do not prove complete absence of contamination. State this limitation and avoid conflating unseen generated rows with genuinely novel reasoning structures.

The experiment should make the easiest alternative explanations visible: larger-model assistance, improved information selection, extra calls, richer candidate actions, task-specific training, or formatting reliability. Finding that one of these explains the result is useful research, not a reason to hide it.

## 16. Primary-source register

Sources below were checked on September 26, 2026. They establish the external model/benchmark facts referenced above; proposed architecture, tests, budgets, metrics selections, and ContextRules are study design recommendations. Reverify mutable APIs at implementation and pin the actual revisions used. Do not copy vendor benchmark claims into the study's own results table.

**[S1] DeepSeek: DeepSeek-R1-0528-Qwen3-8B model card.** Checkpoint identity, publisher-reported 8B benchmark scores, decoding settings, and tokenizer/configuration warning.  
`https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`

**[S2] Qwen: Qwen3-8B model card.** Architecture's reported 8.2B parameter count and optional replication model.  
`https://huggingface.co/Qwen/Qwen3-8B`

**[S3] ConvAI Innovations: Laya model card.** Checkpoint family, documented context budgets, fixed checkpoint loading, and server behavior.  
`https://huggingface.co/convaiinnovations/laya`

**[S4] Laya: Agent API reference.** Loader and single-checkpoint inference integration; use the revision matching the installed SDK.  
`https://nandhakishorm.github.io/laya/reference/agent/`

**[S5] Laya issue #156.** Reported `noul` label-pair sensitivity; verify against the pinned revision rather than assuming a permanent defect.  
`https://github.com/NandhaKishorM/laya/issues/156`

**[S6] Laya issue #185.** Reported act-head saturation and unreliable `act_probability`; verify against the pinned revision.  
`https://github.com/NandhaKishorM/laya/issues/185`

**[S7] ShARC: data documentation.** Source field definitions and explicit exclusion of `answer` and `evidence` from prediction inputs.  
`https://sharc-data.github.io/data.html`

**[S8] ShARC: project page and original paper.** Benchmark purpose and public-test-set release notice. Saeidi et al., “Interpretation of Natural Language Rules in Conversational Machine Reading,” EMNLP 2018.  
`https://sharc-data.github.io/`  
`https://aclanthology.org/D18-1233/`

**[S9] ALFWorld: official repository.** Text environment integration and `admissible_commands`; note the difference from unrestricted command generation.  
`https://github.com/alfworld/alfworld`

**[S10] ALFWorld: project page.** Shridhar et al., “ALFWorld: Aligning Text and Embodied Environments for Interactive Learning,” ICLR 2021.  
`https://alfworld.github.io/`

**[S11] ProofWriter: original paper.** Tafjord, Dalvi, and Clark, “ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language,” Findings of ACL-IJCNLP 2021.  
`https://aclanthology.org/2021.findings-acl.317/`  
`https://arxiv.org/abs/2012.13048`

**[S12] RouteLLM: original paper.** Ong et al., “RouteLLM: Learning to Route LLMs with Preference Data.” Routing precedent, not validation of Laya.  
`https://arxiv.org/abs/2406.18665`

**[S13] SwiftSage: original paper.** Lin et al., “SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks,” NeurIPS 2023. Fast/slow planning precedent using a different fast-model type.  
`https://arxiv.org/abs/2305.17390`

**[S14] ToolSandbox: original paper.** “ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities,” Findings of NAACL 2025. Optional later workflow benchmark.  
`https://aclanthology.org/2025.findings-naacl.65/`

**[S15] Typed Decisions: dataset documentation.** Author-stated distinction between teacher agreement and correctness.  
`https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/main/README.md`

**[S16] Turpin et al.: explanation-faithfulness study.** “Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.” A rationale can omit causally influential prompt factors.  
`https://arxiv.org/abs/2305.04388`

**[S17] Laya: prediction hooks documentation.** Hooks support telemetry but can also alter requests or replace results, which matters for reproducibility.  
`https://nandhakishorm.github.io/laya/hooks/`

---

**Implementation target:** a tested, inspectable experiment harness that can establish whether Laya and a strong 8B reasoner make complementary contributions to context-sensitive decisions—and that can equally demonstrate when a simpler baseline is preferable.
