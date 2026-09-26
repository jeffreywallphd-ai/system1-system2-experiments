# Research protocol v1

This is a software-supported draft protocol, not a preregistered completed study. Source: [provided brief](experiment-brief.md). No empirical effect size or sample-size adequacy is assumed.

## Estimands

The initial primary contrast is H2_mixed minus B1_laya on ShARC balanced accuracy, ContextRules all-variants-correct family rate, and ALFWorld episode success. Preserve benchmark-specific scores and apply Holm correction across exactly these three contrasts. Comparisons to other policies are exploratory until separately frozen. No pooled cross-benchmark score is constructed.

Invalid infrastructure outcomes count as wrong against the true class; they do not create a fifth recall class. Missing classes yield undefined balanced accuracy. ContextRules requires all selected variants, with whole-family common-fit exclusion. Pair accuracy requires both base and variant to be correct. ALFWorld failures receive the step cap in capped-steps diagnostics; successful-only steps are separate.

Repeated seeds measure stochastic variation. ShARC and episode success retain item weighting; ContextRules averages all-variants correctness across seeds within family. Bootstrap preserves entire families and seeds. Permutation swaps whole paired families: up to 12 families uses exact enumeration, otherwise seeded Monte Carlo with the +1 correction. Confidence level is 95%. Any undefined ShARC bootstrap draw causes the interval to be withheld rather than conditioned on favorable draws.

## Data use and freezing

Training labels may determine the majority reference or optional adaptation. Calibration labels determine scaling; development labels determine gates and prompt/budget choices. Test/seen/unseen/OOD labels cannot enter fitting functions. Record canonical families and overlap before inference. Official ShARC overlap is reported under replication; the family-disjoint preparation option has separate provenance.

Before held-out execution, resolve and lock revisions, runtime versions, tokenizer/configuration files, exact data, partitions, eligibility, action interface, candidate ordering, seeds, inference settings, prompts, resource ceilings, routing parameters, and analysis. The `freeze` command binds code, configuration, data, and lock files. `run` rejects changes to that binding. A settings freeze is not capability validation, independent protocol review, or external preregistration.

The generator reports achieved canonical families, which can be fewer than requested. Depth-four shapes are OOD. Inspect train/calibration/development/test distributions and use development-based precision/power planning before selecting the final sample. English templates need independent human review before natural-language capability claims.

## Information parity and resources

Policies see the same original evidence and initial action interface. Model hypotheses are not observations. Supervisor evidence selection is part of a charged treatment; originals remain available. Single-model interactive policies get full history and repeated action opportunities.

S1 calls and questions both count; this version issues one question per call. S2 token totals include reasoning even though private reasoning text is not saved. Unknown usage retains its reservation. GPU memory is measured when available; CPU peak memory, energy, and FLOPs are not fabricated. Per-success resource totals include failures; no successes yields undefined cost per success.

The reference profile gives the reasoner higher generation headroom. Interactive call ceilings allow one decision plus repair allowance at every step, with a separate episode token ceiling. Equal ceilings do not imply equal compute. Loading and shared encoding preflight are outside per-trial timing and are disclosed; real warmed measurements remain to be designed.

## Routing labels

Static calibration uses temperature scaling on actual four-way probabilities, not claimed logits or `act_probability`. Development gate utility is accuracy minus 0.05 times the escalation fraction by default. Fixed grids and fitting hashes are saved.

Interactive calibration is a separate artifact fitted by `calibrate-actions`. Its labels are measured fast/slow episode outcomes from paired training-game continuations, with a single frozen continuation-protocol hash. They are not oracle next actions. Five fixed selected-probability bins use a declared Beta(1,1) posterior mean; empty bins escalate. This estimates fast-workflow suitability under that continuation protocol, not universal action correctness. The development threshold is fitted on separate game families. Generating suitable paired rollout labels remains part of real-study preparation; never use evaluation walkthroughs.

## Audit interpretation

E0 sees input and outcome; E1 adds permitted trace records; E2 adds safe probes and their later outcomes. Predicted probe labels are saved before calls. Initial and revised audits remain separate. Auditing cannot mutate scored decisions.

Automatic support checking is limited to bounded ContextRules evidence and its private rule oracle. Quotes, references, and support differ. Arbitrary prose entailment, semantic trace consistency, and causal claims require further human review. Challenge samples are separate and are not prevalence estimates.

Public benchmarks may have appeared in pretraining. New rows do not prove new structures. Apparent gains may reflect larger-model assistance, extra calls, plan quality, or evidence selection. C1 and matched-plan replay expose some alternatives without proving internal causality. Negative results remain valid outcomes.
