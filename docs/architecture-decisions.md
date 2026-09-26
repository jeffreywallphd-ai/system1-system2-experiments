# Architecture decisions

## ADR 001 — Standard-library Python core

The repository initially had only a README and GPLv3 license, with no harness or repository-specific AGENTS.md. Use dataclasses, JSON, argparse, and unittest so offline work needs no dependency installation. JSON replaces the suggested YAML. Responsibilities live in a few substantive modules rather than one class per trivial operation. `requirements-offline.lock` records the empty third-party runtime dependency set.

## ADR 002 — One local backend per role

Use Laya SDK 0.3.20 and direct Transformers with the selected DeepSeek snapshot. This identifies actual local weights, tokenizer, and template. API paths were inspected upstream, but real capability checks remain required. Original Qwen configuration is never substituted.

Laya private encoding methods are isolated in one version-guarded compatibility adapter. Public context parameters alone cannot prove retention. Compare configured and untruncated sequences and refuse SDK per-option truncation. Revising the SDK requires revisiting that adapter and real encoding checks.

Real models and ALFWorld use separate spawned processes with bounded RPC. Timeouts terminate and join the worker before execution continues. Logical workers are sequential, with concurrency fixed to one. There are no arbitrary agent tools or recursive execution privileges.

## ADR 003 — Different public and evaluator artifacts

Preparation writes `public.jsonl`, administrative `cases.jsonl`, and `private/labels.jsonl`. Model payloads omit administrative IDs, option meanings, and gold metadata. ShARC history is rebuilt from two permitted scalar fields. Observations are immutable, chronological, and versioned; model claims have different provenance.

Policies receive neither evaluator nor hidden-environment handles. Nested-sentinel and import-boundary tests protect the interface. This is an API boundary in the parent harness, not an adversarial OS sandbox. Confirmatory deployments may additionally restrict private-label file permissions. Model processes receive only public requests.

## ADR 004 — Immutable trial commits and retained failures

Journals are appended before calls/actions. An atomic checksum file written last is the trial commit marker; this avoids directory rename limitations in the Windows workspace. Completed trials are checked and reused. Reports and aggregate tables are derived views that can be rebuilt.

On resume, partial journals become interrupted outcomes with conservative usage. They are not retried automatically. An intentional retry belongs in a separate run with a declared retry policy. A leftover run lock requires verifying the old process stopped before removing that lock alone.

## ADR 005 — Explicit limits

The core includes frozen-model policies, bounded graphs, allocation-based interactive subgoals, three audit conditions, and replay. Optional adaptation, trained triage, hierarchies/windowing, extra benchmarks, prose entailment, power planning, and energy measurement are deferred. Real readiness remains unverified until local assets and opt-in checks exist. A fake result is never substituted for missing real capability.

Static and interactive calibration have different schemas. The action gate estimates observed rollout success under a recorded continuation protocol. Training paired rollouts must be supplied; labels are not inferred from hidden evaluation plans.

## External API references inspected

- [Laya agent API](https://nandhakishorm.github.io/laya/reference/agent/) and [source](https://github.com/NandhaKishorM/laya/blob/main/laya/agent.py): fixed loading, typed choices, encoding, response normalization.
- [Laya sequence construction](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py): state, question, and candidate retention.
- [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B): checkpoint-specific tokenizer/configuration and inference setup.
- [ALFWorld text environment](https://github.com/alfworld/alfworld/blob/master/alfworld/agents/environment/alfred_tw_env.py): admissible commands, public demangling, and environment success.

These mutable links document inspected integration points; actual runtime pins come from fetch/lock and capability records.
