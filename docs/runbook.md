# Runbook

Run commands from the repository root. Paths in JSON configuration are relative to that working directory. Run from source without installation; `pip install -e .` is optional. Python 3.11/3.12 is the suggested real-backend environment; the offline implementation was also verified on the provided Python 3.14 host.

## Offline reproduction

```console
python -m unittest discover -s tests -v
python -m s1s2lab.cli demo --out artifacts/mock-study-v2
python -m s1s2lab.cli demo --out artifacts/mock-study-v2
```

The second command invocation verifies resume: committed trials are reused. If code/config/data changed, choose a **new** run directory. Do not change hashes to make old runs appear compatible. For a crash leaving `runner.lock`, verify that its recorded PID is no longer running before removing that one file. Partial trials are retained as cancelled outcomes; completed trials are not edited.

The study index links reports; each run also has manifest/config/readiness records, intended trial and case manifests, predictions, event journals, usage, encoding audits, errors, exclusions, metrics, per-case CSV, comparisons, and checksums. Audit/replay studies are separate directories. A report never upgrades `execution_mode=mock` into empirical performance.

## Prepare data

```console
python -m s1s2lab.cli prepare --benchmark context_rules --families 200 --seed 17 --out data/context_rules_v1
```

Inspect `manifest.json` for achieved families and partition sizes. Changing predicate names or seeds does not create a new structural family. The mock smoke study deliberately uses only two development families; it is not a useful sample-size estimate.

ShARC requires a local JSON release manifest. Download official releases separately, review their terms, and hash the actual files. Example schema (replace placeholders):

```json
{
  "release": "a verified release identifier",
  "source": "https://sharc-data.github.io/data.html",
  "files": {
    "train": {"path": "data/raw/sharc_train.json", "sha256": "..."},
    "development": {"path": "data/raw/sharc_dev.json", "sha256": "..."},
    "test": {"path": "data/raw/sharc_test.json", "sha256": "..."}
  }
}
```

```console
python -m s1s2lab.cli prepare --benchmark sharc --release-manifest manifests/sharc_release.json --out data/sharc_official
python -m s1s2lab.cli prepare --benchmark sharc --release-manifest manifests/sharc_release.json --family-disjoint --out data/sharc_disjoint
```

Training rules are deterministically carved into training/calibration. Official development/test stay untouched in the replication track. Review the overlap audit before fitting; fitting refuses shared calibration/development families. The separate disjoint track merges audited aliases before splitting. Nonterminal source answers must be well-formed follow-up questions; malformed rows fail preparation instead of becoming MORE.

For ALFWorld, follow the [official installation and data instructions](https://github.com/alfworld/alfworld) in a supported environment (Linux/WSL2 may be necessary). Never assume game counts. Provide an enumerated release manifest:

```json
{
  "release": "pinned ALFWorld code/data versions",
  "games": [{
    "case_id": "opaque-game-id", "family_id": "underlying-game-family",
    "partition": "seen", "task_family": "pick_and_place_simple",
    "game_path": "data/alfworld/a/game.tw-pddl", "sha256": "..."
  }]
}
```

```console
python -m s1s2lab.cli prepare --benchmark alfworld --release-manifest manifests/alfworld_release.json --out data/alfworld_prepared
```

Use separate configs for `seen` and `unseen`. For 50-step episodes set `s2_calls` to at least 100 with one repair allowed, `generated_tokens` to an explicitly chosen episode budget (e.g. 131072), and `context_track` to `full_coverage`. Increase those budgets for a separately reported 100-step condition. Dynamic context failures stay in the denominator. Every policy/seed resets a new game environment.

## Explicit model setup

Do not use the host's unrelated application environment as an implicit dependency lock. Create a dedicated environment and install a suitable PyTorch build plus [requirements-models.in](../requirements-models.in). NF4 additionally requires bitsandbytes; ALFWorld has separate dependencies. These inputs are starting constraints, not a tested platform-wide lock. Resolve conflicts there, then capture exact installed versions.

The checkpoint licenses are Apache-2.0 (Laya) and MIT (DeepSeek); inspect the model cards and any dependency/data licenses. The 8B model's two-byte weights alone are roughly 16.4 GB, with runtime/KV caches additional. Quantization does not guarantee a particular context will fit.

The following commands are **opt-in setup** and were not executed for the delivered study:

```console
python -m s1s2lab.cli fetch --role s1 --subfolder multilingual --out data/models/laya --lock manifests/laya_assets.json --download
python -m s1s2lab.cli fetch --role s2 --out data/models/deepseek --lock manifests/deepseek_assets.json --download
python -m s1s2lab.cli lock-runtime --out manifests/runtime_local.json
```

Without `--download`, fetch only prints setup information. The download step resolves a concrete Hub commit, uses resumable official downloads, and records local file hashes. Copy the resulting revisions into a copy of `configs/local_low_vram.json` or `configs/reference.json`, and set local paths/devices/precision. English root Laya is a separate short-context condition: omit `subfolder`, fetch/lock that root snapshot, and freeze its own context setting. Never select the checkpoint using test accuracy.

```console
python -m s1s2lab.cli doctor --config configs/my_pilot.json
python -m s1s2lab.cli doctor --config configs/my_pilot.json --load-models
```

The default doctor does not load models or query a hosted endpoint. `--load-models` explicitly probes local loading and reports blocked reasons. Capability tests additionally exercise inference; unavailable assets are blocked/failing checks, never passed fake results.

In PowerShell, opt into the real tests with:

```powershell
$env:S1S2_RUN_REAL_MODELS = '1'
$env:S1S2_REAL_CONFIG = 'configs/my_pilot.json'
$env:S1S2_REAL_GAME = 'data/alfworld/a/game.tw-pddl'
python -m unittest tests.test_real_models -v
Remove-Item Env:S1S2_RUN_REAL_MODELS
```

On Linux, supply those variables as command-local environment variables. No test expects stochastic answers to always be correct; it checks encoding and final contracts. A malformed structured plan is useful compatibility evidence that needs resolving before a study.

## Fit the simple baseline and routing

```console
python -m s1s2lab.cli fit-majority --dataset data/context_rules_v1 --out manifests/context_majority.json
```

For static calibration, run and score B1/B2 on calibration and development partitions separately. Fit from one prespecified seed per example (the fitting API rejects duplicate case IDs). Input JSONL rows contain `case_id`, `family_id`, `partition`, `gold`, `fast_prediction`, four-way semantic `probabilities`, and development `slow_prediction`. Save the S1 config object as model identity. Retain run fingerprints alongside the fitting rows.

```console
python -m s1s2lab.cli calibrate --calibration-rows data/fitting/calibration.jsonl --development-rows data/fitting/development.jsonl --model-identity manifests/s1_identity.json --out manifests/static_gate.json
```

For interactive routing use `calibrate-actions` with training-game observations under a single paired continuation protocol. Rows contain `case_id`, `family_id`, `partition`, `selected_probability`, Boolean `fast_success` and `slow_success`, and `rollout_protocol_hash`. This is observed workflow suitability, not a gold action dataset. Use independent game families for calibration/development and record the cost of generating those labels. A static gate is refused for real ALFWorld. The mock static gate can cross schemas only as an explicit fake integration fixture.

Point the run config's `baseline_artifact` and `routing_artifact` at the fitted files. Missing gates produce `NOT_READY`; they do not silently disable escalation while calling the result a cascade.

## Run, score, compare, audit

```console
python -m s1s2lab.cli validate --config configs/my_pilot.json
python -m s1s2lab.cli plan-run --config configs/my_pilot.json
python -m s1s2lab.cli run --config configs/my_pilot.json --out artifacts/my-pilot --real
python -m s1s2lab.cli score --run artifacts/my-pilot
python -m s1s2lab.cli report --run artifacts/my-pilot
python -m s1s2lab.cli replay --run artifacts/my-pilot --out artifacts/my-replay --limit 6 --real
python -m s1s2lab.cli audit --run artifacts/my-pilot --out artifacts/my-audit --limit 6 --real
```

Auditing is initially restricted to ContextRules. Add `--challenge` for a separate corrupted-decision challenge bundle. No audit edits primary predictions. A failed audit still has its usage and failure status.

After development choices and external protocol review, change the partition to the held-out target in your final config and freeze it:

```console
python -m s1s2lab.cli freeze --config configs/final_candidate.json --protocol manifests/final_protocol.json --out configs/frozen_final.json
python -m s1s2lab.cli run --config configs/frozen_final.json --out artifacts/final-study --real
```

Freeze rejects unresolved revisions, locks, and required fitting artifacts. It does not certify power, capability, or preregistration. Runtime checks also reject changed code/data/artifacts after freezing. Keep final analysis free of prompt/budget retuning.

```console
python -m s1s2lab.cli compare --runs artifacts/sharc-final artifacts/context-final artifacts/alfworld-final --out reports/primary-comparisons.json
```

The runs must share checkpoint/configuration, source/prompts, policy list, seeds, and context track. Benchmark-specific budgets can differ by design. You may supply both seen and unseen ALFWorld runs if their budgets match; they are combined for the overall endpoint with bootstrap resampling stratified by seen/unseen. Their separate reports remain available. Comparing precision profiles requires a separate series, not bypassing compatibility checks.
