from pathlib import Path
from s1s2lab.cli import demo, mock_config
from s1s2lab.runner import run
from s1s2lab.domain import digest, Failure
from s1s2lab.storage import read_json, write_json, read_jsonl, write_jsonl, verify_tree, file_hash
from s1s2lab.models.fake import ScriptedModel
from s1s2lab.auditing import safe_probes, score_audit
from tests.support import OfflineTest


class EndToEnd(OfflineTest):
    def test_full_offline_study_known_scores_and_resume(self):
        demo(self.root / "study")
        study = self.root / "study"
        total = 0
        hashes = {}
        for benchmark in ("context_rules", "sharc", "alfworld"):
            root = study / benchmark
            verify_tree(root)
            predictions = read_jsonl(root / "predictions.jsonl")
            total += len(predictions)
            self.assertEqual(len(predictions), len({r["trial_id"] for r in predictions}))
            self.assertTrue(all(r["execution_mode"] == "mock" for r in predictions))
            self.assertTrue(all(r["status"] == "COMPLETED" for r in predictions))
            metrics = read_json(root / "metrics.json")["results"]
            endpoint = metrics[f"{benchmark}/B1_laya"]["primary_endpoint"]
            self.assertEqual(endpoint, {"sharc": 0.25, "context_rules": 0.0, "alfworld": 1.0}[benchmark])
            self.assertEqual(metrics[f"{benchmark}/B1_laya"]["resource_totals"]["s2_calls"], 0)
            self.assertIn("SYNTHETIC RESULTS", (root / "report.md").read_text(encoding="utf-8"))
            hashes[benchmark] = file_hash(root / "predictions.jsonl")
            self.assertNotIn("DO_NOT_EXPOSE_THIS_SENTINEL", (root / "events.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(total, 176)
        compare = read_json(study / "study_comparisons.json")
        self.assertEqual(len(compare["contrasts"]), 3)
        self.assertTrue(all(r["holm_adjusted_p"] == 1 for r in compare["contrasts"]))
        replay = read_jsonl(study / "matched_replay" / "replay_results.jsonl")
        self.assertEqual(len(replay), 4)
        self.assertEqual(replay[0]["plan_hash"], replay[1]["plan_hash"])
        self.assertEqual(replay[0]["allocated_initial_plan_calls"], 1)
        audit = read_jsonl(study / "audit_random" / "audit_results.jsonl")
        self.assertEqual(len(audit), 6)
        self.assertEqual({r["condition"] for r in audit}, {"E0_posthoc", "E1_trace", "E2_probe"})
        demo(study)
        for benchmark, old_hash in hashes.items():
            self.assertEqual(file_hash(study / benchmark / "predictions.jsonl"), old_hash)
        config = read_json(study / "sharc" / "resolved_config.json")
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            run(config | {"seeds": [99]}, study / "sharc")

    def test_interruption_keeps_attempt_and_resumes_other_cases(self):
        from s1s2lab.benchmarks.prepare import prepare_mock
        from s1s2lab.config import validate
        paths = prepare_mock(self.root / "data")
        c = validate(dict(schema_version=1, experiment_id="interruption", execution_mode="mock", stage="smoke",
                          policies=["B1_laya"], seeds=[1], dataset=str(paths["sharc"]), partition="development"))
        root = self.root / "run"
        bad = ScriptedModel("s1", [KeyboardInterrupt()])
        # ScriptedModel accepts ordinary Exceptions; inject cancellation at infer directly.
        def cancel(*args): raise KeyboardInterrupt()
        bad.infer = cancel
        run(c, root, models={"s1": bad})
        self.assertEqual(len(read_jsonl(root / "predictions.jsonl")), 1)
        run(c, root, models={"s1": ScriptedModel("s1")})
        rows = read_jsonl(root / "predictions.jsonl")
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(r["status"] == "CANCELLED" for r in rows), 1)

    def test_common_fit_exclusions_shared_and_label_free(self):
        from s1s2lab.benchmarks.prepare import prepare_mock
        from s1s2lab.config import validate
        paths = prepare_mock(self.root / "data")
        c = validate(dict(schema_version=1, experiment_id="fit", execution_mode="mock", stage="smoke", policies=["B1_laya", "B2_s2"],
                          seeds=[1], dataset=str(paths["sharc"]), partition="development", context_track="common_fit"))
        model = ScriptedModel("s1", [Failure("CONTEXT_LIMIT", "known overlength fixture")])
        run(c, self.root / "run", models={"s1": model, "s2": ScriptedModel("s2")})
        self.assertEqual(len(read_jsonl(self.root / "run" / "exclusions.jsonl")), 1)
        self.assertEqual(len(read_jsonl(self.root / "run" / "predictions.jsonl")), 6)

    def test_citation_existence_does_not_prove_support(self):
        from s1s2lab.benchmarks.context_rules import generate
        views, rows, _ = generate(1, 3)
        view, row = views[0], rows[0]
        content = dict(supporting_source_ids=["s0"], supporting_event_ids=[], claims=[], decision_assessment="SUPPORTED")
        scores = score_audit(content, view, [], "YES", row)
        self.assertEqual(scores["valid_reference_rate"], 1)
        self.assertFalse(scores["decision_supported_by_cited_evidence"])
        self.assertEqual(scores["prose_claim_entailment"], "UNVERIFIED")

    def test_partial_journal_tail_is_retained(self):
        from s1s2lab.runner import _recover_journal
        path = self.root / "pending"
        path.mkdir()
        (path / "events.jsonl").write_text('{"kind":"start"}\n{"kind":', encoding="utf-8")
        self.assertEqual(_recover_journal(path), [{"kind": "start"}])
        self.assertTrue((path / "interrupted_journal.txt").is_file())

    def test_freeze_binds_configuration_and_locks_without_model_loading(self):
        from s1s2lab.runner import freeze_configuration
        from s1s2lab.benchmarks.prepare import prepare_mock
        from s1s2lab.config import load
        paths = prepare_mock(self.root / "data")
        c = load("configs/local_low_vram.json")
        c.update(dataset=str(paths["sharc"]), policies=["B1_laya"], partition="development")
        for role in ("s1", "s2"):
            asset, runtime = self.root / f"{role}_asset.json", self.root / f"{role}_runtime.json"
            write_json(asset, {"fixture": True})
            write_json(runtime, {"fixture": True})
            c["models"][role].update(revision="a" * 40, asset_lock=str(asset), runtime_lock=str(runtime))
        final = freeze_configuration(c, self.root / "protocol.json", self.root / "frozen.json")
        frozen = read_json(final)
        self.assertEqual(frozen["stage"], "confirmatory")
        self.assertEqual(read_json(self.root / "protocol.json")["config_hash"], digest(frozen))
        with self.assertRaisesRegex(ValueError, "Frozen protocol"):
            run(frozen | {"seeds": [99]}, self.root / "changed", models={})

    def test_mock_adapter_cannot_be_labelled_real(self):
        from s1s2lab.benchmarks.prepare import prepare_mock
        from s1s2lab.config import load
        paths = prepare_mock(self.root / "data")
        c = load("configs/local_low_vram.json")
        c.update(dataset=str(paths["sharc"]), policies=["B1_laya"])
        with self.assertRaisesRegex(ValueError, "substitution refused"):
            run(c, self.root / "bad-mode", models={"s1": ScriptedModel("s1")})
