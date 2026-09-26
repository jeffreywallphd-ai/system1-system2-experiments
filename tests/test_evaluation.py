from copy import deepcopy
from s1s2lab.domain import LABELS
from s1s2lab.evaluation import metrics, quality, paired_comparison, holm
from s1s2lab.calibration import fit_gate, fit_action_gate, majority, validate_gate
from s1s2lab.models.fake import ScriptedModel
from tests.support import OfflineTest


def scored(label, prediction, family="f", seed=17, variant="base"):
    return dict(case_id=family + label + variant, family_id=family, seed=seed, variant=variant, gold=label,
                prediction=prediction, correct=int(label == prediction), status="COMPLETED" if prediction else "INVALID_OUTPUT",
                usage={"s1_calls": 1, "s2_calls": 0}, wall_seconds=0.1, depth=1, probabilities=None)


class Evaluation(OfflineTest):
    def test_invalid_predictions_use_four_class_denominator(self):
        rows = [scored(label, label) for label in LABELS]
        rows[0] = scored("YES", None)
        result = metrics(rows, "sharc", 50)
        self.assertEqual(result["balanced_accuracy"], 0.75)
        self.assertEqual(result["invalid_rate"], 0.25)
        self.assertEqual(result["confusion"]["YES"]["INVALID"], 1)
        self.assertEqual(len(result["class_recall"]), 4)

    def test_missing_predictions_cannot_improve_accuracy(self):
        rows = [scored(label, label) for label in LABELS]
        before = quality(rows, "sharc")
        for i in range(len(rows)):
            rows[i] = rows[i] | {"prediction": None, "correct": 0, "status": "MISSING"}
            after = quality(rows, "sharc")
            self.assertLessEqual(after, before)
            before = after

    def test_invariant_wrong_answers_are_not_robust(self):
        rows = [scored("YES", "NO", variant="base"), scored("YES", "NO", variant="distractor")]
        result = metrics(rows, "context_rules", 50)
        self.assertEqual(result["primary_endpoint"], 0)
        self.assertEqual(result["pair_accuracy"]["distractor"], 0)
        self.assertEqual(result["unnecessary_decision_changes"]["distractor"], 0)

    def test_exact_family_test_and_repeated_seeds(self):
        left = [scored("YES", "NO", family=str(i)) for i in range(4)]
        right = [scored("YES", "YES", family=str(i)) for i in range(4)]
        result = paired_comparison(left, right, "context_rules", 100, 1)
        self.assertEqual(result["difference"], 1)
        self.assertEqual(result["ci95"], [1, 1])
        self.assertEqual(result["p_value"], 0.125)
        doubled = paired_comparison(left + [r | {"seed": 29} for r in left], right + [r | {"seed": 29} for r in right], "context_rules", 100, 1)
        self.assertEqual(doubled["p_value"], result["p_value"])
        self.assertEqual(doubled["clusters"], 4)

    def test_pair_mismatch_refused(self):
        with self.assertRaises(ValueError):
            paired_comparison([scored("YES", "YES", "a")], [scored("YES", "YES", "b")], "context_rules")

    def test_holm_known_example(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_alfworld_bootstrap_preserves_seen_unseen_strata(self):
        left = [scored("YES", "NO", family=str(i)) | {"partition": "seen" if i < 2 else "unseen"} for i in range(4)]
        right = [r | {"prediction": "YES", "correct": 1} for r in left]
        result = paired_comparison(left, right, "alfworld", 100, 1)
        self.assertEqual(result["strata"], {"seen": 2, "unseen": 2})
        self.assertEqual(result["ci95"], [1, 1])

    def test_no_success_cost_is_undefined(self):
        row = scored("YES", "NO")
        self.assertIsNone(metrics([row], "context_rules", 50)["s2_calls_per_success"])

    def test_calibration_partition_and_checkpoint_guards(self):
        identity = ScriptedModel("s1").identity
        calibration = [dict(case_id="cal", family_id="cal-f", partition="calibration", gold="YES", fast_prediction="YES",
                            probabilities={label: 0.25 for label in LABELS})]
        development = [calibration[0] | dict(case_id="dev", family_id="dev-f", partition="development", slow_prediction="YES")]
        artifact = fit_gate(calibration, development, identity)
        validate_gate(artifact, identity, [{"case_id": "fresh", "family_id": "fresh-f"}], "test")
        with self.assertRaises(ValueError):
            fit_gate([calibration[0] | {"partition": "test"}], development, identity)
        with self.assertRaises(ValueError):
            fit_gate(calibration, [development[0] | {"family_id": "cal-f"}], identity)
        with self.assertRaises(ValueError):
            validate_gate(artifact, identity | {"revision": "other"}, [], "test")
        with self.assertRaises(ValueError):
            validate_gate(artifact, identity, [{"case_id": "cal", "family_id": "fresh"}], "test")
        with self.assertRaises(ValueError):
            majority([{"partition": "test", "gold": "YES"}])

    def test_action_suitability_calibration_and_schema_separation(self):
        identity = {"revision": "real-fixture"}
        cal = [dict(case_id="cal", family_id="cal-f", partition="calibration", selected_probability=0.7,
                    fast_success=False, slow_success=True, rollout_protocol_hash="paired-fixed-protocol")]
        dev = [cal[0] | dict(case_id="dev", family_id="dev-f", partition="development")]
        artifact = fit_action_gate(cal, dev, identity)
        self.assertGreater(artifact["threshold"], artifact["bins"][3]["probability"])
        self.assertIsNone(artifact["bins"][0]["probability"])
        validate_gate(artifact, identity, [{"case_id": "new", "family_id": "new", "benchmark": "alfworld"}], "seen")
        with self.assertRaises(ValueError):
            validate_gate(artifact, identity, [{"case_id": "new", "family_id": "new", "benchmark": "sharc"}], "test")
        with self.assertRaises(ValueError):
            fit_action_gate([cal[0] | {"partition": "test"}], dev, identity)
