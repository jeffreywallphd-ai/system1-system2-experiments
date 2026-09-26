from types import SimpleNamespace, ModuleType
from unittest.mock import patch
from s1s2lab.domain import Request, Failure, Option
from s1s2lab.models.local import LayaModel
from tests.support import OfflineTest, fixture_view


class AdapterContracts(OfflineTest):
    def make_adapter(self, truncate=False, drift=False):
        adapter = LayaModel.__new__(LayaModel)
        adapter.config = {"max_len": 8192, "head_max_len": 512}
        def encode(text, **kwargs): return text.split()
        def actual(state, ids, internal, max_len, head_max_len):
            return [{"ids": [1, 2] if truncate and max_len == 8192 else [1, 2, 3], "markers": [1]}]
        def predict(*args, **kwargs):
            if drift: return {"new_sdk_field": True}
            return {"answers": {"q": {"choice": "o0", "probabilities": {f"o{i}": 0.25 for i in range(4)},
                                       "action": {"act_probability": 1.0}}}, "usage": {"input_tokens": 3}}
        adapter.agent = SimpleNamespace(_to_internal=lambda q: q, tok=SimpleNamespace(mask_token="[MASK]", encode=encode),
                                        _encode_state=actual, predict=predict, device="cpu")
        module = ModuleType("laya.common")
        module.encode_text = lambda tok, text, **kwargs: {"input_ids": encode(text)}
        module.render_options = lambda q: [f"{k}: {v}" for k, v in q["criteria"].items()]
        self.module_patch = patch.dict("sys.modules", {"laya.common": module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        return adapter

    def test_laya_response_shape_and_encoding(self):
        model = self.make_adapter()
        view = fixture_view()
        result = model.infer(Request("decision", view.question, view, view.options, {}, 17), 0, 1)
        self.assertEqual(result.content["selected_option_id"], "o0")
        self.assertNotIn("act_probability", result.content)
        self.assertEqual(result.encoding["status"], "VERIFIED")
        self.assertEqual(result.output_tokens, 0)

    def test_laya_actual_truncation_refused(self):
        model = self.make_adapter(truncate=True)
        view = fixture_view()
        with self.assertRaises(Failure) as exc:
            model.infer(Request("decision", view.question, view, view.options, {}, 17), 0, 1)
        self.assertEqual(exc.exception.status, "CONTEXT_LIMIT")

    def test_individual_option_cap_refused(self):
        model = self.make_adapter()
        view = fixture_view()
        options = (Option("o0", "very " * 60, "YES"), Option("o1", "short", "NO"))
        with self.assertRaisesRegex(Failure, "48 tokens"):
            model.infer(Request("decision", view.question, view, options, {}, 17), 0, 1)

    def test_sdk_response_drift_not_silently_normalized(self):
        model = self.make_adapter(drift=True)
        view = fixture_view()
        with self.assertRaises(KeyError):
            model.infer(Request("decision", view.question, view, view.options, {}, 17), 0, 1)
