from dataclasses import replace
from itertools import product
import math
from concurrent.futures import ThreadPoolExecutor
from s1s2lab.config import validate, DEFAULT_BUDGETS
from s1s2lab.domain import Failure, Response, canonical, digest, validate_probabilities, validate_choice, decision_options
from s1s2lab.governor import Budget
from s1s2lab.models.local import parse_final
from s1s2lab.plans import validate_plan, combine, validate_interactive
from tests.support import OfflineTest, fixture_view, fixture_plan, governor


class Contracts(OfflineTest):
    def test_canonical_serialization(self):
        self.assertEqual(digest({"b": 2, "a": 1}), digest({"a": 1, "b": 2}))
        with self.assertRaises(ValueError):
            canonical({"x": float("nan")})

    def test_probabilities(self):
        options = decision_options()
        validate_probabilities(None, options)
        validate_probabilities({f"o{i}": 0.25 for i in range(4)}, options)
        for value in (float("nan"), float("inf"), -0.1, 1.1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_probabilities({"o0": value, "o1": 0.25, "o2": 0.25, "o3": 0.25}, options)
        for invalid in ({"o0": 1.0}, {f"o{i}": 0.1 for i in range(4)}):
            with self.assertRaises(ValueError):
                validate_probabilities(invalid, options)

    def test_unknown_choice_not_more(self):
        with self.assertRaises(ValueError):
            validate_choice({"selected_option_id": "MORE"}, decision_options())

    def test_candidate_permutation_remaps_meaning(self):
        view = fixture_view()
        swapped = replace(view, options=decision_options(("MORE", "NO", "YES", "IRRELEVANT")))
        self.assertEqual(view.meaning("o0"), swapped.meaning("o2"))

    def test_strict_final_parser(self):
        self.assertEqual(parse_final('<think>private</think>{"x":1}'), {"x": 1})
        self.assertEqual(parse_final('reasoning without opening tag</think>```json\n{"x":1}\n```'), {"x": 1})
        for text in ('{"x":1}{"x":2}', 'prose {"x":1}', '<think>unfinished', '[]', '{', '</think>{}</think>{}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_final(text)

    def test_config_rejects_typos_and_bad_budgets(self):
        base = dict(schema_version=1, execution_mode="mock", stage="smoke", policies=["B1_laya"], seeds=[1])
        self.assertEqual(validate(base)["budgets"]["repairs"], 1)
        for extra in ({"bogus": True}, {"budgets": {"replans": -1}}, {"budgets": {"repairs": 2}}, {"seeds": [1, 1]}, {"policies": ["unknown"]}):
            with self.assertRaises(ValueError):
                validate(base | extra)
        with self.assertRaises(ValueError):
            validate(base | {"stage": "confirmatory"})

    def test_plan_graph_rejections(self):
        validate_plan(fixture_plan(), fixture_view(), DEFAULT_BUDGETS)
        for mutation in ("cycle", "duplicate", "source", "code", "depth", "decorative", "stale", "missing"):
            plan = fixture_plan()
            if mutation == "cycle": plan["nodes"][0]["dependencies"] = ["c"]
            if mutation == "duplicate": plan["nodes"][1]["id"] = "a"
            if mutation == "source": plan["nodes"][0]["sources"] = ["private-answer"]
            if mutation == "code": plan["nodes"][2]["operation"] = "__import__('os')"
            if mutation == "depth": plan["nodes"][1]["dependencies"] = ["a"]
            if mutation == "decorative": plan["root"] = "a"
            if mutation == "stale": plan["evidence_version"] = 1
            if mutation == "missing": plan["root"] = "absent"
            limits = DEFAULT_BUDGETS | ({"max_depth": 1} if mutation == "depth" else {})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_plan(plan, fixture_view(), limits)

    def test_budget_atomic_reservations_and_unknown_usage(self):
        budget = Budget(DEFAULT_BUDGETS | {"s2_calls": 2, "generated_tokens": 8192})
        def reserve():
            try:
                return budget.reserve("s2")
            except Failure:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: reserve(), range(20)))
        self.assertEqual(sum(r is not None for r in results), 2)
        budget.reconcile("s2", 4096, Response({}, 20, 50))
        budget.reconcile("s2", 4096, None)
        self.assertEqual(budget.used["generated_tokens"], 4146)
        self.assertEqual(budget.used["unknown_usage_calls"], 1)

    def test_overreported_usage_is_error(self):
        budget = Budget(DEFAULT_BUDGETS)
        with self.assertRaises(Failure):
            budget.reconcile("s2", 10, Response({}, 0, 11))

    def test_stale_actions_cannot_execute(self):
        g, view = governor(), fixture_view()
        options_hash = digest([o.payload() for o in view.options])
        with self.assertRaises(Failure):
            g.authorize_action(replace(view, version=1), "o0", 0, options_hash)
        with self.assertRaises(Failure):
            g.authorize_action(replace(view, options=decision_options(("NO", "YES", "MORE", "IRRELEVANT"))), "o0", 0, options_hash)
        self.assertEqual(g.budget.used["actions"], 0)
