from copy import deepcopy
from s1s2lab.domain import Failure
from s1s2lab.policies import Policy
from tests.support import OfflineTest, fixture_view, fixture_plan, governor


class Policies(OfflineTest):
    def test_laya_only_has_no_s2_calls(self):
        g = governor()
        Policy("B1_laya", g).static(fixture_view())
        self.assertEqual((g.budget.used["s1_calls"], g.budget.used["s2_calls"]), (1, 0))

    def test_cascade_escalates_and_charges_both(self):
        g = governor()
        p = Policy("B3_cascade", g, {"temperature": 1.0, "threshold": 0.9})
        p.static(fixture_view())
        self.assertEqual((g.budget.used["s1_calls"], g.budget.used["s2_calls"]), (1, 1))

    def test_unresolved_cascade_is_not_ready(self):
        g = governor()
        with self.assertRaises(Failure) as error:
            Policy("B3_cascade", g).static(fixture_view())
        self.assertEqual(error.exception.status, "NOT_READY")
        self.assertEqual(g.budget.used["s1_calls"], 0)

    def test_triage_reasoning_route(self):
        g = governor(s1=[{"selected_option_id": "o1"}])
        Policy("B4_triage", g).static(fixture_view())
        self.assertEqual((g.budget.used["s1_calls"], g.budget.used["s2_calls"]), (1, 1))

    def test_same_frozen_plan_substitutes_workers_and_leaves_matter(self):
        plan, view = fixture_plan(), fixture_view()
        yes = {"selected_option_id": "o0"}
        no = {"selected_option_id": "o1"}
        mixed = governor(s1=[yes, no])
        control = governor(s2=[yes, yes])
        mixed_choice, _ = Policy("H2_mixed", mixed).execute_plan(view, plan)
        control_choice, _ = Policy("C1_s2_workers", control).execute_plan(view, plan)
        self.assertEqual(view.meaning(mixed_choice), "NO")
        self.assertEqual(view.meaning(control_choice), "YES")
        self.assertEqual(mixed.budget.used["laya_leaves"], 2)
        self.assertEqual(control.budget.used["s2_leaves"], 2)
        self.assertEqual(mixed.models["s1"].requests, control.models["s2"].requests[:1] + [control.models["s2"].requests[1]])

    def test_adaptive_replans_after_actual_uncertainty(self):
        plan = fixture_plan()
        g = governor(s1=[{"selected_option_id": "o2"}, {"selected_option_id": "o0"},
                         {"selected_option_id": "o0"}, {"selected_option_id": "o0"}], s2=[plan, plan])
        choice = Policy("H2_mixed", g).static(fixture_view())
        self.assertEqual(choice, "o0")
        self.assertEqual(g.budget.used["replans"], 1)
        self.assertIn("worker_uncertainty", str(g.models["s2"].requests[1]))

    def test_plan_once_does_not_replan_uncertainty(self):
        g = governor(s1=[{"selected_option_id": "o2"}, {"selected_option_id": "o0"}], s2=[fixture_plan()])
        self.assertEqual(Policy("H1_plan_once", g).static(fixture_view()), "o3")
        self.assertEqual(g.budget.used["supervisor_calls"], 1)

    def test_bounded_repair_counts_original_and_repair(self):
        g = governor(s2=[{"wrong": True}, {"selected_option_id": "o0"}])
        Policy("B2_s2", g).static(fixture_view())
        self.assertEqual(g.budget.used["s2_calls"], 2)
        self.assertEqual(g.budget.used["repairs"], 1)
        self.assertIn("repair", g.models["s2"].requests[1]["details"])
        g = governor(s2=[{"wrong": True}, {"wrong": True}])
        with self.assertRaises(Failure) as error:
            Policy("B2_s2", g).static(fixture_view())
        self.assertEqual(error.exception.status, "INVALID_OUTPUT")
        self.assertEqual(g.budget.used["s2_calls"], 2)

    def test_budget_stops_before_another_call(self):
        g = governor(s2=[{"wrong": True}], s2_calls=1)
        with self.assertRaises(Failure) as error:
            Policy("B2_s2", g).static(fixture_view())
        self.assertEqual(error.exception.status, "BUDGET_EXHAUSTED")
        self.assertEqual(len(g.models["s2"].requests), 1)

    def test_reasoner_unavailable_never_becomes_mock_success(self):
        g = governor()
        del g.models["s2"]
        self.assertEqual(Policy("B1_laya", g).static(fixture_view()), "o0")
        with self.assertRaises(Failure) as error:
            Policy("H2_mixed", g).static(fixture_view())
        self.assertEqual(error.exception.status, "MODEL_UNAVAILABLE")

    def test_failed_call_keeps_reserved_usage(self):
        g = governor(s2=[Failure("TIMEOUT", "timed out")])
        with self.assertRaises(Failure):
            Policy("B2_s2", g).static(fixture_view())
        self.assertEqual(g.budget.used["generated_tokens"], 4096)
        self.assertEqual(g.budget.used["unknown_usage_calls"], 1)

    def test_interactive_worker_error_causes_bounded_replan(self):
        g = governor(s1=[{"invalid": True}, {"selected_option_id": "o0"}])
        Policy("H2_mixed", g).action(fixture_view())
        self.assertEqual(g.budget.used["replans"], 1)
        self.assertEqual(g.budget.used["supervisor_calls"], 2)
        self.assertEqual(g.models["s2"].requests[1]["details"]["trigger"]["kind"], "worker_error")
