"""All eight policies share Requests, model access, memory, and action options.

No benchmark/evaluator imports are allowed here. The governor is the only
model-call path; each consequential worker result is recorded before use.
"""
from __future__ import annotations
from dataclasses import replace
import random
from .domain import Failure, LEAF_OPTIONS, ROUTE_OPTIONS, digest, validate_choice
from .plans import validate_plan, validate_interactive, combine


class Policy:
    def __init__(self, policy_id, governor, routing=None, baseline=None):
        self.id, self.g = policy_id, governor
        self.routing, self.baseline = routing, baseline
        self.rng = random.Random(governor.seed)
        self.plan = None
        self.subgoal_index = self.subgoal_steps = 0
        self.last_probabilities = None

    def decide(self, view, role, objective=None, options=None, purpose="decision", details=None):
        options = options or view.options
        choice, response = self.g.request(role, purpose, objective or view.question, view, options,
                                          details or {}, lambda c: validate_choice(c, options))
        self.last_probabilities = response.content.get("probabilities")
        return choice

    def simple_or_routed(self, view):
        if self.id == "B0_simple":
            if view.benchmark == "alfworld":
                return self.rng.choice(view.options).option_id
            if self.baseline is None:
                raise Failure("NOT_READY", "Majority baseline needs a training-only artifact")
            return next(o.option_id for o in view.options if o.meaning == self.baseline["majority_label"])
        if self.id in {"B1_laya", "B2_s2"}:
            return self.decide(view, "s1" if self.id == "B1_laya" else "s2")
        if self.id == "B3_cascade":
            if self.routing is None:
                raise Failure("NOT_READY", "Cascade calibration and development threshold are unresolved")
            choice = self.decide(view, "s1")
            probs = self.last_probabilities
            calibrated = None
            if probs:
                if self.routing.get("artifact_kind") == "action_gate":
                    calibrated = self.routing["bins"][min(4, int(probs[choice] * 5))]["probability"]
                else:
                    # Temperature scaling on probabilities; no invented encoder logits.
                    values = {k: max(v, 1e-12) ** (1 / self.routing["temperature"]) for k, v in probs.items()}
                    calibrated = values[choice] / sum(values.values())
            route = "FAST" if calibrated is not None and calibrated >= self.routing["threshold"] else "REASON"
            self.g.event("route", policy=self.id, route=route, calibrated_selected_probability=calibrated)
            return choice if route == "FAST" else self.decide(view, "s2")
        if self.id == "B4_triage":
            try:
                route_id = self.decide(view, "s1", "Choose a workflow for the current goal.", ROUTE_OPTIONS,
                                       "triage", {"capabilities": {"FAST": "typed choices over supplied context",
                                                                  "REASON": "reasoning and typed choices"}})
                route = next(o.meaning for o in ROUTE_OPTIONS if o.option_id == route_id)
            except Failure as exc:
                if exc.status != "INVALID_OUTPUT":
                    raise
                route = "REASON"
                self.g.event("routing_fallback", reason="invalid triage contract")
            self.g.event("route", policy=self.id, route=route)
            return self.decide(view, "s1" if route == "FAST" else "s2")
        raise ValueError("Not a simple/routing policy")

    def propose(self, view, interactive=False, trigger=None):
        purpose = "interactive_plan" if interactive else "static_plan"
        validator = (lambda p: validate_interactive(p, view, self.g.budget.limits)) if interactive else (
            lambda p: (validate_plan(p, view, self.g.budget.limits), p)[1])
        details = {"limits": self.g.budget.limits, "trigger": trigger,
                   "worker_restriction": "LAYA leaves only" if self.id == "H1_plan_once" else "LAYA and S2",
                   "previous_plan": self.plan}
        plan, _ = self.g.request("s2", purpose, "Construct task contracts, without completed worker answers.",
                                  view, (), details, validator)
        if self.id == "H1_plan_once":
            leaves = plan["subgoals"] if interactive else [n for n in plan["nodes"] if n["kind"] != "COMBINE"]
            if any(n["worker"] != "LAYA" for n in leaves):
                raise Failure("INVALID_OUTPUT", "Plan-once requires eligible Laya workers; no hidden S2 fallback")
        self.g.event("plan", plan=plan, source="supervisor", trigger=trigger)
        self.plan = plan
        return plan

    def execute_plan(self, view, plan):
        """Also used by matched-plan replay, with identical validation/contracts."""
        ordered = validate_plan(plan, view, self.g.budget.limits)
        values = {}
        for node in ordered:
            if node["kind"] == "COMBINE":
                value = combine(node["operation"], [values[d] for d in node["dependencies"]])
                role = "deterministic"
            else:
                role = "s2" if self.id == "C1_s2_workers" or node["worker"] == "S2" else "s1"
                sources = tuple(s for s in view.sources if s.source_id in node["sources"])
                leaf_view = replace(view, sources=sources)
                result = self.decide(leaf_view, role, node["objective"], LEAF_OPTIONS, "leaf",
                                     {"dependencies": {d: values[d] for d in node["dependencies"]},
                                      "provenance": "dependencies are model hypotheses, not observations"})
                value = next(o.meaning for o in LEAF_OPTIONS if o.option_id == result)
                self.g.budget.used["laya_leaves" if role == "s1" else "s2_leaves"] += 1
            values[node["id"]] = value
            self.g.event("task_result", task_id=node["id"], worker=role, result=value,
                         evidence_version=view.version, working_version=len(values))
        relevant = values.get(plan["relevance"], "SUPPORTED")
        decision = "IRRELEVANT" if relevant == "CONTRADICTED" else "MORE" if relevant == "UNKNOWN" else {
            "SUPPORTED": "YES", "CONTRADICTED": "NO", "UNKNOWN": "MORE"}[values[plan["root"]]]
        self.g.event("aggregation", method="deterministic_three_valued", values=values, decision=decision)
        return next(o.option_id for o in view.options if o.meaning == decision), values

    def static(self, view):
        if self.id.startswith("B"):
            return self.simple_or_routed(view)
        trigger = None
        for attempt in range(1 + (0 if self.id == "H1_plan_once" else self.g.budget.limits["replans"])):
            try:
                plan = self.propose(view, trigger=trigger)
                choice, values = self.execute_plan(view, plan)
                if "UNKNOWN" not in values.values() or self.id == "H1_plan_once" or attempt == self.g.budget.limits["replans"]:
                    return choice
                trigger = {"kind": "worker_uncertainty", "results": values}
            except Failure as exc:
                if self.id == "H1_plan_once" or exc.status not in {"INVALID_OUTPUT", "MODEL_ERROR"} or attempt == self.g.budget.limits["replans"]:
                    raise
                trigger = {"kind": "worker_or_plan_error", "status": exc.status, "message": str(exc)}
            trigger["event_id"] = self.g.event("replan_trigger", trigger=trigger.copy())
            self.g.budget.used["replans"] += 1
        raise AssertionError("Unreachable")

    def action(self, view, trigger=None):
        if self.id.startswith("B"):
            choice = self.simple_or_routed(view)
        else:
            exhausted = self.plan is not None and self.subgoal_index >= len(self.plan["subgoals"])
            can_replan = self.id != "H1_plan_once" and self.g.budget.used["replans"] < self.g.budget.limits["replans"]
            if self.plan is None or ((trigger or exhausted) and can_replan):
                if self.plan is not None:
                    self.g.budget.used["replans"] += 1
                self.propose(view, interactive=True, trigger=trigger or ("subgoals_expired" if exhausted else None))
                self.subgoal_index = self.subgoal_steps = 0
            if self.subgoal_index >= len(self.plan["subgoals"]):
                raise Failure("PLAN_EXHAUSTED", "Subgoal allocations exhausted without environment success")
            goal = self.plan["subgoals"][self.subgoal_index]
            role = "s2" if self.id == "C1_s2_workers" or goal["worker"] == "S2" else "s1"
            try:
                choice = self.decide(view, role, goal["objective"], details={"plan": self.plan,
                                     "subgoal_index": self.subgoal_index, "goal": view.question})
            except Failure as exc:
                if (self.id == "H1_plan_once" or exc.status not in {"INVALID_OUTPUT", "MODEL_ERROR"}
                        or self.g.budget.used["replans"] >= self.g.budget.limits["replans"]):
                    raise
                trigger = {"kind": "worker_error", "status": exc.status, "error": str(exc)}
                trigger["event_id"] = self.g.event("replan_trigger", trigger=trigger.copy())
                return self.action(view, trigger)
            self.g.budget.used["laya_leaves" if role == "s1" else "s2_leaves"] += 1
            self.subgoal_steps += 1
            if self.subgoal_steps >= goal["steps"]:
                self.subgoal_index += 1
                self.subgoal_steps = 0
        return choice, view.version, digest([o.payload() for o in view.options])
