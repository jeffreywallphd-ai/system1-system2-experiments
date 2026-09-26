"""Synthetic interface fixtures, intentionally NOT an oracle or model emulator.

The default always selects the first option and proposes a generic one-leaf
plan. Result quality is meaningless; tests can supply explicit response queues
to exercise errors, routing, and worker influence through the real interfaces.
"""
from collections import deque
from ..domain import Response, canonical, digest


class ScriptedModel:
    def __init__(self, role, responses=()):
        self.role = role
        self.responses = deque(responses)
        self.requests = []
        self.identity = {"adapter": "scripted_fake_v1", "role": role, "revision": "mock-v1",
                         "execution_mode": "mock", "precision": "not_applicable"}

    def infer(self, request, max_tokens, timeout):
        self.requests.append(request.payload())
        if self.responses:
            item = self.responses.popleft()
            if isinstance(item, Exception):
                raise item
            content = item(request) if callable(item) else item
        elif request.purpose == "static_plan":
            content = {"evidence_version": request.view.version, "nodes": [
                {"id": "a", "kind": "DECIDE", "worker": "LAYA",
                 "objective": "The supplied evidence establishes eligibility for this request.",
                 "sources": [s.source_id for s in request.view.sources], "dependencies": []}],
                "root": "a", "relevance": None}
        elif request.purpose == "interactive_plan":
            content = {"evidence_version": request.view.version, "subgoals": [
                {"objective": "Take the next admissible step toward the goal.", "worker": "LAYA", "steps": 3}]}
        elif request.purpose == "audit":
            content = dict(decision_assessment="INSUFFICIENT_EVIDENCE", supporting_source_ids=[],
                           supporting_event_ids=[], claims=[], alternative_explanations=[],
                           limitations=["Scripted fake; no evidence assessment performed."],
                           probe_predictions={key: "YES" for key in request.details.get("probes", {})},
                           explanation_kind="EVIDENCE_AND_BEHAVIOR_NOT_INTERNAL_THOUGHT")
        else:
            ids = [o.option_id for o in request.options]
            content = {"selected_option_id": ids[0]}
            if self.role == "s1":
                content["probabilities"] = {key: (0.7 if i == 0 else 0.3 / (len(ids) - 1))
                                            for i, key in enumerate(ids)} if len(ids) > 1 else {ids[0]: 1.0}
        return Response(content, len(canonical(request.payload()).split()),
                        min(max_tokens, 32) if self.role == "s2" else 0,
                        canonical(content), {"status": "MOCK", "request_hash": digest(request.payload()),
                                             "token_unit": "synthetic_whitespace_units"})

    def close(self):
        pass
