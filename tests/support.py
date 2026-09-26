from copy import deepcopy
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from s1s2lab.domain import CaseView, Source, decision_options
from s1s2lab.config import DEFAULT_BUDGETS
from s1s2lab.governor import Governor
from s1s2lab.models.fake import ScriptedModel


def fixture_view():
    return CaseView("case", "family", "sharc", "May I enter?", (Source("s0", "Entry requires training."),
                    Source("s1", "Training completed.")), decision_options())


def fixture_plan():
    return {"evidence_version": 0, "nodes": [
        {"id": "a", "kind": "DECIDE", "worker": "LAYA", "objective": "Training completed.", "sources": ["s0", "s1"], "dependencies": []},
        {"id": "b", "kind": "DECIDE", "worker": "LAYA", "objective": "No overdue items.", "sources": ["s0", "s1"], "dependencies": []},
        {"id": "c", "kind": "COMBINE", "operation": "AND", "dependencies": ["a", "b"]}], "root": "c", "relevance": None}


def governor(s1=(), s2=(), **limits):
    return Governor({"s1": ScriptedModel("s1", s1), "s2": ScriptedModel("s2", s2)}, DEFAULT_BUDGETS | limits, 17)


class OfflineTest(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("Offline test attempted network access"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
