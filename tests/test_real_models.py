"""Explicit real capability checks. Ordinary CI skips these without imports/downloads.

Set S1S2_RUN_REAL_MODELS=1 and S1S2_REAL_CONFIG to a locked local config.
These check interface readiness, not stochastic benchmark correctness.
"""
import os
import unittest
from s1s2lab.config import load
from s1s2lab.domain import Request, validate_choice, decision_options
from s1s2lab.models.process import ProcessModel
from s1s2lab.plans import validate_plan
from tests.support import fixture_view


@unittest.skipUnless(os.getenv("S1S2_RUN_REAL_MODELS") == "1", "Real model checks are opt-in; no live inference executed")
class RealCapabilities(unittest.TestCase):
    def setUp(self):
        path = os.getenv("S1S2_REAL_CONFIG")
        if not path:
            self.skipTest("BLOCKED: S1S2_REAL_CONFIG not provided")
        self.config = load(path)
        if self.config["execution_mode"] != "real":
            self.fail("Real checks cannot use mock adapters")

    def test_laya_typed_choice_and_encoding(self):
        model = ProcessModel("s1", self.config["models"]["s1"])
        self.addCleanup(model.close)
        view = fixture_view()
        result = model.infer(Request("decision", view.question, view, view.options, {}, 17), 0, 600)
        validate_choice(result.content, view.options)
        self.assertEqual(result.encoding["status"], "VERIFIED")

    def test_reasoner_parsable_plan(self):
        model = ProcessModel("s2", self.config["models"]["s2"])
        self.addCleanup(model.close)
        view = fixture_view()
        result = model.infer(Request("static_plan", "Propose a plan.", view, (), {"limits": self.config["budgets"]}, 17), 4096, 600)
        validate_plan(result.content, view, self.config["budgets"])

    def test_alfworld_reset_step(self):
        from s1s2lab.benchmarks.alfworld import AlfworldEnvironment
        path = os.getenv("S1S2_REAL_GAME")
        if not path:
            self.skipTest("BLOCKED: S1S2_REAL_GAME not provided")
        env = AlfworldEnvironment({"game_path": path}, seed=17, max_steps=3)
        self.addCleanup(env.close)
        observation, commands = env.reset()
        self.assertTrue(observation and commands)
        observation, commands, done, won = env.step(commands[0])
        self.assertIsInstance(won, bool)
