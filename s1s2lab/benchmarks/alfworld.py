"""Observation-only ALFWorld boundary and a tiny synthetic contract environment.

Official game paths and success flags stay runner-side. Every reset creates an
independent environment. No expert wrapper or walkthrough is requested.
"""
from __future__ import annotations
from pathlib import Path
from ..domain import CaseView, Source, Option, Failure


class ObservationMemory:
    def __init__(self, case_id, family_id, goal):
        self.case_id, self.family_id, self.goal = case_id, family_id, goal
        self.sources = []
        self.version = -1

    def append(self, observation, commands, previous_action=None):
        if not isinstance(observation, str) or not isinstance(commands, (list, tuple)):
            raise Failure("ENVIRONMENT_ERROR", "Malformed public observation/action list")
        if not commands or any(not isinstance(c, str) or not c for c in commands) or len(commands) != len(set(commands)):
            raise Failure("ENVIRONMENT_ERROR", "Missing/duplicate admissible commands")
        if previous_action is not None:
            self.sources.append(Source(f"s{len(self.sources)}", previous_action, "selected_action"))
        self.sources.append(Source(f"s{len(self.sources)}", observation, "observation"))
        self.version += 1
        return CaseView(self.case_id, self.family_id, "alfworld", self.goal, tuple(self.sources),
                        tuple(Option(f"o{i}", c, c) for i, c in enumerate(commands)), self.version)


class FakeEnvironment:
    """Three visible next steps. Never reports success from an agent's claim."""
    def __init__(self, game=None, seed=0, max_steps=50):
        self.position, self.steps, self.cap = 0, 0, max_steps
        self.commands = ["go to table 1", "take cup 1 from table 1", "put cup 1 in cabinet 1"]

    def reset(self):
        self.position = self.steps = 0
        return self._observation(), self._commands()

    def _commands(self):
        return [self.commands[min(self.position, 2)], "look"]

    def _observation(self):
        return f"Synthetic room. Next useful command: {self.commands[min(self.position, 2)]}."

    def step(self, command):
        self.steps += 1
        progress = command == self.commands[min(self.position, 2)]
        if progress:
            self.position += 1
        success = self.position == 3
        observation = "The cup is in cabinet 1." if success else self._observation()
        if not progress:
            observation = "Nothing happens. " + observation
        return observation, self._commands(), success or self.steps >= self.cap, success

    def close(self):
        pass


class AlfworldEnvironment:
    def __init__(self, game, seed=0, max_steps=50):
        if not Path(game["game_path"]).is_file():
            raise Failure("ENVIRONMENT_SETUP_FAILURE", "Pinned game file is missing")
        # Uses the official text backend directly, with only the public demangler.
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
        infos = textworld.EnvInfos(won=True, admissible_commands=True)
        env_id = textworld.gym.register_games([game["game_path"]], infos, batch_size=1,
            asynchronous=False, max_episode_steps=max_steps, wrappers=[AlfredDemangler(shuffle=False)])
        self.env = textworld.gym.make(env_id)
        self.env.seed(seed)

    def reset(self):
        observations, infos = self.env.reset()
        return observations[0], list(infos["admissible_commands"][0])

    def step(self, command):
        observations, _rewards, done, infos = self.env.step([command])
        # `won` never enters policy memory; it is the environment-owned endpoint.
        return observations[0], list(infos["admissible_commands"][0]), bool(done[0]), bool(infos["won"][0])

    def close(self):
        self.env.close()
