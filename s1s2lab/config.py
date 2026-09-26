"""Strict configuration: unresolved research choices cannot become silent defaults."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import re
from .storage import read_json

POLICIES = ("B0_simple", "B1_laya", "B2_s2", "B3_cascade", "B4_triage",
            "H1_plan_once", "H2_mixed", "C1_s2_workers")
DEFAULT_BUDGETS = dict(s1_calls=64, s2_calls=10, generated_tokens=16384,
                       tokens_per_call=4096, max_leaves=8, max_depth=2,
                       replans=2, repairs=1, steps=50, model_seconds=600, case_seconds=1800)


def validate(config: dict) -> dict:
    c = deepcopy(config)
    allowed = {"schema_version", "experiment_id", "execution_mode", "stage", "policies",
               "seeds", "budgets", "models", "dataset", "partition", "context_track",
               "routing_artifact", "baseline_artifact", "analysis", "notes", "frozen_protocol"}
    if set(c) - allowed:
        raise ValueError(f"Unknown config keys: {set(c) - allowed}")
    if c.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    if c.get("execution_mode") not in {"mock", "real"}:
        raise ValueError("execution_mode must be explicit")
    if c.get("stage") not in {"smoke", "pilot", "confirmatory"}:
        raise ValueError("Invalid study stage")
    if not c.get("policies") or set(c["policies"]) - set(POLICIES):
        raise ValueError("Unknown/empty policies")
    if len(set(c["policies"])) != len(c["policies"]):
        raise ValueError("Duplicate policies")
    if not c.get("seeds") or any(type(s) is not int or s < 0 for s in c["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if len(c["seeds"]) != len(set(c["seeds"])):
        raise ValueError("Duplicate seeds")
    if set(c.get("budgets", {})) - set(DEFAULT_BUDGETS):
        raise ValueError("Unknown budget")
    c["budgets"] = DEFAULT_BUDGETS | c.get("budgets", {})
    if any(type(v) is not int or v < 0 for v in c["budgets"].values()):
        raise ValueError("Budgets must be nonnegative integers")
    if any(c["budgets"][k] == 0 for k in ("tokens_per_call", "model_seconds", "case_seconds", "steps")):
        raise ValueError("Token, timeout, and action limits must be positive")
    if c["budgets"]["repairs"] > 1:
        raise ValueError("At most one format repair per result")
    if c.get("context_track", "full_coverage") not in {"full_coverage", "common_fit"}:
        raise ValueError("Unknown context track")
    c.setdefault("context_track", "full_coverage")
    c.setdefault("analysis", {"resamples": 1000, "seed": 812})
    if type(c["analysis"].get("resamples")) is not int or c["analysis"]["resamples"] < 100:
        raise ValueError("Use at least 100 resamples")
    c.setdefault("partition", "development")
    if c["partition"] not in {"train", "calibration", "development", "test", "seen", "unseen", "ood"}:
        raise ValueError("Unknown partition")
    if c["execution_mode"] == "real":
        for role in ("s1", "s2"):
            model = c.get("models", {}).get(role, {})
            expected = "convaiinnovations/laya" if role == "s1" else "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            if model.get("model_id") != expected:
                raise ValueError(f"Primary {role} checkpoint must be {expected}")
            if model.get("adapter") != ("laya_sdk" if role == "s1" else "transformers"):
                raise ValueError("Unsupported real adapter; no fallback is available")
    if c["stage"] == "confirmatory":
        if c["execution_mode"] != "real":
            raise ValueError("Mock runs cannot be confirmatory")
        for model in c["models"].values():
            if not re.fullmatch(r"[0-9a-f]{40}", model.get("revision") or ""):
                raise ValueError("Confirmatory model revisions must be immutable 40-character SHAs")
            if not model.get("asset_lock") or not model.get("runtime_lock"):
                raise ValueError("Confirmatory models require asset and runtime locks")
        if not c.get("frozen_protocol"):
            raise ValueError("Confirmatory run requires a frozen protocol file")
        if "B3_cascade" in c["policies"] and not c.get("routing_artifact"):
            raise ValueError("Confirmatory cascade requires a fitted gate")
        if "B0_simple" in c["policies"] and not c.get("baseline_artifact"):
            raise ValueError("Confirmatory simple baseline requires a declared fitting artifact")
    return c


def load(path: str | Path) -> dict:
    """Paths in configs are relative to the invoking repository directory."""
    return validate(read_json(path))
