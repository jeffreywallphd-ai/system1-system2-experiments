"""Evaluator-side calibration, development-only gate selection, and provenance.

This module is never imported by policies. They consume a validated frozen
artifact containing only the fitted parameters, not the fitting examples.
"""
from __future__ import annotations
from collections import Counter
import math
from .domain import LABELS, digest

TEMPERATURE_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
THRESHOLD_GRID = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def _partition(rows, expected):
    if not rows or any(r.get("partition") != expected for r in rows):
        raise ValueError(f"Fitting requires only {expected} rows")
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated seed rows must be aggregated or fitted separately")


def scale(probabilities, temperature):
    values = {k: max(v, 1e-12) ** (1 / temperature) for k, v in probabilities.items()}
    total = sum(values.values())
    return {k: v / total for k, v in values.items()}


def fit_gate(calibration_rows, development_rows, model_identity, penalty=0.05):
    _partition(calibration_rows, "calibration")
    _partition(development_rows, "development")
    if {r["family_id"] for r in calibration_rows} & {r["family_id"] for r in development_rows}:
        raise ValueError("Calibration/development families overlap")
    for row in calibration_rows + development_rows:
        if set(row["probabilities"]) != set(LABELS):
            raise ValueError("Calibration requires the fixed four-way semantic schema")
        if row["gold"] not in LABELS or row["fast_prediction"] not in LABELS:
            raise ValueError("Invalid fitting label")
        from .domain import validate_probabilities, Option
        validate_probabilities(row["probabilities"], tuple(Option(x, x, x) for x in LABELS))
    def loss(temperature):
        return sum(-math.log(max(scale(r["probabilities"], temperature)[r["gold"]], 1e-12))
                   for r in calibration_rows) / len(calibration_rows)
    temperature = min(TEMPERATURE_GRID, key=lambda t: (loss(t), t))
    choices = []
    for threshold in THRESHOLD_GRID:
        correct = escalations = retained = errors = 0
        for row in development_rows:
            confidence = scale(row["probabilities"], temperature)[row["fast_prediction"]]
            escalate = confidence < threshold
            prediction = row["slow_prediction"] if escalate else row["fast_prediction"]
            correct += prediction == row["gold"]
            escalations += escalate
            retained += not escalate
            errors += not escalate and prediction != row["gold"]
        choices.append(dict(threshold=threshold, utility=(correct - penalty * escalations) / len(development_rows),
                            coverage=retained / len(development_rows), retained_risk=errors / retained if retained else None,
                            cascade_accuracy=correct / len(development_rows)))
    best = max(choices, key=lambda x: (x["utility"], -x["threshold"]))
    return dict(schema_version=1, artifact_kind="calibrated_gate", execution_mode=model_identity.get("execution_mode", "real"),
                model_identity=model_identity, label_order=list(LABELS), temperature=temperature,
                threshold=best["threshold"], temperature_grid=TEMPERATURE_GRID, threshold_grid=THRESHOLD_GRID,
                objective="accuracy_minus_reasoner_call_penalty", penalty=penalty, risk_coverage=choices,
                calibration_hash=digest(calibration_rows), development_hash=digest(development_rows),
                calibration_ids=[r["case_id"] for r in calibration_rows], development_ids=[r["case_id"] for r in development_rows],
                calibration_families=sorted({r["family_id"] for r in calibration_rows}),
                development_families=sorted({r["family_id"] for r in development_rows}))


def majority(rows):
    _partition(rows, "train")
    if any(r["gold"] not in LABELS for r in rows):
        raise ValueError("Invalid training label")
    counts = Counter(r["gold"] for r in rows)
    return dict(artifact_kind="majority", majority_label=max(LABELS, key=lambda label: counts[label]),
                training_ids=[r["case_id"] for r in rows], training_hash=digest(rows),
                training_families=sorted({r["family_id"] for r in rows}), tie_break=list(LABELS))


def validate_gate(artifact, model_identity, cases, partition):
    if artifact["artifact_kind"] not in {"calibrated_gate", "action_gate"}:
        raise ValueError("Gate schema mismatch")
    if artifact["artifact_kind"] == "calibrated_gate" and artifact["label_order"] != list(LABELS):
        raise ValueError("Gate label ordering mismatch")
    if artifact["model_identity"] != model_identity:
        raise ValueError("Gate checkpoint/configuration identity mismatch")
    if not 0 <= artifact["threshold"] <= 1:
        raise ValueError("Invalid gate parameters")
    if artifact["artifact_kind"] == "calibrated_gate" and not 0 < artifact["temperature"]:
        raise ValueError("Invalid temperature")
    if model_identity.get("execution_mode") != "mock":
        interactive = any(c["benchmark"] == "alfworld" for c in cases)
        if interactive != (artifact["artifact_kind"] == "action_gate"):
            raise ValueError("Static and interactive calibration artifacts are not interchangeable")
    if partition in {"test", "seen", "unseen", "ood"}:
        fitted = set(artifact["calibration_ids"] + artifact["development_ids"])
        families = set(artifact["calibration_families"] + artifact["development_families"])
        if any(c["case_id"] in fitted or c["family_id"] in families for c in cases):
            raise ValueError("Fitted gate overlaps held-out cases/families")


def fit_action_gate(calibration_rows, development_rows, model_identity, penalty=0.05):
    """Calibrate fast-workflow suitability from paired training-game rollouts.

    A label is observed episode success after the proposed fast/slow action and
    a fixed continuation protocol, NOT an invented gold next action. Five fixed
    confidence bins use a declared Beta(1,1) posterior mean. Empty calibration
    bins always escalate. Development then selects the threshold separately.
    """
    _partition(calibration_rows, "calibration")
    _partition(development_rows, "development")
    if {r["family_id"] for r in calibration_rows} & {r["family_id"] for r in development_rows}:
        raise ValueError("Calibration/development families overlap")
    protocols = {r["rollout_protocol_hash"] for r in calibration_rows + development_rows}
    if len(protocols) != 1 or not next(iter(protocols)):
        raise ValueError("Action labels require one declared paired continuation protocol")
    for row in calibration_rows + development_rows:
        confidence = row["selected_probability"]
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid observed selected-option probability")
        if type(row["fast_success"]) is not bool or type(row["slow_success"]) is not bool:
            raise ValueError("Action suitability labels must be observed Boolean episode outcomes")
    bins = [{"count": 0, "successes": 0, "probability": None} for _ in range(5)]
    for row in calibration_rows:
        bucket = bins[min(4, int(row["selected_probability"] * 5))]
        bucket["count"] += 1
        bucket["successes"] += row["fast_success"]
    for bucket in bins:
        if bucket["count"]:
            bucket["probability"] = (bucket["successes"] + 1) / (bucket["count"] + 2)
    grid = []
    for threshold in THRESHOLD_GRID:
        total = calls = retained = 0
        for row in development_rows:
            probability = bins[min(4, int(row["selected_probability"] * 5))]["probability"]
            fast = probability is not None and probability >= threshold
            total += row["fast_success"] if fast else row["slow_success"]
            calls += not fast
            retained += fast
        grid.append(dict(threshold=threshold, utility=(total - penalty * calls) / len(development_rows),
                         coverage=retained / len(development_rows), observed_rollout_success=total / len(development_rows)))
    best = max(grid, key=lambda row: (row["utility"], -row["threshold"]))
    return dict(schema_version=1, artifact_kind="action_gate", model_identity=model_identity,
                threshold=best["threshold"], bins=bins, prior="Beta(1,1)", empty_bin="escalate",
                target="episode_success_after_action_under_fixed_continuation", action_interface="full_list",
                rollout_protocol_hash=next(iter(protocols)), development_grid=grid, penalty=penalty,
                calibration_hash=digest(calibration_rows), development_hash=digest(development_rows),
                calibration_ids=[r["case_id"] for r in calibration_rows], development_ids=[r["case_id"] for r in development_rows],
                calibration_families=sorted({r["family_id"] for r in calibration_rows}),
                development_families=sorted({r["family_id"] for r in development_rows}))
