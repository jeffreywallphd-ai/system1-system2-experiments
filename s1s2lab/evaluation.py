"""Post-run scoring and paired family-clustered inference, using saved records.

Invalid, failed, and missing predictions stay in the denominator. Repeated
seeds are paired and averaged within families; they are not independent cases.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import itertools
import math
from pathlib import Path
import random
import statistics
from .domain import LABELS, digest
from .storage import (read_json, write_json, read_jsonl, write_jsonl, checksum_tree, verify_tree)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def quality(rows, benchmark):
    if not rows:
        return None
    if benchmark == "alfworld":
        return mean(r["correct"] for r in rows)
    if benchmark == "sharc":
        recalls = [mean(r["correct"] for r in rows if r["gold"] == label) for label in LABELS]
        return mean(recalls) if all(v is not None for v in recalls) else None
    families = defaultdict(list)
    for row in rows:
        families[(row["family_id"], row["seed"])].append(row["correct"])
    by_family = defaultdict(list)
    for (family, _), correct in families.items():
        by_family[family].append(int(all(correct)))
    return mean(mean(values) for values in by_family.values())


def probability_metrics(rows):
    valid = [r for r in rows if r.get("probabilities") is not None]
    if not valid:
        return {"probability_cases": 0, "brier": None, "log_loss": None, "ece_10_bins": None}
    brier, loss, bins = [], [], [[] for _ in range(10)]
    for row in valid:
        probs = row["probabilities"]
        brier.append(sum((probs[k] - (row["gold"] == k)) ** 2 for k in LABELS))
        loss.append(-math.log(max(probs[row["gold"]], 1e-12)))
        confidence = probs.get(row["prediction"], 0)
        bins[min(9, int(confidence * 10))].append((confidence, row["correct"]))
    ece = sum(len(b) / len(valid) * abs(mean(x[0] for x in b) - mean(x[1] for x in b)) for b in bins if b)
    return dict(probability_cases=len(valid), brier=mean(brier), log_loss=mean(loss), ece_10_bins=ece,
                log_loss_clip=1e-12, reliability_bins=[dict(count=len(b), confidence=mean(x[0] for x in b),
                    accuracy=mean(x[1] for x in b)) for b in bins])


def metrics(rows, benchmark, step_cap):
    n = len(rows)
    result = dict(benchmark=benchmark, intended=n, completed=sum(r["status"] == "COMPLETED" for r in rows),
                  attempted=sum(r["status"] not in {"MISSING", "NOT_READY"} for r in rows),
                  missing=sum(r["status"] == "MISSING" for r in rows),
                  primary_endpoint=quality(rows, benchmark), accuracy=mean(r["correct"] for r in rows),
                  failures=dict(Counter(r["status"] for r in rows if r["status"] != "COMPLETED")),
                  families=len({r["family_id"] for r in rows}), cases=len({r["case_id"] for r in rows}))
    usage_keys = sorted({k for r in rows for k in r.get("usage", {})})
    result["resource_totals"] = {k: sum(r.get("usage", {}).get(k, 0) for r in rows) for k in usage_keys}
    result["mean_wall_seconds"] = mean(r["wall_seconds"] for r in rows if r.get("wall_seconds") is not None)
    successes = sum(r["correct"] for r in rows)
    result["s2_calls_per_success"] = result["resource_totals"].get("s2_calls", 0) / successes if successes else None
    result["wall_time_complete"] = all(r.get("wall_seconds") is not None for r in rows)
    if benchmark == "alfworld":
        result.update(mean_capped_steps=mean(r.get("usage", {}).get("actions", 0) if r["correct"] else step_cap for r in rows),
                      successful_only_steps=mean(r["usage"]["actions"] for r in rows if r["correct"]),
                      repeated_actions=sum(r.get("repeated_actions", 0) for r in rows),
                      by_task_family={f: mean(r["correct"] for r in rows if r.get("task_family") == f)
                                      for f in sorted({r.get("task_family", "unknown") for r in rows})})
        return result
    confusion = {label: {pred: 0 for pred in (*LABELS, "INVALID")} for label in LABELS}
    for r in rows:
        confusion[r["gold"]][r["prediction"] if r["prediction"] in LABELS else "INVALID"] += 1
    recalls, f1 = {}, []
    for label in LABELS:
        true_positive = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[g][label] for g in LABELS)
        recalls[label] = true_positive / actual if actual else None
        f1.append(2 * true_positive / (actual + predicted) if actual + predicted else 0)
    more_actual = sum(confusion["MORE"].values())
    more_predicted = sum(confusion[g]["MORE"] for g in LABELS)
    terminal_actual = n - more_actual
    result.update(confusion=confusion, class_recall=recalls,
                  balanced_accuracy=mean(recalls.values()) if None not in recalls.values() else None,
                  macro_f1=mean(f1), more_precision=confusion["MORE"]["MORE"] / more_predicted if more_predicted else None,
                  more_recall=recalls["MORE"], invalid_rate=sum(confusion[g]["INVALID"] for g in LABELS) / n,
                  unnecessary_clarification=sum(confusion[g]["MORE"] for g in LABELS if g != "MORE") / terminal_actual if terminal_actual else None,
                  inappropriate_terminal=sum(confusion["MORE"][p] for p in LABELS if p != "MORE") / more_actual if more_actual else None,
                  **probability_metrics(rows))
    if benchmark == "context_rules":
        families = defaultdict(dict)
        for r in rows:
            families[(r["family_id"], r["seed"])][r["variant"]] = r
        pairs = defaultdict(list)
        unnecessary_changes = defaultdict(list)
        for variants in families.values():
            base = variants.get("base")
            if not base:
                continue
            for kind, variant in variants.items():
                if kind != "base":
                    pairs[kind].append(int(base["correct"] and variant["correct"]))
                if kind in {"distractor", "paraphrase", "option_order"}:
                    unnecessary_changes[kind].append(int(base["prediction"] != variant["prediction"]))
        result.update(pair_accuracy={k: mean(v) for k, v in pairs.items()},
                      unnecessary_decision_changes={k: mean(v) for k, v in unnecessary_changes.items()},
                      by_depth={str(d): mean(r["correct"] for r in rows if r.get("depth") == d)
                                for d in sorted({r["depth"] for r in rows})})
    return result


def _paired_groups(left, right):
    key = lambda row: (row["case_id"], row["seed"])
    a, b = {key(r): r for r in left}, {key(r): r for r in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError("Paired comparison requires identical unique cases and seeds")
    groups = defaultdict(list)
    for k in a:
        if a[k]["family_id"] != b[k]["family_id"]:
            raise ValueError("Paired family identities disagree")
        groups[a[k]["family_id"]].append((a[k], b[k]))
    return list(groups.values())


def paired_comparison(left, right, benchmark, resamples=1000, seed=812):
    """Difference = right minus left; bootstrap/permutation unit = whole family."""
    groups = _paired_groups(left, right)
    a, b = quality(left, benchmark), quality(right, benchmark)
    if a is None or b is None or not groups:
        return {"difference": None, "ci95": None, "p_value": None, "reason": "Undefined endpoint (e.g. absent ShARC class)"}
    delta = b - a
    rng, draws, incomplete = random.Random(seed), [], 0
    strata = defaultdict(list)
    for group in groups:
        partitions = {row.get("partition", "unspecified") for pair in group for row in pair}
        if benchmark == "alfworld" and len(partitions) != 1:
            raise ValueError("ALFWorld family crosses seen/unseen strata; resolve game grouping before comparison")
        strata[next(iter(partitions)) if benchmark == "alfworld" else "all"].append(group)
    ratios = []
    for _ in range(resamples):
        sample = [rng.choice(stratum) for stratum in strata.values() for _ in stratum]
        left_sample, right_sample = [], []
        for i, group in enumerate(sample):
            left_sample.extend(x | {"family_id": str(i)} for x, _ in group)
            right_sample.extend(y | {"family_id": str(i)} for _, y in group)
        qa, qb = quality(left_sample, benchmark), quality(right_sample, benchmark)
        if qa is None or qb is None:
            incomplete += 1
        else:
            draws.append(qb - qa)
        denominator = sum(x.get("usage", {}).get("s2_calls", 0) for x in left_sample)
        if denominator:
            ratios.append(sum(x.get("usage", {}).get("s2_calls", 0) for x in right_sample) / denominator)
    def interval(values):
        values = sorted(values)
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]] if values else None
    exact = len(groups) <= 12
    masks = itertools.product((False, True), repeat=len(groups)) if exact else (
        [rng.choice((False, True)) for _ in groups] for _ in range(resamples))
    extreme = count = 0
    for mask in masks:
        perm_a, perm_b = [], []
        for swap, group in zip(mask, groups):
            perm_a.extend(y if swap else x for x, y in group)
            perm_b.extend(x if swap else y for x, y in group)
        difference = quality(perm_b, benchmark) - quality(perm_a, benchmark)
        extreme += abs(difference) >= abs(delta) - 1e-12
        count += 1
    return dict(difference=delta, ci95=interval(draws) if not incomplete and len(groups) > 1 else None,
                p_value=extreme / count if exact else (extreme + 1) / (count + 1),
                bootstrap_resamples=resamples, undefined_bootstrap_draws=incomplete,
                ci_limitation="CI withheld if any draw lacks a required class or fewer than two families" if incomplete or len(groups) < 2 else None,
                s2_call_ratio_ci95=interval(ratios) if len(ratios) == resamples and len(groups) > 1 else None,
                permutation="exact_family_swap" if exact else "monte_carlo_family_swap",
                clusters=len(groups), strata={key: len(value) for key, value in strata.items()}, seed=seed, confidence_level=0.95)


def holm(p_values):
    """Holm step-down adjusted p-values in input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, previous = [None] * len(order), 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * p_values[i]))
        adjusted[i] = previous
    return adjusted


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_run(run_path):
    root = Path(run_path)
    verify_tree(root)
    manifest = read_json(root / "manifest.json")
    config = manifest["binding"]["config"]
    dataset = Path(config["dataset"])
    verify_tree(dataset)
    if read_json(dataset / "checksums.json") != manifest["binding"]["dataset_checksums"]:
        raise ValueError("Scoring data differs from the frozen run data")
    labels = {r["case_id"]: r for r in read_jsonl(dataset / "private" / "labels.jsonl")}
    predictions = read_jsonl(root / "predictions.jsonl")
    key = lambda r: (r["case_id"], r["policy_id"], r["seed"])
    recorded = {key(r): r for r in predictions}
    if len(recorded) != len(predictions):
        raise ValueError("Duplicate predictions")
    rows = []
    for expected in read_jsonl(root / "intended_trials.jsonl"):
        result = recorded.get(key(expected), expected | dict(prediction=None, success=None, status="MISSING", usage={}, wall_seconds=None))
        private = labels[expected["case_id"]]
        correct = bool(result.get("success")) if expected["benchmark"] == "alfworld" else (
            result["status"] == "COMPLETED" and result["prediction"] == private["gold"])
        rows.append(result | {k: private[k] for k in ("gold", "variant", "partition", "depth", "task_family") if k in private} | {"correct": int(correct)})
    all_metrics, comparisons = {}, []
    for benchmark in sorted({r["benchmark"] for r in rows}):
        for policy in config["policies"]:
            subset = [r for r in rows if r["policy_id"] == policy and r["benchmark"] == benchmark]
            all_metrics[f"{benchmark}/{policy}"] = metrics(subset, benchmark, config["budgets"]["steps"])
        if "B1_laya" in config["policies"]:
            left = [r for r in rows if r["policy_id"] == "B1_laya" and r["benchmark"] == benchmark]
            for policy in config["policies"]:
                if policy == "B1_laya":
                    continue
                right = [r for r in rows if r["policy_id"] == policy and r["benchmark"] == benchmark]
                comparisons.append(dict(benchmark=benchmark, left="B1_laya", right=policy,
                    designation="primary_candidate" if policy == "H2_mixed" else "exploratory",
                    **paired_comparison(left, right, benchmark, **config["analysis"])))
    write_json(root / "metrics.json", {"execution_mode": config["execution_mode"], "results": all_metrics})
    write_jsonl(root / "scored.jsonl", rows)
    write_csv(root / "per_case_results.csv", [{k: r.get(k) for k in ("case_id", "family_id", "benchmark", "policy_id", "seed", "gold", "prediction", "correct", "status", "wall_seconds")} | r["usage"] for r in rows])
    write_json(root / "comparisons.json", comparisons)
    write_csv(root / "comparisons.csv", comparisons)
    write_json(root / "checksums.json", checksum_tree(root))
    return all_metrics


def compare_study(paths, destination):
    manifests = [read_json(Path(p) / "manifest.json") for p in paths]
    for path in paths:
        verify_tree(Path(path))
    def compatibility(m):
        c = m["binding"]["config"]
        return (c["execution_mode"], c.get("models"), c["context_track"], c["policies"],
                c["seeds"], c["stage"], c["analysis"],
                {path: sha for path, sha in m["binding"]["asset_locks"].items() if path != c.get("frozen_protocol")},
                m["binding"]["software"]["python"], m["binding"]["software"]["source_hash"], m["binding"]["software"]["prompt_hash"])
    if any(compatibility(m) != compatibility(manifests[0]) for m in manifests):
        raise ValueError("Incompatible model, context, policy, seed, or software manifests")
    by_benchmark = defaultdict(list)
    for path, manifest in zip(paths, manifests):
        scored = read_jsonl(Path(path) / "scored.jsonl")
        for benchmark in {r["benchmark"] for r in scored}:
            by_benchmark[benchmark].append((manifest, [r for r in scored if r["benchmark"] == benchmark]))
    if set(by_benchmark) != {"sharc", "context_rules", "alfworld"}:
        raise ValueError("The prespecified Holm family requires exactly three benchmark contrasts")
    primary = []
    for benchmark, entries in by_benchmark.items():
        if len(entries) > 1:
            partitions = [m["binding"]["config"]["partition"] for m, _ in entries]
            budgets = [m["binding"]["config"]["budgets"] for m, _ in entries]
            if benchmark != "alfworld" or sorted(partitions) != ["seen", "unseen"] or budgets[0] != budgets[1]:
                raise ValueError("Only matched seen/unseen ALFWorld runs may be combined within a benchmark")
        rows = [r for _, records in entries for r in records]
        left = [r for r in rows if r["policy_id"] == "B1_laya"]
        right = [r for r in rows if r["policy_id"] == "H2_mixed"]
        if not left or not right:
            raise ValueError("Primary policies are missing")
        primary.append(dict(benchmark=benchmark, left="B1_laya", right="H2_mixed", designation="primary_candidate",
            **paired_comparison(left, right, benchmark, **entries[0][0]["binding"]["config"]["analysis"])))
    if any(row["p_value"] is None for row in primary):
        raise ValueError("A primary endpoint is undefined")
    for row, adjusted in zip(primary, holm([r["p_value"] for r in primary])):
        row["holm_adjusted_p"] = adjusted
    output = {"execution_mode": manifests[0]["execution_mode"], "contrasts": primary,
              "family": "H2_mixed versus B1_laya on the three prespecified primary endpoints",
              "run_fingerprints": [m["fingerprint"] for m in manifests]}
    write_json(destination, output)
    return output
