"""Explicit preparation commands. Public inputs and evaluator data are separate."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from ..domain import digest, decision_options, CaseView, Source
from ..storage import write_json, write_jsonl, checksum_tree, read_json, file_hash
from . import context_rules, sharc


def save_dataset(destination, views, private, provenance, games=None):
    root = Path(destination)
    if root.exists():
        raise ValueError("Prepared datasets are immutable; choose a new output directory")
    root.mkdir(parents=True)
    if len({v.case_id for v in views}) != len(views):
        raise ValueError("Duplicate case IDs")
    if {v.case_id for v in views} != {p["case_id"] for p in private}:
        raise ValueError("Public/private case sets disagree")
    write_jsonl(root / "public.jsonl", [asdict(v) for v in views])
    write_jsonl(root / "private" / "labels.jsonl", private)
    write_jsonl(root / "cases.jsonl", [{k: p[k] for k in ("case_id", "family_id", "benchmark", "partition")}
                                      for p in private])
    if games is not None:
        write_json(root / "games.json", games)
    write_json(root / "manifest.json", {"schema_version": 1, "provenance": provenance,
                                        "cases": len(views), "partitions": dict(Counter(p["partition"] for p in private)),
                                        "families": len({p["family_id"] for p in private})})
    write_json(root / "checksums.json", checksum_tree(root))
    return root


def prepare_context(destination, families=60, seed=17):
    views, private, provenance = context_rules.generate(families, seed)
    return save_dataset(destination, views, private, provenance)


def prepare_sharc(destination, release_manifest, family_disjoint=False):
    manifest = read_json(release_manifest)
    views, private = [], []
    for partition, item in manifest["files"].items():
        if partition not in {"train", "development", "test"} or file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Invalid ShARC release partition or hash")
        records = read_json(item["path"])
        for record in records:
            view, gold = sharc.adapt(record, partition)
            # Calibration is carved from training by normalized rule family.
            if partition == "train" and int(digest(view.family_id)[:8], 16) % 5 == 0:
                gold["partition"] = "calibration"
            views.append(view)
            private.append(gold)
    audit = sharc.split_audit(private)
    if family_disjoint:
        # Union all exact, tree, and near-duplicate aliases before hash splitting.
        parents = {row["family_id"]: row["family_id"] for row in private}
        def find(key):
            while parents[key] != key:
                parents[key] = parents[parents[key]]
                key = parents[key]
            return key
        aliases = sharc.split_audit(private, include_within=True)
        for pair in aliases["exact_rule_overlap"] + aliases["tree_overlap"] + aliases["near_duplicate_overlap"]:
            a, b = find(pair[0]), find(pair[1])
            parents[max(a, b)] = min(a, b)
        from dataclasses import replace
        views = [replace(view, family_id=find(view.family_id)) for view in views]
        for row in private:
            row["family_id"] = find(row["family_id"])
            row["official_partition"] = row["partition"]
            bucket = int(digest(row["family_id"])[:8], 16) % 10
            row["partition"] = "train" if bucket < 5 else "calibration" if bucket < 7 else "development" if bucket < 8 else "test"
    return save_dataset(destination, views, private, {"benchmark_variant": "sharc_four_way_decision_only",
                        "release": manifest, "official_overlap_audit": audit,
                        "track": "family_disjoint_generalization" if family_disjoint else "official_replication_with_training_calibration_carveout",
                        "normalization": "casefold_whitespace_terminal_else_question_v1"})


def prepare_alfworld(destination, release_manifest):
    """An explicit enumerated manifest avoids invented split names or game counts."""
    manifest = read_json(release_manifest)
    views, private, games = [], [], {}
    for game in manifest["games"]:
        if game["partition"] not in {"train", "development", "seen", "unseen"}:
            raise ValueError("Unsupported ALFWorld partition")
        if file_hash(game["game_path"]) != game["sha256"]:
            raise ValueError("ALFWorld game integrity failure")
        cid = game["case_id"]
        fid = game["family_id"]
        views.append(CaseView(cid, fid, "alfworld", "Complete the task described in the environment observation.",
                              (Source("s0", "Initial observation is supplied by reset."),), decision_options()))
        private.append(dict(case_id=cid, family_id=fid, benchmark="alfworld", partition=game["partition"],
                            gold=None, variant="episode", task_family=game["task_family"]))
        games[cid] = game
    return save_dataset(destination, views, private, {"release": manifest, "interface": "text_admissible_commands"}, games)


def prepare_mock(destination):
    """Tiny, hand-authored integration fixtures; no public benchmark is downloaded."""
    root = Path(destination)
    paths = {}
    views, private, provenance = context_rules.generate(60, 17)
    development = sorted({p["family_id"] for p in private if p["partition"] == "development"})[:2]
    rows = [p for p in private if p["family_id"] in development]
    ids = {p["case_id"] for p in rows}
    paths["context_rules"] = save_dataset(root / "context_rules", [v for v in views if v.case_id in ids], rows,
        provenance | {"execution_mode": "mock", "selection": "first_two_development_family_hashes"})
    views, private = [], []
    for index, answer in enumerate(("Yes", "No", "Irrelevant", "Have you completed the course?")):
        view, row = sharc.adapt(dict(utterance_id=f"fixture-{index}", tree_id=f"fixture-tree-{index}",
            snippet=f"Policy {index + 1}: workshop entry requires completed training.",
            scenario=("Training completed." if index == 0 else "Training not completed." if index == 1 else ""),
            question="May I enter?" if index != 2 else "Can I obtain a fishing permit?", history=[], answer=answer,
            evidence={"private": "DO_NOT_EXPOSE_THIS_SENTINEL"}), "development")
        views.append(view)
        private.append(row)
    paths["sharc"] = save_dataset(root / "sharc", views, private,
                                   {"execution_mode": "mock", "benchmark_variant": "synthetic_sharc_schema_fixtures"})
    views, private, games = [], [], {}
    for i in range(2):
        cid = f"fake-game-{i}"
        views.append(CaseView(cid, cid, "alfworld", "Put the cup in cabinet 1.", (Source("s0", "Synthetic game."),), decision_options()))
        private.append(dict(case_id=cid, family_id=cid, benchmark="alfworld", partition="development",
                            gold=None, variant="episode", task_family="synthetic_pick_place"))
        games[cid] = {"execution_mode": "mock"}
    paths["alfworld"] = save_dataset(root / "alfworld", views, private,
                                     {"execution_mode": "mock", "interface": "synthetic_admissible_commands"}, games)
    return paths
