"""Four-way decision-only ShARC adapter with a recursive public allowlist."""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from ..domain import CaseView, Source, decision_options, digest


def text(value, field, allow_empty=False):
    if not isinstance(value, str) or (not value.strip() and not allow_empty):
        raise ValueError(f"Malformed ShARC {field}")
    return value.strip()


def normalize_answer(answer) -> str:
    answer = text(answer, "answer")
    normalized = " ".join(answer.casefold().split())
    terminal = {"yes": "YES", "no": "NO", "irrelevant": "IRRELEVANT"}
    if normalized in terminal:
        return terminal[normalized]
    # Conservative adaptation: malformed terminal strings must not become MORE.
    if not answer.endswith("?") or len(answer.split()) < 2:
        raise ValueError("Nonterminal ShARC answer must be a nonempty follow-up question")
    return "MORE"


def adapt(record: dict, partition: str) -> tuple[CaseView, dict]:
    snippet = text(record.get("snippet"), "snippet")
    question = text(record.get("question"), "question")
    scenario = text(record.get("scenario"), "scenario", allow_empty=True)
    history = record.get("history")
    if not isinstance(history, list):
        raise ValueError("ShARC history must be a list")
    blocks = [Source("s0", snippet, "rule"), Source("s1", scenario, "scenario")]
    for item in history:
        if not isinstance(item, dict):
            raise ValueError("Malformed history record")
        # Reconstruct two scalar strings; never serialize the history dictionary.
        q = text(item.get("follow_up_question"), "history question")
        a = text(item.get("follow_up_answer"), "history answer")
        blocks.append(Source(f"s{len(blocks)}", f"Question: {q}\nAnswer: {a}", "dialogue"))
    normalized = re.sub(r"\s+", " ", snippet.casefold()).strip()
    fid = digest(normalized)[:20]
    cid = str(record.get("utterance_id") or digest([question, scenario, [s.text for s in blocks]]))
    view = CaseView(cid, fid, "sharc", question, tuple(blocks), decision_options())
    private = dict(case_id=cid, family_id=fid, benchmark="sharc", partition=partition,
                   gold=normalize_answer(record.get("answer")), tree_id=str(record.get("tree_id", "")),
                   rule_text=normalized, variant="base")
    return view, private


def split_audit(private: list[dict], threshold=0.95, include_within=False) -> dict:
    """Report official overlap without silently relabeling the replication track."""
    exact, tree, near, duplicate_ids = [], [], [], []
    seen = {}
    for row in private:
        if row["case_id"] in seen:
            duplicate_ids.append(row["case_id"])
        seen[row["case_id"]] = row["partition"]
    rules = list({(r["partition"], r["family_id"], r.get("tree_id", ""), r["rule_text"])
                  for r in private})
    for i, left in enumerate(rules):
        for right in rules[i + 1:]:
            if left[0] == right[0] and not include_within:
                continue
            pair = [left[1], right[1], left[0], right[0]]
            if left[1] == right[1]:
                exact.append(pair)
            elif SequenceMatcher(None, left[3], right[3], autojunk=False).ratio() >= threshold:
                near.append(pair)
            if left[2] and left[2] == right[2]:
                tree.append(pair)
    return dict(exact_rule_overlap=exact, tree_overlap=tree, near_duplicate_overlap=near,
                duplicate_case_ids=duplicate_ids, near_duplicate_threshold=threshold)
