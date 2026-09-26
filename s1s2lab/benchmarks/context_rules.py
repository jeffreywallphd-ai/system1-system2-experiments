"""ContextRules v1: read-once independent Boolean rules with explicit corrections.

This is a synthetic diagnostic. Semantic family = canonical expression shape,
including negation; renaming predicates does not create new families. The
oracle below is never imported by any policy or model adapter.
"""
from __future__ import annotations
from dataclasses import asdict
from itertools import product
import random
from ..domain import CaseView, Source, decision_options, digest


def variables(expr) -> list[str]:
    if isinstance(expr, str):
        return [expr]
    return [v for child in expr[1:] for v in variables(child)]


def validate_expression(expr) -> None:
    if isinstance(expr, str):
        if not expr.startswith("p") or not expr[1:].isdigit():
            raise ValueError("Invalid predicate")
        return
    if not isinstance(expr, (tuple, list)) or not expr:
        raise ValueError("Invalid expression")
    if expr[0] not in {"AND", "OR", "NOT"} or len(expr) != (2 if expr[0] == "NOT" else 3):
        raise ValueError("Invalid Boolean operator/arity")
    for child in expr[1:]:
        validate_expression(child)
    names = variables(expr)
    if len(names) != len(set(names)):
        raise ValueError("ContextRules v1 prohibits repeated/dependent predicates")


def evaluate(expr, facts: dict[str, bool | None]) -> bool | None:
    if isinstance(expr, str):
        return facts.get(expr)
    values = [evaluate(child, facts) for child in expr[1:]]
    if expr[0] == "NOT":
        return None if values[0] is None else not values[0]
    if expr[0] == "AND":
        return False if False in values else (None if None in values else True)
    return True if True in values else (None if None in values else False)


def completion_oracle(expr, facts: dict) -> bool | None:
    """Independent two-valued evaluator, enumerating every unknown completion."""
    def binary(node, assignment):
        if isinstance(node, str):
            return assignment[node]
        if node[0] == "NOT":
            return not binary(node[1], assignment)
        left, right = binary(node[1], assignment), binary(node[2], assignment)
        return (left and right) if node[0] == "AND" else (left or right)
    unknown = [v for v in variables(expr) if facts.get(v) is None]
    outputs = {binary(expr, facts | dict(zip(unknown, values)))
               for values in product((False, True), repeat=len(unknown))}
    return next(iter(outputs)) if len(outputs) == 1 else None


def label(expr, facts, relevant=True) -> str:
    if not relevant:
        return "IRRELEVANT"
    return {True: "YES", False: "NO", None: "MORE"}[evaluate(expr, facts)]


def shape(expr) -> str:
    if isinstance(expr, str):
        return "P"
    children = [shape(x) for x in expr[1:]]
    if expr[0] in {"AND", "OR"}:
        children.sort()  # commutative reordering does not create independent families
    return expr[0] + "(" + ",".join(children) + ")"


def depth(expr) -> int:
    return 0 if isinstance(expr, str) else 1 + max(depth(x) for x in expr[1:])


def render(expr, paraphrase=False) -> str:
    if isinstance(expr, str):
        return f"requirement {int(expr[1:]) + 1} is met"
    if expr[0] == "NOT":
        return f"it is not the case that ({render(expr[1], paraphrase)})"
    word = ("and also" if paraphrase else "and") if expr[0] == "AND" else "or"
    return f"({render(expr[1], paraphrase)}) {word} ({render(expr[2], paraphrase)})"


def render_case(expr, facts, case_id, family_id, variant, original=None, relevant=True):
    rule = "This policy governs workshop access only. Access is granted exactly when " + render(expr, variant == "paraphrase") + ". Missing facts are unknown. An explicit correction replaces the identified earlier fact."
    blocks = [Source("s0", rule, "rule")]
    for name in variables(expr):
        value = (original or facts).get(name)
        if value is not None:
            blocks.append(Source(f"s{len(blocks)}", f"Requirement {int(name[1:]) + 1} is {'met' if value else 'not met'}."))
    if original:
        for name in facts:
            if original.get(name) != facts[name]:
                blocks.append(Source(f"s{len(blocks)}", f"Correction: replace the previous statement about requirement {int(name[1:]) + 1}. That requirement is now {'met' if facts[name] else 'not met'}.", "correction"))
    if variant == "distractor":
        blocks.append(Source(f"s{len(blocks)}", "The applicant's notebook is purple; notebook color is unrelated to this policy."))
    order = ("MORE", "IRRELEVANT", "YES", "NO") if variant == "option_order" else ("YES", "NO", "IRRELEVANT", "MORE")
    question = "Can the applicant access the workshop?" if relevant else "Does this policy grant a fishing permit?"
    return CaseView(case_id, family_id, "context_rules", question, tuple(blocks), decision_options(order))


def generate(families=60, seed=17) -> tuple[list[CaseView], list[dict], dict]:
    if type(families) is not int or families < 1:
        raise ValueError("Requested family count must be positive")
    rng = random.Random(seed)
    expressions = {}
    for _ in range(max(1000, families * 300)):
        counter = [0]
        def tree(level):
            if level == 0 or rng.random() < 0.30:
                name = f"p{counter[0]}"
                counter[0] += 1
                return name
            if rng.random() < 0.23:
                return ("NOT", tree(level - 1))
            return (rng.choice(("AND", "OR")), tree(level - 1), tree(level - 1))
        expr = tree(rng.randint(1, 4))
        if 1 <= len(variables(expr)) <= 8:
            expressions.setdefault(shape(expr), expr)
        if len(expressions) >= families:
            break
    views, private = [], []
    for canonical_shape, expr in sorted(expressions.items()):
        validate_expression(expr)
        fid = digest(canonical_shape)[:16]
        # Structural OOD: all depth-four families. Remaining shapes are hash-split.
        bucket = int(digest(["split-v1", canonical_shape])[:8], 16) % 10
        partition = "ood" if depth(expr) == 4 else ("train" if bucket < 4 else "calibration" if bucket < 6 else "development" if bucket < 8 else "test")
        names = variables(expr)
        decisive = None
        for bits in product((False, True), repeat=len(names)):
            base = dict(zip(names, bits))
            for name in names:
                flipped = base | {name: not base[name]}
                removed = base | {name: None}
                if evaluate(expr, base) != evaluate(expr, flipped) and evaluate(expr, removed) is None:
                    decisive = base, flipped, removed
                    break
            if decisive:
                break
        if decisive is None:
            raise AssertionError("A read-once nonconstant expression needs an influential variable")
        base, flipped, removed = decisive
        variants = [("base", base, None, True), ("flip", flipped, None, True),
                    ("missing", removed, None, True), ("distractor", base, None, True),
                    ("paraphrase", base, None, True), ("correction", flipped, base, True),
                    ("option_order", base, None, True), ("irrelevant", base, None, False)]
        for variant, facts, original, relevant in variants:
            assert evaluate(expr, facts) == completion_oracle(expr, facts)
            cid = digest([fid, variant, seed])[:20]
            view = render_case(expr, facts, cid, fid, variant, original, relevant)
            views.append(view)
            private.append(dict(case_id=cid, family_id=fid, benchmark="context_rules",
                                partition=partition, gold=label(expr, facts, relevant),
                                variant=variant, expression=expr, facts=facts, depth=depth(expr),
                                relevant=relevant, canonical_shape=canonical_shape))
    return views, private, {"generator": "context_rules_v1", "seed": seed,
                            "requested_families": families, "achieved_families": len(expressions),
                            "variants_per_family": 8, "linguistic_review": "not_yet_human_reviewed"}
