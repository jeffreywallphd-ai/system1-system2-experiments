"""A finite task-graph language. No code execution or arbitrary tool references."""
from __future__ import annotations
from .domain import CaseView


def validate_plan(plan: dict, view: CaseView, limits: dict) -> list[dict]:
    if set(plan) != {"evidence_version", "nodes", "root", "relevance"}:
        raise ValueError("Unexpected static plan fields")
    if plan["evidence_version"] != view.version:
        raise ValueError("Stale plan evidence version")
    nodes = plan["nodes"]
    if not isinstance(nodes, list) or not nodes or len(nodes) > 2 * limits["max_leaves"] + 2:
        raise ValueError("Invalid/oversized task graph")
    by_id = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise ValueError("Malformed node")
        if node["id"] in by_id:
            raise ValueError("Duplicate task IDs")
        by_id[node["id"]] = node
        if node.get("kind") in {"DECIDE", "REASON"}:
            if set(node) != {"id", "kind", "worker", "objective", "sources", "dependencies"}:
                raise ValueError("Unexpected leaf fields")
            if node["worker"] not in {"LAYA", "S2"} or (node["kind"] == "REASON" and node["worker"] != "S2"):
                raise ValueError("Unsupported worker capability")
            if not isinstance(node["objective"], str) or not node["objective"].strip():
                raise ValueError("Leaf requires a proposition")
            if not isinstance(node["sources"], list) or not node["sources"] or set(node["sources"]) - {s.source_id for s in view.sources}:
                raise ValueError("Unknown/empty source references")
        elif node.get("kind") == "COMBINE":
            if set(node) != {"id", "kind", "operation", "dependencies"}:
                raise ValueError("Unexpected combination fields")
            if node["operation"] not in {"AND", "OR", "NOT", "IDENTITY"}:
                raise ValueError("Unknown deterministic operation")
            n = len(node["dependencies"])
            wrong_arity = n != 1 if node["operation"] in {"NOT", "IDENTITY"} else n < 2
            if wrong_arity:
                raise ValueError("Invalid operation arity")
        else:
            raise ValueError("Unknown task kind")
        deps = node["dependencies"]
        if not isinstance(deps, list) or len(deps) != len(set(deps)):
            raise ValueError("Invalid dependency list")
    if sum(n["kind"] != "COMBINE" for n in nodes) > limits["max_leaves"]:
        raise ValueError("Too many leaf tasks")
    ordered, visiting, levels = [], set(), {}
    def visit(key):
        if key in visiting:
            raise ValueError("Cyclic task graph")
        if key in levels:
            return levels[key]
        if key not in by_id:
            raise ValueError("Missing dependency/root")
        visiting.add(key)
        level = 0
        for dep in by_id[key]["dependencies"]:
            level = max(level, visit(dep) + 1)
        if level > limits["max_depth"]:
            raise ValueError("Task graph is too deep")
        visiting.remove(key)
        levels[key] = level
        ordered.append(by_id[key])
        return level
    visit(plan["root"])
    if plan["relevance"] is not None:
        visit(plan["relevance"])
    if len(ordered) != len(nodes):
        raise ValueError("Unreachable/decorative tasks are prohibited")
    return ordered


def combine(operation, values):
    if operation == "IDENTITY":
        return values[0]
    if operation == "NOT":
        return {"SUPPORTED": "CONTRADICTED", "CONTRADICTED": "SUPPORTED", "UNKNOWN": "UNKNOWN"}[values[0]]
    if operation == "AND":
        return "CONTRADICTED" if "CONTRADICTED" in values else "UNKNOWN" if "UNKNOWN" in values else "SUPPORTED"
    if operation == "OR":
        return "SUPPORTED" if "SUPPORTED" in values else "UNKNOWN" if "UNKNOWN" in values else "CONTRADICTED"
    raise ValueError("Unknown operation")


def validate_interactive(plan, view, limits):
    if set(plan) != {"evidence_version", "subgoals"} or plan["evidence_version"] != view.version:
        raise ValueError("Invalid/stale interactive plan")
    goals = plan["subgoals"]
    if not isinstance(goals, list) or not 1 <= len(goals) <= limits["max_leaves"]:
        raise ValueError("Invalid subgoal count")
    for goal in goals:
        if set(goal) != {"objective", "worker", "steps"} or goal["worker"] not in {"LAYA", "S2"}:
            raise ValueError("Unsupported interactive task contract")
        if not isinstance(goal["objective"], str) or not goal["objective"].strip():
            raise ValueError("Missing subgoal objective")
        if type(goal["steps"]) is not int or not 1 <= goal["steps"] <= limits["steps"]:
            raise ValueError("Invalid subgoal step budget")
    if sum(g["steps"] for g in goals) > limits["steps"]:
        raise ValueError("Plan exceeds episode action ceiling")
    return plan
