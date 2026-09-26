"""Small, immutable contracts shared by adapters, policies, and the governor.

There are deliberately no gold labels, private ASTs, or environment handles in
these types. `CaseView.payload` is the ONLY case-to-model serialization path.
Administrative IDs and option meanings are never included in that payload.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Protocol

LABELS = ("YES", "NO", "IRRELEVANT", "MORE")
DESCRIPTIONS = (
    "The rule applies and the supplied evidence establishes eligibility.",
    "The rule applies and the supplied evidence establishes ineligibility.",
    "The question is outside the scope of the supplied rule.",
    "The rule applies but further information is needed to decide.",
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Source:
    source_id: str
    text: str
    kind: str = "observation"


@dataclass(frozen=True)
class Option:
    option_id: str
    description: str
    meaning: str

    def payload(self) -> dict:
        return {"option_id": self.option_id, "description": self.description}


def decision_options(order: tuple[str, ...] = LABELS) -> tuple[Option, ...]:
    return tuple(Option(f"o{i}", DESCRIPTIONS[LABELS.index(label)], label)
                 for i, label in enumerate(order))


LEAF_OPTIONS = (
    Option("o0", "The proposition is supported by the available evidence.", "SUPPORTED"),
    Option("o1", "The proposition is contradicted by the available evidence.", "CONTRADICTED"),
    Option("o2", "The evidence is insufficient to determine the proposition.", "UNKNOWN"),
)
ROUTE_OPTIONS = (
    Option("o0", "The supported fast typed-decision workflow is adequate.", "FAST"),
    Option("o1", "Reasoning execution is needed.", "REASON"),
)


@dataclass(frozen=True)
class CaseView:
    case_id: str
    family_id: str
    benchmark: str
    question: str
    sources: tuple[Source, ...]
    options: tuple[Option, ...]
    version: int = 0

    def __post_init__(self):
        if not self.sources or not self.options or not self.question.strip():
            raise ValueError("A case needs evidence, options, and a question")
        for ids in ([s.source_id for s in self.sources], [o.option_id for o in self.options]):
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate source or option ID")

    def payload(self) -> dict:
        return {"question": self.question, "sources": [asdict(s) for s in self.sources],
                "options": [o.payload() for o in self.options], "state_version": self.version}

    def meaning(self, option_id: str) -> str:
        for option in self.options:
            if option.option_id == option_id:
                return option.meaning
        raise ValueError("Unknown option ID")


@dataclass(frozen=True)
class Request:
    purpose: str
    objective: str
    view: CaseView
    options: tuple[Option, ...]
    details: dict
    seed: int

    def payload(self) -> dict:
        public = self.view.payload()
        public["options"] = [o.payload() for o in self.options]
        return {"purpose": self.purpose, "objective": self.objective,
                "input": public, "details": self.details}


@dataclass
class Response:
    content: dict
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_final: str | None = None
    encoding: dict | None = None
    metadata: dict | None = None


class Model(Protocol):
    identity: dict

    def infer(self, request: Request, max_tokens: int, timeout: float) -> Response: ...
    def close(self) -> None: ...


class Failure(Exception):
    """An explicit experimental outcome, not a replacement prediction."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def validate_probabilities(probabilities: dict | None, options: tuple[Option, ...]) -> None:
    if probabilities is None:
        return
    if not isinstance(probabilities, dict) or set(probabilities) != {o.option_id for o in options}:
        raise ValueError("Probability keys do not match the option set")
    values = list(probabilities.values())
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
           or not 0 <= p <= 1 for p in values):
        raise ValueError("Probabilities must be finite and within [0,1]")
    # Laya's SDK rounds individual probabilities to four decimal places.
    if abs(sum(values) - 1) > max(0.001, len(values) * 0.000051):
        raise ValueError("Probabilities do not sum to one within rounding tolerance")


def validate_choice(content: dict, options: tuple[Option, ...]) -> str:
    if set(content) - {"selected_option_id", "probabilities"}:
        raise ValueError("Unexpected decision fields")
    choice = content.get("selected_option_id")
    if choice not in {o.option_id for o in options}:
        raise ValueError("Selected option is not in the current option set")
    validate_probabilities(content.get("probabilities"), options)
    return choice


def view_from_dict(row: dict) -> CaseView:
    """Read our prepared public schema, never an arbitrary benchmark record."""
    expected = {"case_id", "family_id", "benchmark", "question", "sources", "options", "version"}
    if set(row) - expected:
        raise ValueError("Unexpected prepared case fields")
    return CaseView(row["case_id"], row["family_id"], row["benchmark"], row["question"],
                    tuple(Source(**s) for s in row["sources"]),
                    tuple(Option(**o) for o in row["options"]), row.get("version", 0))
