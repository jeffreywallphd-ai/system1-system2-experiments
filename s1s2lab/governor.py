"""Deterministic admission, usage accounting, event journaling, and validation."""
from __future__ import annotations
from dataclasses import replace
import os
import threading
import time
from .domain import Failure, Request, canonical, digest


class Budget:
    """Reserve worst-case generation atomically, then reconcile known usage.

    Unknown usage consumes the entire reservation and is separately marked.
    A provider exceeding the reservation is an accounting error, never silently
    clipped into a seemingly budget-compliant measurement.
    """
    def __init__(self, limits):
        self.limits = limits
        self.used = dict(s1_calls=0, s1_questions=0, s1_encoder_tokens=0,
                         s2_calls=0, s2_prompt_tokens=0, generated_tokens=0,
                         observed_generated_tokens=0, unknown_usage_calls=0,
                         repairs=0, supervisor_calls=0, actions=0, replans=0,
                         laya_leaves=0, s2_leaves=0, service_seconds=0.0)
        self.lock = threading.Lock()
        self.started = time.perf_counter()

    def remaining_seconds(self):
        return self.limits["case_seconds"] - (time.perf_counter() - self.started)

    def reserve(self, role):
        with self.lock:
            amount = self.limits["tokens_per_call"] if role == "s2" else 0
            if (self.remaining_seconds() <= 0 or self.used[role + "_calls"] >= self.limits[role + "_calls"]
                    or self.used["generated_tokens"] + amount > self.limits["generated_tokens"]):
                raise Failure("BUDGET_EXHAUSTED", f"Cannot reserve another {role} call")
            self.used[role + "_calls"] += 1
            self.used["s1_questions"] += int(role == "s1")
            self.used["generated_tokens"] += amount
            return amount

    def reconcile(self, role, reserved, response):
        with self.lock:
            if response is None or response.output_tokens is None:
                self.used["unknown_usage_calls"] += 1
                return
            actual = response.output_tokens
            if type(actual) is not int or actual < 0 or actual > reserved:
                raise Failure("ACCOUNTING_ERROR", "Reported generation violates its reservation")
            self.used["generated_tokens"] += actual - reserved
            self.used["observed_generated_tokens"] += actual
            if response.input_tokens is not None:
                if type(response.input_tokens) is not int or response.input_tokens < 0:
                    raise Failure("ACCOUNTING_ERROR", "Invalid input token count")
                self.used["s1_encoder_tokens" if role == "s1" else "s2_prompt_tokens"] += response.input_tokens
            else:
                self.used["unknown_usage_calls"] += 1

    def admit_action(self):
        with self.lock:
            if self.remaining_seconds() <= 0 or self.used["actions"] >= self.limits["steps"]:
                raise Failure("BUDGET_EXHAUSTED", "No remaining environment action budget")
            self.used["actions"] += 1


class Governor:
    def __init__(self, models, limits, seed, journal=None):
        self.models, self.budget, self.seed = models, Budget(limits), seed
        self.events = []
        self.journal = journal

    def event(self, kind, **fields):
        event = {"event_id": f"e{len(self.events)}", "kind": kind, **fields}
        self.events.append(event)
        if self.journal:
            with self.journal.open("a", encoding="utf-8") as stream:
                stream.write(canonical(event) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return event["event_id"]

    def request(self, role, purpose, objective, view, options, details, validator):
        request = Request(purpose, objective, view, options, details, self.seed + len(self.events))
        for attempt in range(1 + (self.budget.limits["repairs"] if role == "s2" else 0)):
            if role not in self.models:
                raise Failure("MODEL_UNAVAILABLE", f"{role} is unavailable; no fallback")
            reserved = self.budget.reserve(role)
            if attempt:
                self.budget.used["repairs"] += 1
            if purpose.endswith("plan"):
                self.budget.used["supervisor_calls"] += 1
            call_id = self.event("model_request", role=role, purpose=purpose, attempt=attempt,
                                 state_version=view.version, fingerprint=digest(request.payload()),
                                 payload=request.payload(), reserved_generated_tokens=reserved)
            started, response = time.perf_counter(), None
            try:
                response = self.models[role].infer(request, reserved,
                    min(self.budget.limits["model_seconds"], self.budget.remaining_seconds()))
            except Exception as exc:
                encoding = {"status": "REJECTED", "reason": str(exc), "request_hash": digest(request.payload())} if getattr(exc, "status", None) == "CONTEXT_LIMIT" else None
                self.event("model_failure", request_event=call_id,
                           status=getattr(exc, "status", "MODEL_ERROR"), error=str(exc), usage_known=False, encoding=encoding)
                raise Failure(getattr(exc, "status", "MODEL_ERROR"), str(exc)) from exc
            finally:
                self.budget.used["service_seconds"] += time.perf_counter() - started
                self.budget.reconcile(role, reserved, response)
            self.event("model_response", request_event=call_id, role=role, content=response.content,
                       input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                       encoding=response.encoding, metadata=response.metadata, raw_final=response.raw_final)
            try:
                result = validator(response.content)
                return result, response
            except (ValueError, KeyError, TypeError) as exc:
                error_event = self.event("validation_failure", request_event=call_id, error=str(exc))
                if attempt == self.budget.limits["repairs"] or role == "s1":
                    raise Failure("INVALID_OUTPUT", str(exc)) from exc
                request = replace(request, details=details | {"repair": {"invalid_contract": response.content,
                                   "validation_error": str(exc), "event": error_event}})
        raise AssertionError("Unreachable")

    def authorize_action(self, view, choice, version, options_hash):
        if version != view.version or options_hash != digest([o.payload() for o in view.options]):
            raise Failure("STALE_RESULT", "Action selection refers to a superseded state/action set")
        try:
            command = view.meaning(choice)
        except ValueError as exc:
            raise Failure("INVALID_OUTPUT", str(exc)) from exc
        self.budget.admit_action()
        self.event("action", state_version=version, option_id=choice, command=command,
                   options_hash=options_hash)
        return command
