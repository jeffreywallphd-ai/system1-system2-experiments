"""Persistent, sequential model processes with hard timeout/cancellation.

Only public Requests cross the process boundary. A timed-out process is killed
and joined before another request can be admitted; no untracked generation is
left running. Loading is separate from per-case inference timing.
"""
from __future__ import annotations
import multiprocessing as mp
import os
from ..domain import Failure


def _worker(connection, role, config):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        from .local import LayaModel, TransformersModel
        model = (LayaModel if role == "s1" else TransformersModel)(config)
        connection.send(("ready", model.identity))
        while True:
            message = connection.recv()
            if message is None:
                break
            try:
                connection.send(("ok", model.infer(*message)))
            except Exception as exc:
                connection.send(("error", (getattr(exc, "status", "MODEL_ERROR"), str(exc))))
    except Exception as exc:
        connection.send(("error", (getattr(exc, "status", "MODEL_UNAVAILABLE"), str(exc))))
    finally:
        connection.close()


class ProcessModel:
    def __init__(self, role, config, timeout=600):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, role, config), daemon=True)
        self.process.start()
        child.close()
        try:
            status, value = self._receive(timeout)
            if status != "ready":
                raise Failure(*value)
            self.identity = value
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            self.close()
            raise Failure("TIMEOUT", "Model process timed out and was terminated; usage is unknown")
        try:
            return self.connection.recv()
        except EOFError as exc:
            self.close()
            raise Failure("MODEL_ERROR", "Model process exited without usage") from exc

    def infer(self, request, max_tokens, timeout):
        if not self.process.is_alive():
            raise Failure("MODEL_UNAVAILABLE", "Model process is no longer running")
        self.connection.send((request, max_tokens, timeout))
        status, value = self._receive(timeout)
        if status == "error":
            raise Failure(*value)
        return value

    def close(self):
        if getattr(self, "process", None) and self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        if getattr(self, "connection", None):
            self.connection.close()
