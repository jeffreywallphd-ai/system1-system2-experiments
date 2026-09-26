"""Killable ALFWorld process; only reset/step are exposed to the runner."""
import multiprocessing as mp
from ..domain import Failure


def _environment_worker(connection, game, seed, max_steps):
    environment = None
    try:
        from .alfworld import AlfworldEnvironment
        environment = AlfworldEnvironment(game, seed, max_steps)
        connection.send(("ready", None))
        while True:
            operation, argument = connection.recv()
            if operation == "close":
                break
            if operation == "reset":
                value = environment.reset()
            elif operation == "step":
                value = environment.step(argument)
            else:
                raise ValueError("Unknown environment operation")
            connection.send(("ok", value))
    except Exception as exc:
        connection.send(("error", str(exc)))
    finally:
        if environment is not None:
            environment.close()
        connection.close()


class IsolatedAlfworld:
    def __init__(self, game, seed, max_steps, timeout):
        self.timeout = timeout
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_environment_worker, args=(child, game, seed, max_steps), daemon=True)
        self.process.start()
        child.close()
        try:
            self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if self.timeout <= 0 or not self.connection.poll(self.timeout):
            self.close()
            raise Failure("TIMEOUT", "ALFWorld process timed out and was terminated")
        try:
            status, result = self.connection.recv()
        except EOFError as exc:
            raise Failure("ENVIRONMENT_ERROR", "ALFWorld process exited") from exc
        if status == "error":
            raise Failure("ENVIRONMENT_ERROR", result)
        return result

    def reset(self):
        self.connection.send(("reset", None))
        return self._receive()

    def step(self, command):
        self.connection.send(("step", command))
        return self._receive()

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        self.connection.close()
