import time

from pyscipopt import Model


class TimeSinceLastStep:
    """Information function that returns the wall-clock time delta since the last step."""

    def __init__(self) -> None:
        self.previous_timestamp = time.monotonic()

    def reset(self, model: Model) -> None:
        self.previous_timestamp = time.monotonic()

    def extract(self, model: Model, done: bool) -> float:
        current_timestamp = time.monotonic()
        delta = current_timestamp - self.previous_timestamp
        self.previous_timestamp = current_timestamp
        return delta
