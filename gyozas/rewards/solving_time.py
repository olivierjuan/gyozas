from pyscipopt import Model

from .arithmetic import ArithmeticMixin


class SolvingTime(ArithmeticMixin):
    """Reward based on the wall-clock solving time elapsed since the last step."""

    def __init__(self) -> None:
        self.solving_time = 0

    def reset(self, model: Model) -> None:
        self.solving_time = 0

    def extract(self, model: Model, done: bool) -> float:
        solving_time = model.getSolvingTime()
        delta = solving_time - self.solving_time
        self.solving_time = solving_time
        return delta
