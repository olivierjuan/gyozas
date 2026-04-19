from pyscipopt import Model

from .arithmetic import ArithmeticMixin


class LPIterations(ArithmeticMixin):
    """Reward based on the change in LP iteration count since the last step."""

    def __init__(self) -> None:
        self.n_lp_iterations = 0

    def reset(self, model: Model) -> None:
        self.n_lp_iterations = 0

    def extract(self, model: Model, done: bool) -> int:
        n_lp_iterations = model.getNLPIterations()
        delta = n_lp_iterations - self.n_lp_iterations
        self.n_lp_iterations = n_lp_iterations
        return delta
