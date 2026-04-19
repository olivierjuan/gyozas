from pyscipopt import Model

from .arithmetic import ArithmeticMixin


class Done(ArithmeticMixin):
    """Reward that returns 1 when the solver finds an optimal solution, 0 otherwise."""

    def reset(self, model: Model) -> None:
        pass

    def extract(self, model: Model, done: bool) -> float:
        return 1.0 if model.getStatus() == "optimal" else 0.0
