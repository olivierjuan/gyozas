from pyscipopt import Model

from .arithmetic import ArithmeticMixin


class NNodes(ArithmeticMixin):
    """Reward based on the change in number of explored nodes since the last step."""

    def __init__(self) -> None:
        self.last_n_nodes = 0

    def reset(self, model: Model) -> None:
        self.last_n_nodes = 0

    def extract(self, model: Model, done: bool) -> int:
        n_nodes = model.getNNodes()
        delta = n_nodes - self.last_n_nodes
        self.last_n_nodes = n_nodes
        return delta
