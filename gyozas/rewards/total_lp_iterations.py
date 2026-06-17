from pyscipopt import Model

from .arithmetic import ArithmeticMixin


class TotalLPIterations(ArithmeticMixin):
    """Reward based on the change in total LP iterations since the last step.

    Sums the node-relaxation LP iterations (``getNLPIterations``) and the
    strong-branching LP iterations (``getNStrongbranchLPIterations``), which SCIP
    tracks in separate counters. It therefore captures the full LP work of an
    episode — including the probing cost incurred by strong-branching dynamics such
    as ``ProbingDynamics`` — which plain :class:`LPIterations` (node LPs only) does
    not account for.
    """

    def __init__(self) -> None:
        self.n_total_lp_iterations = 0

    def reset(self, model: Model) -> None:
        self.n_total_lp_iterations = 0

    def extract(self, model: Model, done: bool) -> int:
        n = model.getNLPIterations() + model.getNStrongbranchLPIterations()
        delta = n - self.n_total_lp_iterations
        self.n_total_lp_iterations = n
        return delta
