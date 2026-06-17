from pyscipopt import Model

from .arithmetic import ArithmeticMixin


class StrongBranchingLPIterations(ArithmeticMixin):
    """Reward based on the change in strong-branching LP iterations since the last step.

    Where :class:`LPIterations` counts node-relaxation LP iterations
    (``getNLPIterations``), this counts only the LP iterations spent on strong
    branching (``getNStrongbranchLPIterations``) — a counter SCIP keeps separately.
    It is the natural cost signal for dynamics that probe variables with strong
    branching, e.g. ``ProbingDynamics``, where it makes the agent's probing budget
    an explicit trade-off rather than a free action.
    """

    def __init__(self) -> None:
        self.n_strong_branching_lp_iterations = 0

    def reset(self, model: Model) -> None:
        self.n_strong_branching_lp_iterations = 0

    def extract(self, model: Model, done: bool) -> int:
        n = model.getNStrongbranchLPIterations()
        delta = n - self.n_strong_branching_lp_iterations
        self.n_strong_branching_lp_iterations = n
        return delta
