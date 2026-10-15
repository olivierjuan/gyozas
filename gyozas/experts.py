import math
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pyscipopt import Model

from gyozas.observations.pseudo_cost import Pseudocosts
from gyozas.observations.strong_branching_scores import StrongBranchingScores


@runtime_checkable
class BranchingExpert(Protocol):
    """A pluggable expert branching policy.

    An expert is a parameter of the data-collection setup, not of the evaluated
    method: it selects branching decisions the same way a policy would, so the
    harness can drive the solver with either. ``select`` receives the current
    model and the action set and returns the chosen action, or ``None`` when it
    cannot decide (no candidate, all scores invalid).
    """

    def reset(self, model: Model) -> None: ...

    def select(self, model: Model, action_set: NDArray[np.int64]) -> int | None: ...


def _argmax_action(scores: dict[int, float], action_set) -> int | None:
    """Action from ``action_set`` with the highest score; NaN scores ignored."""
    best_action = None
    best_score = -math.inf
    for action in action_set:
        score = scores.get(int(action))
        if score is not None and math.isfinite(score) and score > best_score:
            best_score = score
            best_action = int(action)
    return best_action


class ScoreExpert:
    """Expert that branches on the candidate with the best score of a scoring function.

    Wraps any observation function returning a per-candidate score array (e.g.
    :class:`~gyozas.observations.Pseudocosts`,
    :class:`~gyozas.observations.StrongBranchingScores`).
    """

    def __init__(self, scoring, name: str) -> None:
        self._scoring = scoring
        self.name = name

    def reset(self, model: Model) -> None:
        self._scoring.reset(model)

    def select(self, model: Model, action_set: NDArray[np.int64]) -> int | None:
        scores = self._scoring.extract(model, done=False)
        if scores is None:
            return None
        return _argmax_action({i: float(scores[i]) for i in range(len(scores))}, action_set)


class RelpscostExpert:
    """Expert branching on SCIP's native pseudocost score (the ``relpscost`` family).

    Uses SCIP's own pseudocost score for each LP branching candidate — the
    scoring behind SCIP's default reliability pseudocost branching rule. This is
    the scalable expert (track A): O(1) per candidate, no LP re-solves.
    """

    name = "relpscost"

    def __init__(self) -> None:
        pass

    def reset(self, _model: Model) -> None:
        pass

    def select(self, model: Model, action_set: NDArray[np.int64]) -> int | None:
        return _argmax_action(self.scores(model), action_set)

    def scores(self, model: Model) -> dict[int, float]:
        """SCIP's per-candidate pseudocost score, keyed by LP column position.

        The full label vector behind :meth:`select` — an imitation-learning
        collector records it as the expert's scores (R6.4: the GCNN trained
        on these labels prices the switch to the scalable expert).
        """
        cands, lp_vals, _, n_cands, _, _ = model.getLPBranchCands()
        cands, lp_vals = cands[:n_cands], lp_vals[:n_cands]
        return {
            int(var.getCol().getLPPos()): model.getVarPseudocostScore(var, val)
            for var, val in zip(cands, lp_vals, strict=False)
        }


class StrongBranchingExpert:
    """Expert branching on full strong-branching scores (the track-B expert).

    Strong branching re-solves the LP for each candidate child: accurate but
    expensive, and the natural expert for the synthetic bipartite-LP track.
    """

    name = "strong_branching"

    def __init__(self, itlim: int = -1) -> None:
        self._scores = StrongBranchingScores(itlim=itlim)

    def reset(self, model: Model) -> None:
        self._scores.reset(model)

    def select(self, model: Model, action_set: NDArray[np.int64]) -> int | None:
        scores = self._scores.extract(model, done=False)
        if scores is None:
            return None
        return _argmax_action({i: float(scores[i]) for i in range(len(scores))}, action_set)


class PseudocostExpert(ScoreExpert):
    """Expert branching on locally-observed pseudocost scores (Ecole-style).

    Unlike :class:`RelpscostExpert` (which reads SCIP's internal pseudocost
    tables), this expert accumulates pseudocosts from the branchings it observes,
    mirroring Ecole's ``Pseudocosts`` observation.
    """

    name = "pseudocost"

    def __init__(self) -> None:
        super().__init__(Pseudocosts(), name=self.name)
