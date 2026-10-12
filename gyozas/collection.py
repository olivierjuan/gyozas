import math

import numpy as np
from numpy.typing import NDArray


class CollectionMetadata:
    """Metadata recorder for expert data collection.

    The essential record is the **multi-optimal node distribution**: for each
    collected node, how many candidate actions reach the expert's best score.
    Without it, the top-1 accuracy (acc@1) of a learned policy is unreadable —
    a policy can take a different-but-equally-optimal action, and acc@1 penalizes
    it although the node counts (nodes to optimality) would not.

    Usage: at every collected node, feed the per-candidate expert scores to
    :meth:`record`. Scores can be any mapping from action to score (or an array
    indexed by action).
    """

    def __init__(self, tie_tolerance: float = 1e-9) -> None:
        self.tie_tolerance = tie_tolerance
        self.n_decisions = 0
        self._node_optimal_counts: list[int] = []

    def reset(self) -> None:
        self.n_decisions = 0
        self._node_optimal_counts.clear()

    def record(self, scores: NDArray[np.float64] | dict[int, float], action_set: NDArray[np.int64]) -> int:
        """Record one collected node and return its number of optimal actions.

        An action is *optimal* when its score ties the best score within
        ``tie_tolerance`` (relative to the best score's magnitude). Non-finite
        scores are ignored.
        """
        best_action, best_score = _best(scores, action_set)
        if best_action is None:
            self.n_decisions += 1
            self._node_optimal_counts.append(0)
            return 0
        tol = self.tie_tolerance * max(1.0, abs(best_score))
        n_optimal = sum(
            1
            for action in action_set
            if _score_at(scores, int(action)) is not None
            and math.isfinite(_score_at(scores, int(action)))
            and _score_at(scores, int(action)) >= best_score - tol
        )
        self.n_decisions += 1
        self._node_optimal_counts.append(n_optimal)
        return n_optimal

    def multi_optimal_distribution(self) -> dict[int, int]:
        """Distribution ``{number of optimal actions -> number of nodes}``."""
        distribution: dict[int, int] = {}
        for count in self._node_optimal_counts:
            distribution[count] = distribution.get(count, 0) + 1
        return dict(sorted(distribution.items()))

    @property
    def frac_multi_optimal(self) -> float:
        """Fraction of collected nodes with more than one optimal action."""
        if not self._node_optimal_counts:
            return 0.0
        return sum(1 for c in self._node_optimal_counts if c > 1) / len(self._node_optimal_counts)

    def to_dict(self) -> dict:
        return {
            "n_decisions": self.n_decisions,
            "multi_optimal_distribution": self.multi_optimal_distribution(),
            "frac_multi_optimal": self.frac_multi_optimal,
        }


def _score_at(scores: NDArray[np.float64] | dict[int, float], action: int) -> float | None:
    if isinstance(scores, dict):
        return scores.get(action)
    if 0 <= action < len(scores):
        return float(scores[action])
    return None


def _best(scores: NDArray[np.float64] | dict[int, float], action_set) -> tuple[int | None, float]:
    best_action = None
    best_score = -math.inf
    for action in action_set:
        score = _score_at(scores, int(action))
        if score is not None and math.isfinite(score) and score > best_score:
            best_score = score
            best_action = int(action)
    return best_action, best_score
