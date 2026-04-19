import random
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from pyscipopt import Model


class Dynamics(ABC):
    """Base class for solver dynamics that control agent-solver interaction.

    Subclasses must implement :meth:`reset`, :meth:`step`, and :meth:`close`.
    """

    min_seed = 0
    max_seed = 2**31 - 1

    def __init__(self) -> None:
        self._rng = random.Random()

    def seed(self, value: int) -> None:
        """Seed the dynamics' internal RNG for reproducible SCIP randomization."""
        self._rng.seed(value)

    def set_seed_on_model(self, model: Model) -> None:
        def draw() -> int:
            return self._rng.randint(self.min_seed, self.max_seed)

        model.setParams(
            {
                "randomization/permuteconss": True,
                "randomization/permutevars": True,
                "randomization/permutationseed": draw(),
                "randomization/randomseedshift": draw(),
                "randomization/lpseed": draw(),
            }
        )

    @abstractmethod
    def reset(self, model: Model) -> tuple[bool, NDArray[np.int64] | None]:
        """Reset the dynamics for a new episode and return (done, action_set)."""
        ...

    @abstractmethod
    def step(self, action) -> tuple[bool, NDArray[np.int64] | None]:
        """Apply an action and return (done, action_set)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the dynamics."""
        ...

    @abstractmethod
    def add_action_reward_to_branching_tree(self, _branching_tree, _action, _reward) -> None:
        """Record an action and its reward in the branching tree visualisation."""
        ...
