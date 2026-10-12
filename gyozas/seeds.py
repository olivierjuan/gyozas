import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedTriplet:
    """The three seeds that make a run reproducible.

    A run's identity carries all three: the solver seed (SCIP randomization),
    the Python/NumPy seed (instance generation, shuffling, model init), and the
    augmentation seed (data augmentation draws). Replicating a run means
    replicating the full triplet — re-rolling only one of the three is a
    different run.

    Environment.seed covers the solver seed; :meth:`seed_python` covers the
    Python/NumPy pair; the augmentation seed belongs to the augmentation
    module that consumes it.
    """

    solver: int
    python: int
    augmentation: int

    def to_dict(self) -> dict[str, int]:
        return {"solver": self.solver, "python": self.python, "augmentation": self.augmentation}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "SeedTriplet":
        return cls(solver=int(data["solver"]), python=int(data["python"]), augmentation=int(data["augmentation"]))

    @classmethod
    def from_base_seed(cls, base: int) -> "SeedTriplet":
        """Derive a deterministic triplet from a single base seed."""
        rng = random.Random(base)
        return cls(
            solver=rng.randint(0, 2**31 - 1), python=rng.randint(0, 2**31 - 1), augmentation=rng.randint(0, 2**31 - 1)
        )

    def seed_python(self) -> None:
        """Seed Python's ``random`` and NumPy's global RNG."""
        random.seed(self.python)
        np.random.seed(self.python % 2**32)
