from dataclasses import dataclass

from pyscipopt import Model


@dataclass(frozen=True)
class ZarpellonAugmentation:
    """Search-tree augmentation (seed s, k) — Zarpellon et al. 2021, as used by T-BranT.

    The first ``k`` branching decisions of the episode are taken by a random
    explorer instead of the policy, and the solver's variable permutation is
    seeded with ``s`` (``randomization/permutevars`` + ``permutationseed``):
    each (s, k) pair yields a different search-tree prefix and therefore
    different states for the policy to be collected on.

    The augmentation is **opt-in**: it is a parameter of the data-collection
    setup, never imposed — methods that do not use it are unaffected, methods
    that do keep their advantage.

    References: Zarpellon et al., "Parameterizing branch-and-bound search trees
    to learn branching policies" (2021); github.com/ds4dm/branch-search-trees.
    """

    seed: int
    k: int
    explorer: str = "random"

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError(f"Augmentation k must be >= 0, got {self.k}.")

    def apply(self, model: Model) -> None:
        """Seed the solver's variable permutation for the episode."""
        model.setBoolParam("randomization/permutevars", True)
        model.setIntParam("randomization/permutationseed", self.seed)

    def to_dict(self) -> dict[str, int | str]:
        return {"seed": self.seed, "k": self.k, "explorer": self.explorer}

    @classmethod
    def from_dict(cls, data: dict[str, int | str]) -> "ZarpellonAugmentation":
        return cls(seed=int(data["seed"]), k=int(data["k"]), explorer=str(data.get("explorer", "random")))
