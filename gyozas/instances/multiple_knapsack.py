import numpy as np
from numpy.typing import NDArray
from pyscipopt import Model, quicksum

from gyozas.instances.instance_generator import InstanceGenerator, sanitize_rng


class MultipleKnapsackGenerator(InstanceGenerator):
    """Generator for random instances of the Multiple Knapsack problem.

    Items with a weight and a profit must be assigned to at most one of several
    capacitated knapsacks so as to maximize the total profit of packed items.
    Instances follow the schemes described in section 2.1 of

        Fukunaga, Alex S. (2011). A branch-and-bound algorithm for hard multiple
        knapsack problems. Annals of Operations Research, 184, 97-119.

    Parameters
    ----------
    n_items : int
        Number of items. Default is 100.
    n_knapsacks : int
        Number of knapsacks. Default is 6.
    min_range : int
        Lower bound (inclusive) of the range used to sample item weights. Default is 10.
    max_range : int
        Upper bound (exclusive) of the range used to sample item weights. Default is 20.
        Must be strictly greater than ``min_range``.
    scheme : str
        Weight/profit correlation scheme, one of ``"uncorrelated"``,
        ``"weakly correlated"``, ``"strongly correlated"``, or ``"subset-sum"``.
        Default is ``"weakly correlated"``.
    rng : numpy.random.Generator | int | None
        Random number generator (or seed) for reproducibility.
    """

    def __init__(
        self,
        n_items=100,
        n_knapsacks=6,
        min_range=10,
        max_range=20,
        scheme="weakly correlated",
        rng=None,
    ) -> None:
        super().__init__(rng=rng)
        self.n_items = n_items
        self.n_knapsacks = n_knapsacks
        self.min_range = min_range
        self.max_range = max_range
        self.scheme = scheme

    def __next__(self) -> Model:
        return self.generate_instance(
            n_items=self.n_items,
            n_knapsacks=self.n_knapsacks,
            min_range=self.min_range,
            max_range=self.max_range,
            scheme=self.scheme,
            rng=self.rng,
        )

    @staticmethod
    def _sample_profits(
        weights: NDArray[np.int64], min_range: int, max_range: int, scheme: str, rng: np.random.Generator
    ) -> NDArray:
        """Sample item profits correlated to ``weights`` according to ``scheme``."""
        n_items = len(weights)
        spread = max_range - min_range
        match scheme.lower():
            case "uncorrelated":
                return rng.integers(min_range, max_range, n_items)
            case "weakly correlated":
                low = np.maximum(weights - spread, 1)
                high = weights + spread
                return rng.integers(low, high)
            case "strongly correlated":
                return weights + spread / 10
            case "subset-sum":
                return weights
            case _:
                raise ValueError(
                    f"Unknown scheme {scheme!r}. Expected one of 'uncorrelated', "
                    "'weakly correlated', 'strongly correlated', 'subset-sum'."
                )

    @staticmethod
    def _sample_capacities(weights: NDArray[np.int64], n_knapsacks: int, rng: np.random.Generator) -> NDArray[np.int64]:
        """Sample knapsack capacities summing to roughly half the total item weight."""
        total_weight = int(weights.sum())
        capacities = np.zeros(n_knapsacks, dtype=int)
        capacities[:-1] = rng.integers(
            int(0.4 * total_weight // n_knapsacks),
            int(0.6 * total_weight // n_knapsacks),
            n_knapsacks - 1,
        )
        capacities[-1] = int(0.5 * total_weight) - capacities[:-1].sum()
        return capacities

    def generate_instance(
        self,
        n_items=100,
        n_knapsacks=6,
        min_range=10,
        max_range=20,
        scheme="weakly correlated",
        rng=None,
    ) -> Model:
        if min_range >= max_range:
            raise ValueError(f"min_range ({min_range}) must be strictly less than max_range ({max_range}).")
        rng = sanitize_rng(rng, default=self.rng)

        weights = rng.integers(min_range, max_range, n_items)
        profits = self._sample_profits(weights, min_range, max_range, scheme, rng)
        capacities = self._sample_capacities(weights, n_knapsacks, rng)

        model = Model(problemName=f"MultipleKnapsack-{n_items}-{n_knapsacks}")
        model.setMaximize()

        # x[i, k] = 1 if item i is placed in knapsack k.
        x = {
            (i, k): model.addVar(vtype="B", lb=0.0, ub=1.0, name=f"x_{i}_{k}", obj=profits[i])
            for i in range(n_items)
            for k in range(n_knapsacks)
        }

        # Each knapsack must respect its capacity.
        for k in range(n_knapsacks):
            model.addCons(
                quicksum(weights[i] * x[i, k] for i in range(n_items)) <= capacities[k],
                name=f"capacity_{k}",
            )

        # Each item is placed in at most one knapsack.
        for i in range(n_items):
            model.addCons(
                quicksum(x[i, k] for k in range(n_knapsacks)) <= 1,
                name=f"assign_{i}",
            )

        return model
