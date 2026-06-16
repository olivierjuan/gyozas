import unittest

import numpy as np
from pyscipopt import Model

from gyozas.instances.multiple_knapsack import MultipleKnapsackGenerator


class TestMultipleKnapsackGenerator(unittest.TestCase):
    def setUp(self):
        self.default_params = {"n_items": 8, "n_knapsacks": 3, "min_range": 10, "max_range": 20}
        self.generator = MultipleKnapsackGenerator(**self.default_params, rng=42)

    def test_generate_instance_returns_model(self):
        model = self.generator.generate_instance(**self.default_params, rng=self.generator.rng)
        self.assertIsInstance(model, Model)

    def test_variable_and_constraint_counts(self):
        model = self.generator.generate_instance(**self.default_params, rng=42)
        n_items, n_knapsacks = self.default_params["n_items"], self.default_params["n_knapsacks"]
        # One binary variable per (item, knapsack) pair.
        self.assertEqual(model.getNVars(), n_items * n_knapsacks)
        # One capacity constraint per knapsack plus one assignment constraint per item.
        self.assertEqual(model.getNConss(), n_knapsacks + n_items)

    def test_is_maximization(self):
        model = self.generator.generate_instance(**self.default_params, rng=42)
        self.assertEqual(model.getObjectiveSense(), "maximize")

    def test_all_variables_binary(self):
        model = self.generator.generate_instance(**self.default_params, rng=42)
        self.assertTrue(all(v.vtype() == "BINARY" for v in model.getVars()))

    def test_solves_to_optimality(self):
        model = self.generator.generate_instance(**self.default_params, rng=7)
        model.hideOutput()
        model.optimize()
        self.assertEqual(model.getStatus(), "optimal")

    def test_supported_schemes(self):
        for scheme in ("uncorrelated", "weakly correlated", "strongly correlated", "subset-sum"):
            gen = MultipleKnapsackGenerator(**self.default_params, scheme=scheme, rng=1)
            self.assertIsInstance(next(gen), Model)

    def test_unknown_scheme_raises(self):
        gen = MultipleKnapsackGenerator(**self.default_params, scheme="nonexistent", rng=1)
        with self.assertRaises(ValueError):
            next(gen)

    def test_equal_range_raises(self):
        gen = MultipleKnapsackGenerator(n_items=8, n_knapsacks=3, min_range=10, max_range=10, rng=1)
        with self.assertRaises(ValueError):
            next(gen)

    def test_inverted_range_raises(self):
        gen = MultipleKnapsackGenerator(n_items=8, n_knapsacks=3, min_range=20, max_range=10, rng=1)
        with self.assertRaises(ValueError):
            next(gen)

    def test_capacities_sum_to_half_total_weight(self):
        rng = np.random.default_rng(0)
        weights = rng.integers(10, 20, 8)
        capacities = MultipleKnapsackGenerator._sample_capacities(weights, 3, rng)
        self.assertEqual(int(capacities.sum()), int(0.5 * weights.sum()))

    def test_reproducibility(self):
        gen1 = MultipleKnapsackGenerator(**self.default_params, rng=123)
        gen2 = MultipleKnapsackGenerator(**self.default_params, rng=123)
        m1 = gen1.generate_instance(**self.default_params, rng=gen1.rng)
        m2 = gen2.generate_instance(**self.default_params, rng=gen2.rng)
        self.assertEqual(m1.getNVars(), m2.getNVars())
        self.assertEqual(m1.getNConss(), m2.getNConss())
        self.assertEqual(
            [v.getObj() for v in m1.getVars()],
            [v.getObj() for v in m2.getVars()],
        )

    def test_iter_and_next(self):
        model = next(iter(self.generator))
        self.assertIsInstance(model, Model)

    def test_seed_sets_rng(self):
        gen = MultipleKnapsackGenerator(**self.default_params)
        gen.seed(1234)
        self.assertIsNotNone(gen.rng)


if __name__ == "__main__":
    unittest.main()
