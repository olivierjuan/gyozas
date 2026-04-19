import unittest

from pyscipopt import Model

from gyozas.instances.combinatorial_auction import CombinatorialAuctionGenerator


class TestCombinatorialAuctionGenerator(unittest.TestCase):
    def setUp(self):
        self.params = {"n_items": 10, "n_bids": 20, "min_value": 1, "max_value": 10}
        self.generator = CombinatorialAuctionGenerator(**self.params, rng=42)

    def test_generate_instance_returns_model(self):
        model = next(self.generator)
        self.assertIsInstance(model, Model)
        self.assertGreater(model.getNVars(), 0)
        self.assertGreater(model.getNConss(), 0)

    def test_seed_reproducibility(self):
        gen1 = CombinatorialAuctionGenerator(**self.params)
        gen2 = CombinatorialAuctionGenerator(**self.params)
        gen1.seed(123)
        gen2.seed(123)
        m1 = gen1.generate_instance()
        m2 = gen2.generate_instance()
        self.assertEqual(m1.getNVars(), m2.getNVars())
        self.assertEqual(m1.getNConss(), m2.getNConss())
        c1 = [v.getObj() for v in m1.getVars()]
        c2 = [v.getObj() for v in m2.getVars()]
        self.assertEqual(c1, c2)

    def test_seed_changes_output(self):
        gen1 = CombinatorialAuctionGenerator(**self.params)
        gen2 = CombinatorialAuctionGenerator(**self.params)
        gen1.seed(1)
        gen2.seed(2)
        m1 = gen1.generate_instance()
        m2 = gen2.generate_instance()
        c1 = [v.getObj() for v in m1.getVars()]
        c2 = [v.getObj() for v in m2.getVars()]
        self.assertNotEqual(c1, c2)

    def test_callable_parameters(self):
        gen = CombinatorialAuctionGenerator(
            n_items=lambda rng: rng.integers(5, 15),
            n_bids=lambda rng: rng.integers(10, 30),
            rng=42,
        )
        model = next(gen)
        self.assertIsInstance(model, Model)

    def test_invalid_value_range_raises(self):
        gen = CombinatorialAuctionGenerator(rng=42)
        with self.assertRaises(ValueError):
            gen.generate_instance(min_value=10, max_value=1)

    def test_invalid_add_item_prob_raises(self):
        gen = CombinatorialAuctionGenerator(rng=42)
        with self.assertRaises(ValueError):
            gen.generate_instance(add_item_prob=1.5)

    def test_iter_and_next(self):
        gen = CombinatorialAuctionGenerator(**self.params, rng=42)
        it = iter(gen)
        model = next(it)
        self.assertIsInstance(model, Model)

    def test_problem_name(self):
        model = self.generator.generate_instance()
        self.assertIn("CombinatorialAuction", model.getProbName())


if __name__ == "__main__":
    unittest.main()
