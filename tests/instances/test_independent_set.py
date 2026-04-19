import unittest

import networkx as nx
from pyscipopt import Model

from gyozas.instances.independent_set import (
    CliqueIndex,
    IndependentSetGenerator,
)


class TestIndependentSetGenerator(unittest.TestCase):
    def setUp(self):
        self.default_params = {"n_nodes": 10, "edge_probability": 0.3, "affinity": 2, "graph_type": "erdos_renyi"}
        self.generator = IndependentSetGenerator(**self.default_params, rng=42)

    def test_generate_instance_returns_model(self):
        model = self.generator.generate_instance(**self.default_params, rng=self.generator.rng)
        self.assertIsInstance(model, Model)
        self.assertEqual(model.getNVars(), self.default_params["n_nodes"])

    def test_generate_instance_constraints(self):
        model = self.generator.generate_instance(**self.default_params, rng=42)
        graph = nx.erdos_renyi_graph(self.default_params["n_nodes"], self.default_params["edge_probability"], seed=42)
        if graph.number_of_edges() > 0:
            self.assertGreater(model.getNConss(), 0)

    def test_generate_instance_with_barabasi_albert(self):
        params = {"n_nodes": 8, "affinity": 2, "graph_type": "barabasi_albert"}
        gen = IndependentSetGenerator(**params, rng=123)
        model = gen.generate_instance(**params, rng=gen.rng)
        self.assertIsInstance(model, Model)
        self.assertEqual(model.getNVars(), params["n_nodes"])

    def test_generate_instance_isolated_node(self):
        params = {"n_nodes": 3, "edge_probability": 0.0, "graph_type": "erdos_renyi"}
        gen = IndependentSetGenerator(**params, rng=1)
        model = gen.generate_instance(**params, rng=gen.rng)
        self.assertEqual(model.getNConss(), params["n_nodes"])

    def test_generate_instance_reproducibility(self):
        gen1 = IndependentSetGenerator(**self.default_params, rng=123)
        gen2 = IndependentSetGenerator(**self.default_params, rng=123)
        model1 = gen1.generate_instance(**self.default_params, rng=gen1.rng)
        model2 = gen2.generate_instance(**self.default_params, rng=gen2.rng)
        self.assertEqual(model1.getNVars(), model2.getNVars())
        self.assertEqual(model1.getNConss(), model2.getNConss())

    def test_make_graph_raises_on_unknown_type(self):
        params = {"n_nodes": 5, "graph_type": "unknown"}
        with self.assertRaises(ValueError):
            self.generator._make_graph(**params, rng=self.generator.rng)

    def test_clique_index_are_in_same_clique(self):
        partition = [[0, 1], [2, 3]]
        ci = CliqueIndex(partition, 4)
        self.assertTrue(ci.are_in_same_clique(0, 1))
        self.assertTrue(ci.are_in_same_clique(2, 3))
        self.assertFalse(ci.are_in_same_clique(0, 2))

    def test_iter_and_next(self):
        gen = IndependentSetGenerator(**self.default_params, rng=42)
        it = iter(gen)
        model = next(it)
        self.assertIsInstance(model, Model)

    def test_seed_sets_rng(self):
        gen = IndependentSetGenerator(**self.default_params)
        gen.seed(1234)
        self.assertIsNotNone(gen.rng)


if __name__ == "__main__":
    unittest.main()
