import unittest

import numpy as np
from pyscipopt import Model

from gyozas.instances.set_cover import SetCoverGenerator


class TestSetCoverGenerator(unittest.TestCase):
    def setUp(self):
        self.params = {"n_rows": 100, "n_cols": 150, "density": 0.2, "max_coef": 10}
        self.generator = SetCoverGenerator(**self.params)

    def test_seed_reproducibility(self):
        gen1 = SetCoverGenerator(**self.params)
        gen2 = SetCoverGenerator(**self.params)
        gen1.seed(123)
        gen2.seed(123)
        inst1 = gen1.generate_instance()
        inst2 = gen2.generate_instance()
        c1 = [v.getObj() for v in inst1.getVars()]
        c2 = [v.getObj() for v in inst2.getVars()]
        self.assertEqual(c1, c2)

    def test_get_counts(self):
        indices = np.array([0, 1, 1, 2, 2, 2, 3])
        counts = SetCoverGenerator._get_counts(indices, 4)
        np.testing.assert_array_equal(counts, np.array([1, 2, 3, 1]))

    def test_get_choice_in_range(self):
        samples = self.generator._get_choice_in_range(0, 10, 5)
        self.assertEqual(len(samples), 5)
        self.assertEqual(len(set(samples)), 5)
        self.assertTrue(all(0 <= s < 10 for s in samples))

    def test_convert_csc_to_csr_identity(self):
        indices = np.array([0, 1, 2])
        indptr = np.array([0, 1, 2, 3])
        n_rows, n_cols = 3, 3
        indptr_csr, indices_csr = SetCoverGenerator._convert_csc_to_csr(indices, indptr, n_rows, n_cols)
        np.testing.assert_array_equal(indptr_csr, np.array([0, 1, 2, 3]))
        self.assertEqual(set(indices_csr), {0, 1, 2})

    def test_generate_instance_returns_model(self):
        model = next(self.generator)
        self.assertIsInstance(model, Model)
        vars_ = model.getVars()
        cons_ = model.getConss()
        self.assertEqual(len(vars_), self.params["n_cols"])
        self.assertEqual(len(cons_), self.params["n_rows"])
        for v in vars_:
            self.assertEqual(v.vtype(), "BINARY")

    def test_generate_instance_objective_range(self):
        model = self.generator.generate_instance(**self.params, rng=self.generator.rng)
        vars_ = model.getVars()
        for v in vars_:
            self.assertGreaterEqual(v.getObj(), 1)
            self.assertLessEqual(v.getObj(), self.params["max_coef"])

    def test_seed_changes_output(self):
        gen1 = SetCoverGenerator(**self.params)
        gen2 = SetCoverGenerator(**self.params)
        gen1.seed(1)
        gen2.seed(2)
        inst1 = gen1.generate_instance()
        inst2 = gen2.generate_instance()
        c1 = [v.getObj() for v in inst1.getVars()]
        c2 = [v.getObj() for v in inst2.getVars()]
        self.assertNotEqual(c1, c2)


if __name__ == "__main__":
    unittest.main()
