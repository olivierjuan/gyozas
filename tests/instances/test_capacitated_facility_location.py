import unittest

from pyscipopt import Model

from gyozas.instances.capacitated_facility_location import CapacitatedFacilityLocationGenerator


class TestCapacitatedFacilityLocationGenerator(unittest.TestCase):
    def setUp(self):
        self.params = {"n_customers": 10, "n_facilities": 5}
        self.generator = CapacitatedFacilityLocationGenerator(**self.params, rng=42)

    def test_generate_instance_returns_model(self):
        model = next(self.generator)
        self.assertIsInstance(model, Model)
        self.assertGreater(model.getNVars(), 0)
        self.assertGreater(model.getNConss(), 0)

    def test_seed_reproducibility(self):
        gen1 = CapacitatedFacilityLocationGenerator(**self.params)
        gen2 = CapacitatedFacilityLocationGenerator(**self.params)
        gen1.seed(123)
        gen2.seed(123)
        m1 = gen1.generate_instance()
        m2 = gen2.generate_instance()
        self.assertEqual(m1.getNVars(), m2.getNVars())
        self.assertEqual(m1.getNConss(), m2.getNConss())
        c1 = sorted(v.getObj() for v in m1.getVars())
        c2 = sorted(v.getObj() for v in m2.getVars())
        self.assertEqual(c1, c2)

    def test_seed_changes_output(self):
        gen1 = CapacitatedFacilityLocationGenerator(**self.params)
        gen2 = CapacitatedFacilityLocationGenerator(**self.params)
        gen1.seed(1)
        gen2.seed(2)
        m1 = gen1.generate_instance()
        m2 = gen2.generate_instance()
        c1 = sorted(v.getObj() for v in m1.getVars())
        c2 = sorted(v.getObj() for v in m2.getVars())
        self.assertNotEqual(c1, c2)

    def test_continuous_assignment(self):
        gen = CapacitatedFacilityLocationGenerator(**self.params, continuous_assignment=True, rng=42)
        model = next(gen)
        serving_vars = [v for v in model.getVars() if v.name.startswith("s_")]
        for v in serving_vars:
            self.assertEqual(v.vtype(), "CONTINUOUS")

    def test_binary_assignment(self):
        gen = CapacitatedFacilityLocationGenerator(**self.params, continuous_assignment=False, rng=42)
        model = next(gen)
        serving_vars = [v for v in model.getVars() if v.name.startswith("s_")]
        for v in serving_vars:
            self.assertEqual(v.vtype(), "BINARY")

    def test_variable_counts(self):
        n_c, n_f = 10, 5
        gen = CapacitatedFacilityLocationGenerator(n_customers=n_c, n_facilities=n_f, rng=42)
        model = next(gen)
        facility_vars = [v for v in model.getVars() if v.name.startswith("f_")]
        serving_vars = [v for v in model.getVars() if v.name.startswith("s_")]
        self.assertEqual(len(facility_vars), n_f)
        self.assertEqual(len(serving_vars), n_c * n_f)

    def test_iter_and_next(self):
        gen = CapacitatedFacilityLocationGenerator(**self.params, rng=42)
        it = iter(gen)
        model = next(it)
        self.assertIsInstance(model, Model)

    def test_problem_name(self):
        model = self.generator.generate_instance()
        self.assertIn("CapacitatedFacilityLocation", model.getProbName())


if __name__ == "__main__":
    unittest.main()
