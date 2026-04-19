import unittest

from pyscipopt import Model

from gyozas.instances import InstanceGenerator
from gyozas.instances.modifiers.embed_objective import EmbedObjective


class DummyInstanceGenerator(InstanceGenerator):
    def __init__(self):
        super().__init__()
        self._seed = None
        self._called = False
        self._model = Model()
        x = self._model.addVar("x", vtype="C", obj=2.0)
        y = self._model.addVar("y", vtype="C", obj=0.0)
        self._model.setObjective(x + y)

    def seed(self, seed):
        self._seed = seed

    def generate_instance(self, *args, **kwargs):
        self._called = True
        return self._model

    def __next__(self):
        return self._model


class TestEmbedObjective(unittest.TestCase):
    def setUp(self):
        self.dummy_generator = DummyInstanceGenerator()

    def test_seed_calls_inner_generator(self):
        embed = EmbedObjective(self.dummy_generator)
        embed.seed(42)
        self.assertEqual(self.dummy_generator._seed, 42)

    def test_generate_instance_calls_inner(self):
        embed = EmbedObjective(self.dummy_generator)
        model = embed.generate_instance()
        self.assertTrue(self.dummy_generator._called)
        self.assertIsInstance(model, Model)

    def test_next_embeds_objective(self):
        embed = EmbedObjective(self.dummy_generator)
        model = embed.__next__()
        # Check that the new variable exists
        fobj_vars = [v for v in model.getVars() if v.name == "_fobj_"]
        self.assertEqual(len(fobj_vars), 1)
        # Check that the objective is set to the new variable
        variables = [var.vartuple[0].name for var in model.getObjective()]
        self.assertNotIn("_fobj_", variables)
        # Check that the constraint is present
        cons_names = [c.name for c in model.getConss()]
        self.assertIn("objective_function_constraint", cons_names)

    def test_next_embeds_objective_and_replace_fobj(self):
        embed = EmbedObjective(self.dummy_generator, replace_fobj=True)
        model = embed.__next__()
        # Check that the new variable exists
        fobj_vars = [v for v in model.getVars() if v.name == "_fobj_"]
        self.assertEqual(len(fobj_vars), 1)
        # Check that the objective is set to the new variable
        variables = [var.vartuple[0] for var in model.getObjective()]
        self.assertEqual(variables[0].name, "_fobj_")
        # Check that the constraint is present
        cons_names = [c.name for c in model.getConss()]
        self.assertIn("objective_function_constraint", cons_names)

    def test_next_and_next_alias(self):
        embed = EmbedObjective(self.dummy_generator)
        model1 = embed.__next__()
        model2 = embed.next()
        self.assertIsInstance(model1, Model)
        self.assertIsInstance(model2, Model)


if __name__ == "__main__":
    unittest.main()
