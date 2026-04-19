import unittest

from pyscipopt import Model

from gyozas.instances.files import FileGenerator


class TestFileGenerator(unittest.TestCase):
    def test_loads_lp_file(self):
        gen = FileGenerator(directory="tests", pattern="*.lp", rng=42)
        model = next(gen)
        self.assertIsInstance(model, Model)

    def test_iter_and_next(self):
        gen = FileGenerator(directory="tests", pattern="*.lp", rng=42)
        it = iter(gen)
        model = next(it)
        self.assertIsInstance(model, Model)

    def test_seed_resets_file_list(self):
        gen = FileGenerator(directory="tests", pattern="*.lp", rng=42)
        initial_remaining = gen.files_remaining
        gen.seed(123)
        self.assertEqual(gen.files_remaining, initial_remaining)

    def test_done_on_empty_directory(self):
        gen = FileGenerator(directory="tests", pattern="*.nonexistent", rng=42)
        self.assertTrue(gen.done())

    def test_done_remove_mode(self):
        gen = FileGenerator(directory="tests", pattern="*.lp", sampling_mode="remove", rng=42)
        self.assertFalse(gen.done())
        n_files = len(gen.files)
        for _ in range(n_files):
            next(gen)
        self.assertTrue(gen.done())

    def test_replace_mode_never_exhausts(self):
        gen = FileGenerator(directory="tests", pattern="*.lp", sampling_mode="replace", rng=42)
        for _ in range(10):
            model = next(gen)
            self.assertIsInstance(model, Model)

    def test_generate_instance_with_filepath(self):
        gen = FileGenerator(directory="tests", pattern="*.lp", rng=42)
        filepath = gen.files[0]
        model = gen.generate_instance(filepath)
        self.assertIsInstance(model, Model)


if __name__ == "__main__":
    unittest.main()
