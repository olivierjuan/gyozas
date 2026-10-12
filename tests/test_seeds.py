"""Tests for the seed triplet (R2.4)."""

import random

import numpy as np
import pytest

from gyozas.seeds import SeedTriplet


class TestSeedTriplet:
    def test_to_dict_roundtrip(self):
        triplet = SeedTriplet(solver=1, python=2, augmentation=3)
        assert SeedTriplet.from_dict(triplet.to_dict()) == triplet

    def test_from_base_seed_is_deterministic(self):
        assert SeedTriplet.from_base_seed(42) == SeedTriplet.from_base_seed(42)
        assert SeedTriplet.from_base_seed(42) != SeedTriplet.from_base_seed(43)

    def test_three_components_distinct(self):
        triplet = SeedTriplet.from_base_seed(7)
        values = {triplet.solver, triplet.python, triplet.augmentation}
        assert len(values) == 3

    def test_seeds_in_scip_range(self):
        for triplet in (SeedTriplet.from_base_seed(i) for i in range(20)):
            for seed in (triplet.solver, triplet.python, triplet.augmentation):
                assert 0 <= seed <= 2**31 - 1

    def test_seed_python_makes_rngs_reproducible(self):
        SeedTriplet(solver=0, python=11, augmentation=0).seed_python()
        py_draw = random.random()
        np_draw = np.random.rand()
        SeedTriplet(solver=0, python=11, augmentation=0).seed_python()
        assert random.random() == py_draw
        assert np.random.rand() == np_draw

    def test_different_python_seed_different_draws(self):
        SeedTriplet(solver=0, python=1, augmentation=0).seed_python()
        first = random.random()
        SeedTriplet(solver=0, python=2, augmentation=0).seed_python()
        assert random.random() != first

    def test_frozen(self):
        triplet = SeedTriplet(solver=1, python=2, augmentation=3)
        with pytest.raises((AttributeError, TypeError)):
            triplet.solver = 4
