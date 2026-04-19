"""Unit tests for gyozas.dynamics.Dynamics base class."""

import pytest
from pyscipopt import Model

from gyozas.dynamics import Dynamics

# ---------------------------------------------------------------------------
# Abstract base class checks
# ---------------------------------------------------------------------------


class TestAbstractBase:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Dynamics()

    def test_has_abstract_methods(self):
        abstracts = Dynamics.__abstractmethods__
        assert "reset" in abstracts
        assert "step" in abstracts
        assert "close" in abstracts
        assert "add_action_reward_to_branching_tree" in abstracts


# ---------------------------------------------------------------------------
# Concrete subclass for testing non-abstract methods
# ---------------------------------------------------------------------------


class _ConcreteDynamics(Dynamics):
    def reset(self, model):
        return True, None

    def step(self, action):
        return True, None

    def close(self):
        pass

    def add_action_reward_to_branching_tree(self, _branching_tree, _action, _reward) -> None:
        pass


class TestSeed:
    def test_seed_sets_rng(self):
        d = _ConcreteDynamics()
        d.seed(42)
        val1 = d._rng.randint(0, 1000)
        d.seed(42)
        val2 = d._rng.randint(0, 1000)
        assert val1 == val2

    def test_different_seeds_give_different_values(self):
        d = _ConcreteDynamics()
        d.seed(1)
        val1 = d._rng.randint(0, 2**31)
        d.seed(2)
        val2 = d._rng.randint(0, 2**31)
        assert val1 != val2


class TestSetSeedOnModel:
    def test_sets_randomization_params(self):
        d = _ConcreteDynamics()
        d.seed(42)
        m = Model()
        m.setParam("display/verblevel", 0)
        d.set_seed_on_model(m)
        # Just check it doesn't raise — params are set internally


class TestMinMaxSeed:
    def test_seed_bounds(self):
        assert Dynamics.min_seed == 0
        assert Dynamics.max_seed == 2**31 - 1
