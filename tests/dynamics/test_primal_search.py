"""Unit and integration tests for gyozas.dynamics.primal_search."""

import numpy as np
import pytest
from pyscipopt import Model

from gyozas.dynamics import Dynamics
from gyozas.dynamics.primal_search import PrimalSearchDynamics

_INSTANCE = "tests/instance.lp"
_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}


def make_model() -> Model:
    m = Model()
    m.setParams(_PARAMS)
    m.readProblem(_INSTANCE)
    return m


# ---------------------------------------------------------------------------
# Protocol / subclass
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_is_dynamics_subclass(self):
        assert issubclass(PrimalSearchDynamics, Dynamics)

    def test_isinstance(self):
        assert isinstance(PrimalSearchDynamics(), Dynamics)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self):
        d = PrimalSearchDynamics()
        assert d.trials_per_node == 1
        assert d.depth_freq == 1
        assert d.depth_start == 0
        assert d.depth_stop == -1

    def test_custom_params(self):
        d = PrimalSearchDynamics(trials_per_node=3, depth_freq=2, depth_start=1, depth_stop=10)
        assert d.trials_per_node == 3
        assert d.depth_freq == 2
        assert d.depth_start == 1
        assert d.depth_stop == 10

    def test_invalid_trials_raises(self):
        with pytest.raises(ValueError, match="trials_per_node"):
            PrimalSearchDynamics(trials_per_node=-2)

    def test_unlimited_trials(self):
        d = PrimalSearchDynamics(trials_per_node=-1)
        assert d.trials_per_node == -1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_tuple(self):
        d = PrimalSearchDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        assert isinstance(done, bool)
        if not done:
            assert isinstance(action_set, np.ndarray)
        d.close()

    def test_zero_trials_solves_immediately(self):
        d = PrimalSearchDynamics(trials_per_node=0)
        m = make_model()
        done, action_set = d.reset(m)
        assert done is True
        assert action_set is None


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_with_empty_assignment(self):
        d = PrimalSearchDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        if done:
            pytest.skip("Instance solved at root")
        done2, action_set2 = d.step((np.array([], dtype=np.int64), np.array([], dtype=np.float64)))
        assert isinstance(done2, bool)
        d.close()

    def test_step_without_reset_raises(self):
        d = PrimalSearchDynamics()
        with pytest.raises(RuntimeError, match="No action set"):
            d.step((np.array([], dtype=np.int64), np.array([], dtype=np.float64)))

    def test_step_mismatched_lengths_raises(self):
        d = PrimalSearchDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        if done:
            pytest.skip("Instance solved at root")
        with pytest.raises(ValueError, match="same length"):
            d.step((np.array([0, 1], dtype=np.int64), np.array([0.5], dtype=np.float64)))
        d.close()

    def test_step_with_partial_assignment(self):
        d = PrimalSearchDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        if done or action_set is None or len(action_set) == 0:
            pytest.skip("Instance solved at root")
        # Assign first variable to 0.0
        done2, action_set2 = d.step((np.array([action_set[0]], dtype=np.int64), np.array([0.0], dtype=np.float64)))
        assert isinstance(done2, bool)
        d.close()

    def test_full_episode_skip_all(self):
        d = PrimalSearchDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        steps = 0
        while not done:
            done, action_set = d.step((np.array([], dtype=np.int64), np.array([], dtype=np.float64)))
            steps += 1
            if steps > 200:
                break
        d.close()


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_after_reset(self):
        d = PrimalSearchDynamics()
        m = make_model()
        d.reset(m)
        d.close()


# ---------------------------------------------------------------------------
# add_action_reward_to_branching_tree
# ---------------------------------------------------------------------------


class TestBranchingTree:
    def test_noop(self):
        d = PrimalSearchDynamics()
        d.add_action_reward_to_branching_tree(None, None, None)
