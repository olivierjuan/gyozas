"""Unit and integration tests for gyozas.dynamics.configuring."""

import pytest
from pyscipopt import Model

from gyozas.dynamics import Dynamics
from gyozas.dynamics.configuring import ConfiguringDynamics

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
        assert issubclass(ConfiguringDynamics, Dynamics)

    def test_isinstance(self):
        assert isinstance(ConfiguringDynamics(), Dynamics)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_initial_state(self):
        d = ConfiguringDynamics()
        assert d.model is None
        assert d.done is False


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_not_done(self):
        d = ConfiguringDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        assert done is False
        assert action_set is None

    def test_reset_stores_model(self):
        d = ConfiguringDynamics()
        m = make_model()
        d.reset(m)
        assert d.model is m

    def test_reset_clears_done(self):
        d = ConfiguringDynamics()
        m = make_model()
        d.reset(m)
        d.step({})
        assert d.done is True
        m2 = make_model()
        d.reset(m2)
        assert d.done is False


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_with_empty_dict(self):
        d = ConfiguringDynamics()
        m = make_model()
        d.reset(m)
        done, action_set = d.step({})
        assert done is True
        assert action_set is None

    def test_step_with_params(self):
        d = ConfiguringDynamics()
        m = make_model()
        d.reset(m)
        done, action_set = d.step({"limits/time": 60.0})
        assert done is True

    def test_step_without_reset_raises(self):
        d = ConfiguringDynamics()
        with pytest.raises(RuntimeError, match="No model"):
            d.step({})

    def test_double_step_raises(self):
        d = ConfiguringDynamics()
        m = make_model()
        d.reset(m)
        d.step({})
        with pytest.raises(RuntimeError, match="already done"):
            d.step({})

    def test_model_solved_after_step(self):
        d = ConfiguringDynamics()
        m = make_model()
        d.reset(m)
        d.step({})
        status = m.getStatus()
        assert status in ("optimal", "nodelimit", "timelimit", "infeasible", "unbounded")


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_clears_model(self):
        d = ConfiguringDynamics()
        m = make_model()
        d.reset(m)
        d.step({})
        d.close()
        assert d.model is None

    def test_close_without_reset(self):
        d = ConfiguringDynamics()
        d.close()  # should not raise


# ---------------------------------------------------------------------------
# add_action_reward_to_branching_tree
# ---------------------------------------------------------------------------


class TestBranchingTree:
    def test_noop(self):
        d = ConfiguringDynamics()
        d.add_action_reward_to_branching_tree(None, None, None)  # should not raise


# ---------------------------------------------------------------------------
# Full episode
# ---------------------------------------------------------------------------


class TestFullEpisode:
    def test_reset_step_close(self):
        d = ConfiguringDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        assert not done
        done, action_set = d.step({})
        assert done
        d.close()

    def test_multiple_episodes(self):
        d = ConfiguringDynamics()
        for _ in range(3):
            m = make_model()
            d.reset(m)
            d.step({})
        d.close()
