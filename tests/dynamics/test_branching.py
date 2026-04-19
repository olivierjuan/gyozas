"""Unit and integration tests for gyozas.dynamics.branching."""

import numpy as np
import pytest
from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics, ExtraBranchingActions

_INSTANCE = "tests/instance.lp"
_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}


def make_model(scip_params: dict | None = None) -> Model:
    m = Model()
    m.setParams(_PARAMS)
    if scip_params:
        m.setParams(scip_params)
    m.readProblem(_INSTANCE)
    return m


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_no_extra_actions(self):
        d = BranchingDynamics()
        assert d.with_extra_actions is None

    def test_extra_actions_sorted(self):
        d = BranchingDynamics(
            with_extra_actions=[
                ExtraBranchingActions.CUT_OFF,
                ExtraBranchingActions.SKIP,
            ]
        )
        assert d.with_extra_actions is not None
        assert d.with_extra_actions.tolist() == sorted(
            [
                ExtraBranchingActions.CUT_OFF,
                ExtraBranchingActions.SKIP,
            ]
        )


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_tuple(self):
        d = BranchingDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        assert isinstance(done, bool)
        if not done:
            assert isinstance(action_set, np.ndarray)
            assert len(action_set) > 0
        d.close()

    def test_reset_not_done_on_nontrivial_instance(self):
        d = BranchingDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        assert not done
        assert action_set is not None
        d.close()

    def test_action_set_contains_variable_indices(self):
        d = BranchingDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        if not done:
            assert action_set is not None
            for idx in action_set:
                assert isinstance(idx, int | np.integer)
                assert idx >= 0
        d.close()


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_returns_tuple(self):
        d = BranchingDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        if done:
            pytest.skip("Instance solved at root")
        assert action_set is not None
        done2, action_set2 = d.step(action_set[0])
        assert isinstance(done2, bool)
        d.close()

    def test_step_without_reset_raises(self):
        d = BranchingDynamics()
        with pytest.raises(RuntimeError):
            d.step(0)

    def test_invalid_action_raises(self):
        d = BranchingDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        if done:
            pytest.skip("Instance solved at root")
        with pytest.raises(ValueError, match="not in action set"):
            d.step(-99999)
        d.close()

    def test_full_episode(self):
        d = BranchingDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        steps = 0
        while not done:
            assert action_set is not None
            done, action_set = d.step(action_set[0])
            steps += 1
        assert steps > 0
        d.close()

    def test_action_set_none_when_done(self):
        d = BranchingDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        while not done:
            assert action_set is not None
            done, action_set = d.step(action_set[0])
        assert action_set is None
        d.close()


# ---------------------------------------------------------------------------
# Extra actions
# ---------------------------------------------------------------------------


class TestExtraActions:
    def test_extra_actions_prepended(self):
        extras = [ExtraBranchingActions.SKIP]
        d = BranchingDynamics(with_extra_actions=extras)
        m = make_model()
        done, action_set = d.reset(m)
        if done:
            pytest.skip("Instance solved at root")
        assert action_set is not None
        assert action_set[0] == ExtraBranchingActions.SKIP
        d.close()

    def test_skip_action_continues(self):
        extras = [ExtraBranchingActions.SKIP]
        d = BranchingDynamics(with_extra_actions=extras)
        m = make_model()
        done, action_set = d.reset(m)
        if done:
            pytest.skip("Instance solved at root")
        # Use SKIP, should not crash
        done2, action_set2 = d.step(ExtraBranchingActions.SKIP)
        assert isinstance(done2, bool)
        d.close()


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_after_reset(self):
        d = BranchingDynamics()
        m = make_model()
        d.reset(m)
        d.close()  # should not raise

    def test_close_mid_episode(self):
        d = BranchingDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        if not done:
            assert action_set is not None
            d.step(action_set[0])
        d.close()  # should not raise


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class TestSeeding:
    def test_seed_does_not_raise(self):
        d = BranchingDynamics()
        d.seed(42)

    def test_set_seed_on_model(self):
        d = BranchingDynamics()
        d.seed(42)
        m = make_model()
        d.set_seed_on_model(m)  # should not raise


# ---------------------------------------------------------------------------
# ExtraBranchingActions enum
# ---------------------------------------------------------------------------


class TestExtraBranchingActionsEnum:
    def test_skip_value(self):
        assert ExtraBranchingActions.SKIP == -1

    def test_cutoff_value(self):
        assert ExtraBranchingActions.CUT_OFF == -2

    def test_reduce_domain_value(self):
        assert ExtraBranchingActions.REDUCE_DOMAIN == -3

    def test_is_int(self):
        assert isinstance(ExtraBranchingActions.SKIP, int)
