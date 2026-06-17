"""Unit and integration tests for gyozas.dynamics.probing."""

import numpy as np
import pytest
from pyscipopt import Model

from gyozas.dynamics.probing import ProbingDynamics

_INSTANCE = "tests/instance.lp"
_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}


def make_model(scip_params: dict | None = None) -> Model:
    m = Model()
    m.setParams(_PARAMS)
    if scip_params:
        m.setParams(scip_params)
    m.readProblem(_INSTANCE)
    return m


def run_episode(d: ProbingDynamics, m: Model, subset_of):
    """Drive a full episode, choosing the probe subset via ``subset_of(action_set)``."""
    done, action_set = d.reset(m)
    steps = 0
    while not done:
        assert action_set is not None
        done, action_set = d.step(subset_of(action_set))
        steps += 1
    return steps


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self):
        d = ProbingDynamics()
        assert d.scoring == "product"
        assert d.itlim == -1
        assert d.pseudocost_weight == 1.0

    @pytest.mark.parametrize("scoring", ["product", "sum", "min"])
    def test_valid_scoring(self, scoring):
        assert ProbingDynamics(scoring=scoring).scoring == scoring

    def test_invalid_scoring_raises(self):
        with pytest.raises(ValueError, match="scoring must be one of"):
            ProbingDynamics(scoring="nonexistent")

    def test_invalid_pseudocost_weight_raises(self):
        with pytest.raises(ValueError, match="pseudocost_weight"):
            ProbingDynamics(pseudocost_weight=0.0)
        with pytest.raises(ValueError, match="pseudocost_weight"):
            ProbingDynamics(pseudocost_weight=1.5)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_tuple(self):
        d = ProbingDynamics()
        done, action_set = d.reset(make_model())
        assert isinstance(done, bool)
        if not done:
            assert isinstance(action_set, np.ndarray)
            assert len(action_set) > 0
        d.close()

    def test_action_set_contains_variable_indices(self):
        d = ProbingDynamics()
        done, action_set = d.reset(make_model())
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
    def test_step_without_reset_raises(self):
        with pytest.raises(RuntimeError):
            ProbingDynamics().step([0])

    def test_invalid_action_raises(self):
        d = ProbingDynamics()
        done, action_set = d.reset(make_model())
        if done:
            pytest.skip("Instance solved at root")
        with pytest.raises(ValueError, match="not in action set"):
            d.step([-99999])
        d.close()

    def test_full_episode_probe_all(self):
        d = ProbingDynamics()
        steps = run_episode(d, make_model(), subset_of=lambda a: a)
        assert steps > 0
        d.close()

    def test_full_episode_probe_one(self):
        d = ProbingDynamics()
        steps = run_episode(d, make_model(), subset_of=lambda a: a[:1])
        assert steps > 0
        d.close()

    def test_full_episode_empty_subset(self):
        # Empty subset => pure pseudocost branching, still a valid episode.
        d = ProbingDynamics()
        steps = run_episode(d, make_model(), subset_of=lambda a: [])
        assert steps > 0
        d.close()

    def test_action_set_none_when_done(self):
        d = ProbingDynamics()
        done, action_set = d.reset(make_model())
        while not done:
            assert action_set is not None
            done, action_set = d.step(action_set)
        assert action_set is None
        d.close()

    @pytest.mark.parametrize("scoring", ["product", "sum", "min"])
    def test_scoring_variants_complete(self, scoring):
        d = ProbingDynamics(scoring=scoring)
        steps = run_episode(d, make_model(), subset_of=lambda a: a)
        assert steps > 0
        d.close()


# ---------------------------------------------------------------------------
# Strong-branching side effects
# ---------------------------------------------------------------------------


class TestStrongBranchingEffects:
    def test_probing_consumes_strong_branch_iterations(self):
        d = ProbingDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        if done:
            pytest.skip("Instance solved at root")
        run_episode_done = done
        while not run_episode_done:
            assert action_set is not None
            run_episode_done, action_set = d.step(action_set)
        assert m.getNStrongbranchLPIterations() > 0
        d.close()

    def test_empty_subset_does_no_strong_branching(self):
        d = ProbingDynamics()
        m = make_model()
        done, action_set = d.reset(m)
        if done:
            pytest.skip("Instance solved at root")
        while not done:
            assert action_set is not None
            done, action_set = d.step([])
        assert m.getNStrongbranchLPIterations() == 0
        d.close()


# ---------------------------------------------------------------------------
# Close & seeding
# ---------------------------------------------------------------------------


class TestCloseAndSeed:
    def test_close_after_reset(self):
        d = ProbingDynamics()
        d.reset(make_model())
        d.close()

    def test_close_mid_episode(self):
        d = ProbingDynamics()
        done, action_set = d.reset(make_model())
        if not done:
            assert action_set is not None
            d.step(action_set)
        d.close()

    def test_seed_does_not_raise(self):
        ProbingDynamics().seed(42)
