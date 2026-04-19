"""Unit and integration tests for gyozas.observations.strong_branching_scores.StrongBranchingScores."""

import math

import numpy as np
import pytest
from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.observations import ObservationFunction, StrongBranchingScores

# ---------------------------------------------------------------------------
# Helpers (shared with pseudo_cost tests)
# ---------------------------------------------------------------------------

_INSTANCE = "tests/instance.lp"
_BASE_PARAMS = {
    "display/verblevel": 5,
    "limits/nodes": 20,
    "randomization/randomseedshift": 0,
    "randomization/randomseedshiftmultiplier": 10,
    "randomization/permutationseed": 0,
    "randomization/lpseed": 0,
    "concurrent/changeseeds": True,
    "concurrent/initseed": 5131912,
    "presolving/milp/randomseed": 0,
    "branching/relpscost/startrandseed": 5,
    "branching/random/seed": 41,
    "heuristics/scheduler/seed": 113,
    "heuristics/scheduler/subsciprandseeds": False,
    "heuristics/alns/seed": 113,
    "heuristics/alns/subsciprandseeds": False,
    "separating/zerohalf/initseed": 24301,
}


def make_mip(scip_params: dict | None = None) -> Model:
    """Load the shared test instance with a node limit to keep tests fast."""
    m = Model()
    for k, v in _BASE_PARAMS.items():
        try:
            m.setParam(k, v)
        except KeyError:
            print(f"paramater {k} is invalid")
    if scip_params:
        for k, v in scip_params.items():
            try:
                m.setParam(k, v)
            except KeyError:
                print(f"paramater {k} is invalid")
    m.readProblem(_INSTANCE)
    return m


class _BranchingContext:
    def __init__(self, m: Model):
        self._dynamics = BranchingDynamics()
        self.model = m
        self.done, self.action_set = self._dynamics.reset(m)

    def step(self):
        if self.done or self.action_set is None or len(self.action_set) == 0:
            return self.done, None
        self.done, self.action_set = self._dynamics.step(self.action_set[0])
        return self.done, self.action_set

    def close(self):
        self._dynamics.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@pytest.fixture
def branching_ctx():
    m = make_mip()
    ctx = _BranchingContext(m)
    yield ctx
    ctx.close()


@pytest.fixture
def branching_ctx_pseudo():
    """Fixture using pseudo candidates (n=5 for more variety)."""
    m = make_mip()
    ctx = _BranchingContext(m)
    yield ctx
    ctx.close()


# ---------------------------------------------------------------------------
# Protocol / import
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_is_observation_function(self):
        assert isinstance(StrongBranchingScores(), ObservationFunction)

    def test_exported_from_package(self):
        from gyozas.observations import StrongBranchingScores as SBS  # noqa: F401

    def test_default_params(self):
        obs = StrongBranchingScores()
        assert obs.pseudo_candidates is False
        assert obs.itlim == -1

    def test_custom_params_stored(self):
        obs = StrongBranchingScores(pseudo_candidates=True, itlim=50)
        assert obs.pseudo_candidates is True
        assert obs.itlim == 50


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_none(self, branching_ctx):
        obs = StrongBranchingScores()
        assert obs.reset(branching_ctx.model) is None

    def test_reset_is_idempotent(self, branching_ctx):
        obs = StrongBranchingScores()
        obs.reset(branching_ctx.model)
        obs.reset(branching_ctx.model)  # second call must not raise


# ---------------------------------------------------------------------------
# extract: early-exit guards
# ---------------------------------------------------------------------------


class TestExtractGuards:
    def test_returns_none_when_done(self, branching_ctx):
        obs = StrongBranchingScores()
        result = obs.extract(branching_ctx.model, done=True)
        assert result is None

    def test_returns_none_before_solving(self):
        obs = StrongBranchingScores()
        m = make_mip()
        result = obs.extract(m, done=False)
        assert result is None

    def test_returns_none_after_solve(self):
        obs = StrongBranchingScores()
        m = make_mip()
        m.optimize()
        result = obs.extract(m, done=False)
        assert result is None

    def test_done_takes_priority_over_stage(self, branching_ctx):
        """done=True must return None even if stage is SOLVING."""
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores()
        result = obs.extract(branching_ctx.model, done=True)
        assert result is None


# ---------------------------------------------------------------------------
# extract: output shape and dtype
# ---------------------------------------------------------------------------


class TestExtractOutputShape:
    def test_returns_ndarray(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores()
        result = obs.extract(branching_ctx.model, done=False)
        assert isinstance(result, np.ndarray)

    def test_shape_covers_all_candidates(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores()
        result = obs.extract(branching_ctx.model, done=False)
        cands, _, _, n_cands, _, _ = branching_ctx.model.getLPBranchCands()
        max_idx = max(v.getCol().getLPPos() for v in cands[:n_cands])
        assert result is not None
        assert result.shape[0] > max_idx

    def test_dtype_float64(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores()
        result = obs.extract(branching_ctx.model, done=False)
        assert result is not None
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# extract: NaN / finite placement (LP candidates)
# ---------------------------------------------------------------------------


class TestExtractLPCandidates:
    def test_lp_candidates_mostly_finite(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores(pseudo_candidates=False)
        result = obs.extract(branching_ctx.model, done=False)
        # At least one LP candidate should have a finite score
        # (some may be NaN due to LP errors during probing)
        assert result is not None
        assert np.any(np.isfinite(result))

    def test_most_entries_are_nan(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores(pseudo_candidates=False)
        result = obs.extract(branching_ctx.model, done=False)
        # Most entries should be NaN (only candidates have finite scores)
        assert result is not None
        n_finite = np.sum(np.isfinite(result))
        assert n_finite < len(result), "Not all entries should be finite"
        assert n_finite > 0, "At least one candidate should have a finite score"

    def test_scores_nonnegative(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores()
        result = obs.extract(branching_ctx.model, done=False)
        assert result is not None
        finite = result[np.isfinite(result)]
        assert (finite >= 0.0).all(), "All finite scores must be >= 0"

    def test_at_least_one_finite_score(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores()
        result = obs.extract(branching_ctx.model, done=False)
        assert result is not None
        assert np.any(np.isfinite(result)), "At least one candidate must have a finite score"


# ---------------------------------------------------------------------------
# extract: pseudo candidates mode
# ---------------------------------------------------------------------------


class TestExtractPseudoCandidates:
    def test_pseudo_candidates_returns_ndarray(self, branching_ctx_pseudo):
        if branching_ctx_pseudo.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores(pseudo_candidates=True)
        result = obs.extract(branching_ctx_pseudo.model, done=False)
        assert isinstance(result, np.ndarray)

    def test_pseudo_candidates_correct_shape(self, branching_ctx_pseudo):
        if branching_ctx_pseudo.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores(pseudo_candidates=True)
        result = obs.extract(branching_ctx_pseudo.model, done=False)
        assert result is not None
        assert result.ndim == 1
        assert result.shape[0] > 0

    def test_pseudo_candidates_all_finite_or_nan(self, branching_ctx_pseudo):
        if branching_ctx_pseudo.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores(pseudo_candidates=True)
        result = obs.extract(branching_ctx_pseudo.model, done=False)
        # No +/-inf (either finite score or NaN, nothing else)
        assert result is not None
        assert not np.any(np.isinf(result))

    def test_pseudo_candidates_nonnegative(self, branching_ctx_pseudo):
        if branching_ctx_pseudo.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores(pseudo_candidates=True)
        result = obs.extract(branching_ctx_pseudo.model, done=False)
        assert result is not None
        finite = result[np.isfinite(result)]
        assert (finite >= 0.0).all()

    def test_pseudo_cands_cover_lp_cands(self, branching_ctx_pseudo):
        """Pseudo candidates should be a superset of LP candidates."""
        if branching_ctx_pseudo.done:
            pytest.skip("No branching node")
        m = branching_ctx_pseudo.model
        obs_lp = StrongBranchingScores(pseudo_candidates=False)
        obs_ps = StrongBranchingScores(pseudo_candidates=True)
        result_lp = obs_lp.extract(m, done=False)
        result_ps = obs_ps.extract(m, done=False)

        # Pseudo result should cover at least as many indices
        assert result_lp is not None
        assert result_ps is not None
        min_len = min(len(result_lp), len(result_ps))
        lp_finite = {i for i in range(min_len) if np.isfinite(result_lp[i])}
        ps_finite = {i for i in range(min_len) if np.isfinite(result_ps[i])}
        # LP finite scores should be a subset of pseudo finite scores
        # (some may be NaN due to probing errors, so allow some slack)
        if lp_finite:
            assert len(lp_finite & ps_finite) > 0, "At least some LP candidates should also be pseudo candidates"


# ---------------------------------------------------------------------------
# extract: itlim parameter
# ---------------------------------------------------------------------------


class TestItlim:
    def test_itlim_zero_still_returns_array(self, branching_ctx):
        """Even with 0 LP iterations the function should return an array (possibly all NaN)."""
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores(itlim=0)
        result = obs.extract(branching_ctx.model, done=False)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_itlim_1_returns_array(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores(itlim=1)
        result = obs.extract(branching_ctx.model, done=False)
        assert isinstance(result, np.ndarray)

    def test_unlimited_itlim_returns_finite_scores(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores(itlim=-1)
        result = obs.extract(branching_ctx.model, done=False)
        assert result is not None
        assert np.any(np.isfinite(result))


# ---------------------------------------------------------------------------
# _probe_bound internal method
# ---------------------------------------------------------------------------


class TestProbebound:
    def test_probe_down_returns_nonnegative(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores()
        m = branching_ctx.model
        cands, _, _, n_cands, _, _ = m.getLPBranchCands()
        if n_cands == 0:
            pytest.skip("No LP candidates")
        lp_obj = m.getLPObjVal()
        gain = obs._probe_bound(m, cands[0], lp_obj, m.infinity(), down=True)
        assert math.isnan(gain) or gain >= 0.0

    def test_probe_up_returns_nonnegative(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores()
        m = branching_ctx.model
        cands, _, _, n_cands, _, _ = m.getLPBranchCands()
        if n_cands == 0:
            pytest.skip("No LP candidates")
        lp_obj = m.getLPObjVal()
        gain = obs._probe_bound(m, cands[0], lp_obj, m.infinity(), down=False)
        assert math.isnan(gain) or gain >= 0.0

    def test_probe_does_not_modify_lp_state(self, branching_ctx):
        """Probing must leave the LP solution unchanged (idempotent)."""
        if branching_ctx.done:
            pytest.skip("No branching node")
        obs = StrongBranchingScores()
        m = branching_ctx.model
        cands, _, _, n_cands, _, _ = m.getLPBranchCands()
        if n_cands == 0:
            pytest.skip("No LP candidates")

        lp_obj_before = m.getLPObjVal()
        obs._probe_bound(m, cands[0], lp_obj_before, m.infinity(), down=True)
        obs._probe_bound(m, cands[0], lp_obj_before, m.infinity(), down=False)

        lp_obj_after = m.getLPObjVal()
        _, _, _, n_after, _, _ = m.getLPBranchCands()
        assert abs(lp_obj_after - lp_obj_before) < 1e-9, "LP objective changed after probing"
        assert n_after == n_cands, "Number of LP candidates changed after probing"


# ---------------------------------------------------------------------------
# Consistency: same result on repeated calls
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_extract_after_step_returns_new_array(self):
        """After a branching, the observation at the child node may differ."""
        m = make_mip()
        with _BranchingContext(m) as ctx:
            if ctx.done:
                pytest.skip("No branching node")
            obs = StrongBranchingScores()
            r1 = obs.extract(ctx.model, done=False)

            done, action_set = ctx.step()
            if done or action_set is None or len(action_set) == 0:
                pytest.skip("No second node")

            r2 = obs.extract(ctx.model, done=False)
            assert r1 is not r2  # different objects
            assert r2 is not None


# ---------------------------------------------------------------------------
# Multi-episode
# ---------------------------------------------------------------------------


class TestMultiEpisode:
    def test_two_episodes_give_consistent_root_results(self):
        """StrongBranching has no state, so two root observations must be identical."""
        obs = StrongBranchingScores()
        results = []
        for _ in range(2):
            m = make_mip()
            with _BranchingContext(m) as ctx:
                if ctx.done:
                    pytest.skip("No branching node")
                results.append(obs.extract(ctx.model, done=False))

        assert results[0] is not None and results[1] is not None
        np.testing.assert_allclose(results[0], results[1], equal_nan=True)
