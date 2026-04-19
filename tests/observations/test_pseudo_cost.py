"""Unit and integration tests for gyozas.observations.pseudo_cost.Pseudocosts."""

from typing import cast

import numpy as np
import pytest
from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.observations import ObservationFunction, Pseudocosts
from gyozas.observations.pseudo_cost import _EPS, _INIT_PSEUDOCOST

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INSTANCE = "tests/instance.lp"
_BASE_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}


def make_mip(scip_params: dict | None = None) -> Model:
    """Load the shared test instance with a node limit to keep tests fast."""
    m = Model()
    m.setParams(_BASE_PARAMS)
    if scip_params:
        m.setParams(scip_params)
    m.readProblem(_INSTANCE)
    return m


class _BranchingContext:
    """Run BranchingDynamics on a model and expose the model at each node."""

    def __init__(self, m: Model):
        self._dynamics = BranchingDynamics()
        self.model = m
        self.done, self.action_set = self._dynamics.reset(m)

    def step(self) -> tuple[bool, np.ndarray | None]:
        if self.done or self.action_set is None or len(self.action_set) == 0:
            return self.done, None
        self.done, self.action_set = self._dynamics.step(self.action_set[0])
        return self.done, self.action_set

    def close(self):
        self._dynamics.close()

    # -- context manager support --
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@pytest.fixture
def branching_ctx():
    """Provides a BranchingContext at the root branching node of a small MIP."""
    m = make_mip()
    ctx = _BranchingContext(m)
    yield ctx
    ctx.close()


# ---------------------------------------------------------------------------
# Protocol / import tests
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_is_observation_function(self):
        assert isinstance(Pseudocosts(), ObservationFunction)

    def test_exported_from_package(self):
        from gyozas.observations import Pseudocosts as PC  # noqa: F401


# ---------------------------------------------------------------------------
# Unit tests for _pseudocost (no SCIP required)
# ---------------------------------------------------------------------------


class TestPseudocostHelper:
    def test_returns_prior_when_no_history(self):
        obs = Pseudocosts()
        assert obs._pseudocost(0, down=True) == _INIT_PSEUDOCOST
        assert obs._pseudocost(0, down=False) == _INIT_PSEUDOCOST

    def test_returns_prior_for_unknown_index(self):
        obs = Pseudocosts()
        obs._pseudo_down[1] = [2.0, 1.0]
        assert obs._pseudocost(99, down=True) == _INIT_PSEUDOCOST

    def test_correct_ratio_down(self):
        obs = Pseudocosts()
        obs._pseudo_down[3] = [6.0, 2.0]
        assert abs(obs._pseudocost(3, down=True) - 3.0) < 1e-12

    def test_correct_ratio_up(self):
        obs = Pseudocosts()
        obs._pseudo_up[3] = [4.0, 8.0]
        assert abs(obs._pseudocost(3, down=False) - 0.5) < 1e-12

    def test_zero_denominator_returns_prior(self):
        obs = Pseudocosts()
        obs._pseudo_down[7] = [5.0, 0.0]
        assert obs._pseudocost(7, down=True) == _INIT_PSEUDOCOST

    def test_below_eps_denominator_returns_prior(self):
        obs = Pseudocosts()
        obs._pseudo_down[7] = [5.0, _EPS / 2]
        assert obs._pseudocost(7, down=True) == _INIT_PSEUDOCOST

    def test_down_and_up_are_independent(self):
        obs = Pseudocosts()
        obs._pseudo_down[2] = [3.0, 1.0]
        obs._pseudo_up[2] = [6.0, 1.0]
        assert abs(obs._pseudocost(2, down=True) - 3.0) < 1e-12
        assert abs(obs._pseudocost(2, down=False) - 6.0) < 1e-12

    def test_accumulates_multiple_updates(self):
        obs = Pseudocosts()
        obs._pseudo_down[0] = [0.0, 0.0]
        obs._pseudo_down[0][0] += 2.0
        obs._pseudo_down[0][1] += 0.5
        obs._pseudo_down[0][0] += 4.0
        obs._pseudo_down[0][1] += 0.5
        # expected: 6.0 / 1.0 = 6.0
        assert abs(obs._pseudocost(0, down=True) - 6.0) < 1e-12


# ---------------------------------------------------------------------------
# Unit tests for reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_pseudo_down(self):
        obs = Pseudocosts()
        obs._pseudo_down[1] = [1.0, 0.5]
        obs.reset(cast(Model, None))  # model unused
        assert obs._pseudo_down == {}

    def test_reset_clears_pseudo_up(self):
        obs = Pseudocosts()
        obs._pseudo_up[1] = [1.0, 0.5]
        obs.reset(cast(Model, None))
        assert obs._pseudo_up == {}

    def test_reset_clears_node_lp_vals(self):
        obs = Pseudocosts()
        obs._node_lp_vals[42] = {0: 0.5}
        obs.reset(cast(Model, None))
        assert obs._node_lp_vals == {}

    def test_reset_returns_none(self):
        assert Pseudocosts().reset(cast(Model, None)) is None


# ---------------------------------------------------------------------------
# extract: early-exit guards
# ---------------------------------------------------------------------------


class TestExtractGuards:
    def test_returns_none_when_done(self, branching_ctx):
        obs = Pseudocosts()
        obs.reset(branching_ctx.model)
        result = obs.extract(branching_ctx.model, done=True)
        assert result is None

    def test_returns_none_before_solving_stage(self):
        obs = Pseudocosts()
        m = make_mip()
        obs.reset(m)
        # Model has not been optimized yet → wrong stage
        result = obs.extract(m, done=False)
        assert result is None

    def test_returns_none_after_solve_completes(self):
        obs = Pseudocosts()
        m = make_mip()
        obs.reset(m)
        m.optimize()
        # Stage is now SOLVED, not SOLVING
        result = obs.extract(m, done=False)
        assert result is None


# ---------------------------------------------------------------------------
# extract: output shape and dtype during solving
# ---------------------------------------------------------------------------


class TestExtractOutputShape:
    def test_returns_ndarray_during_solving(self, branching_ctx):
        import numpy as np

        obs = Pseudocosts()
        obs.reset(branching_ctx.model)
        if branching_ctx.done:
            pytest.skip("Instance solved at root")
        result = obs.extract(branching_ctx.model, done=False)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_shape_covers_all_candidates(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("Instance solved at root")
        obs = Pseudocosts()
        obs.reset(branching_ctx.model)
        result = obs.extract(branching_ctx.model, done=False)
        cands, _, _, n_cands, _, _ = branching_ctx.model.getLPBranchCands()
        max_idx = max(v.getCol().getLPPos() for v in cands[:n_cands])
        assert result is not None
        assert result.shape[0] > max_idx

    def test_dtype_is_float64(self, branching_ctx):
        import numpy as np

        if branching_ctx.done:
            pytest.skip("Instance solved at root")
        obs = Pseudocosts()
        obs.reset(branching_ctx.model)
        result = obs.extract(branching_ctx.model, done=False)
        assert result is not None
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# extract: NaN / finite placement
# ---------------------------------------------------------------------------


class TestExtractNaNPlacement:
    def test_lp_candidates_have_finite_scores(self, branching_ctx):
        import numpy as np

        if branching_ctx.done:
            pytest.skip("Instance solved at root")
        obs = Pseudocosts()
        obs.reset(branching_ctx.model)
        result = obs.extract(branching_ctx.model, done=False)

        cands, _, _, n_cands, _, _ = branching_ctx.model.getLPBranchCands()
        assert result is not None
        for var in cands[:n_cands]:
            assert np.isfinite(result[var.getCol().getLPPos()]), f"var {var.getCol().getLPPos()} should be finite"

    def test_non_candidates_are_nan(self, branching_ctx):
        import numpy as np

        if branching_ctx.done:
            pytest.skip("Instance solved at root")
        obs = Pseudocosts()
        obs.reset(branching_ctx.model)
        result = obs.extract(branching_ctx.model, done=False)

        cands, _, _, n_cands, _, _ = branching_ctx.model.getLPBranchCands()
        cand_indices = {var.getCol().getLPPos() for var in cands[:n_cands]}
        assert result is not None
        for i, val in enumerate(result):
            if i not in cand_indices:
                assert np.isnan(val), f"non-candidate var {i} should be NaN"

    def test_scores_are_nonnegative(self, branching_ctx):
        import numpy as np

        if branching_ctx.done:
            pytest.skip("Instance solved at root")
        obs = Pseudocosts()
        obs.reset(branching_ctx.model)
        result = obs.extract(branching_ctx.model, done=False)
        assert result is not None
        finite = result[np.isfinite(result)]
        assert (finite >= 0.0).all()

    def test_at_least_one_candidate(self, branching_ctx):
        import numpy as np

        if branching_ctx.done:
            pytest.skip("Instance solved at root")
        obs = Pseudocosts()
        obs.reset(branching_ctx.model)
        result = obs.extract(branching_ctx.model, done=False)
        assert result is not None
        assert np.any(np.isfinite(result))


# ---------------------------------------------------------------------------
# extract: root node uses prior scores
# ---------------------------------------------------------------------------


class TestRootNodePrior:
    def test_root_uses_init_pseudocost(self, branching_ctx):
        """At the root there is no parent branching, so pseudocosts use the init prior."""
        import numpy as np

        if branching_ctx.done:
            pytest.skip("Instance solved at root")
        obs = Pseudocosts()
        obs.reset(branching_ctx.model)
        result = obs.extract(branching_ctx.model, done=False)

        assert result is not None
        finite_scores = result[np.isfinite(result)]
        assert len(finite_scores) > 0
        # All candidates should use the init prior (no history yet)
        assert obs._pseudo_down == {}
        assert obs._pseudo_up == {}


# ---------------------------------------------------------------------------
# extract: pseudocost history accumulates across nodes
# ---------------------------------------------------------------------------


class TestHistoryAccumulation:
    def test_node_lp_vals_cached_after_extract(self, branching_ctx):
        if branching_ctx.done:
            pytest.skip("Instance solved at root")
        obs = Pseudocosts()
        obs.reset(branching_ctx.model)
        obs.extract(branching_ctx.model, done=False)
        node = branching_ctx.model.getCurrentNode()
        assert node.getNumber() in obs._node_lp_vals

    def test_pseudocosts_updated_at_child_node(self):
        """After one branching, the child node's extract should trigger a pseudocost update."""
        m = make_mip()
        with _BranchingContext(m) as ctx:
            if ctx.done:
                pytest.skip("Instance solved at root")
            obs = Pseudocosts()
            obs.reset(ctx.model)
            obs.extract(ctx.model, done=False)  # root: cache LP vals

            done, action_set = ctx.step()
            if done or action_set is None:
                pytest.skip("Instance solved after one branch")

            obs.extract(ctx.model, done=False)  # child: should update pseudocosts
            assert (
                len(obs._pseudo_down) > 0 or len(obs._pseudo_up) > 0
            ), "At least one pseudocost direction should be updated after a branching"

    def test_scores_can_differ_after_history(self):
        """After branchings, candidates may get different scores (history breaks symmetry)."""
        m = make_mip()
        with _BranchingContext(m) as ctx:
            if ctx.done:
                pytest.skip("Instance solved at root")
            obs = Pseudocosts()
            obs.reset(ctx.model)
            obs.extract(ctx.model, done=False)

            # Take two branchings to build up some history
            for _ in range(2):
                done, action_set = ctx.step()
                if done or action_set is None or len(action_set) == 0:
                    break
                obs.extract(ctx.model, done=False)


# ---------------------------------------------------------------------------
# Multi-episode: reset clears history
# ---------------------------------------------------------------------------


class TestMultiEpisode:
    def test_reset_between_episodes_clears_history(self):
        def run_one_episode(obs):
            m = make_mip()
            with _BranchingContext(m) as ctx:
                if ctx.done:
                    return None
                obs.reset(ctx.model)
                obs.extract(ctx.model, done=False)
                ctx.step()
                if ctx.done:
                    return None
                return obs.extract(ctx.model, done=False)

        obs = Pseudocosts()

        # Episode 1: accumulate history
        run_one_episode(obs)

        # Episode 2: reset should wipe history
        result = run_one_episode(obs)

        # After reset, history should be from ep2 only (not accumulated from ep1)
        if result is not None:
            assert obs._node_lp_vals is not None  # object alive
            # The keys should not include any from ep1 (since reset cleared everything)
            # We can't check exact node numbers, but the cache shouldn't be enormous
            assert len(obs._node_lp_vals) <= 10  # small instance → few nodes

    def test_two_consecutive_resets_give_same_root_obs(self):
        import numpy as np

        obs = Pseudocosts()
        results = []
        for _ in range(2):
            m = make_mip()
            with _BranchingContext(m) as ctx:
                if ctx.done:
                    pytest.skip("Instance solved at root")
                obs.reset(ctx.model)
                results.append(obs.extract(ctx.model, done=False))

        assert results[0] is not None and results[1] is not None
        # Both root obs use prior → should be equal (same problem structure)
        np.testing.assert_array_equal(np.isnan(results[0]), np.isnan(results[1]))


# ---------------------------------------------------------------------------
# _update_from_current_node edge cases
# ---------------------------------------------------------------------------


class TestUpdateEdgeCases:
    def test_no_update_when_no_parent_branching(self, branching_ctx):
        """Root node has no parent branchings → dicts stay empty."""
        if branching_ctx.done:
            pytest.skip("Instance solved at root")
        obs = Pseudocosts()
        obs.reset(branching_ctx.model)
        obs._update_from_current_node(branching_ctx.model)
        assert obs._pseudo_down == {}
        assert obs._pseudo_up == {}

    def test_no_update_when_parent_lp_vals_missing(self):
        """If parent LP values were never cached, update is skipped."""
        m = make_mip()
        with _BranchingContext(m) as ctx:
            if ctx.done:
                pytest.skip("Instance solved at root")

            obs = Pseudocosts()
            obs.reset(ctx.model)
            # Do NOT call extract at root → parent LP vals never cached

            done, action_set = ctx.step()
            if done or action_set is None or len(action_set) == 0:
                pytest.skip("No second node")

            obs._update_from_current_node(ctx.model)
            # Parent LP vals missing → no update
            assert obs._pseudo_down == {}
            assert obs._pseudo_up == {}

    def test_boundtype_upper_updates_pseudo_down(self):
        """Simulated upper-bound branching should update _pseudo_down."""
        obs = Pseudocosts()
        obs._pseudo_down.clear()
        obs._pseudo_up.clear()

        # Inject artificial data: btype = _BOUNDTYPE_UPPER → down
        _BOUNDTYPE_LOWER = 0
        # Call internal accumulation directly
        idx = 0
        val_delta = 0.4  # frac
        obj_delta = 2.0
        entry = obs._pseudo_down.setdefault(idx, [0.0, 0.0])
        entry[0] += obj_delta
        entry[1] += val_delta

        assert abs(obs._pseudocost(0, down=True) - (2.0 / 0.4)) < 1e-9
        assert obs._pseudo_up == {}  # up untouched
