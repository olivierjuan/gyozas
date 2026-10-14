"""Tests for BranchingStateObservation (R1.2b — the Lin 2022 solver-stats triple).

The state is the T-BranT feature layout (25 per-candidate + 8 node + 53 tree),
reimplemented on stock PySCIPOpt: unavailable statistics are exact zeros and
per-slot availability is recorded in the feature tables.
"""

import numpy as np
import pytest

from gyozas.environment import Environment
from gyozas.instances.files import FileGenerator
from gyozas.observations import (
    CANDIDATE_FEATURES,
    MIP_FEATURES,
    NODE_FEATURES,
    BranchingState,
    BranchingStateObservation,
)
from gyozas.observations.branching_state import rel_distance, rel_position


@pytest.fixture
def env():
    env = Environment(
        instance_generator=FileGenerator(directory="tests", pattern="*.lp"),
        observation_function=BranchingStateObservation(),
        scip_params={"display/verblevel": 0, "limits/nodes": 50},
    )
    yield env
    env.close()


class TestFeatureTables:
    """The Lin 2022 layout and its per-slot availability record."""

    def test_layout_dimensions(self):
        assert len(CANDIDATE_FEATURES) == 25
        assert len(NODE_FEATURES) == 8
        assert len(MIP_FEATURES) == 53

    def test_every_slot_is_named(self):
        for spec in CANDIDATE_FEATURES + NODE_FEATURES + MIP_FEATURES:
            assert spec.name
            assert isinstance(spec.available, bool)

    def test_unavailable_slots_cite_the_missing_scip_call(self):
        for spec in CANDIDATE_FEATURES + NODE_FEATURES + MIP_FEATURES:
            if not spec.available:
                assert spec.note, f"{spec.name} is unavailable but cites no SCIP call"

    def test_availability_summary(self):
        """What stock PySCIPOpt can actually fill (documented reduction)."""
        assert sum(s.available for s in NODE_FEATURES) == 8
        assert sum(s.available for s in CANDIDATE_FEATURES) == 3
        assert sum(s.available for s in MIP_FEATURES) == 30


class TestNormalizers:
    def test_rel_distance_straddling_zero_is_zero(self):
        assert rel_distance(-1.0, 2.0) == 0.0

    def test_rel_distance_equal_values(self):
        assert rel_distance(3.5, 3.5) == 0.0

    def test_rel_distance_scales_by_the_larger_magnitude(self):
        assert rel_distance(1.0, 3.0) == pytest.approx(2.0 / 3.0)

    def test_rel_position_zero_when_ub_infinite(self):
        assert rel_position(5.0, float("inf"), 0.0, True) == 0.0

    def test_rel_position_between_bounds(self):
        assert rel_position(5.0, 10.0, 0.0, False) == pytest.approx(0.5)

    def test_rel_position_zero_when_bounds_coincide(self):
        assert rel_position(5.0, 7.0, 7.0, False) == 0.0


class TestBranchingState:
    def test_reset_returns_the_three_blocks(self, env):
        state, action_set, reward, done, info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        assert isinstance(state, BranchingState)
        assert state.node_features.shape == (8,)
        assert state.mip_features.shape == (53,)
        assert state.candidate_features.shape == (len(state.candidate_indices), 25)

    def test_candidates_align_with_the_action_set(self, env):
        state, action_set, reward, done, info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        assert np.array_equal(state.candidate_indices, action_set)

    def test_all_features_are_finite(self, env):
        state, action_set, reward, done, info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        assert np.isfinite(state.candidate_features).all()
        assert np.isfinite(state.node_features).all()
        assert np.isfinite(state.mip_features).all()

    def test_root_zeroes_the_depth_slots(self, env):
        state, _action_set, _reward, done, _info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        assert state.node_features[0] == 0.0  # relative depth
        assert state.node_features[1] == 0.0  # relative plunge depth
        assert state.mip_features[:16] == pytest.approx(np.zeros(16))

    def test_candidate_slots(self, env):
        state, _action_set, _reward, done, _info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        # Slot 0: LP fractionality of a branching candidate — strictly fractional.
        assert (state.candidate_features[:, 0] > 0).all()
        assert (state.candidate_features[:, 0] < 1).all()
        # Slot 8: the average-normalized pseudocost score proxy — in [0, 1).
        assert (state.candidate_features[:, 8] >= 0).all()
        assert (state.candidate_features[:, 8] < 1).all()
        # Slots the stock API cannot fill are exact zeros.
        unavailable = [i for i, spec in enumerate(CANDIDATE_FEATURES) if not spec.available]
        if unavailable:
            assert (state.candidate_features[:, unavailable] == 0).all()

    def test_step_keeps_alignment_and_fills_tree_features(self, env):
        state, action_set, _reward, done, _info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        while not done:
            action = action_set[len(action_set) // 2]
            state, action_set, _reward, done, _info = env.step(action)
            if done:
                break
            assert np.array_equal(state.candidate_indices, action_set)
            assert np.isfinite(state.mip_features).all()
            # Past the root, the tree-level block carries signal.
            if state.node_features[0] > 0:
                assert state.mip_features[4] > 0  # nodes-left ratio
                return

    def test_returns_none_when_done(self, env):
        obs, action_set, _reward, done, _info = env.reset()
        while not done:
            obs, action_set, _reward, done, _info = env.step(action_set[len(action_set) // 2])
        assert obs is None

    def test_unavailable_mip_slots_are_zero(self, env):
        state, _action_set, _reward, done, _info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        unavailable = [i for i, spec in enumerate(MIP_FEATURES) if not spec.available]
        assert (state.mip_features[unavailable] == 0).all()
