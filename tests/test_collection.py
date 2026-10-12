"""Tests for collection metadata: multi-optimal node distribution (R4.3)."""

import numpy as np
import pytest

from gyozas.collection import CollectionMetadata


class TestCollectionMetadataUnit:
    def test_unique_optimum(self):
        meta = CollectionMetadata()
        n = meta.record(np.array([1.0, 3.0, 2.0]), np.array([0, 1, 2]))
        assert n == 1
        assert meta.multi_optimal_distribution() == {1: 1}
        assert meta.frac_multi_optimal == 0.0

    def test_two_optima_with_ties(self):
        meta = CollectionMetadata()
        n = meta.record(np.array([1.0, 3.0, 2.0, 3.0]), np.array([0, 1, 2, 3]))
        assert n == 2
        assert meta.multi_optimal_distribution() == {2: 1}
        assert meta.frac_multi_optimal == 1.0

    def test_tie_tolerance_relative(self):
        meta = CollectionMetadata(tie_tolerance=1e-6)
        n = meta.record({0: 1e6, 1: 1e6 - 1e-3}, np.array([0, 1]))
        assert n == 2

    def test_nan_scores_ignored(self):
        meta = CollectionMetadata()
        n = meta.record(np.array([np.nan, 2.0, np.nan]), np.array([0, 1, 2]))
        assert n == 1

    def test_all_non_finite_scores(self):
        meta = CollectionMetadata()
        n = meta.record(np.array([np.nan, np.inf]), np.array([0, 1]))
        assert n == 0
        assert meta.multi_optimal_distribution() == {0: 1}

    def test_distribution_over_many_nodes(self):
        meta = CollectionMetadata()
        meta.record(np.array([1.0, 2.0]), np.array([0, 1]))  # unique
        meta.record(np.array([2.0, 2.0]), np.array([0, 1]))  # 2 optima
        meta.record(np.array([2.0, 2.0]), np.array([0, 1]))  # 2 optima
        assert meta.multi_optimal_distribution() == {1: 1, 2: 2}
        assert meta.n_decisions == 3
        assert meta.frac_multi_optimal == pytest.approx(2 / 3)

    def test_reset(self):
        meta = CollectionMetadata()
        meta.record(np.array([1.0]), np.array([0]))
        meta.reset()
        assert meta.n_decisions == 0
        assert meta.multi_optimal_distribution() == {}
        assert meta.frac_multi_optimal == 0.0

    def test_to_dict(self):
        meta = CollectionMetadata()
        meta.record(np.array([1.0, 2.0]), np.array([0, 1]))
        d = meta.to_dict()
        assert d["n_decisions"] == 1
        assert d["multi_optimal_distribution"] == {1: 1}
        assert d["frac_multi_optimal"] == 0.0

    def test_action_outside_score_range_ignored(self):
        meta = CollectionMetadata()
        n = meta.record(np.array([1.0]), np.array([5]))
        assert n == 0


class TestCollectionMetadataIntegration:
    def test_expert_collection_records_distribution(self):
        from gyozas.environment import Environment
        from gyozas.instances.files import FileGenerator

        env = Environment(
            instance_generator=FileGenerator(directory="tests", pattern="*.lp"),
            scip_params={"display/verblevel": 0, "limits/nodes": 50},
        )
        meta = CollectionMetadata()
        try:
            obs, action_set, reward, done, info = env.reset()
            while not done:
                cands, lp_vals, _, n_cands, _, _ = env.model.getLPBranchCands()
                scores = {
                    int(var.getCol().getLPPos()): env.model.getVarPseudocostScore(var, val)
                    for var, val in zip(cands[:n_cands], lp_vals[:n_cands], strict=False)
                }
                meta.record(scores, action_set)
                obs, action_set, reward, done, info = env.step(int(action_set[len(action_set) // 2]))
            assert meta.n_decisions > 0
            assert sum(meta.multi_optimal_distribution().values()) == meta.n_decisions
            assert 0.0 <= meta.frac_multi_optimal <= 1.0
        finally:
            env.close()
