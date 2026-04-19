"""Unit and integration tests for gyozas.observations.node_bipartite_python."""

import numpy as np
import pytest
from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.observations import BipartiteGraph
from gyozas.observations.node_bipartite_ecole import (
    NodeBipartiteEcole,
    _put,
)

_INSTANCE = "tests/instance.lp"
_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}


def make_model() -> Model:
    m = Model()
    m.setParams(_PARAMS)
    m.readProblem(_INSTANCE)
    return m


@pytest.fixture
def branching_ctx():
    d = BranchingDynamics()
    m = make_model()
    done, action_set = d.reset(m)
    yield m, done, action_set, d
    d.close()


# ---------------------------------------------------------------------------
# _put helper
# ---------------------------------------------------------------------------


class TestPut:
    def test_assigns_when_present(self):
        row = np.zeros(3)
        _put(row, {"x": 1}, "x", 5.0)
        assert row[1] == 5.0

    def test_noop_when_missing(self):
        row = np.zeros(3)
        _put(row, {"x": 1}, "y", 5.0)
        assert row[1] == 0.0

    def test_empty_feature_map(self):
        row = np.zeros(3)
        _put(row, {}, "x", 5.0)
        np.testing.assert_array_equal(row, np.zeros(3))


# # ---------------------------------------------------------------------------
# # _set_static_feature_for_var
# # ---------------------------------------------------------------------------

# class TestSetStaticFeatureForVar:
#     def test_binary_one_hot(self):
#         row = np.zeros(5)
#         feature_map = {"binary": 0, "integer": 1, "implicit_integer": 2, "continuous": 3, "obj_coef": 4}

#         class _FakeVar:
#             def vtype(self):
#                 return "BINARY"
#             def getObj(self):
#                 return 2.0

#         _set_static_feature_for_var(row, _FakeVar(), 1.0, feature_map)
#         assert row[0] == 1.0  # binary
#         assert row[1] == 0.0  # integer
#         assert row[2] == 0.0  # implicit_integer
#         assert row[3] == 0.0  # continuous
#         assert row[4] == 2.0  # obj_coef / obj_norm

#     def test_integer_one_hot(self):
#         row = np.zeros(5)
#         feature_map = {"binary": 0, "integer": 1, "implicit_integer": 2, "continuous": 3, "obj_coef": 4}

#         class _FakeVar:
#             def vtype(self):
#                 return "INTEGER"
#             def getObj(self):
#                 return 0.0

#         _set_static_feature_for_var(row, _FakeVar(), 1.0, feature_map)
#         assert row[0] == 0.0
#         assert row[1] == 1.0

#     def test_obj_coef_normalized(self):
#         row = np.zeros(5)
#         feature_map = {"binary": 0, "integer": 1, "implicit_integer": 2, "continuous": 3, "obj_coef": 4}

#         class _FakeVar:
#             def vtype(self):
#                 return "CONTINUOUS"
#             def getObj(self):
#                 return 6.0

#         _set_static_feature_for_var(row, _FakeVar(), 2.0, feature_map)
#         assert row[4] == 3.0  # 6.0 / 2.0

#     def test_partial_feature_map(self):
#         row = np.zeros(2)
#         feature_map = {"binary": 0, "obj_coef": 1}

#         class _FakeVar:
#             def vtype(self):
#                 return "BINARY"
#             def getObj(self):
#                 return 1.0

#         _set_static_feature_for_var(row, _FakeVar(), 1.0, feature_map)
#         assert row[0] == 1.0
#         assert row[1] == 1.0


# ---------------------------------------------------------------------------
# NodeBipartiteEcole construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_instantiation(self):
        obs = NodeBipartiteEcole()
        assert obs is not None

    def test_reset_does_not_raise(self):
        obs = NodeBipartiteEcole()
        obs.reset(Model())


# ---------------------------------------------------------------------------
# NodeBipartiteEcole.extract during solving
# ---------------------------------------------------------------------------


class TestExtract:
    def test_returns_bipartite_graph(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = NodeBipartiteEcole()
        obs.reset(m)
        result = obs.extract(m, done=False)
        assert isinstance(result, BipartiteGraph)

    def test_col_features_shape(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = NodeBipartiteEcole()
        result = obs.extract(m, done=False)
        assert result is not None
        col_features = result.variable_features
        assert isinstance(col_features, np.ndarray)
        assert col_features.ndim == 2
        assert col_features.shape[0] > 0
        assert col_features.shape[1] > 0

    def test_edge_features_structure(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = NodeBipartiteEcole()
        result = obs.extract(m, done=False)
        assert result is not None
        idx_features = result.edge_features.indices
        coeff_features = result.edge_features.values
        assert isinstance(idx_features, np.ndarray)
        assert isinstance(coeff_features, np.ndarray)
        assert idx_features.shape[0] == 2
        assert idx_features.shape[1] == coeff_features.shape[0]

    def test_row_features_shape(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = NodeBipartiteEcole()
        result = obs.extract(m, done=False)
        assert result is not None
        row_features = result.row_features
        assert isinstance(row_features, np.ndarray)
        assert row_features.ndim == 2
        assert row_features.shape[0] > 0

    def test_no_nans_in_col_features(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = NodeBipartiteEcole()
        result = obs.extract(m, done=False)
        assert result is not None
        assert not np.any(np.isnan(result.variable_features))

    def test_no_nans_in_row_features(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = NodeBipartiteEcole()
        result = obs.extract(m, done=False)
        assert result is not None
        assert not np.any(np.isnan(result.row_features))

    def test_returns_none_outside_solving_stage(self):
        obs = NodeBipartiteEcole()
        m = make_model()
        result = obs.extract(m, done=False)
        assert result is None

    def test_get_repr_raises_outside_solving_stage(self):
        m = make_model()
        with pytest.raises(RuntimeError, match="SOLVING"):
            NodeBipartiteEcole.getBipartiteGraphRepresentation(m)


# ---------------------------------------------------------------------------
# getBipartiteGraphRepresentation with custom features
# ---------------------------------------------------------------------------


class TestCustomFeatures:
    def test_static_only(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        result = NodeBipartiteEcole.getBipartiteGraphRepresentation(
            m,
            static_only=True,
            dynamic_col_features=(),
            dynamic_row_features=(),
        )
        col_features = result[0]
        assert col_features.shape[1] == 5  # 5 static col features

    def test_subset_of_features(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        result = NodeBipartiteEcole.getBipartiteGraphRepresentation(
            m,
            static_col_features=("binary",),
            dynamic_col_features=("sol_val",),
            static_row_features=("bias",),
            dynamic_row_features=(),
        )
        col_features = result[0]
        assert col_features.shape[1] == 2  # binary + sol_val
        row_features = result[2]
        assert row_features.shape[1] == 1  # bias only


# ---------------------------------------------------------------------------
# Incremental update (prev_* features)
# ---------------------------------------------------------------------------


class TestIncrementalUpdate:
    def test_prev_col_features_reused(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        result1 = NodeBipartiteEcole.getBipartiteGraphRepresentation(m)
        col1 = result1[0]
        result2 = NodeBipartiteEcole.getBipartiteGraphRepresentation(m, prev_col_features=col1)
        col2 = result2[0]
        # When shape matches, prev is reused (same object)
        assert col2 is col1

    def test_prev_col_wrong_shape_warns(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        wrong_prev = np.zeros((1, 1))
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NodeBipartiteEcole.getBipartiteGraphRepresentation(m, prev_col_features=wrong_prev)
            assert len(w) >= 1

    def test_suppress_warnings(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        wrong_prev = np.zeros((1, 1))
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NodeBipartiteEcole.getBipartiteGraphRepresentation(m, prev_col_features=wrong_prev, suppress_warnings=True)
            col_warnings = [x for x in w if "column" in str(x.message).lower()]
            assert len(col_warnings) == 0
