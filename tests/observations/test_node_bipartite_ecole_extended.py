"""Extended tests for gyozas.observations.node_bipartite_python.NodeBipartiteEcole."""

import numpy as np
import pytest
from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.observations import BipartiteGraph
from gyozas.observations.node_bipartite_ecole import NodeBipartiteEcole

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
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_instantiation(self):
        obs = NodeBipartiteEcole()
        assert obs is not None

    def test_reset_noop(self):
        obs = NodeBipartiteEcole()
        obs.reset(Model())  # no-op


# ---------------------------------------------------------------------------
# Extract
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
        assert col_features.dtype == np.float64

    def test_row_features_shape(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        result = NodeBipartiteEcole().extract(m, done=False)
        assert result is not None
        row_features = result.row_features
        assert isinstance(row_features, np.ndarray)
        assert row_features.ndim == 2
        assert row_features.dtype == np.float64

    def test_edge_features_structure(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        result = NodeBipartiteEcole().extract(m, done=False)
        assert result is not None
        idx_features = result.edge_features.indices
        coeff_features = result.edge_features.values
        assert isinstance(idx_features, np.ndarray)
        assert isinstance(coeff_features, np.ndarray)
        assert idx_features.shape[0] == 2
        assert idx_features.shape[1] == coeff_features.shape[0]

    def test_get_repr_returns_metadata_dict(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        result = NodeBipartiteEcole.getBipartiteGraphRepresentation(m)
        assert isinstance(result[3], dict)


# ---------------------------------------------------------------------------
# getBipartiteGraphRepresentation: error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_raises_outside_solving_stage(self):
        m = make_model()
        with pytest.raises(RuntimeError, match="SOLVING"):
            NodeBipartiteEcole.getBipartiteGraphRepresentation(m)


# ---------------------------------------------------------------------------
# Incremental updates (prev_ features)
# ---------------------------------------------------------------------------


class TestIncremental:
    def test_prev_col_features_same_shape(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = NodeBipartiteEcole()
        r1 = obs.extract(m, done=False)
        assert r1 is not None
        # Pass previous features back via the static interface
        col = r1.variable_features
        idx = r1.edge_features.indices
        coeff = r1.edge_features.values
        row = r1.row_features
        r2 = NodeBipartiteEcole.getBipartiteGraphRepresentation(
            m, prev_col_features=col, prev_edge_features=(idx, coeff), prev_row_features=row
        )
        assert r2[0].shape == col.shape
        assert r2[2].shape == row.shape

    def test_prev_col_features_different_shape_warns(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        # Pass wrong-shaped previous features
        wrong_col = np.zeros((1, 1))
        with pytest.warns(UserWarning, match="Variable features shape changed"):
            NodeBipartiteEcole.getBipartiteGraphRepresentation(m, prev_col_features=wrong_col)

    def test_prev_row_features_different_shape_warns(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        wrong_row = np.zeros((1, 1))
        with pytest.warns(UserWarning, match="Row features shape changed"):
            NodeBipartiteEcole.getBipartiteGraphRepresentation(m, prev_row_features=wrong_row)

    def test_prev_edge_features_different_shape_warns(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        wrong_idx = np.zeros((2, 1), dtype=np.int64)
        wrong_coeff = np.zeros(1)
        with pytest.warns(UserWarning, match="Edge features shape changed"):
            NodeBipartiteEcole.getBipartiteGraphRepresentation(m, prev_edge_features=(wrong_idx, wrong_coeff))

    def test_suppress_warnings(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        wrong_col = np.zeros((1, 1))
        # Should not warn with suppress_warnings=True
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            NodeBipartiteEcole.getBipartiteGraphRepresentation(m, prev_col_features=wrong_col, suppress_warnings=True)


# ---------------------------------------------------------------------------
# Static-only mode
# ---------------------------------------------------------------------------


class TestStaticOnly:
    def test_static_only_mode(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        result = NodeBipartiteEcole.getBipartiteGraphRepresentation(m, static_only=True)
        col_features = result[0]
        assert isinstance(col_features, np.ndarray)
        assert col_features.shape[0] > 0
