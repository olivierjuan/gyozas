"""Unit and integration tests for gyozas.observations.node_bipartite."""

import pytest
from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.observations import BipartiteGraph
from gyozas.observations.node_bipartite_scip import NodeBipartiteSCIP

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
        obs = NodeBipartiteSCIP()
        assert obs is not None


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_does_not_raise(self):
        obs = NodeBipartiteSCIP()
        obs.reset(Model())


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


class TestExtract:
    def test_returns_tuple_during_solving(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = NodeBipartiteSCIP()
        obs.reset(m)
        result = obs.extract(m, done=False)
        assert result is not None

    def test_result_has_expected_structure(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = NodeBipartiteSCIP()
        obs.reset(m)
        result = obs.extract(m, done=False)
        assert isinstance(result, BipartiteGraph)
        assert result.variable_features is not None
        assert result.row_features is not None
        assert result.edge_features is not None

    def test_col_features_are_sequence(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = NodeBipartiteSCIP()
        obs.reset(m)
        result = obs.extract(m, done=False)
        col_features = result.variable_features
        assert col_features.ndim == 2
        assert col_features.shape[0] > 0
        assert col_features.shape[1] > 0

    def test_row_features_are_sequence(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = NodeBipartiteSCIP()
        obs.reset(m)
        result = obs.extract(m, done=False)
        row_features = result.row_features
        assert row_features.ndim == 2
        assert row_features.shape[0] > 0
        assert row_features.shape[1] > 0
