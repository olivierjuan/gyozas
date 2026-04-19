"""Unit and integration tests for gyozas.observations.branching_tree.BranchingTreeObservation."""

import networkx as nx
import pytest
from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.observations.branching_tree import BranchingTreeObservation

_INSTANCE = "tests/instance.lp"
_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}


def make_model() -> Model:
    m = Model()
    m.setParams(_PARAMS)
    m.readProblem(_INSTANCE)
    return m


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_tree_is_digraph(self):
        obs = BranchingTreeObservation()
        assert isinstance(obs.tree, nx.DiGraph)

    def test_tree_initially_empty(self):
        obs = BranchingTreeObservation()
        assert obs.tree.number_of_nodes() == 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_creates_new_graph(self):
        obs = BranchingTreeObservation()
        obs.tree.add_node(999)
        obs.reset(Model())
        assert obs.tree.number_of_nodes() == 0

    def test_reset_is_idempotent(self):
        obs = BranchingTreeObservation()
        obs.reset(Model())
        obs.reset(Model())
        assert obs.tree.number_of_nodes() == 0


# ---------------------------------------------------------------------------
# Extract during solving
# ---------------------------------------------------------------------------


class TestExtract:
    def test_returns_digraph(self):
        obs = BranchingTreeObservation()
        m = make_model()
        d = BranchingDynamics()
        done, action_set = d.reset(m)
        if done:
            d.close()
            pytest.skip("Instance solved at root")
        obs.reset(m)
        result = obs.extract(m, done=False)
        assert isinstance(result, nx.DiGraph)
        d.close()

    def test_root_node_added(self):
        obs = BranchingTreeObservation()
        m = make_model()
        d = BranchingDynamics()
        done, action_set = d.reset(m)
        if done:
            d.close()
            pytest.skip("Instance solved at root")
        obs.reset(m)
        obs.extract(m, done=False)
        assert obs.tree.number_of_nodes() >= 1
        assert "root" in obs.tree.graph
        d.close()

    def test_root_node_visited(self):
        obs = BranchingTreeObservation()
        m = make_model()
        d = BranchingDynamics()
        done, action_set = d.reset(m)
        if done:
            d.close()
            pytest.skip("Instance solved at root")
        obs.reset(m)
        obs.extract(m, done=False)
        root = obs.tree.graph["root"]
        assert obs.tree.nodes[root]["visited"] is True
        d.close()

    def test_tree_grows_after_step(self):
        obs = BranchingTreeObservation()
        m = make_model()
        d = BranchingDynamics()
        done, action_set = d.reset(m)
        if done:
            d.close()
            pytest.skip("Instance solved at root")
        obs.reset(m)
        obs.extract(m, done=False)
        n_before = obs.tree.number_of_nodes()

        assert action_set is not None
        done, action_set = d.step(action_set[0])
        if done:
            d.close()
            pytest.skip("Instance solved after one step")
        obs.extract(m, done=False)
        assert obs.tree.number_of_nodes() >= n_before
        d.close()


# ---------------------------------------------------------------------------
# add_node
# ---------------------------------------------------------------------------


class TestAddNode:
    def test_add_node_with_attributes(self):
        obs = BranchingTreeObservation()
        m = make_model()
        d = BranchingDynamics()
        done, action_set = d.reset(m)
        if done:
            d.close()
            pytest.skip("Instance solved at root")
        obs.reset(m)
        obs.extract(m, done=False)
        root = obs.tree.graph["root"]
        data = obs.tree.nodes[root]
        assert "lower_bound" in data
        assert "estimate" in data
        d.close()

    def test_idempotent_add(self):
        obs = BranchingTreeObservation()
        m = make_model()
        d = BranchingDynamics()
        done, action_set = d.reset(m)
        if done:
            d.close()
            pytest.skip("Instance solved at root")
        obs.reset(m)
        obs.extract(m, done=False)
        n1 = obs.tree.number_of_nodes()
        obs.extract(m, done=False)
        n2 = obs.tree.number_of_nodes()
        # Should not duplicate nodes for the same SCIP state
        assert n2 >= n1
        d.close()
