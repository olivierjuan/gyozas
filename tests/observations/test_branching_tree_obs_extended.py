"""Tests for gyozas.observations.branching_tree.BranchingTreeObservation."""

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
    def test_initial_tree_empty(self):
        obs = BranchingTreeObservation()
        assert isinstance(obs.tree, nx.DiGraph)
        assert obs.tree.number_of_nodes() == 0

    def test_reset_creates_fresh_tree(self):
        obs = BranchingTreeObservation()
        obs.tree.add_node(1)
        obs.reset(Model())
        assert obs.tree.number_of_nodes() == 0


# ---------------------------------------------------------------------------
# Extract during solving
# ---------------------------------------------------------------------------


class TestExtract:
    def test_returns_digraph(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = BranchingTreeObservation()
        obs.reset(m)
        result = obs.extract(m, done=False)
        assert isinstance(result, nx.DiGraph)

    def test_tree_has_nodes(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = BranchingTreeObservation()
        obs.reset(m)
        result = obs.extract(m, done=False)
        assert result.number_of_nodes() > 0

    def test_root_node_set(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = BranchingTreeObservation()
        obs.reset(m)
        result = obs.extract(m, done=False)
        assert "root" in result.graph

    def test_current_node_visited(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = BranchingTreeObservation()
        obs.reset(m)
        result = obs.extract(m, done=False)
        node = m.getCurrentNode()
        if node is not None:
            nid = node.getNumber()
            assert result.nodes[nid].get("visited") is True

    def test_tree_grows_after_step(self, branching_ctx):
        m, done, action_set, d = branching_ctx
        if done:
            pytest.skip("Instance solved at root")
        obs = BranchingTreeObservation()
        obs.reset(m)
        r1 = obs.extract(m, done=False)
        n1 = r1.number_of_nodes()

        done, action_set = d.step(action_set[0])
        if done:
            return
        r2 = obs.extract(m, done=False)
        n2 = r2.number_of_nodes()
        assert n2 >= n1


# ---------------------------------------------------------------------------
# add_node
# ---------------------------------------------------------------------------


class TestAddNode:
    def test_add_node_with_lower_bound(self):
        obs = BranchingTreeObservation()

        class FakeNode:
            def getParent(self):
                return None

            def getNumber(self):
                return 1

        obs._add_node(1, FakeNode(), lower_bound=5.0)
        assert obs.tree.nodes[1]["lower_bound"] == 5.0

    def test_add_node_with_estimate(self):
        obs = BranchingTreeObservation()

        class FakeNode:
            def getParent(self):
                return None

            def getNumber(self):
                return 1

        obs._add_node(1, FakeNode(), estimate=3.0)
        assert obs.tree.nodes[1]["estimate"] == 3.0

    def test_first_node_becomes_root(self):
        obs = BranchingTreeObservation()

        class FakeNode:
            def getParent(self):
                return None

            def getNumber(self):
                return 42

        obs._add_node(42, FakeNode())
        assert obs.tree.graph["root"] == 42
