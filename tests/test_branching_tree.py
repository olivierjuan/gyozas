"""Unit tests for gyozas.branching_tree.BranchingTree, focusing on _render_rich."""

import unittest.mock as mock

from gyozas.branching_tree import BranchingTree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tree(nodes=None, edges=None, root=None) -> BranchingTree:
    """Build a BranchingTree from scratch without a SCIP model."""
    bt = BranchingTree()
    for node_id, data in nodes or []:
        bt.tree.add_node(node_id, **data)
    for src, dst in edges or []:
        bt.tree.add_edge(src, dst)
    if root is not None:
        bt.tree.graph["root"] = root
    return bt


def _simple_tree() -> BranchingTree:
    """Single root (visited) with two children (unvisited)."""
    bt = _make_tree(
        nodes=[
            (1, {"visited": True, "step": 0, "lower_bound": 0.0, "estimate": 1.5}),
            (2, {"visited": False, "estimate": 2.0}),
            (3, {"visited": False, "estimate": 3.0}),
        ],
        edges=[(1, 2), (1, 3)],
        root=1,
    )
    return bt


def _status_tree() -> BranchingTree:
    """Root with children having different statuses."""
    bt = _make_tree(
        nodes=[
            (1, {"visited": True, "step": 1}),
            (2, {"status": "infeasible", "lower_bound": 5.0}),
            (3, {"status": "feasible", "obj": 7.0}),
        ],
        edges=[(1, 2), (1, 3)],
        root=1,
    )
    return bt


# ---------------------------------------------------------------------------
# _render_rich: basic smoke tests
# ---------------------------------------------------------------------------


class TestRenderRich:
    def test_runs_without_error_on_simple_tree(self, capsys):
        bt = _simple_tree()
        bt._render_rich()  # must not raise

    def test_runs_without_error_on_empty_tree(self, capsys):
        bt = BranchingTree()
        bt._render_rich()  # must not raise

    def test_runs_on_status_tree(self, capsys):
        bt = _status_tree()
        bt._render_rich()  # must not raise

    def test_renders_node_id_in_output(self, capsys):
        bt = _simple_tree()
        bt._render_rich()
        captured = capsys.readouterr()
        # node #1 should appear somewhere in the rich output
        assert "1" in captured.out

    def test_renders_step_in_output(self, capsys):
        bt = _simple_tree()
        bt._render_rich()
        captured = capsys.readouterr()
        assert "step=0" in captured.out

    def test_renders_lower_bound_in_output(self, capsys):
        bt = _simple_tree()
        bt._render_rich()
        captured = capsys.readouterr()
        assert "lb=" in captured.out

    def test_renders_estimate_in_output(self, capsys):
        bt = _simple_tree()
        bt._render_rich()
        captured = capsys.readouterr()
        assert "est=" in captured.out

    def test_renders_obj_in_output(self, capsys):
        bt = _status_tree()
        bt._render_rich()
        captured = capsys.readouterr()
        assert "obj=" in captured.out

    def test_renders_status_in_output(self, capsys):
        bt = _status_tree()
        bt._render_rich()
        captured = capsys.readouterr()
        assert "infeasible" in captured.out
        assert "feasible" in captured.out


# ---------------------------------------------------------------------------
# _render_rich: fallback when rich is unavailable
# ---------------------------------------------------------------------------


class TestRenderRichFallback:
    def test_falls_back_to_network_text_when_rich_missing(self, capsys):
        """If rich is not importable, _render_rich must fall back gracefully."""
        import builtins

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("rich not available")
            return real_import(name, *args, **kwargs)

        bt = _simple_tree()
        with mock.patch("builtins.__import__", side_effect=_mock_import):
            bt._render_rich()  # must not raise


# ---------------------------------------------------------------------------
# _render_rich: root detection fallback
# ---------------------------------------------------------------------------


class TestRenderRichRootDetection:
    def test_no_root_key_picks_root_by_in_degree(self, capsys):
        """When tree.graph has no 'root' key, the node with in-degree 0 is used."""
        bt = _make_tree(
            nodes=[
                (10, {"visited": True, "step": 5}),
                (11, {"visited": False}),
            ],
            edges=[(10, 11)],
            root=None,  # no root key
        )
        bt._render_rich()  # must not raise
        captured = capsys.readouterr()
        assert "10" in captured.out

    def test_single_node_no_root_key(self, capsys):
        bt = _make_tree(nodes=[(42, {"visited": True})], root=None)
        bt._render_rich()  # must not raise


# ---------------------------------------------------------------------------
# render() dispatch
# ---------------------------------------------------------------------------


class TestRenderDispatch:
    def test_render_rich_mode_calls_render_rich(self, capsys):
        bt = _simple_tree()
        bt.render("rich")
        captured = capsys.readouterr()
        assert "1" in captured.out

    def test_render_ansi_mode(self, capsys):
        bt = _simple_tree()
        bt.render("ansi")  # must not raise

    def test_render_case_insensitive(self, capsys):
        bt = _simple_tree()
        bt.render("RICH")  # must not raise


# ---------------------------------------------------------------------------
# _get_node_groups
# ---------------------------------------------------------------------------


class TestGetNodeGroups:
    def test_visited_node_grouped_as_visited(self):
        bt = _make_tree(nodes=[(1, {"visited": True})], root=1)
        groups = bt._get_node_groups()
        assert 1 in groups["visited"]

    def test_unvisited_node_grouped_as_unvisited(self):
        bt = _make_tree(nodes=[(1, {"visited": False})], root=1)
        groups = bt._get_node_groups()
        assert 1 in groups["unvisited"]

    def test_status_overrides_visited(self):
        bt = _make_tree(nodes=[(1, {"visited": True, "status": "infeasible"})], root=1)
        groups = bt._get_node_groups()
        assert 1 in groups["infeasible"]
        assert 1 not in groups.get("visited", [])
