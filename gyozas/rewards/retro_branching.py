"""RetroBranching's ``retro_binary_fathomed`` reward (Parsonson & Deisenroth 2022).

Faithful port of the official repository's reward machinery
(github.com/ds4dm/retro_branching): :class:`RetroSearchTree` (the official
``SearchTree``) tracks only the nodes the brancher actually visited plus the
currently open nodes — nodes pruned, infeasible or cut off by bounds are never
added, which is what makes the binary "fathomed" score well-defined. The
:class:`RetroBinaryFathomed` reward scores each branching decision
retrospectively (-1 if the node has successors in the tree, 0 if it was
closed) and, at the end of the episode, decomposes the B&B tree into
sub-tree episodes (:meth:`RetroBinaryFathomed.subtree_episodes`) that
:meth:`subtree_transitions` then turns into DQN transitions.

Deviations from the official code, both forced by stock PySCIPOpt:
``getOpenNodes`` raises once the problem is solved, so the tree is not
updated on the final (``done=True``) extraction — at optimal termination the
open set is empty anyway, so only time-limited episodes differ (their last
decision is scored 0 instead of -1). And reward extraction in this harness
always happens with the SCIP thread parked in the branch-and-bound callback,
which is exactly the stage the official queries assume.

Like the official implementation, the tracker assumes SCIP node numbers are
stable: a SCIP *restart* renumbers the search tree, which orphans the
experiences taken before it. The official experiments (and this benchmark)
run under the ``gasse_2019`` parameters, which disable restarts.
"""

from typing import Any

import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from pyscipopt import Model


class RetroSearchTree:
    """The official ``SearchTree``: visited + open nodes, with parent edges.

    Nodes pruned, infeasible, or cut off by the search's bounds are never
    stored by SCIP and therefore never appear here. Node attributes:
    ``lower_bound``, ``step_visited`` (set on visited nodes), and later
    ``score`` / ``closed_by_agent`` (set by :class:`RetroBinaryFathomed`).
    """

    def __init__(self, model: Model) -> None:
        self.tree = nx.DiGraph()
        self.tree.graph["root_node"] = None
        self.tree.graph["visited_node_ids"] = []  # ordered, unique
        self.tree.graph["optimum_node_ids"] = []
        self.tree.graph["fathomed_node_ids"] = []
        self.step_idx = 0
        if model.getCurrentNode() is not None:
            self._incumbent_primal_bound = model.getPrimalbound()
            self.update_tree(model)

    def update_tree(self, model: Model) -> None:
        """Record the current focus node and the currently open nodes."""
        curr_node = model.getCurrentNode()
        curr_node_id = curr_node.getNumber() if curr_node is not None else None
        visited = self.tree.graph["visited_node_ids"]

        if len(visited) >= 1:
            # Branching at the previous node may have found a new incumbent.
            if model.getPrimalbound() < self._incumbent_primal_bound:
                self.tree.graph["optimum_node_ids"].append(visited[-1])
                self._incumbent_primal_bound = model.getPrimalbound()

        if curr_node_id is not None and curr_node_id not in visited:
            self._add_node(curr_node_id, curr_node)
            visited.append(curr_node_id)
            self.tree.nodes[curr_node_id]["step_visited"] = self.step_idx

        # Add open nodes (leaves, children, siblings) with their parent edges.
        open_leaves, open_children, open_siblings = model.getOpenNodes()
        for node in (*open_leaves, *open_children, *open_siblings):
            self._add_node(node.getNumber(), node)

        if len(visited) > 2 or curr_node_id is None:
            # Track nodes the brancher closed: if the node visited before the
            # current one has no successor in the tree, it was fathomed.
            if curr_node_id is not None:
                prev_node_id = visited[-2]
            else:
                prev_node_id = visited[-1]
            if prev_node_id is not None and len(list(self.tree.successors(prev_node_id))) == 0:
                if prev_node_id != curr_node_id:
                    self.tree.graph["fathomed_node_ids"].append(prev_node_id)

        self.step_idx += 1

    def _add_node(self, node_id: int, node: Any) -> None:
        """Add a node (and its parent edge) if not already in the tree."""
        if node_id in self.tree:
            return
        self.tree.add_node(node_id, lower_bound=node.getLowerbound())
        parent_node = node.getParent()
        if parent_node is not None:
            self.tree.add_edge(parent_node.getNumber(), node_id)
        else:
            # Root node, has no parent.
            self.tree.graph["root_node"] = node_id


class RetroBinaryFathomed:
    """The official ``retro_binary_fathomed`` reward as a harness reward function.

    Per step, returns the binary score of the *previous* branching decision:
    -1 when the previous node has successors in the search tree (the brancher
    kept the sub-tree open), 0 when it was closed by the brancher. At the end
    of the episode, ``subtree_episodes`` holds the retrospective sub-tree
    decomposition: a list of ``{step_idx: reward}`` maps, one per sub-tree
    episode, built with the official ``max_leaf_lp_gain`` construction
    (``min_subtree_depth=1``, ``force_include_branching_fathomed_transitions``
    enabled — the ``retro.yaml`` configuration).
    """

    def __init__(
        self,
        min_subtree_depth: int = 1,
        retro_trajectory_construction: str = "max_leaf_lp_gain",
        force_include_branching_fathomed_transitions: bool = True,
    ) -> None:
        self.min_subtree_depth = min_subtree_depth
        self.retro_trajectory_construction = retro_trajectory_construction
        self.force_include_branching_fathomed_transitions = force_include_branching_fathomed_transitions
        self.search_tree: RetroSearchTree | None = None
        self.subtree_episodes: list[dict[int, float]] | None = None
        self._prev_node_id: int | None = None

    def reset(self, model: Model) -> None:
        self.search_tree = None
        self.subtree_episodes = None
        self._prev_node_id = None

    def extract(self, model: Model, done: bool) -> float:
        if self._prev_node_id is None:
            # First extraction: create the tree tracker and record the root.
            self.search_tree = RetroSearchTree(model)
            if self.search_tree.tree.graph["visited_node_ids"]:
                # The root is the first visited node (avoid re-querying the
                # model: on a pre-solved instance that prints a SCIP error).
                self._prev_node_id = self.search_tree.tree.graph["visited_node_ids"][0]
            if done:
                # Pre-solved at reset: no branching decision was ever made.
                self.subtree_episodes = self._build_subtree_episodes()
            return 0.0

        assert self.search_tree is not None
        if not done:
            # Record the current node and the open nodes, then score the
            # previous decision: -1 if its node has successors, else 0.
            self.search_tree.update_tree(model)
            score = self._score_prev_node()
            curr_node = model.getCurrentNode()
            if curr_node is not None:
                self._prev_node_id = curr_node.getNumber()
            return score

        # Final extraction: the open set is no longer queryable (the problem
        # is solved), so the tree is not updated and the last decision node,
        # whose children were never added, is scored 0 — closed by the brancher.
        score = self._score_prev_node()
        self.subtree_episodes = self._build_subtree_episodes()
        return score

    def _score_prev_node(self) -> float:
        assert self.search_tree is not None
        tree = self.search_tree.tree
        prev_node_id = self._prev_node_id
        if len(list(tree.successors(prev_node_id))) > 0:
            # Previous branching decision kept the sub-tree open.
            score = -1.0
        else:
            # All children were pruned, infeasible, or outside bounds.
            score = 0.0
        tree.nodes[prev_node_id]["score"] = score
        tree.nodes[prev_node_id]["closed_by_agent"] = score == 0.0
        return score

    def _build_subtree_episodes(self) -> list[dict[int, float]]:
        """The retrospective sub-tree decomposition, as in ``RetroBranching.extract``."""
        assert self.search_tree is not None
        tree = self.search_tree.tree
        if tree.graph["root_node"] is None:
            # Instance was pre-solved.
            return [{0: 0}]

        # Remove nodes which were never visited and therefore have no score
        # (SCIP sometimes reports a stray node id when the episode finishes).
        for node in list(tree.nodes):
            if "score" not in tree.nodes[node]:
                tree.remove_node(node)
                if node in tree.graph["visited_node_ids"]:
                    tree.graph["visited_node_ids"].remove(node)

        # Map each visited node to its position in the visitation order —
        # the index of the corresponding experience in the episode.
        visited_nodes_to_step_idx = {
            node: idx for idx, node in enumerate(tree.graph["visited_node_ids"])
        }

        subtrees: list[dict[int, float]] = []
        nodes_added: set[int] = set()
        root_node = tree.graph["root_node"]

        while True:
            # Build DFS sub-trees from the nodes not yet covered by an episode.
            nx_subtrees = []
            if len(nodes_added) > 0:
                for node in nodes_added:
                    for child in tree.successors(node):
                        if child not in nodes_added:
                            nx_subtrees.append(dfs_tree(tree, child))
            else:
                # No episode yet: the whole B&B tree is the first sub-tree.
                nx_subtrees.append(dfs_tree(tree, root_node))

            for subtree in nx_subtrees:
                # networkx dfs_tree does not carry node attributes over.
                for node in subtree.nodes:
                    subtree.nodes[node]["score"] = tree.nodes[node]["score"]
                    subtree.nodes[node]["lower_bound"] = tree.nodes[node]["lower_bound"]

                path = self._select_path_in_subtree(subtree)
                if len(path) > 0:
                    episode = self._conv_path_to_step_idx_reward_map(
                        path, tree, visited_nodes_to_step_idx, nodes_added
                    )
                    if episode is not None:
                        subtrees.append(episode)
                else:
                    # No valid path: do not consider these nodes again.
                    nodes_added.update(subtree.nodes)

            if len(nx_subtrees) == 0:
                # All sub-trees added.
                break

        if len(subtrees) == 0:
            # Solved at the root: no sub-tree was ever deep enough.
            return [{0: 0}]
        return subtrees

    def _select_path_in_subtree(self, subtree: nx.DiGraph) -> list[int]:
        """Choose a root-to-leaf path through the sub-tree (max_leaf_lp_gain)."""
        for root_node in subtree.nodes:
            if subtree.in_degree(root_node) == 0:
                break

        leaf_nodes = [node for node in subtree.nodes if subtree.out_degree(node) == 0]
        if len(leaf_nodes) == 0:
            return []

        if self.retro_trajectory_construction == "max_leaf_lp_gain":
            # Leaf with the greatest lower bound.
            lp_gains = [subtree.nodes[leaf]["lower_bound"] for leaf in leaf_nodes]
            final_node = leaf_nodes[lp_gains.index(max(lp_gains))]
        else:
            raise ValueError(
                f"Unrecognised retro_trajectory_construction {self.retro_trajectory_construction}"
            )

        assert self.search_tree is not None
        return shortest_path(self.search_tree.tree, source=root_node, target=final_node)

    def _conv_path_to_step_idx_reward_map(
        self,
        path: list[int],
        tree: nx.DiGraph,
        visited_nodes_to_step_idx: dict[int, int],
        nodes_added: set[int],
    ) -> dict[int, float] | None:
        """Register the path nodes and map each to its episode step and score."""
        for node in path:
            nodes_added.add(node)

        if len(path) < self.min_subtree_depth:
            if self.force_include_branching_fathomed_transitions:
                if tree.nodes[path[-1]]["score"] == 0:
                    # The brancher fathomed this sub-tree: include it anyway.
                    pass
                else:
                    return None
            else:
                return None

        rewards = [tree.nodes[node]["score"] for node in path]
        step_indices = [visited_nodes_to_step_idx[node] for node in path]
        return {step_idx: reward for step_idx, reward in zip(step_indices, rewards)}


def subtree_transitions(
    experiences: list[dict[str, Any]], subtree_episodes: list[dict[int, float]]
) -> list[list[dict[str, Any]]]:
    """Assemble DQN transitions from sub-tree episodes.

    Port of the official ``process_episodes_into_subtree_episodes`` (no
    intrinsic reward, no backtracking). ``experiences`` is the episode's list
    of experience dicts — one per branching decision, in visitation order —
    with at least the keys ``prev_state`` (the state acted on), ``state``
    (the resulting next state) and ``done``. For each sub-tree episode:

    - the per-transition reward is the sub-tree reward at that step index;
    - the next state of a transition is the following path node's state
      (official ``CHANGE``);
    - the final transition is terminal (official ``CHANGE``);
    - the final transition is only kept when its reward differs from the
      per-step reward (official ``CHANGE #2``).

    Unlike the official code, which mutates the shared experience dicts, the
    returned transitions are shallow copies — sub-tree episodes are disjoint,
    so the result is identical.
    """
    episodes: list[list[dict[str, Any]]] = []
    for subtree_episode in subtree_episodes:
        experiences_sub = [dict(experiences[i]) for i in subtree_episode]
        for idx, reward in enumerate(subtree_episode.values()):
            experiences_sub[idx]["reward"] = reward
            if idx < len(experiences_sub) - 1:
                # Set next state as the state the following path node acted on.
                experiences_sub[idx]["state"] = experiences_sub[idx + 1]["prev_state"]  # CHANGE
        if len(experiences_sub) > 0:
            experiences_sub[-1]["done"] = True  # CHANGE: final sub-tree step is terminal

            # CHANGE #2: only add the final transition if its reward differs
            # from the per-step reward of the rest of the sub-tree episode.
            if len(experiences_sub) > 1:
                per_step_reward = experiences_sub[0]["reward"]
            else:
                per_step_reward = None
            if experiences_sub[-1]["reward"] != per_step_reward:
                episodes.append(experiences_sub)
            else:
                episodes.append(experiences_sub[:-1])
    return episodes
