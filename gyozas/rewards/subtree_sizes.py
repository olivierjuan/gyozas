"""The tree-MDP credit assignment of Scavuzzo et al. 2022 (rl2branch).

Port of the official ``TreeRecorder`` (``agent.py`` of
github.com/lascavana/rl2branch): during an episode every branching decision's
focus node is recorded with its parent and depth; once the instance is solved,
the size of the sub-tree each decision created — its children and all their
descendants — is accumulated bottom-up over depth groups. The tree-MDP return
of a decision is ``-(subtree_size) - 1``: a decision whose sub-tree closes
immediately scores -1, every extra node in the sub-tree costs one more.

Two protocol facts the porter must know (both mirror the official code):

- The tree is only well-defined because the official training regime completes
  it: under ``tmdp+ObjLim`` the instance is solved with an objective limit set
  to its optimal value (+0.1 for a minimisation problem), so every node worse
  than the optimum is cut when created; under ``tmdp+DFS`` node selection is
  forced depth-first. Setting the objective limit is the caller's job — it is
  a property of the training instance, not of the reward.
- A SCIP restart renumbers the search tree and orphans the recording, so the
  training solver must disable restarts (the paper's ``gasse_2019`` parameters
  do).
"""

from typing import Any

from pyscipopt import Model


class SubtreeSizes:
    """Record the branching tree; at done, expose each decision's sub-tree size.

    Per step, ``extract`` records the current focus node (one per branching
    decision, in visitation order — ``node_ids``). When the episode finishes,
    ``subtree_sizes`` maps every recorded node number to the number of nodes in
    the sub-tree it created; the per-decision return is
    ``-(subtree_sizes[node]) - 1``. The step-wise reward itself is always 0:
    the official environment's ``NNodes`` reward is only consumed by the
    paper's ``mdp`` ablation, not by the tree-MDP variants.
    """

    def __init__(self, children_per_branch: int = 2) -> None:
        # lp_cand=True in the official recorder: a branching decision creates
        # two children.
        self.children_per_branch = children_per_branch
        self.reset(None)

    def reset(self, model: Model) -> None:
        self._tree: dict[int, dict[str, Any]] = {}
        self._depth_groups: list[list[int]] = []
        self.node_ids: list[int] = []  # visitation order, one entry per decision
        self.subtree_sizes: dict[int, int] | None = None

    def extract(self, model: Model, done: bool) -> float:
        if not done:
            # On a finished (or pre-solved) instance the current node is not
            # queryable — and there is no decision left to record anyway.
            node = model.getCurrentNode()
            if node is not None:
                self._record(node)
        else:
            self.subtree_sizes = self._calculate_subtree_sizes()
        return 0.0

    def _record(self, node: Any) -> None:
        node_id = node.getNumber()
        parent = node.getParent()
        self._tree[node_id] = {
            "parent": parent.getNumber() if parent is not None else None,
            "lowerbound": node.getLowerbound(),
        }
        self.node_ids.append(node_id)
        depth = node.getDepth()
        while len(self._depth_groups) <= depth:
            self._depth_groups.append([])
        self._depth_groups[depth].append(node_id)

    def _calculate_subtree_sizes(self) -> dict[int, int]:
        """Sum sub-tree sizes bottom-up, deepest depth group first."""
        subtree_sizes = {node_id: 0 for node_id in self._tree}
        for group in reversed(self._depth_groups):
            for node_id in group:
                subtree_sizes[node_id] += self.children_per_branch
                parent_id = self._tree[node_id]["parent"]
                if parent_id is not None:
                    subtree_sizes[parent_id] += subtree_sizes[node_id]
        return subtree_sizes
