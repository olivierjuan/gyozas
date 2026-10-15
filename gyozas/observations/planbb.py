"""PlanB&B's 43-variable-feature observation (Strang et al., AAAI 2026).

Faithful port of ``MuZeroNodeBipartite`` from the official repository
(github.com/abfariah/planbb): the standard bipartite-LP observation (19
variable features) plus 24 extra per-variable features broadcast from
solver statistics and focus-node attributes.

The 24 features are the RetroBranching 43-variable observation with two
deviations, both from the official code (not choices of this port):

- ``is_best_sibling_none`` / ``is_best_sibling_best_node`` are replaced by a
  single ``is_best_sibling`` flag (1 when the focus node has siblings — or is
  the root — 0 otherwise), and
- a constant ``branchability`` slot (always 1 at extraction) closes the
  block; the *predicted* branchability target is produced by the collector
  from the search tree, never from the observation.

The initial dual and primal bounds are captured at the first extraction,
exactly like the official code (and the retro observation it derives from).
"""

import numpy as np
from numpy.typing import NDArray
from pyscipopt import Model

from .node_bipartite_ecole import NodeBipartiteEcole
from .structs import BipartiteGraph

#: Number of extra per-variable features added on top of the 19 bipartite ones.
N_PLANBB_FEATURES = 24

#: The names of the 24 extra features, in the official order.
PLANBB_FEATURE_NAMES = (
    "dual_bound_frac_change",
    "primal_bound_frac_change",
    "max_primal_bound_frac_change",
    "max_dual_bound_frac_change",
    "curr_primal_dual_bound_gap_frac",
    "num_leaves_frac",
    "num_feasible_leaves_frac",
    "num_infeasible_leaves_frac",
    "num_lp_iterations_frac",
    "num_siblings_frac",
    "is_curr_node_best",
    "is_curr_node_parent_best",
    "curr_node_depth",
    "curr_node_lower_bound_relative_to_init_dual_bound",
    "curr_node_lower_bound_relative_to_curr_dual_bound",
    "branching_changes_frac",
    "constraint_prop_changes_frac",
    "prop_changes_frac",
    "parent_branching_changes_frac",
    "best_sibling_lower_bound_relative_to_init_dual_bound",
    "best_sibling_lower_bound_relative_to_curr_dual_bound",
    "best_sibling_lower_bound_relative_to_curr_node_lower_bound",
    "is_best_sibling",
    "branchability",
)


def _planbb_features(model: Model, init_dual_bound: float, init_primal_bound: float) -> NDArray[np.float64]:
    """The 24 extra features, each broadcast to every variable."""
    dual_bound_frac_change = abs(init_dual_bound - model.getDualbound()) / init_dual_bound
    primal_bound_frac_change = abs(init_primal_bound - model.getPrimalbound()) / init_primal_bound

    primal_dual_gap = abs(model.getPrimalbound() - model.getDualbound())
    max_dual_bound_frac_change = primal_dual_gap / init_dual_bound
    max_primal_bound_frac_change = primal_dual_gap / init_primal_bound

    curr_primal_dual_bound_gap_frac = model.getGap()

    # Global tree features.
    num_leaves_frac = model.getNLeaves() / model.getNNodes()
    num_feasible_leaves_frac = model.getNFeasibleLeaves() / model.getNNodes()
    num_infeasible_leaves_frac = model.getNInfeasibleLeaves() / model.getNNodes()
    num_lp_iterations_frac = model.getNNodes() / model.getNLPIterations()

    # Focus node features.
    num_siblings_frac = model.getNSiblings() / model.getNNodes()
    curr_node = model.getCurrentNode()
    best_node = model.getBestNode()
    if best_node is not None:
        is_curr_node_best = float(curr_node.getNumber() == best_node.getNumber())
    else:
        # No best node found yet.
        is_curr_node_best = 0
    parent_node = curr_node.getParent()
    if parent_node is not None and best_node is not None:
        is_curr_node_parent_best = float(parent_node.getNumber() == best_node.getNumber())
    else:
        is_curr_node_parent_best = 0
    curr_node_depth = model.getDepth() / model.getNNodes()
    curr_node_lower_bound_relative_to_init_dual_bound = init_dual_bound / curr_node.getLowerbound()
    curr_node_lower_bound_relative_to_curr_dual_bound = model.getDualbound() / curr_node.getLowerbound()
    num_branching_changes, num_constraint_prop_changes, num_prop_changes = curr_node.getNDomchg()
    total_num_changes = num_branching_changes + num_constraint_prop_changes + num_prop_changes
    try:
        branching_changes_frac = num_branching_changes / total_num_changes
    except ZeroDivisionError:
        branching_changes_frac = 0
    try:
        constraint_prop_changes_frac = num_constraint_prop_changes / total_num_changes
    except ZeroDivisionError:
        constraint_prop_changes_frac = 0
    try:
        prop_changes_frac = num_prop_changes / total_num_changes
    except ZeroDivisionError:
        prop_changes_frac = 0
    parent_branching_changes_frac = curr_node.getNParentBranchings() / model.getNNodes()
    n_siblings = model.getNSiblings()
    best_sibling = model.getBestSibling()
    if n_siblings > 0:
        is_best_sibling = 1
    elif curr_node.getParent() is None:
        is_best_sibling = 1
    else:
        is_best_sibling = 0
    if best_sibling is not None:
        best_sibling_lower_bound_relative_to_init_dual_bound = init_dual_bound / best_sibling.getLowerbound()
        best_sibling_lower_bound_relative_to_curr_dual_bound = model.getDualbound() / best_sibling.getLowerbound()
        best_sibling_lower_bound_relative_to_curr_node_lower_bound = (
            best_sibling.getLowerbound() / curr_node.getLowerbound()
        )
    else:
        best_sibling_lower_bound_relative_to_init_dual_bound = 0
        best_sibling_lower_bound_relative_to_curr_dual_bound = 0
        best_sibling_lower_bound_relative_to_curr_node_lower_bound = 0
    branchability = 1

    return np.array(
        [
            dual_bound_frac_change,
            primal_bound_frac_change,
            max_primal_bound_frac_change,
            max_dual_bound_frac_change,
            curr_primal_dual_bound_gap_frac,
            num_leaves_frac,
            num_feasible_leaves_frac,
            num_infeasible_leaves_frac,
            num_lp_iterations_frac,
            num_siblings_frac,
            is_curr_node_best,
            is_curr_node_parent_best,
            curr_node_depth,
            curr_node_lower_bound_relative_to_init_dual_bound,
            curr_node_lower_bound_relative_to_curr_dual_bound,
            branching_changes_frac,
            constraint_prop_changes_frac,
            prop_changes_frac,
            parent_branching_changes_frac,
            best_sibling_lower_bound_relative_to_init_dual_bound,
            best_sibling_lower_bound_relative_to_curr_dual_bound,
            best_sibling_lower_bound_relative_to_curr_node_lower_bound,
            is_best_sibling,
            branchability,
        ],
        dtype=np.float64,
    )


class NodeBipartitePlanBB:
    """The bipartite-LP observation with PlanB&B's 24 extra variable features.

    Wraps a :class:`~gyozas.observations.NodeBipartiteEcole` base (the 19
    Gasse variable features) and appends the 24 features of the official
    ``MuZeroNodeBipartite`` observation to every variable column, so each
    variable carries 43 features.
    """

    def __init__(self, base: NodeBipartiteEcole | None = None) -> None:
        self.base = base or NodeBipartiteEcole()
        self._init_dual_bound: float | None = None
        self._init_primal_bound: float | None = None

    def reset(self, model: Model) -> None:
        self.base.reset(model)
        self._init_dual_bound = None
        self._init_primal_bound = None

    def extract(self, model: Model, done: bool) -> BipartiteGraph | None:
        obs = self.base.extract(model, done)
        if obs is None:
            return None
        if self._init_dual_bound is None:
            # Not yet started: capture the initial bounds for the frac-change
            # features (the official code does this on its first extract).
            self._init_dual_bound = model.getDualbound()
            self._init_primal_bound = model.getPrimalbound()
        feats = _planbb_features(model, self._init_dual_bound, self._init_primal_bound)
        obs.variable_features = np.column_stack(
            (obs.variable_features, np.broadcast_to(feats, (obs.variable_features.shape[0], len(feats))))
        )
        return obs
