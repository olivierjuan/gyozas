"""The solver-statistics branching state of Lin 2022 (T-BranT).

The structured solver-stats view of R1.2(b): three aligned feature blocks
describing the current branching decision — a 25-dim vector per LP branching
candidate, an 8-dim vector for the current node, and a 53-dim vector for the
whole search tree (the MIP state). TGPPO 2025 reuses the same triple.

Ported from the official repository (``linjc16/TBranT``), whose authors
expose the state through a *patched* PySCIPOpt with Cython-level access to
SCIP internals. This module reimplements the triple on stock PySCIPOpt
(SCIP 10): every feature slot of the original layout is present, but the
statistics SCIP does not expose through its Python API are exact zeros.
Per-slot availability is recorded in the ``*_FEATURES`` tables below (and
asserted by the tests) so downstream method ports know exactly which inputs
carry signal:

- candidates: 3 of 25 (LP fractionality, average solution value, and a proxy
  of the average-normalized pseudocost score — the normalizer
  ``SCIPgetAvgPseudocostScore`` is not exposed, so the candidate-set mean
  stands in);
- node: 8 of 8;
- tree: 30 of 53 (the missing ones are the conflict/inference/cutoff score
  family and the run-internal counters of ``SCIPstat``).

Returns ``None`` outside the solving stage or when the episode is done.
"""

from dataclasses import dataclass

import numpy as np
from pyscipopt import Model
from pyscipopt.scip import PY_SCIP_STAGE

from gyozas._utils import is_fixed_domain

# --------------------------------------------------------------------------
# The feature layouts, with per-slot availability under stock PySCIPOpt.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """One feature slot: its name and whether stock PySCIPOpt can compute it."""

    name: str
    available: bool = True
    note: str = ""


CANDIDATE_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("lp_fractionality"),
    FeatureSpec("avg_solution_value"),
    FeatureSpec("branch_depth_up", available=False, note="SCIPvarGetAvgBranchdepthCurrentRun"),
    FeatureSpec("branch_depth_down", available=False, note="SCIPvarGetAvgBranchdepthCurrentRun"),
    FeatureSpec("conflict_score", available=False, note="SCIPgetVarConflictScore"),
    FeatureSpec("conflictlength_score", available=False, note="SCIPgetVarConflictlengthScore"),
    FeatureSpec("inference_score", available=False, note="SCIPgetVarAvgInferenceScore"),
    FeatureSpec("cutoff_score", available=False, note="SCIPgetVarAvgCutoffScore"),
    FeatureSpec(
        "pseudocost_score",
        note="proxy: SCIPgetAvgPseudocostScore not exposed — candidate-set mean normalizes",
    ),
    FeatureSpec("pseudocost_count_ratio_up", available=False, note="SCIPgetVarPseudocostCountCurrentRun"),
    FeatureSpec("pseudocost_count_ratio_down", available=False, note="SCIPgetVarPseudocostCountCurrentRun"),
    FeatureSpec("pseudocost_per_branching_up", available=False, note="SCIPgetVarPseudocostCountCurrentRun"),
    FeatureSpec("pseudocost_per_branching_down", available=False, note="SCIPgetVarPseudocostCountCurrentRun"),
    FeatureSpec("pseudocost_count_per_branch_up", available=False, note="SCIPgetVarPseudocostCountCurrentRun"),
    FeatureSpec("pseudocost_count_per_branch_down", available=False, note="SCIPgetVarPseudocostCountCurrentRun"),
    FeatureSpec("n_implications_down", available=False, note="SCIPvarGetNImpls"),
    FeatureSpec("n_implications_up", available=False, note="SCIPvarGetNImpls"),
    FeatureSpec("clique_ratio_down", available=False, note="SCIPvarGetNCliques"),
    FeatureSpec("clique_ratio_up", available=False, note="SCIPvarGetNCliques"),
    FeatureSpec("avg_cutoffs_current_run_up", available=False, note="SCIPgetVarAvgCutoffsCurrentRun"),
    FeatureSpec("avg_cutoffs_current_run_down", available=False, note="SCIPgetVarAvgCutoffsCurrentRun"),
    FeatureSpec("avg_conflictlength_current_run_up", available=False, note="SCIPgetVarAvgConflictlengthCurrentRun"),
    FeatureSpec("avg_conflictlength_current_run_down", available=False, note="SCIPgetVarAvgConflictlengthCurrentRun"),
    FeatureSpec("avg_inferences_current_run_up", available=False, note="SCIPgetVarAvgInferencesCurrentRun"),
    FeatureSpec("avg_inferences_current_run_down", available=False, note="SCIPgetVarAvgInferencesCurrentRun"),
)

NODE_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("relative_depth"),
    FeatureSpec("relative_plunge_depth"),
    FeatureSpec("lp_obj_vs_dual_bound"),
    FeatureSpec("lp_obj_vs_root_dual_bound"),
    FeatureSpec("lp_obj_vs_primal_bound"),
    FeatureSpec("bound_position"),
    FeatureSpec("candidate_fraction"),
    FeatureSpec("domain_change_fraction"),
)

MIP_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("objlim_leaf_fraction"),
    FeatureSpec("infeasible_leaf_fraction"),
    FeatureSpec("feasible_leaf_fraction"),
    FeatureSpec("dynamic_factor"),
    FeatureSpec("nodes_left_ratio"),
    FeatureSpec("leaf_ratio"),
    FeatureSpec("internal_node_ratio", available=False, note="SCIPstat.ninternalnodes"),
    FeatureSpec("processed_created_ratio", available=False, note="SCIPstat.ncreatednodes"),
    FeatureSpec("activated_node_ratio", available=False, note="SCIPstat.nactivatednodes"),
    FeatureSpec("deactivated_node_ratio", available=False, note="SCIPstat.ndeactivatednodes"),
    FeatureSpec("plunge_depth_ratio"),
    FeatureSpec("backtrack_ratio"),
    FeatureSpec("log_lp_iterations_per_node"),
    FeatureSpec("log_lps_per_node"),
    FeatureSpec("nodes_per_lp"),
    FeatureSpec("node_lp_fraction", available=False, note="SCIPgetNNodeLPs"),
    FeatureSpec("log_primal_dual_integral", available=False, note="SCIPstat.primaldualintegral"),
    FeatureSpec("gap_vs_last_sol_gap", available=False, note="SCIPstat.lastsolgap"),
    FeatureSpec("gap_vs_first_sol_gap", available=False, note="SCIPstat.firstsolgap"),
    FeatureSpec("last_first_sol_gap_ratio", available=False, note="SCIPstat.firstsolgap/lastsolgap"),
    FeatureSpec("dual_bound_gain"),
    FeatureSpec("avg_dual_bound_gain"),
    FeatureSpec("primal_dual_gap"),
    FeatureSpec("primal_bound_is_sol", available=False, note="SCIPisPrimalboundSol"),
    FeatureSpec("nodes_before_first_sol", available=False, note="SCIPstat.nnodesbeforefirst"),
    FeatureSpec("avg_conflict_score", available=False, note="SCIPgetAvgConflictScore"),
    FeatureSpec("avg_conflictlength_score", available=False, note="SCIPgetAvgConflictlengthScore"),
    FeatureSpec("avg_inference_score", available=False, note="SCIPgetAvgInferenceScore"),
    FeatureSpec("avg_cutoff_score", available=False, note="SCIPgetAvgCutoffScore"),
    FeatureSpec("avg_pseudocost_score", available=False, note="SCIPgetAvgPseudocostScore"),
    FeatureSpec("avg_cutoffs_up", available=False, note="SCIPgetAvgCutoffs"),
    FeatureSpec("avg_cutoffs_down", available=False, note="SCIPgetAvgCutoffs"),
    FeatureSpec("avg_inferences_up", available=False, note="SCIPgetAvgInferences"),
    FeatureSpec("avg_inferences_down", available=False, note="SCIPgetAvgInferences"),
    FeatureSpec("pseudocost_variance_up", available=False, note="SCIPgetPseudocostVariance"),
    FeatureSpec("pseudocost_variance_down", available=False, note="SCIPgetPseudocostVariance"),
    FeatureSpec("conflict_constraints_applied", available=False, note="SCIPgetNConflictConssApplied"),
    FeatureSpec("min_bound_open_fraction"),
    FeatureSpec("max_bound_open_fraction"),
    FeatureSpec("dual_bound_vs_max_open"),
    FeatureSpec("open_bound_spread"),
    FeatureSpec("min_open_vs_primal"),
    FeatureSpec("max_open_vs_primal"),
    FeatureSpec("mean_open_bound_position"),
    FeatureSpec("min_open_bound_position"),
    FeatureSpec("max_open_bound_position"),
    FeatureSpec("open_bound_iqr"),
    FeatureSpec("open_bound_cv"),
    FeatureSpec("open_bound_qcd"),
    FeatureSpec("mean_open_depth_ratio"),
    FeatureSpec("open_depth_iqr"),
    FeatureSpec("open_depth_cv"),
    FeatureSpec("open_depth_qcd"),
)

N_CANDIDATE_FEATURES = len(CANDIDATE_FEATURES)
N_NODE_FEATURES = len(NODE_FEATURES)
N_MIP_FEATURES = len(MIP_FEATURES)

_EPS = 1e-10


# --------------------------------------------------------------------------
# Normalizers (faithful ports of the reference implementation's helpers).
# --------------------------------------------------------------------------


def rel_distance(x: float, y: float) -> float:
    """Relative distance between two quantities (0 when they straddle zero)."""
    if x * y < 0:
        return 0.0
    return float(abs(x - y) / max(abs(x), abs(y), _EPS))


def rel_position(node_bound: float, ub: float, lb: float, ub_infinite: bool) -> float:
    """Position of ``node_bound`` between the global bounds (0 if UB unset)."""
    if ub_infinite or ub == lb:
        return 0.0
    return float(abs(ub - node_bound) / abs(ub - lb))


# --------------------------------------------------------------------------
# The state object.
# --------------------------------------------------------------------------


@dataclass
class BranchingState:
    """The three feature blocks of one branching decision.

    ``candidate_features`` holds one row per LP branching candidate, in
    ``candidate_indices`` order — the same candidates, same order, same
    fixed-domain filter as the branching action set, so row *i* is the
    feature vector of the variable ``candidate_indices[i]`` the policy may
    branch on. ``node_features`` describes the current node and
    ``mip_features`` the whole search tree so far.
    """

    candidate_features: np.ndarray  # (n_candidates, 25)
    candidate_indices: np.ndarray  # (n_candidates,) LP column positions
    node_features: np.ndarray  # (8,)
    mip_features: np.ndarray  # (53,)


class BranchingStateObservation:
    """Observation function extracting :class:`BranchingState` (R1.2b).

    The solver-statistics view used by the T-BranT and TGPPO regimes: no LP
    graph, just the three feature blocks of Lin 2022 — per-candidate solver
    statistics, current-node features, and tree-level features. Slots that
    stock PySCIPOpt cannot compute are exact zeros (see the ``*_FEATURES``
    tables for the per-slot availability).
    """

    def reset(self, _model: Model) -> None:
        pass

    def extract(self, model: Model, done: bool) -> BranchingState | None:
        if done or model.getStage() != PY_SCIP_STAGE.SOLVING:
            return None

        cand_vars, cand_vals, cand_fracs, n_cands, _n_prio, _n_impl = model.getLPBranchCands()
        cand_vars, cand_vals, cand_fracs = cand_vars[:n_cands], cand_vals[:n_cands], cand_fracs[:n_cands]
        # Same filter as the branching dynamics' action set — rows stay
        # aligned with the variables the policy may actually branch on.
        keep = [i for i, var in enumerate(cand_vars) if not is_fixed_domain(var)]
        cand_vars = [cand_vars[i] for i in keep]
        cand_vals = [cand_vals[i] for i in keep]
        cand_fracs = [cand_fracs[i] for i in keep]

        return BranchingState(
            candidate_features=self._candidate_features(model, cand_vars, cand_vals, cand_fracs),
            candidate_indices=np.array([var.getCol().getLPPos() for var in cand_vars], dtype=np.int64),
            node_features=self._node_features(model, cand_vars),
            mip_features=self._mip_features(model),
        )

    # ------------------------------------------------------------------
    # Feature blocks
    # ------------------------------------------------------------------

    def _candidate_features(self, model: Model, cand_vars, cand_vals, cand_fracs) -> np.ndarray:
        features = np.zeros((len(cand_vars), N_CANDIDATE_FEATURES))
        # The reference normalizes each variable's pseudocost score by the
        # run-wide average score; stock PySCIPOpt exposes neither that average
        # nor the other four score families. We fill slot 8 with the same
        # normalization formula applied against the candidate-set mean (a
        # documented proxy), and leave the rest zero.
        scores = [
            model.getVarPseudocostScore(var, val) for var, val in zip(cand_vars, cand_vals, strict=True)
        ]
        avg = max(float(np.mean(scores)), 0.1) if scores else 0.1
        for i, (var, _val, frac) in enumerate(zip(cand_vars, cand_vals, cand_fracs, strict=True)):
            features[i, 0] = frac  # LP fractionality
            features[i, 1] = var.getAvgSol()  # across the feasible primal solutions found
            features[i, 8] = 1.0 - 1.0 / (1.0 + scores[i] / avg)
        return features

    def _node_features(self, model: Model, cand_vars) -> np.ndarray:
        node = model.getCurrentNode()
        features = np.zeros(N_NODE_FEATURES)
        if node is None:
            return features
        depth = node.getDepth()
        max_depth = model.getMaxDepth()
        lp_obj = model.getLPObjVal()
        lb = model.getLowerbound()
        ub = model.getUpperbound()
        ub_infinite = model.isInfinity(ub)
        is_root = depth == 0

        if not is_root:
            if max_depth > 0:
                features[0] = depth / max_depth
            if depth > 0:
                features[1] = model.getPlungeDepth() / depth
        features[2] = rel_distance(lb, lp_obj)
        features[3] = rel_distance(model.getLowerboundRoot(), lp_obj)
        if not ub_infinite:
            features[4] = rel_distance(ub, lp_obj)
            features[5] = rel_position(lp_obj, ub, lb, ub_infinite)
        n_discrete = model.getNBinVars() + model.getNIntVars() + model.getNImplVars()
        if n_discrete > 0:
            features[6] = len(cand_vars) / n_discrete
        n_vars = model.getNVars()
        if n_vars > 0:
            # SCIPdomchgGetNBoundchgs (branching + constraint + propagation
            # bound changes) — the stock getNDomchg tuple sums to the same
            # quantity.
            features[7] = sum(node.getNDomchg()) / n_vars
        return features

    def _mip_features(self, model: Model) -> np.ndarray:
        features = np.zeros(N_MIP_FEATURES)
        n_nodes = model.getNNodes()
        is_root = model.getDepth() == 0

        if not is_root and n_nodes > 0:
            # Nodes and leaves.
            all_leaves = (
                model.getNObjlimLeaves() + model.getNFeasibleLeaves() + model.getNInfeasibleLeaves()
            )
            if all_leaves > 0:
                features[0] = model.getNObjlimLeaves() / all_leaves
                features[1] = model.getNInfeasibleLeaves() / all_leaves
                features[2] = model.getNFeasibleLeaves() / all_leaves
            features[3] = (model.getNInfeasibleLeaves() + 1.0) / (model.getNObjlimLeaves() + 1.0)
            features[4] = model.getNNodesLeft() / n_nodes
            if all_leaves > 0:
                features[5] = all_leaves / n_nodes
            # Depth and backtracks.
            max_depth = model.getMaxDepth()
            if max_depth > 0:
                features[10] = model.getPlungeDepth() / max_depth
            features[11] = model.getNBacktracks() / n_nodes
            # LP iterations.
            n_lps = model.getNLPs()
            if model.getNLPIterations() > 0:
                features[12] = float(np.log(model.getNLPIterations() / n_nodes))
            if n_lps > 0:
                features[13] = float(np.log(n_lps / n_nodes))
                features[14] = n_nodes / n_lps

        # Bounds and gap.
        lb = model.getLowerbound()
        ub = model.getUpperbound()
        ub_infinite = model.isInfinity(ub)
        features[20] = rel_distance(model.getLowerboundRoot(), lb)
        features[21] = rel_distance(model.getLowerboundRoot(), model.getAvgLowerbound())
        if not ub_infinite:
            features[22] = rel_distance(ub, lb)

        # Open nodes (leaves + children + siblings).
        open_nodes = model.getLeaves() + model.getChildren() + model.getSiblings()
        if open_nodes:
            open_lowerbounds = np.array([node.getLowerbound() for node in open_nodes], dtype=float)
            open_depths = np.array([node.getDepth() for node in open_nodes], dtype=float)
            features[37] = float(np.mean(open_lowerbounds == open_lowerbounds.min()))
            features[38] = float(np.mean(open_lowerbounds == open_lowerbounds.max()))
            features[39] = rel_distance(lb, float(open_lowerbounds.max()))
            features[40] = rel_distance(float(open_lowerbounds.min()), float(open_lowerbounds.max()))
            if not ub_infinite:
                features[41] = rel_distance(float(open_lowerbounds.min()), ub)
                features[42] = rel_distance(float(open_lowerbounds.max()), ub)
                features[43] = rel_position(float(open_lowerbounds.mean()), ub, lb, ub_infinite)
                features[44] = rel_position(float(open_lowerbounds.min()), ub, lb, ub_infinite)
                features[45] = rel_position(float(open_lowerbounds.max()), ub, lb, ub_infinite)
            lb_q1, lb_q3 = np.quantile(open_lowerbounds, [0.25, 0.75])
            features[46] = rel_distance(float(lb_q1), float(lb_q3))
            if open_lowerbounds.mean() != 0:
                features[47] = float(open_lowerbounds.std() / open_lowerbounds.mean())
            if lb_q1 + lb_q3 != 0:
                features[48] = float((lb_q3 - lb_q1) / (lb_q3 + lb_q1))
            max_depth = model.getMaxDepth()
            if max_depth > 0:
                features[49] = float(open_depths.mean()) / max_depth
            d_q1, d_q3 = np.quantile(open_depths, [0.25, 0.75])
            features[50] = rel_distance(float(d_q1), float(d_q3))
            if open_depths.mean() != 0:
                features[51] = float(open_depths.std() / open_depths.mean())
            if d_q1 + d_q3 != 0:
                features[52] = float((d_q3 - d_q1) / (d_q3 + d_q1))
        return features
