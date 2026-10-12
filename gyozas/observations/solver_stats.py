import math

from pyscipopt import Model
from pyscipopt.scip import PY_SCIP_STAGE


def _safe(model: Model, name: str, *args) -> float:
    """Call a pyscipopt getter and fall back to NaN if unavailable at this stage."""
    try:
        value = getattr(model, name)(*args)
        return float(value)
    except Exception:
        return math.nan


def _node_stats(model: Model) -> dict[str, float]:
    node = model.getCurrentNode()
    stats: dict[str, float] = {}
    if node is None:
        return {"node_number": math.nan, "node_depth": math.nan, "node_lowerbound": math.nan}
    stats["node_number"] = float(node.getNumber())
    stats["node_depth"] = float(node.getDepth())
    stats["node_lowerbound"] = float(node.getLowerbound())
    return stats


def _tree_stats(model: Model) -> dict[str, float]:
    getters = {
        # Bounds and gap
        "primal_bound": "getPrimalbound",
        "dual_bound": "getDualbound",
        "gap": "getGap",
        "dualbound_root": "getDualboundRoot",
        "lowerbound_root": "getLowerboundRoot",
        "avg_dualbound": "getAvgDualbound",
        "avg_lowerbound": "getAvgLowerbound",
        "first_primal_bound": "getFirstPrimalBound",
        "cutoff_bound": "getCutoffbound",
        # Tree search progress
        "n_nodes": "getNNodes",
        "n_total_nodes": "getNTotalNodes",
        "n_nodes_left": "getNNodesLeft",
        "max_depth": "getMaxDepth",
        "max_total_depth": "getMaxTotalDepth",
        "plunge_depth": "getPlungeDepth",
        # LP and separation effort
        "n_lps": "getNLPs",
        "n_lp_iterations": "getNLPIterations",
        "n_node_lp_iterations": "getNNodeLPIterations",
        "n_sb_lp_iterations": "getNStrongbranchLPIterations",
        "n_lp_cols": "getNLPCols",
        "n_lp_rows": "getNLPRows",
        "n_cuts": "getNCuts",
        "n_cuts_applied": "getNCutsApplied",
        "n_sepa_rounds": "getNSepaRounds",
        # Primal solutions
        "n_sols_found": "getNSolsFound",
        "n_lim_sols_found": "getNLimSolsFound",
        "n_counted_sols": "getNCountedSols",
        # Time
        "solving_time": "getSolvingTime",
        "presolving_time": "getPresolvingTime",
        "deterministic_time": "getDeterministicTime",
    }
    return {key: _safe(model, getter) for key, getter in getters.items()}


class SolverStats:
    """Solver-statistics view of the current node and the search tree.

    Second view of the state contract (next to the bipartite LP graph): a flat
    dictionary of ~35 O(1) scalars describing both the current node (``node_*``
    keys) and the whole search so far (tree-level keys). Suited to solver-stats
    policies such as TreeGate, T-BranT, TGPPO or Mamba that never look at the LP
    graph.

    Returns ``None`` outside the solving stage or when the episode is done.
    """

    def reset(self, _model: Model) -> None:
        pass

    def extract(self, model: Model, done: bool) -> dict[str, float] | None:
        if done or model.getStage() != PY_SCIP_STAGE.SOLVING:
            return None
        stats = _tree_stats(model)
        stats.update(_node_stats(model))
        return stats
