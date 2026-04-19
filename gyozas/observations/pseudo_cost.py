import numpy as np
from pyscipopt import Model
from pyscipopt.scip import PY_SCIP_STAGE

# SCIP_BOUNDTYPE: 0 = LOWER (branch up), 1 = UPPER (branch down)
_BOUNDTYPE_UPPER = 1

_EPS = 1e-9
_INIT_PSEUDOCOST = 1.0  # default when no branching history is available


class Pseudocosts:
    """Pseudocost scores for LP branching candidates.

    Mirrors ``ecole.observation.Pseudocosts``, implemented in pure pyscipopt.

    Pseudocosts are estimated incrementally from the branching history observed
    during solving.  For each branching decision recorded via
    ``node.getParentBranchings()``, the per-variable up/down pseudocosts are
    updated as::

        pseudocost[dir] = Σ obj_delta_k / Σ |frac_delta_k|

    where *obj_delta* is the LP bound improvement at the child node and
    *frac_delta* is the LP fractionality consumed by the branching.

    Returns a 1-D array of shape ``(n_vars,)`` with the branch score for each
    LP candidate (``NaN`` for non-candidates), or ``None`` outside the solving
    stage.

    Note
    ----
    Only branchings observed while ``extract`` is called are tracked.  Branchings
    that SCIP performs between two calls (e.g. at nodes not visited by the agent)
    are missed; those variables fall back to the ``_INIT_PSEUDOCOST`` prior.
    """

    def __init__(self) -> None:
        self._pseudo_down: dict[int, list[float]] = {}  # var_idx -> [sum_obj, sum_frac]
        self._pseudo_up: dict[int, list[float]] = {}
        self._node_lp_vals: dict[int, dict[int, float]] = {}  # node_num -> {var_idx: lp_val}

    def reset(self, _model: Model) -> None:
        self._pseudo_down.clear()
        self._pseudo_up.clear()
        self._node_lp_vals.clear()

    def extract(self, model: Model, done: bool) -> np.ndarray | None:
        if done or model.getStage() != PY_SCIP_STAGE.SOLVING:
            return None

        self._update_from_current_node(model)

        cands, lp_vals, _, n_cands, _, _ = model.getLPBranchCands()
        cands = cands[:n_cands]
        lp_vals = lp_vals[:n_cands]

        # Cache LP solution values for this node so children can look them up.
        node = model.getCurrentNode()
        if node is not None:
            self._node_lp_vals[node.getNumber()] = {
                var.getCol().getLPPos(): lp_val for var, lp_val in zip(cands, lp_vals, strict=False)
            }

        max_idx = max(var.getCol().getLPPos() for var in cands) + 1
        scores = np.full(max_idx, np.nan, dtype=np.float64)
        for var, lp_val in zip(cands, lp_vals, strict=False):
            scores[var.getCol().getLPPos()] = self._score(model, var, lp_val)

        return scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_from_current_node(self, model: Model) -> None:
        """Update pseudocost estimates using the branching that created the current node."""
        node = model.getCurrentNode()
        if node is None or node.getNParentBranchings() == 0:
            return
        branching = node.getParentBranchings()
        if branching is None:
            return

        parent = node.getParent()
        if parent is None:
            return
        parent_lp_vals = self._node_lp_vals.get(parent.getNumber(), {})
        if not parent_lp_vals:
            return

        parent_lb = parent.getLowerbound()
        child_lp_obj = model.getLPObjVal()
        obj_delta = max(0.0, child_lp_obj - parent_lb)

        vars_, _, btypes = branching
        for var, btype in zip(vars_, btypes, strict=False):
            idx = var.getCol().getLPPos()
            parent_lp_val = parent_lp_vals.get(idx)
            if parent_lp_val is None:
                continue
            frac = model.feasFrac(parent_lp_val)
            if btype == _BOUNDTYPE_UPPER:  # upper bound set → branch down
                val_delta = max(frac, _EPS)
                entry = self._pseudo_down.setdefault(idx, [0.0, 0.0])
            else:  # lower bound set → branch up
                val_delta = max(1.0 - frac, _EPS)
                entry = self._pseudo_up.setdefault(idx, [0.0, 0.0])
            entry[0] += obj_delta
            entry[1] += val_delta

    def _pseudocost(self, idx: int, down: bool) -> float:
        data = self._pseudo_down.get(idx) if down else self._pseudo_up.get(idx)
        if data is None or data[1] < _EPS:
            return _INIT_PSEUDOCOST
        return data[0] / data[1]

    def _score(self, model: Model, var, lp_val: float) -> float:
        frac = model.feasFrac(lp_val)
        idx = var.getCol().getLPPos()
        downgain = self._pseudocost(idx, down=True) * frac
        upgain = self._pseudocost(idx, down=False) * (1.0 - frac)
        return model.getBranchScoreMultiple(var, [downgain, upgain])
