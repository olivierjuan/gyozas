import math

import numpy as np
from pyscipopt import Model
from pyscipopt.scip import PY_SCIP_STAGE

from gyozas._utils import is_fixed_domain


class StrongBranchingScores:
    """Full strong branching scores for LP branching candidates.

    Mirrors ``ecole.observation.StrongBranchingScores``, implemented in pure pyscipopt.

    For each LP branching candidate the observation temporarily enters probing
    mode, solves the down-branch LP (``var <= floor(lp_val)``) and the up-branch
    LP (``var >= ceil(lp_val)``), then combines the bound improvements into a
    branch score via ``model.getBranchScoreMultiple``.

    The probing LPs are solved idempotently (no side-effects on SCIP state).

    Returns a 1-D array of shape ``(n_vars,)`` with the score for each LP
    candidate (``NaN`` for non-candidates), or ``None`` outside the solving stage.

    Parameters
    ----------
    pseudo_candidates
        If ``False`` (default), score LP branching candidates (fractional vars).
        If ``True``, score pseudo-branching candidates (all non-fixed discrete vars).
    itlim
        LP iteration limit for each strong-branching solve. -1 = no limit.
    """

    def __init__(self, pseudo_candidates: bool = False, itlim: int = -1) -> None:
        self.pseudo_candidates = pseudo_candidates
        self.itlim = itlim

    def reset(self, _model: Model) -> None:
        # No per-episode state to reset.
        pass

    def _probe_bound(self, model: Model, var, lp_obj: float, inf: float, down: bool) -> float:
        """Probe one branch direction and return the bound improvement (gain).

        Manages its own probing session so it can be called independently.
        Returns ``float('nan')`` on LP error, ``inf`` on cutoff, otherwise
        ``max(0, new_lp_obj - lp_obj)``.
        """
        lp_val = var.getLPSol()
        model.startProbing()
        try:
            model.newProbingNode()
            if down:
                model.chgVarUbProbing(var, math.floor(lp_val))
            else:
                model.chgVarLbProbing(var, math.ceil(lp_val))
            model.constructLP()
            lperror, cutoff = model.solveProbingLP(self.itlim)
            if lperror:
                return float("nan")
            elif cutoff:
                return inf
            else:
                return max(0.0, model.getLPObjVal() - lp_obj)
        finally:
            model.endProbing()

    def extract(self, model: Model, done: bool) -> np.ndarray | None:
        if done or model.getStage() != PY_SCIP_STAGE.SOLVING:
            return None

        if self.pseudo_candidates:
            cands, n_cands, _ = model.getPseudoBranchCands()
            cands = cands[:n_cands]
            lp_vals = [var.getLPSol() for var in cands]
        else:
            cands, lp_vals, _, n_cands, _, _ = model.getLPBranchCands()
            cands = cands[:n_cands]
            lp_vals = lp_vals[:n_cands]
            lp_vals = [val for val, var in zip(lp_vals, cands, strict=False) if not is_fixed_domain(var)]
            cands = [var for var in cands if not is_fixed_domain(var)]

        if not cands:
            return np.empty(0, dtype=np.float64)

        max_idx = max(var.getCol().getLPPos() for var in cands) + 1
        scores = np.full(max_idx, np.nan, dtype=np.float64)

        lp_obj = model.getLPObjVal()
        inf = model.infinity()

        model.startProbing()
        try:
            for var, lp_val in zip(cands, lp_vals, strict=False):
                # Down branch
                model.newProbingNode()
                model.chgVarUbProbing(var, math.floor(lp_val))
                model.constructLP()
                lperror, cutoff = model.solveProbingLP(self.itlim)
                if lperror:
                    downgain = float("nan")
                elif cutoff:
                    downgain = inf
                else:
                    downgain = max(0.0, model.getLPObjVal() - lp_obj)
                model.backtrackProbing(0)

                # Up branch
                model.newProbingNode()
                model.chgVarLbProbing(var, math.ceil(lp_val))
                model.constructLP()
                lperror, cutoff = model.solveProbingLP(self.itlim)
                if lperror:
                    upgain = float("nan")
                elif cutoff:
                    upgain = inf
                else:
                    upgain = max(0.0, model.getLPObjVal() - lp_obj)
                model.backtrackProbing(0)

                if not (math.isnan(downgain) or math.isnan(upgain)):
                    scores[var.getCol().getLPPos()] = model.getBranchScoreMultiple(var, [downgain, upgain])
        finally:
            model.endProbing()

        return scores
