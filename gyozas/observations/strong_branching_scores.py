import numpy as np
from pyscipopt import Model
from pyscipopt.scip import PY_SCIP_STAGE

from gyozas._utils import is_fixed_domain


class StrongBranchingScores:
    """Full strong branching scores for LP branching candidates.

    Mirrors ``ecole.observation.StrongBranchingScores``, which runs SCIP's native
    *vanilla full strong branching* (``branching/vanillafullstrong`` with
    ``idempotent=true``).  This implementation reproduces that algorithm in
    pyscipopt using the native strong-branching API
    (``startStrongbranch`` / ``getVarStrongbranch`` / ``endStrongbranch``).

    Why not probing
    ---------------
    The previous implementation entered SCIP *probing* mode
    (``startProbing`` / ``solveProbingLP``) to evaluate each branch.  That mutates
    SCIP's search state, which is unsafe when the observation is extracted while
    the solve is suspended inside a branching-rule callback (e.g. gyozas'
    threaded ``BranchingDynamics``): resuming ``optimize()`` afterwards could
    dereference corrupted state and segfault.  Native strong branching with
    ``idempotent=True`` leaves SCIP's state unchanged — it is exactly the API the
    built-in strong-branching rules call from within ``branchexeclp`` — so it is
    safe in that context and matches the ecole reference semantics.

    For each candidate the down/up dual bounds are obtained from SCIP's strong
    branching, converted to bound gains relative to the current LP objective, and
    combined into a branch score via ``getBranchScoreMultiple`` (SCIP's default
    product score), the same scoring ecole/vanillafullstrong uses.

    Returns a 1-D array of shape ``(n_lp_cols,)`` with the score for each
    candidate indexed by LP position (``NaN`` for non-candidates), or ``None``
    outside the solving stage. Like ecole, the array is sized by the number of
    LP columns, so it is indexable by any branching candidate — including the
    pseudo candidates (a superset of the LP ones), which read ``NaN``.

    Parameters
    ----------
    pseudo_candidates
        If ``False`` (default), score LP branching candidates (fractional vars)
        with fractional strong branching.  If ``True``, score pseudo-branching
        candidates (all non-fixed discrete vars) with integral strong branching
        (matching ecole's ``integralcands`` flag).
    itlim
        LP iteration limit for each strong-branching solve. -1 = no limit.
    """

    def __init__(self, pseudo_candidates: bool = False, itlim: int = -1) -> None:
        self.pseudo_candidates = pseudo_candidates
        self.itlim = itlim

    def reset(self, _model: Model) -> None:
        # No per-episode state to reset.
        pass

    def extract(self, model: Model, done: bool) -> np.ndarray | None:
        if done or model.getStage() != PY_SCIP_STAGE.SOLVING:
            return None

        # var.getIndex() -> LP position, built ONLY from real LP columns.
        # getLPColsData() is always valid; var.getCol() on a LOOSE / aggregated
        # variable reads a garbage union pointer (the COLUMN assert is compiled
        # out in release builds) and segfaults — so we never call it directly.
        # Strong branching also requires a column variable, so non-column
        # candidates are skipped here anyway.
        lppos: dict[int, int] = {}
        for col in model.getLPColsData():
            v = col.getVar()
            if v is not None:
                lppos[v.getIndex()] = col.getLPPos()

        if self.pseudo_candidates:
            cands, n_cands, _ = model.getPseudoBranchCands()
            cands = cands[:n_cands]
            cands = [v for v in cands if v.getIndex() in lppos]
        else:
            cands, _, _, n_cands, _, _ = model.getLPBranchCands()
            cands = cands[:n_cands]
            cands = [v for v in cands if not is_fixed_domain(v) and v.getIndex() in lppos]

        scores = np.full(model.getNLPCols(), np.nan, dtype=np.float64)
        if not cands:
            return scores

        lp_obj = model.getLPObjVal()
        inf = model.infinity()
        feastol = 1e-6  # SCIP default feasibility tolerance

        model.startStrongbranch()
        try:
            for var in cands:
                lp_pos = lppos[var.getIndex()]

                # Only genuinely *fractional* candidates are real branchings.  An
                # integral candidate (``pseudo_candidates=True`` includes every
                # non-fixed discrete var, most of which sit at an integer LP value)
                # has no meaningful branch: integral strong branching would round it
                # to v±1, and e.g. a binary at 1 makes the up-branch x>=2 trivially
                # infeasible.  Treating that as an infinite gain made ~half the
                # candidates tie at the top score (acc@1 collapsed to 1/#ties).  We
                # instead give integral candidates the zero-gain score (matching the
                # old probing behaviour, where floor/ceil were no-ops → gain 0), so
                # they stay in the candidate set but never spuriously win.
                lp_val = var.getLPSol()
                if abs(lp_val - round(lp_val)) <= feastol:
                    scores[lp_pos] = model.getBranchScoreMultiple(var, [0.0, 0.0])
                    continue

                # Fractional candidate: real fractional strong branching.  Here an
                # infeasible child is a *genuine* cutoff (rare and informative), so
                # an infinite gain is appropriate and does not create mass ties.
                (down, up, _downvalid, _upvalid, downinf, upinf, _downconflict, _upconflict, lperror) = (
                    model.getVarStrongbranch(var, self.itlim, idempotent=True, integral=False)
                )

                if lperror:
                    # Unresolved LP error: leave NaN (treated as non-candidate).
                    continue

                downgain = inf if downinf else max(0.0, down - lp_obj)
                upgain = inf if upinf else max(0.0, up - lp_obj)
                scores[lp_pos] = model.getBranchScoreMultiple(var, [downgain, upgain])
        finally:
            model.endStrongbranch()

        return scores
