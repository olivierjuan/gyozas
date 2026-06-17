import logging
import math

import numpy as np
from numpy.typing import NDArray
from pyscipopt import SCIP_EVENTTYPE, SCIP_RESULT, Branchrule, Model

from gyozas._utils import is_fixed_domain

from .branching import _NodeEventHandler
from .threaded_dynamics import ThreadedDynamics

# SCIP_BRANCHDIR enum values accepted by getVarPseudocost.
_BRANCHDIR_DOWN = 0
_BRANCHDIR_UP = 1

_SCORING_FUNCTIONS = ("product", "sum", "min")


class ProbeLedger:
    """Per-variable strong-branch evaluation history for the probing setting.

    SCIP's internal pseudocost *count* is not exposed by PySCIPOpt, so ``ProbingDynamics``
    records its own probing history here: for each variable (keyed by the stable transformed
    index ``var.getIndex()``) the number of times it has been strong-branched this episode and
    the node at which it was last evaluated. A ``NodeBipartiteProbing`` observation can share
    the same ledger to expose these as per-variable features.

    The ledger is cleared (not replaced) at the start of each episode, so a reference handed to
    an observation stays valid across episodes.
    """

    def __init__(self) -> None:
        self._count: dict[int, int] = {}
        self._last_node: dict[int, int] = {}

    def clear(self) -> None:
        self._count.clear()
        self._last_node.clear()

    def record(self, var_index: int, node: int) -> None:
        self._count[var_index] = self._count.get(var_index, 0) + 1
        self._last_node[var_index] = node

    def count(self, var_index: int) -> int:
        """Number of times the variable has been strong-branched this episode."""
        return self._count.get(var_index, 0)

    def last_node(self, var_index: int) -> int:
        """Node number at which the variable was last evaluated, or -1 if never."""
        return self._last_node.get(var_index, -1)


class ProbingOracle(Branchrule):
    """Branching rule that asks the agent which fractional variables to strong-branch.

    Each call publishes the LP branching candidates and waits for the agent to return
    a *subset* of them. That subset is evaluated with strong branching (the expensive,
    exact probe); the remaining candidates are scored from pseudocosts. The variable
    with the best combined score is branched on (reliability-style). Strong-branch
    probes run with ``idempotent=False`` and feed ``updateVarPseudocost`` so their
    side effects — LP-iteration accounting, global pseudocost seeding, infeasible-branch
    detection — propagate back into SCIP.
    """

    def __init__(
        self, scip: Model, obs_event, action_event, die_event, scoring, itlim, pseudocost_weight, ledger
    ) -> None:
        self.scip = scip
        self.obs_event = obs_event
        self.action_event = action_event
        self.die_event = die_event
        self.scoring = scoring
        self.itlim = itlim
        self.pseudocost_weight = pseudocost_weight
        self.ledger = ledger
        self.obs: NDArray[np.int64] | None = None
        self.action = None
        self.count = 0
        self.node_order: dict = {}
        # Outcome telemetry (per episode): how the agent's subset decisions resolved.
        self.n_branched = 0
        self.n_reductions = 0
        self.n_cutoffs = 0
        self.n_pseudocost_updates = 0

    def branchexeclp(self, allowaddcons) -> dict:
        if self.die_event.is_set():
            self.scip.interruptSolve()
            return {"result": SCIP_RESULT.DIDNOTRUN}
        self.count += 1
        self.node_order[self.count] = self.scip.getCurrentNode().getNumber()

        cands, sols, fracs, ncands, _nprio, _nimpl = self.scip.getLPBranchCands()
        cand_data = [
            (v, s, f)
            for v, s, f in zip(cands[:ncands], sols[:ncands], fracs[:ncands], strict=True)
            if not is_fixed_domain(v)
        ]
        positions = [v.getCol().getLPPos() for v, _, _ in cand_data]
        self.obs = np.array(positions, dtype=np.int64)
        self.obs_event.set()

        while not self.action_event.wait(timeout=1.0):
            if self.die_event.is_set():
                self.scip.interruptSolve()
                return {"result": SCIP_RESULT.DIDNOTRUN}
        # Re-check after the wait exits — action_event may have been set by
        # _stop_thread() rather than by a real action from the main thread.
        if self.die_event.is_set():
            self.scip.interruptSolve()
            return {"result": SCIP_RESULT.DIDNOTRUN}
        subset = self.action
        self.action_event.clear()

        return self._decide(allowaddcons, cand_data, positions, subset)

    def _decide(self, allowaddcons, cand_data, positions, subset) -> dict:
        if not positions:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        lpobj = self.scip.getLPObjVal()
        pos_to_data = dict(zip(positions, cand_data, strict=True))
        subset_positions = {int(p) for p in subset} if subset is not None else set()

        sb_results = self._strong_branch(positions, pos_to_data, subset_positions, lpobj)

        # Classify the infeasibility flags, then apply the resulting bound changes
        # (only when it is safe to change bounds: allowaddcons and all columns in LP).
        can_change = bool(allowaddcons) and self.scip.allColsInLP()
        cutoff, tighten, forced_pos = self._classify_infeasibilities(sb_results, can_change)
        if cutoff:
            self.n_cutoffs += 1
            return {"result": SCIP_RESULT.CUTOFF}
        reduced = False
        for pos, side in tighten:
            var, sol, _f = pos_to_data[pos]
            if side == "lb":
                infeasible, tightened = self.scip.tightenVarLb(var, math.ceil(sol), force=True)
            else:
                infeasible, tightened = self.scip.tightenVarUb(var, math.floor(sol), force=True)
            if infeasible:
                self.n_cutoffs += 1
                return {"result": SCIP_RESULT.CUTOFF}
            reduced = reduced or tightened
        if reduced:
            self.n_reductions += 1
            return {"result": SCIP_RESULT.REDUCEDDOM}

        best_pos = (
            forced_pos if forced_pos is not None else self._argmax_score(positions, pos_to_data, sb_results, lpobj)
        )
        var, sol, _f = pos_to_data[best_pos]
        down_child, _eq_child, up_child = self.scip.branchVarVal(var, sol)

        # Fold the valid strong-branch dual bounds into the children's lower bounds.
        if best_pos in sb_results:
            res = sb_results[best_pos]
            if down_child is not None and res["downvalid"]:
                self.scip.updateNodeLowerbound(down_child, res["down"])
            if up_child is not None and res["upvalid"]:
                self.scip.updateNodeLowerbound(up_child, res["up"])
        self.n_branched += 1
        return {"result": SCIP_RESULT.BRANCHED}

    @staticmethod
    def _classify_infeasibilities(sb_results: dict, can_change: bool) -> tuple[bool, list, int | None]:
        """Decide what to do with strong-branch infeasibility flags (pure, no solver calls).

        Returns ``(cutoff, tighten, forced_pos)``:
        - ``cutoff``: some probed variable had *both* branches infeasible — the node is infeasible.
        - ``tighten``: list of ``(pos, "lb"|"ub")`` bound reductions to apply (only when ``can_change``).
        - ``forced_pos``: a candidate with a single infeasible branch to branch on when bounds may
          not be changed (``can_change`` is False), so the infeasible child is pruned by branching.
        """
        tighten: list = []
        forced_pos = None
        for pos, res in sb_results.items():
            if res["downinf"] and res["upinf"]:
                return True, [], None
            if res["downinf"] or res["upinf"]:
                if can_change:
                    tighten.append((pos, "lb" if res["downinf"] else "ub"))
                elif forced_pos is None:
                    forced_pos = pos
        return False, tighten, forced_pos

    def _strong_branch(self, positions, pos_to_data, subset_positions, lpobj) -> dict:
        """Strong-branch every chosen candidate, seeding global pseudocosts as a side effect.

        Returns a mapping ``pos -> {down, up, downvalid, upvalid, downinf, upinf, gains}``.
        This is the single seam through which strong-branching information is acquired:
        when PySCIPOpt exposes propagation-based strong branching, only this method needs
        to change (enable propagation and apply the returned bound deductions).
        """
        probed = [p for p in positions if p in subset_positions]
        results: dict = {}
        if not probed:
            return results
        node = self.scip.getCurrentNode().getNumber()
        self.scip.startStrongbranch()
        try:
            for pos in probed:
                var, _sol, frac = pos_to_data[pos]
                down, up, downvalid, upvalid, downinf, upinf, _dconf, _uconf, lperror = self.scip.getVarStrongbranch(
                    var, self.itlim, idempotent=False, integral=False
                )
                if lperror:
                    continue
                self.ledger.record(var.getIndex(), node)
                gain_down = max(down - lpobj, 0.0)
                gain_up = max(up - lpobj, 0.0)
                # Seed global pseudocosts from valid, feasible probe directions.
                if downvalid and not downinf:
                    self.scip.updateVarPseudocost(var, -frac, gain_down, self.pseudocost_weight)
                    self.n_pseudocost_updates += 1
                if upvalid and not upinf:
                    self.scip.updateVarPseudocost(var, 1.0 - frac, gain_up, self.pseudocost_weight)
                    self.n_pseudocost_updates += 1
                results[pos] = {
                    "down": down,
                    "up": up,
                    "downvalid": downvalid,
                    "upvalid": upvalid,
                    "downinf": downinf,
                    "upinf": upinf,
                    "gain_down": gain_down,
                    "gain_up": gain_up,
                }
        finally:
            self.scip.endStrongbranch()
        return results

    def _argmax_score(self, positions, pos_to_data, sb_results, lpobj):
        best_pos, best_score = positions[0], -math.inf
        for pos in positions:
            var, _sol, frac = pos_to_data[pos]
            if pos in sb_results:
                gain_down = sb_results[pos]["gain_down"]
                gain_up = sb_results[pos]["gain_up"]
            else:
                gain_down = self.scip.getVarPseudocost(var, _BRANCHDIR_DOWN) * frac
                gain_up = self.scip.getVarPseudocost(var, _BRANCHDIR_UP) * (1.0 - frac)
            score = self._score(var, gain_down, gain_up)
            if score > best_score:
                best_score, best_pos = score, pos
        return best_pos

    def _score(self, var, gain_down, gain_up) -> float:
        if self.scoring == "product":
            return self.scip.getBranchScoreMultiple(var, [gain_down, gain_up])
        if self.scoring == "sum":
            return gain_down + gain_up
        return min(gain_down, gain_up)


class ProbingDynamics(ThreadedDynamics):
    """Dynamics for agent-guided strong-branching (a learnable reliability rule).

    At each step the agent receives the fractional LP branching candidates and returns
    a **subset** of them to evaluate with strong branching. The chosen subset is probed
    exactly (strong branching), the remaining candidates are scored from pseudocosts,
    and the variable with the best combined score is branched on. An empty subset falls
    back to pure pseudocost branching.

    Probing has a real computational cost (strong-branching LP iterations), so the action
    is only meaningful when paired with a reward that penalises it — use
    ``StrongBranchingLPIterations`` or ``TotalLPIterations`` (plain ``LPIterations`` counts
    node LPs only and will *not* see the probing cost). Without such a penalty the optimal
    policy degenerates to "probe every candidate" (full strong branching).

    This is *not* a faithful reproduction of SCIP's strong-branching rule: it deliberately
    ignores some MIP-solver interactions (primal solutions found mid-probe, cutoff-bound
    updates, deeper probing). Because PySCIPOpt currently exposes only LP-based strong
    branching, propagation-discovered domain reductions are unavailable; LP-detected
    infeasibilities are still applied as bound reductions/cutoffs when ``allowaddcons``
    holds and all columns are in the LP.

    Parameters
    ----------
    scoring
        How to combine the down/up gains into a branching score: ``"product"`` (SCIP's
        default, via ``getBranchScoreMultiple``), ``"sum"``, or ``"min"``.
    itlim
        LP iteration limit per strong-branch call (``-1`` for no limit).
    pseudocost_weight
        Weight in ``(0, 1]`` used when feeding probe gains into the global pseudocosts.
    ledger
        Optional :class:`ProbeLedger` recording per-variable evaluation history. Pass the same
        instance to a ``NodeBipartiteProbing`` observation to expose probe counts as features.
        Defaults to a fresh ledger, available as ``self.ledger``.
    """

    def __init__(
        self,
        scoring: str = "product",
        itlim: int = -1,
        pseudocost_weight: float = 1.0,
        ledger: "ProbeLedger | None" = None,
    ) -> None:
        if scoring not in _SCORING_FUNCTIONS:
            raise ValueError(f"scoring must be one of {_SCORING_FUNCTIONS}, got {scoring!r}.")
        if not 0.0 < pseudocost_weight <= 1.0:
            raise ValueError(f"pseudocost_weight must be in (0, 1], got {pseudocost_weight}.")
        super().__init__()
        self.scoring = scoring
        self.itlim = itlim
        self.pseudocost_weight = pseudocost_weight
        self.ledger = ledger if ledger is not None else ProbeLedger()
        self.action = None
        self.oracle: ProbingOracle
        self.model: Model
        self.infeasible_nodes: list = []
        self.feasible_nodes: list = []
        self.current_node_id = None
        self._last_node_id = None
        self._action_set: NDArray[np.int64] | None = None

    def reset(self, model) -> tuple[bool, NDArray[np.int64] | None]:
        self._stop_thread()
        # Drop caught events and null plugin refs so the old model can be GC'd.
        self._release_plugins()
        self.done = False
        self.model = model
        self._last_node_id = None
        self.current_node_id = None
        self.obs_event.clear()
        self.action_event.clear()
        self.die_event.clear()
        self.infeasible_nodes = []
        self.feasible_nodes = []
        self.ledger.clear()
        self.oracle = ProbingOracle(
            model,
            self.obs_event,
            self.action_event,
            self.die_event,
            self.scoring,
            self.itlim,
            self.pseudocost_weight,
            self.ledger,
        )
        model.includeBranchrule(
            self.oracle,
            "python-probing",
            "agent-guided strong-branching rule",
            priority=10000000,
            maxdepth=-1,
            maxbounddist=1,
        )
        self._node_event_handler = _NodeEventHandler(self)  # ty: ignore[invalid-argument-type]
        model.includeEventhdlr(self._node_event_handler, "probing-node-events", "tracks node feasibility events")

        self._start_solve_thread(model)
        self.obs_event.wait()
        if self.done:
            return self.done, None
        action_set = self.oracle.obs
        self.obs_event.clear()
        self._action_set = action_set
        self.current_node_id = self.model.getCurrentNode().getNumber()
        return self.done, action_set

    def step(self, action) -> tuple[bool, NDArray[np.int64] | None]:
        if self._action_set is None:
            raise RuntimeError("No action set available. Call reset() first.")
        subset = np.asarray(action, dtype=np.int64).reshape(-1)
        invalid = set(subset.tolist()) - set(self._action_set.tolist())
        if invalid:
            raise ValueError(f"Actions {sorted(invalid)} not in action set {self._action_set}")
        self.oracle.action = subset
        self._last_node_id = self.current_node_id
        self.action_event.set()
        self.obs_event.wait()
        if self.done:
            self._action_set = None
            return self.done, None
        action_set = self.oracle.obs
        self._action_set = action_set
        self.obs_event.clear()
        self.current_node_id = self.model.getCurrentNode().getNumber()
        return self.done, action_set

    def close(self) -> None:
        super().close()  # _stop_thread() — joins thread so SCIP is no longer running
        self._release_plugins()

    def _release_plugins(self) -> None:
        """Drop caught events and null plugin refs so the SCIP model can be freed."""
        if hasattr(self, "_node_event_handler"):
            handler = self._node_event_handler
            if handler.model is not None:
                try:
                    handler.model.dropEvent(SCIP_EVENTTYPE.NODEINFEASIBLE, handler)
                    handler.model.dropEvent(SCIP_EVENTTYPE.NODEFEASIBLE, handler)
                except Exception:
                    pass
            handler.model = None
            handler.dynamics = None  # ty: ignore[invalid-assignment]
        if hasattr(self, "oracle"):
            self.oracle.scip = None  # ty: ignore[invalid-assignment]
            self.oracle.model = None

    def add_action_reward_to_branching_tree(self, _branching_tree, _action, _reward) -> None:
        data = _branching_tree.get_node_data(self._last_node_id)
        if data is None:
            logging.error(f"Node {self._last_node_id} not found in branching tree.")
            return
        data.update({"action": _action, "reward": _reward})
