import logging
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from pyscipopt import SCIP_EVENTTYPE, SCIP_RESULT, Branchrule, Eventhdlr, Model

from gyozas._utils import is_fixed_domain

from .threaded_dynamics import ThreadedDynamics


class ExtraBranchingActions(IntEnum):
    SKIP = -1
    CUT_OFF = -2
    REDUCE_DOMAIN = -3


class BranchingOracle(Branchrule):
    def __init__(self, scip: Model, obs_event, action_event, die_event) -> None:
        self.scip = scip
        self.obs_event = obs_event
        self.action_event = action_event
        self.die_event = die_event
        self.obs: NDArray[np.int64] | None = None
        self.action = None
        self.count = 0
        self.node_order = {}

    def branchexeclp(self, allowaddcons) -> dict:
        if self.die_event.is_set():
            self.scip.interruptSolve()
            return {"result": SCIP_RESULT.DIDNOTRUN}
        self.count += 1
        self.node_order[self.count] = self.scip.getCurrentNode().getNumber()
        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.scip.getLPBranchCands()
        branch_cands = branch_cands[:ncands]
        action_set = np.array(
            [var.getCol().getLPPos() for var in branch_cands if not is_fixed_domain(var)], dtype=np.int64
        )
        self.obs = action_set
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
        best_cand_idx = self.action
        self.action_event.clear()
        match best_cand_idx:
            case ExtraBranchingActions.SKIP:
                return {"result": SCIP_RESULT.DIDNOTRUN}
            case ExtraBranchingActions.CUT_OFF:
                return {"result": SCIP_RESULT.CUTOFF}
            case ExtraBranchingActions.REDUCE_DOMAIN:
                return {"result": SCIP_RESULT.REDUCEDDOM}
        chosen_idx = None
        for i, idx in enumerate(action_set):
            if idx == best_cand_idx:
                chosen_idx = i
                break
        if chosen_idx is None:
            raise ValueError(f"Action {best_cand_idx} not found in action set {action_set}")
        self.scip.branchVarVal(branch_cands[chosen_idx], branch_cand_sols[chosen_idx])

        return {"result": SCIP_RESULT.BRANCHED}


class _NodeEventHandler(Eventhdlr):
    """Tracks NODEINFEASIBLE/NODEFEASIBLE events for BranchingDynamics."""

    def __init__(self, dynamics: "BranchingDynamics") -> None:
        self.dynamics = dynamics

    def eventinit(self) -> None:
        self.model.catchEvent(SCIP_EVENTTYPE.NODEINFEASIBLE, self)
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFEASIBLE, self)

    def eventexec(self, event) -> dict:
        node = event.getNode()
        parent = node.getParent() if node is not None else None
        entry = (
            node,
            node.getNumber(),
            parent,
            parent.getNumber() if parent is not None else None,
            node.getLowerbound(),
        )
        if event.getType() == SCIP_EVENTTYPE.NODEINFEASIBLE:
            self.dynamics.infeasible_nodes.append(entry)
        else:
            self.dynamics.feasible_nodes.append(entry)
        return {}


class BranchingDynamics(ThreadedDynamics):
    """Dynamics for variable branching decisions in the branch-and-bound solver.

    At each step, the agent selects which fractional variable to branch on.
    The action set contains variable indices from SCIP's LP branching candidates.

    Parameters
    ----------
    with_extra_actions
        Optional list of extra action IDs (e.g. ``ExtraBranchingActions.SKIP``)
        to prepend to the action set at each step.
    """

    def __init__(
        self,
        with_extra_actions: list[int] | list[ExtraBranchingActions] | list[int | ExtraBranchingActions] | None = None,
    ) -> None:
        super().__init__()
        self.action = None
        self.branch_rule: BranchingOracle
        self.model: Model
        self.infeasible_nodes: list = []
        self.feasible_nodes: list = []
        self.current_node_id = None
        self._last_node_id = None
        self.with_extra_actions: NDArray[np.int64] | None = (
            np.array(sorted(int(x) for x in with_extra_actions), dtype=np.int64)
            if with_extra_actions is not None
            else None
        )
        self._action_set: NDArray[np.int64] | None = None

    def reset(self, model) -> tuple[bool, NDArray[np.int64] | None]:
        self._stop_thread()
        # Drop caught events and null plugin refs so the old model can be GC'd.
        # catchEvent() calls Py_INCREF(model); dropEvent() balances each one.
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
        self.branch_rule = BranchingOracle(model, self.obs_event, self.action_event, self.die_event)
        model.includeBranchrule(
            self.branch_rule,
            "python-mostinf",
            "custom most infeasible branching rule",
            priority=10000000,
            maxdepth=-1,
            maxbounddist=1,
        )
        self._node_event_handler = _NodeEventHandler(self)
        model.includeEventhdlr(self._node_event_handler, "branching-node-events", "tracks node feasibility events")

        self._start_solve_thread(model)
        self.obs_event.wait()
        if self.done:
            return self.done, None
        action_set = self.branch_rule.obs
        if self.with_extra_actions is not None and action_set is not None and len(action_set) > 0:
            action_set = np.concatenate((self.with_extra_actions, action_set))
        self.obs_event.clear()
        self._action_set = action_set
        current_node = self.model.getCurrentNode()
        self.current_node_id = current_node.getNumber()
        return self.done, action_set

    def step(self, action) -> tuple[bool, NDArray[np.int64] | None]:
        if self._action_set is None:
            raise RuntimeError("No action set available. Call reset() first.")
        if action not in self._action_set:
            raise ValueError(f"Action {action} not in action set {self._action_set}")
        self.branch_rule.action = action
        self._last_node_id = self.current_node_id
        self.action_event.set()
        self.obs_event.wait()
        if self.done:
            self._action_set = None
            return self.done, None
        action_set = self.branch_rule.obs
        self._action_set = action_set
        self.obs_event.clear()
        current_node = self.model.getCurrentNode()
        self.current_node_id = current_node.getNumber()
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
        if hasattr(self, "branch_rule"):
            self.branch_rule.scip = None  # ty: ignore[invalid-assignment]
            self.branch_rule.model = None

    def add_action_reward_to_branching_tree(self, _branching_tree, _action, _reward) -> None:
        node_id = self._last_node_id
        data = _branching_tree.get_node_data(node_id)
        if data is None:
            logging.error(f"Node {node_id} not found in branching tree.")
            return
        data.update({"action": _action, "reward": _reward})
