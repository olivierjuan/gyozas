import logging
from dataclasses import dataclass
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


class BranchingDirection(IntEnum):
    """Direction of a branching decision, i.e. which bound of the variable is tightened.

    ``UP`` tightens the lower bound (child node with ``var >= ceil(value)``),
    ``DOWN`` tightens the upper bound (child node with ``var <= floor(value)``).
    """

    UP = 1
    DOWN = -1


# pyscipopt SCIP_BOUNDTYPE: 0 = LOWER bound was set, 1 = UPPER bound was set.
_BOUNDTYPE_UPPER = 1


@dataclass(frozen=True)
class BranchingDecision:
    """Typed branching decision (variable x direction).

    The decision that created a node: which variable was branched on and in which
    direction its bound was tightened. It is the unit of the *decision* counter in
    :class:`PairedCounters` and is kept separate from node counts (see
    :class:`PairedCounters` for why the two must never be aggregated together).

    For a value branching (``branchVarVal``) both children are created at once, so
    a *chosen action* does not carry a direction; the direction is well defined per
    *processed node* through the branching that created it, which is what
    :attr:`BranchingDynamics.last_branching_decision` reports.
    """

    variable_name: str
    variable_index: int
    direction: BranchingDirection | None
    lp_value: float


@dataclass
class PairedCounters:
    """Paired decision/node counters for a branching policy.

    Decisions and nodes are counted **separately** and must never be mixed into a
    single aggregate: a node count aggregates solver-side work while a decision
    count aggregates policy-side work. These two counters measure consultations
    and applied branchings, so in a standard run ``nodes == decisions + 1`` (the
    root is consulted, no branching is applied there) — they do *not* reproduce
    the literature's total-tree-nodes/decisions asymmetry (~1.84-2.00x, which
    counts auxiliary nodes processed without a policy consultation). Report both
    counters, or the ratio explicitly and named for what it counts.

    Attributes
    ----------
    decisions:
        Number of branching decisions taken by the policy (variable branchings
        actually applied).
    nodes:
        Number of distinct nodes at which the policy was consulted.
    """

    decisions: int = 0
    nodes: int = 0

    @property
    def nodes_per_decision(self) -> float:
        if self.decisions == 0:
            return float("nan")
        return self.nodes / self.decisions

    def as_dict(self) -> dict[str, int | float]:
        return {"decisions": self.decisions, "nodes": self.nodes, "nodes_per_decision": self.nodes_per_decision}


class BranchingOracle(Branchrule):
    def __init__(self, scip: Model, obs_event, action_event, die_event, dynamics: "BranchingDynamics") -> None:
        self.scip = scip
        self.obs_event = obs_event
        self.action_event = action_event
        self.die_event = die_event
        self.dynamics = dynamics
        self.obs: NDArray[np.int64] | None = None
        self.action = None
        self.count = 0
        self.node_order = {}

    def branchexeclp(self, allowaddcons) -> dict:
        if self.die_event.is_set():
            self.scip.interruptSolve()
            return {"result": SCIP_RESULT.DIDNOTRUN}
        self.count += 1
        current_node = self.scip.getCurrentNode()
        self.node_order[self.count] = current_node.getNumber()
        self.dynamics.counters.nodes += 1
        self.dynamics.last_branching_decision = self._creating_decision(current_node)
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
        self.dynamics.counters.decisions += 1

        return {"result": SCIP_RESULT.BRANCHED}

    def _creating_decision(self, node) -> BranchingDecision | None:
        """The branching decision (variable x direction) that created this node."""
        if node is None or node.getNParentBranchings() == 0:
            return None
        branching = node.getParentBranchings()
        if branching is None:
            return None
        vars_, vals, btypes = branching
        var, val, btype = vars_[0], vals[0], btypes[0]
        direction = BranchingDirection.DOWN if btype == _BOUNDTYPE_UPPER else BranchingDirection.UP
        return BranchingDecision(
            variable_name=var.name,
            variable_index=var.getCol().getLPPos(),
            direction=direction,
            lp_value=float(val),
        )


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
        self.counters = PairedCounters()
        self.last_branching_decision: BranchingDecision | None = None
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
        self.counters = PairedCounters()
        self.last_branching_decision = None
        self.branch_rule = BranchingOracle(model, self.obs_event, self.action_event, self.die_event, self)
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
        if self.with_extra_actions is not None and action_set is not None and len(action_set) > 0:
            action_set = np.concatenate((self.with_extra_actions, action_set))
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
