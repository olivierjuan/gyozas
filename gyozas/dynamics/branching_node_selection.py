"""Joint variable-branching and node-selection dynamics.

The regime of the sequential branch-and-bound methods (Seq-BnB 2026): one
policy controls BOTH solver decisions — which open node to process next and
which variable to branch on at that node. The two decisions interleave
arbitrarily (a selected node can be fathomed by bound without ever asking
for a branching; consecutive node selections happen), so the step's action
set carries its kind: :class:`Decision` names whether the accepted actions
are open-node numbers or LP column positions, and a policy must read the
kind before interpreting the actions.

Protocol notes (empirically pinned, see tests/dynamics): the Nodesel plugin
must ALWAYS hand SCIP a valid ``selnode`` on its die route — pyscipopt maps
a missing ``selnode`` to NULL, which crashes the C solving loop after an
interrupt (an empty open set, by contrast, occurs right before the loop
exits and tolerates it). Both plugins re-check ``die_event`` after their
action wait: the dying main thread sets ``action_event`` without a fresh
action, and a stale action (a column position read as a node number, or the
reverse) must never be applied.
"""

import logging
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from pyscipopt import SCIP_EVENTTYPE, SCIP_RESULT, Branchrule, Eventhdlr, Model, Nodesel

from .branching import _BOUNDTYPE_UPPER, BranchingDecision, BranchingDirection, PairedCounters
from .threaded_dynamics import ThreadedDynamics


class DecisionKind(IntEnum):
    """Which of the two solver decisions a step's action set refers to."""

    NODE_SELECTION = 0
    BRANCHING = 1


@dataclass(frozen=True)
class Decision:
    """The pending decision: its kind and the accepted actions.

    ``actions`` are open-node *numbers* for
    :attr:`DecisionKind.NODE_SELECTION` and LP *column positions* for
    :attr:`DecisionKind.BRANCHING` — disjoint semantics, hence the kind tag.
    """

    kind: DecisionKind
    actions: NDArray[np.int64]

    def __contains__(self, action) -> bool:
        return bool(np.any(self.actions == action))

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions.tolist())


@dataclass
class SelectedPairedCounters(PairedCounters):
    """:class:`~gyozas.dynamics.branching.PairedCounters` plus the
    node-selection decisions — a third, separately reported kind of
    policy-side work (the branching ``nodes``/``decisions`` pair keeps its
    meaning: consultations at, and branchings applied on, processed nodes)."""

    nodes_selected: int = 0

    def as_dict(self) -> dict[str, int | float]:
        return {**super().as_dict(), "nodes_selected": self.nodes_selected}


class _JointNodesel(Nodesel):
    """Pauses the solver at every node selection with at least one open node."""

    def __init__(
        self, scip: Model, obs_event, action_event, die_event, dynamics: "BranchingNodeSelectionDynamics"
    ) -> None:
        self.scip = scip
        self.obs_event = obs_event
        self.action_event = action_event
        self.die_event = die_event
        self.dynamics = dynamics
        self.obs: NDArray[np.int64] | None = None
        self.action = None
        self.count = 0
        self.empty_calls = 0

    def nodeselect(self) -> dict:
        open_leaves, open_children, open_siblings = self.scip.getOpenNodes()
        nodes = [*open_leaves, *open_children, *open_siblings]
        if not nodes:
            # An empty open set occurs right before the solving loop exits;
            # an empty dict (selnode NULL) is tolerated there. The die route
            # below never takes this path: NULL after an interrupt crashes.
            self.empty_calls += 1
            return {}
        if self.die_event.is_set():
            self.scip.interruptSolve()
            return {"selnode": nodes[0]}
        self.count += 1
        self.dynamics.counters.nodes_selected += 1
        self.obs = np.array([node.getNumber() for node in nodes], dtype=np.int64)
        self.dynamics._pause(DecisionKind.NODE_SELECTION)
        while not self.action_event.wait(timeout=1.0):
            if self.die_event.is_set():
                self.scip.interruptSolve()
                return {"selnode": nodes[0]}
        # Re-check after the wait exits — action_event may have been set by
        # _stop_thread() rather than by a real action from the main thread.
        if self.die_event.is_set():
            self.scip.interruptSolve()
            return {"selnode": nodes[0]}
        self.action_event.clear()
        chosen = None
        for node in nodes:
            if node.getNumber() == self.action:
                chosen = node
                break
        if chosen is None:
            raise ValueError(f"Action {self.action} not found in action set {self.obs}")
        return {"selnode": chosen}

    def nodecomp(self, node1, node2) -> int:
        return 0


class _JointBranchrule(Branchrule):
    """Pauses the solver at every LP branching decision."""

    def __init__(
        self, scip: Model, obs_event, action_event, die_event, dynamics: "BranchingNodeSelectionDynamics"
    ) -> None:
        self.scip = scip
        self.obs_event = obs_event
        self.action_event = action_event
        self.die_event = die_event
        self.dynamics = dynamics
        self.obs: NDArray[np.int64] | None = None
        self.action = None
        self.count = 0

    def branchexeclp(self, allowaddcons) -> dict:
        if self.die_event.is_set():
            self.scip.interruptSolve()
            return {"result": SCIP_RESULT.DIDNOTRUN}
        branch_cands, branch_cand_sols, _fracs, ncands, _npriocands, _nimplcands = self.scip.getLPBranchCands()
        if ncands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        branch_cands = branch_cands[:ncands]
        self.count += 1
        current_node = self.scip.getCurrentNode()
        self.dynamics.counters.nodes += 1
        self.dynamics.last_branching_decision = self._creating_decision(current_node)
        self.obs = np.array([var.getCol().getLPPos() for var in branch_cands], dtype=np.int64)
        self.dynamics._pause(DecisionKind.BRANCHING)
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
        chosen_idx = None
        for i, idx in enumerate(self.obs):
            if idx == best_cand_idx:
                chosen_idx = i
                break
        if chosen_idx is None:
            raise ValueError(f"Action {best_cand_idx} not found in action set {self.obs}")
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
    """Tracks NODEINFEASIBLE/NODEFEASIBLE events (processed nodes)."""

    def __init__(self, dynamics: "BranchingNodeSelectionDynamics") -> None:
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


class BranchingNodeSelectionDynamics(ThreadedDynamics):
    """Dynamics controlling BOTH the node selection and the variable branching.

    At each step the solver pauses at one of its two decision points and the
    action set is a :class:`Decision`: open-node numbers when SCIP asks which
    node to process next, LP column positions when it asks how to branch on
    the current node. Both plugins share the solve thread's synchronisation
    pair and mark :attr:`pending` with their kind before pausing, so the main
    thread always knows which oracle is asking.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action = None
        self.nodesel_rule: _JointNodesel
        self.branch_rule: _JointBranchrule
        self.model: Model
        self.infeasible_nodes: list = []
        self.feasible_nodes: list = []
        self.counters: SelectedPairedCounters = SelectedPairedCounters()
        self.last_branching_decision: BranchingDecision | None = None
        self.pending: DecisionKind | None = None
        self._action_set: Decision | None = None

    def _pause(self, kind: DecisionKind) -> None:
        """Called by a plugin at its decision point, before obs_event fires."""
        self.pending = kind
        self.obs_event.set()

    def _current_decision(self) -> Decision:
        assert self.pending is not None
        rule = self.nodesel_rule if self.pending == DecisionKind.NODE_SELECTION else self.branch_rule
        assert rule.obs is not None
        return Decision(kind=self.pending, actions=rule.obs)

    # The action set is a Decision, not an NDArray — the base contract is
    # "whatever actions this dynamics accepts at this step".
    def reset(self, model: Model) -> tuple[bool, "Decision | None"]:  # ty: ignore[invalid-method-override]
        self._stop_thread()
        # Drop caught events and null plugin refs so the old model can be GC'd.
        # catchEvent() calls Py_INCREF(model); dropEvent() balances each one.
        self._release_plugins()
        self.done = False
        self.model = model
        self.obs_event.clear()
        self.action_event.clear()
        self.die_event.clear()
        self.infeasible_nodes = []
        self.feasible_nodes = []
        self.counters = SelectedPairedCounters()
        self.last_branching_decision = None
        self.pending = None
        self.nodesel_rule = _JointNodesel(model, self.obs_event, self.action_event, self.die_event, self)
        model.includeNodesel(
            self.nodesel_rule,
            "python-joint-nodesel",
            "custom node selection rule",
            stdpriority=1_000_000,
            memsavepriority=1_000_000,
        )
        self.branch_rule = _JointBranchrule(model, self.obs_event, self.action_event, self.die_event, self)
        model.includeBranchrule(
            self.branch_rule,
            "python-joint-branch",
            "custom joint branching rule",
            priority=10000000,
            maxdepth=-1,
            maxbounddist=1,
        )
        self._node_event_handler = _NodeEventHandler(self)
        model.includeEventhdlr(self._node_event_handler, "joint-node-events", "tracks node feasibility events")

        self._start_solve_thread(model)
        self.obs_event.wait()
        if self.done:
            self._action_set = None
            return self.done, None
        self.obs_event.clear()
        self._action_set = self._current_decision()
        return self.done, self._action_set

    def step(self, action) -> tuple[bool, "Decision | None"]:  # ty: ignore[invalid-method-override]
        if self._action_set is None:
            raise RuntimeError("No action set available. Call reset() first.")
        if action not in self._action_set:
            raise ValueError(f"Action {action} not in action set {self._action_set}")
        rule = self.nodesel_rule if self._action_set.kind == DecisionKind.NODE_SELECTION else self.branch_rule
        rule.action = action
        self.action_event.set()
        self.obs_event.wait()
        if self.done:
            self._action_set = None
            return self.done, None
        self.obs_event.clear()
        self._action_set = self._current_decision()
        return self.done, self._action_set

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
        if hasattr(self, "nodesel_rule"):
            self.nodesel_rule.scip = None  # ty: ignore[invalid-assignment]
            self.nodesel_rule.model = None
        if hasattr(self, "branch_rule"):
            self.branch_rule.scip = None  # ty: ignore[invalid-assignment]
            self.branch_rule.model = None

    def add_action_reward_to_branching_tree(self, _branching_tree, _action, _reward) -> None:
        # The branching-tree visualisation is branching-only; the joint
        # regime's node-selection actions have no node data to update.
        if self._action_set is None or self._action_set.kind != DecisionKind.BRANCHING:
            return
        data = _branching_tree.get_node_data(_action)
        if data is None:
            logging.error(f"Node {_action} not found in branching tree.")
            return
        data.update({"action": _action, "reward": _reward})
