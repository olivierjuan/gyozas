import itertools

import numpy as np
from numpy.typing import NDArray
from pyscipopt import SCIP_EVENTTYPE, SCIP_RESULT, Eventhdlr, Model, Nodesel

from gyozas.dynamics.threaded_dynamics import ThreadedDynamics


class NodeSelectionOracle(Nodesel):
    def __init__(self, scip: Model, obs_event, action_event, die_event) -> None:
        self.scip = scip
        self.obs_event = obs_event
        self.action_event = action_event
        self.die_event = die_event
        self.obs: NDArray[np.int64] | None = None
        self.action = None
        self.count = 0

    def _nodeselect(self, nodes) -> dict:
        if self.die_event.is_set():
            self.scip.interruptSolve()
            return {"result": SCIP_RESULT.DIDNOTRUN}
        self.count += 1
        action_set = np.array([node.getNumber() for node in nodes], dtype=np.int64)
        if len(nodes) == 0:
            return {}
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
        node_idx = self.action
        self.action_event.clear()

        chosen_node = None
        for idx, node in zip(action_set, nodes, strict=True):
            if idx == node_idx:
                chosen_node = node
                break
        return {"selnode": chosen_node}

    def nodeselect(self) -> dict:
        open_leaves, open_children, open_siblings = self.scip.getOpenNodes()
        nodes = list(itertools.chain(open_leaves, open_children, open_siblings))
        return self._nodeselect(nodes)

    def nodecomp(self, node1, node2) -> int:
        return 0


class _NodeEventHandler(Eventhdlr):
    """Tracks NODEINFEASIBLE/NODEFEASIBLE events for NodeSelectionDynamics."""

    def __init__(self, dynamics: "NodeSelectionDynamics") -> None:
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


class NodeSelectionDynamics(ThreadedDynamics):
    """Dynamics for node selection decisions in the branch-and-bound solver.

    At each step, the agent selects which open node to explore next.
    The action set contains node IDs (leaves, children, and siblings).
    """

    def __init__(self) -> None:
        super().__init__()
        self.action = None
        self.node_selection_rule: NodeSelectionOracle
        self.model = None
        self.infeasible_nodes = []
        self.feasible_nodes = []
        self.nsteps = 0

    def reset(self, model: Model) -> tuple[bool, NDArray[np.int64] | None]:
        self._stop_thread()
        # Drop caught events and null plugin refs so the old model can be GC'd.
        # catchEvent() calls Py_INCREF(model); dropEvent() balances each one.
        self._release_plugins()
        self.done = False
        self.model = model
        self.nsteps = 0
        self.obs_event.clear()
        self.action_event.clear()
        self.die_event.clear()
        self.infeasible_nodes = []
        self.feasible_nodes = []
        self.node_selection_rule = NodeSelectionOracle(model, self.obs_event, self.action_event, self.die_event)
        model.includeNodesel(
            self.node_selection_rule,
            "python-nodesel",
            "custom node selection rule",
            stdpriority=1_000_000,
            memsavepriority=1_000_000,
        )
        self._node_event_handler = _NodeEventHandler(self)
        model.includeEventhdlr(self._node_event_handler, "nodesel-node-events", "tracks node feasibility events")

        self._start_solve_thread(model)
        self.obs_event.wait()
        if self.done:
            return self.done, None
        action_set = self.node_selection_rule.obs
        if action_set is not None and len(action_set) == 0:
            # No open nodes yet (e.g. SCIP is still in presolving); advance
            # the solver one step by sending a sentinel action.
            self.step(-1)
        self.obs_event.clear()
        return self.done, action_set

    def step(self, action) -> tuple[bool, NDArray[np.int64] | None]:
        self.node_selection_rule.action = action
        self.action_event.set()
        self.obs_event.wait()
        if self.done:
            return self.done, None
        action_set = self.node_selection_rule.obs
        self.obs_event.clear()
        self.nsteps += 1
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
        if hasattr(self, "node_selection_rule"):
            self.node_selection_rule.scip = None  # ty: ignore[invalid-assignment]
            self.node_selection_rule.model = None

    def add_action_reward_to_branching_tree(self, _branching_tree, _action, _reward) -> None:
        node_id = _action
        data = _branching_tree.get_node_data(node_id)
        if data is None:
            return
        data.update({"order": self.nsteps, "reward": _reward})
