from threading import Event

import numpy as np
from numpy.typing import NDArray
from pyscipopt import SCIP_RESULT, Heur, Model

from gyozas.dynamics.threaded_dynamics import ThreadedDynamics

# SCIP_HEURTIMING_AFTERLPLOOP = 4: called after the LP is solved at each node,
# which is when LP column values are available for probing.
_HEURTIMING_AFTERLPLOOP = 4


class PrimalSearchOracle(Heur):
    """SCIP heuristic plugin that pauses at each call to ask an agent for a partial solution.

    The agent provides ``(var_indices, vals)``: a list of variable indices (from
    ``getPseudoBranchCands``) and their proposed values.  The oracle enters probing
    mode, fixes those variables, solves the LP, and tries the result as a feasible
    solution.  This repeats for ``trials_per_node`` trials per heuristic call.
    """

    def __init__(
        self, scip: Model, obs_event: Event, action_event: Event, die_event: Event, trials_per_node: int
    ) -> None:
        self.scip = scip
        self.obs_event = obs_event
        self.action_event = action_event
        self.die_event = die_event
        self.trials_per_node = trials_per_node
        self.obs = None  # action set sent to the agent
        self.action: tuple[NDArray[np.int64], NDArray[np.float64]]  # (var_indices, vals) received from the agent

    def heurexec(self, heurtiming, nodeinfeasible) -> dict:
        if self.die_event.is_set():
            self.scip.interruptSolve()
            return {"result": SCIP_RESULT.DIDNOTRUN}

        pseudo_cands, _, _ = self.scip.getPseudoBranchCands()
        idx_to_var = {v.getCol().getLPPos(): v for v in pseudo_cands}
        action_set = np.array(list(idx_to_var.keys()), dtype=np.int64)
        result = self._run_trials(idx_to_var, action_set)
        return {"result": result}

    def _wait_for_action(self) -> bool:
        """Block until the agent sets action_event; return False if die_event fires first."""
        while not self.action_event.wait(timeout=1.0):
            if self.die_event.is_set():
                return False
        # Re-check after the wait exits — action_event may have been set by
        # _stop_thread() rather than by a real action from the main thread.
        return not self.die_event.is_set()

    def _run_trials(self, idx_to_var: dict, action_set: NDArray[np.int64]):
        """Exchange actions with the agent for up to trials_per_node iterations."""
        result = SCIP_RESULT.DIDNOTRUN
        trials = 0
        while True:
            self.obs = action_set
            self.obs_event.set()

            if not self._wait_for_action():
                self.scip.interruptSolve()
                return SCIP_RESULT.DIDNOTRUN
            var_indices, vals = self.action
            self.action_event.clear()

            if len(var_indices) > 0:
                if self._probe(idx_to_var, var_indices, vals):
                    result = SCIP_RESULT.FOUNDSOL
                elif result == SCIP_RESULT.DIDNOTRUN:
                    result = SCIP_RESULT.DIDNOTFIND
            elif result == SCIP_RESULT.DIDNOTRUN:
                result = SCIP_RESULT.DIDNOTFIND

            trials += 1
            if (
                self.trials_per_node != -1 and trials >= self.trials_per_node
            ) or self.scip.isStopped():  # ty: ignore[unresolved-attribute]
                return result

    def _probe(self, idx_to_var: dict, var_indices: NDArray[np.int64], vals: NDArray[np.float64]) -> bool:
        """Fix variables, solve the LP, and try the LP solution. Returns True if kept."""
        self.scip.startProbing()
        try:
            for idx, val in zip(var_indices, vals, strict=False):
                var = idx_to_var.get(idx)
                if var is not None:
                    self.scip.fixVarProbing(var, val)

            cutoff, _ = self.scip.propagateProbing(0)
            if cutoff:
                return False

            cutoff = self.scip.constructLP()
            if cutoff:
                return False

            lperror, cutoff = self.scip.solveProbingLP()
            if lperror or cutoff:
                return False

            return self._try_lp_solution()
        finally:
            self.scip.endProbing()

    def _try_lp_solution(self) -> bool:
        """Copy the current LP solution into a new SCIP solution and try to accept it."""
        sol = self.scip.createSol(heur=self)
        for col in self.scip.getLPColsData():
            self.scip.setSolVal(sol, col.getVar(), col.getPrimsol())
        return self.scip.trySol(
            sol,
            printreason=False,
            completely=False,
            checkbounds=True,
            checkintegrality=True,
            checklprows=True,
            free=True,
        )


class PrimalSearchDynamics(ThreadedDynamics):
    """Dynamics for primal solution search in the branch-and-bound solver.

    Inspired by ``ecole.dynamics.PrimalSearchDynamics``.  At each step the agent
    provides a *partial assignment* ``(var_indices, vals)`` over the current
    pseudo-branching candidates.  The dynamics tries to complete it into a feasible
    solution via LP probing.

    Parameters
    ----------
    trials_per_node
        Number of agent interactions (probing trials) per heuristic call.
        -1 means unlimited (run until SCIP stops the solve).
    depth_freq
        Heuristic frequency: called every ``depth_freq`` nodes in depth.
    depth_start
        Minimum depth at which the heuristic is called.
    depth_stop
        Maximum depth at which the heuristic is called (-1 = no limit).
    """

    def __init__(
        self,
        trials_per_node: int = 1,
        depth_freq: int = 1,
        depth_start: int = 0,
        depth_stop: int = -1,
    ) -> None:
        if trials_per_node < -1:
            raise ValueError(f"trials_per_node must be >= -1, got {trials_per_node}")
        super().__init__()
        self.trials_per_node = trials_per_node
        self.depth_freq = depth_freq
        self.depth_start = depth_start
        self.depth_stop = depth_stop

        self.model: Model
        self._oracle: PrimalSearchOracle
        self._action_set: NDArray[np.int64] | None = None
        self.infeasible_nodes: list = []
        self.feasible_nodes: list = []

    def close(self) -> None:
        super().close()  # _stop_thread() — joins thread so SCIP is no longer running
        self._release_plugins()

    def _release_plugins(self) -> None:
        """Null oracle plugin refs so the SCIP model can be freed."""
        if hasattr(self, "_oracle"):
            self._oracle.scip = None  # ty: ignore[invalid-assignment]
            self._oracle.model = None

    def reset(self, model: Model) -> tuple[bool, NDArray[np.int64] | None]:
        self._stop_thread()
        self._release_plugins()
        self.done = False
        self.model = model
        self.obs_event.clear()
        self.action_event.clear()
        self.die_event.clear()
        self._action_set = None

        if self.trials_per_node == 0:
            model.optimize()
            return True, None

        self._oracle = PrimalSearchOracle(
            model, self.obs_event, self.action_event, self.die_event, self.trials_per_node
        )
        model.includeHeur(
            self._oracle,
            name="primal-search",
            desc="agent-driven primal solution search via LP probing",
            dispchar="P",
            priority=100_000,
            freq=self.depth_freq,
            freqofs=self.depth_start,
            maxdepth=self.depth_stop,
            timingmask=_HEURTIMING_AFTERLPLOOP,
            usessubscip=False,
        )

        self._start_solve_thread(model)
        self.obs_event.wait()

        if self.done:
            return self.done, None

        action_set = self._oracle.obs
        self.obs_event.clear()
        self._action_set = action_set
        return self.done, action_set

    def step(self, action: tuple[NDArray[np.int64], NDArray[np.float64]]) -> tuple[bool, NDArray[np.int64] | None]:
        """Apply a partial assignment and advance the solve.

        Parameters
        ----------
        action
            ``(var_indices, vals)`` — lists of variable indices and their proposed
            values.  Pass ``([], [])`` to skip without fixing any variable.
        """
        if self._action_set is None:
            raise RuntimeError("No action set available. Call reset() first.")
        var_indices, vals = action
        if len(var_indices) != len(vals):
            raise ValueError(f"var_indices and vals must have the same length, got {len(var_indices)} and {len(vals)}")

        self._oracle.action = (var_indices, vals)
        self.action_event.set()
        self.obs_event.wait()

        if self.done:
            self._action_set = None
            return self.done, None

        action_set = self._oracle.obs
        self.obs_event.clear()
        self._action_set = action_set
        return self.done, action_set

    def add_action_reward_to_branching_tree(self, _branching_tree, _action, _reward) -> None:
        # PrimalSearch actions are partial variable assignments, not node decisions,
        # so there is no natural node to annotate in the branching tree.
        pass
