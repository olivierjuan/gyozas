from pyscipopt import Model

# Statuses meaning the solve terminated on its own (with a definite answer).
_TERMINAL_STATUSES = {"optimal", "infeasible", "unbounded", "inforunbd", "gaplimit"}

# Statuses meaning the solve was stopped by a budget, mapped to the budget kind.
_TRUNCATED_STATUSES = {
    "timelimit": "time",
    "nodelimit": "nodes",
    "totalnodelimit": "total_nodes",
    "stallnodelimit": "stall_nodes",
    "memlimit": "memory",
    "sollimit": "solutions",
    "bestsollimit": "best_solutions",
    "restartlimit": "restarts",
    "primallimit": "primal",
    "duallimit": "dual",
}


class SolveStatus:
    """Always-reported solve outcome (solved vs truncated at which budget).

    The truncation pitfall: a run stopped at its time or node budget is not
    comparable to a completed run, so the end state must be reported on *every*
    run, never implicit. This information function returns, at every step and
    crucially at the last one, a dictionary with:

    - ``status``: the raw SCIP status string (e.g. ``"optimal"``, ``"timelimit"``);
    - ``solved``: whether the solve terminated on its own (including proving
      infeasibility, unboundedness, or reaching a gap target);
    - ``truncated``: whether the solve was stopped by a budget;
    - ``limit``: which budget was hit (``None`` if not truncated), e.g.
      ``"time"``, ``"nodes"``, ``"memory"``.
    """

    def reset(self, _model: Model) -> None:
        pass

    def extract(self, model: Model, done: bool) -> dict[str, str | bool | None]:
        status = model.getStatus()
        truncated = status in _TRUNCATED_STATUSES
        return {
            "status": status,
            "solved": status in _TERMINAL_STATUSES,
            "truncated": truncated,
            "limit": _TRUNCATED_STATUSES.get(status) if truncated else None,
        }
