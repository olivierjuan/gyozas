from pyscipopt import Model

from gyozas.dynamics.dynamics import Dynamics


class ConfiguringDynamics(Dynamics):
    """Dynamics for algorithm configuration: set SCIP parameters then solve.

    Inspired by ``ecole.dynamics.ConfiguringDynamics``.

    The episode has a single step:
    1. ``reset(model)`` returns ``(False, None)`` — the agent may now choose params.
    2. ``step(param_dict)`` sets each ``{name: value}`` pair on the model, calls
       ``model.optimize()``, and returns ``(True, None)``.

    The action is a ``dict[str, Any]`` mapping SCIP parameter names to values,
    e.g. ``{"limits/time": 60.0, "lp/threads": 4}``.
    The action set is always ``None`` (unconstrained).
    """

    def __init__(self) -> None:
        super().__init__()
        self.model: Model | None = None
        self.done: bool = False
        self.infeasible_nodes: list = []
        self.feasible_nodes: list = []

    def __del__(self) -> None:
        self.close()

    def reset(self, model: Model) -> tuple[bool, None]:
        self.model = model
        self.done = False
        return False, None

    def step(self, action: dict) -> tuple[bool, None]:
        """Set SCIP parameters from *action* and solve the instance.

        Parameters
        ----------
        action
            Mapping of SCIP parameter names to values.
            Pass an empty dict to solve with default parameters.
        """
        if self.model is None:
            raise RuntimeError("No model available. Call reset() first.")
        if self.done:
            raise RuntimeError("Episode is already done. Call reset() first.")
        if action:
            self.model.setParams(action)
        self.model.optimize()
        self.done = True
        return True, None

    def add_action_reward_to_branching_tree(self, _branching_tree, _action, _reward) -> None:
        # Configuration is a single global action — there is no per-node decision to annotate.
        pass

    def close(self) -> None:
        self.model = None
