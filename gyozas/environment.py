from collections.abc import Iterator
from typing import Any

from pyscipopt.scip import Model

from gyozas.branching_tree import BranchingTree
from gyozas.dynamics.branching import BranchingDynamics
from gyozas.informations.empty import Empty
from gyozas.instances import InstanceGenerator
from gyozas.observations.meta_observation import MetaObservation
from gyozas.rewards import Constant


class Environment:
    """Partially Observable Markov Decision Process (POMDP) for combinatorial optimization.

    Similar to Gymnasium, environments represent the task that an agent is supposed to solve.
    For maximum customizability, different components are composed/orchestrated in this class.
    """

    def __init__(
        self,
        instance_generator: InstanceGenerator | Iterator[Model],
        observation_function=None,
        reward_function=None,
        information_function=None,
        scip_params=None,
        render_mode=None,
        dynamics=None,
        **dynamics_kwargs,
    ) -> None:
        """Create a new environment object.

        Parameters
        ----------
        instance_generator:
            An iterator yielding PySCIPOpt Model instances to solve.
        observation_function:
            An observation function used to customize the observation returned by
            :meth:`reset` and :meth:`step`. Defaults to Empty.
        reward_function:
            A reward function used to customize the reward returned by :meth:`reset`
            and :meth:`step`. Defaults to Constant(1.0).
        information_function:
            An information function used to customize the additional information
            returned by :meth:`reset` and :meth:`step`. Defaults to Empty.
        scip_params:
            Parameters set on the underlying SCIP Model at the start of every episode.
        render_mode:
            If set, enables rendering of the branching tree.
        dynamics:
            The dynamics controlling the MDP transitions. Defaults to BranchingDynamics.
        **dynamics_kwargs:
            Additional keyword arguments.
        """
        self.reward_function = reward_function or Constant(1.0)
        if observation_function is not None and isinstance(observation_function, list | dict | tuple):
            observation_function = MetaObservation(observation_function)
        self.observation_function = observation_function or Empty()
        if information_function is not None and isinstance(information_function, list | dict | tuple):
            information_function = MetaObservation(information_function)
        self.information_function = information_function or Empty()
        self.scip_params = scip_params or {}
        self.model: Model | None = None
        if isinstance(dynamics, type):
            dynamics = dynamics()
        self.dynamics = dynamics or BranchingDynamics()
        self.can_transition = False
        self.branching_tree: BranchingTree | None = BranchingTree() if render_mode is not None else None
        self.step_idx = 0
        self.render_mode = render_mode
        self.instance_generator = instance_generator

    def reset(self, **dynamics_kwargs) -> tuple[Any, Any, float, bool, Any]:
        """Start a new episode.

        This method brings the environment to a new initial state, *i.e.* starts a new
        episode. The method can be called at any point in time.

        Parameters
        ----------
        dynamics_kwargs:
            Extra arguments are forwarded to the underlying Dynamics.

        Returns
        -------
        observation:
            The observation extracted from the initial state.
        action_set:
            An optional subset that defines which actions are accepted in the next transition.
        reward_offset:
            An offset on the total cumulated reward, a.k.a. the initial reward.
        done:
            A boolean flag indicating whether the current state is terminal.
        info:
            A collection of environment specific information about the transition.
        """
        self.can_transition = True
        try:
            self.model = next(self.instance_generator)
            self.model.setParams(self.scip_params)
            self.dynamics.set_seed_on_model(self.model)

            # Reset data extraction functions
            self.reward_function.reset(self.model)
            self.observation_function.reset(self.model)
            self.information_function.reset(self.model)
            self.step_idx = 0

            # Place the environment in its initial state
            done, action_set = self.dynamics.reset(self.model, **dynamics_kwargs)
            if self.branching_tree is not None:
                self.branching_tree.add_current_node_from_pyscipopt(
                    self.model, step=self.step_idx, action_set=action_set
                )

            return self._extract_mdp_data(action_set, done)
        except Exception as e:
            self.can_transition = False
            raise e

    def step(self, action, **dynamics_kwargs) -> tuple[Any, Any, float, bool, Any]:
        """Transition from one state to another.

        This method takes a user action to transition from the current state to the
        next. The method **cannot** be called if the environment has not been reset
        since its instantiation or since a terminal state has been reached.

        Parameters
        ----------
        action:
            The action to take as part of the Markov Decision Process.
        dynamics_kwargs:
            Extra arguments are forwarded to the underlying Dynamics.

        Returns
        -------
        observation:
            The observation extracted from the new state.
        action_set:
            An optional subset that defines which actions are accepted in the next transition.
        reward:
            A real number to use for reinforcement learning.
        done:
            A boolean flag indicating whether the current state is terminal.
        info:
            A collection of environment specific information about the transition.
        """
        if not self.can_transition:
            raise RuntimeError("Environment needs to be reset.")

        try:
            # Transition the environment to the next state
            done, action_set = self.dynamics.step(action, **dynamics_kwargs)
            self.step_idx += 1
            result = self._extract_mdp_data(action_set, done)
            if self.branching_tree is not None:
                if not done:
                    assert self.model is not None
                    self.branching_tree.add_current_node_from_pyscipopt(
                        self.model, step=self.step_idx, action_set=action_set
                    )
                    self.dynamics.add_action_reward_to_branching_tree(self.branching_tree, action, result[2])
                else:
                    self.branching_tree.add_infeasible_nodes(self.dynamics.infeasible_nodes)
                    self.branching_tree.add_feasible_nodes(self.dynamics.feasible_nodes)
            return result

        except Exception as e:
            self.can_transition = False
            raise e

    def _extract_mdp_data(self, action_set, done) -> tuple[Any, Any, float, bool, Any]:
        assert self.model is not None
        self.can_transition = not done
        reward = self.reward_function.extract(self.model, done)
        if not done:
            observation = self.observation_function.extract(self.model, done)
        else:
            observation = None
        information = self.information_function.extract(self.model, done)
        return observation, action_set, reward, done, information

    def close(self) -> None:
        """Close the environment and free resources."""
        self.dynamics.close()
        for fn in (self.reward_function, self.observation_function, self.information_function):
            close = getattr(fn, "close", None)
            if close is not None:
                close()

    def __del__(self) -> None:
        self.close()

    def render(self) -> None:
        if self.render_mode is None:
            return
        assert self.branching_tree is not None, "Branching tree is not initialized for rendering."
        self.branching_tree.render(self.render_mode)

    def seed(self, value: int) -> None:
        """Set the random seed of the environment for reproducibility.

        Seeds the dynamics' SCIP randomization and the instance generator
        (if it exposes a ``seed`` method).
        """
        self.dynamics.seed(value)
        if hasattr(self.instance_generator, "seed"):
            self.instance_generator.seed(value)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[call-non-callable]
