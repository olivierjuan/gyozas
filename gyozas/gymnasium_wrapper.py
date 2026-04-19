"""Gymnasium-compatible wrapper for gyozas environments.

Provides a standard ``gymnasium.Env`` interface so that gyozas environments
can be used directly with RL libraries like Stable-Baselines3, CleanRL, etc.

Example
-------
>>> from gyozas.gymnasium_wrapper import GymnasiumWrapper
>>> import gyozas
>>>
>>> env = GymnasiumWrapper(
...     instance_generator=gyozas.SetCoverGenerator(n_rows=50, n_cols=100, rng=0),
...     observation_function=gyozas.NodeBipartite(),
...     reward_function=gyozas.NNodes(),
... )
>>> obs, info = env.reset()
>>> obs, reward, terminated, truncated, info = env.step(env.unwrapped.action_set[0])
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces

from gyozas.environment import Environment
from gyozas.instances import InstanceGenerator


class GymnasiumWrapper(gym.Env):
    """Wraps a gyozas ``Environment`` as a standard ``gymnasium.Env``.

    Since gyozas action sets are variable-size (different branching candidates
    at each step), this wrapper uses a ``Discrete`` action space sized to the
    maximum action set seen so far. The ``action_set`` attribute holds the
    valid actions for the current step.

    Parameters
    ----------
    instance_generator
        Iterator yielding PySCIPOpt Model instances.
    observation_function
        Gyozas observation function. Defaults to ``NodeBipartite``.
    reward_function
        Gyozas reward function. Defaults to ``NNodes``.
    information_function
        Gyozas information function. Defaults to ``Empty``.
    dynamics
        Gyozas dynamics. Defaults to ``BranchingDynamics``.
    scip_params
        SCIP parameters dict applied at the start of each episode.
    max_steps
        If set, truncate episodes after this many steps.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance_generator: InstanceGenerator,
        observation_function=None,
        reward_function=None,
        information_function=None,
        dynamics=None,
        scip_params=None,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = Environment(
            instance_generator=instance_generator,
            observation_function=observation_function,
            reward_function=reward_function,
            information_function=information_function,
            dynamics=dynamics,
            scip_params=scip_params,
            render_mode=render_mode,
        )
        self.max_steps = max_steps
        self._step_count = 0
        self.action_set: list[int] | None = None

        # Gymnasium spaces -- observation space is set after first reset
        # Action space starts at 1 and grows as needed
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Dict({})  # placeholder

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        """Reset the environment and return initial observation and info.

        Parameters
        ----------
        seed
            Random seed for reproducibility.
        options
            Unused, for Gymnasium API compatibility.

        Returns
        -------
        observation
            The initial observation.
        info
            Dictionary with ``action_set`` and any information function output.
        """
        if seed is not None:
            self.env.seed(seed)

        obs, action_set, reward, done, info = self.env.reset()
        self._step_count = 0

        # When the instance is solved at the root node (e.g. by presolving),
        # obs and action_set are None.  Keep generating new instances until
        # we get one that actually requires branching decisions.
        _retries = 0
        while done:
            _retries += 1
            if _retries > 100:
                raise RuntimeError(
                    "GymnasiumWrapper.reset() got 100 consecutive instances that were "
                    "solved without any agent decisions (e.g. solved by presolving). "
                    "Use a harder instance generator."
                )
            obs, action_set, reward, done, info = self.env.reset()

        self.action_set = action_set
        self.action_space = spaces.Discrete(len(action_set))

        info_dict = {"action_set": action_set, "gyozas_info": info, "reward_offset": reward}
        return obs, info_dict

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]:
        """Take a step in the environment.

        Parameters
        ----------
        action
            Index into the current ``action_set``. The wrapper translates
            this positional index to the actual gyozas action.

        Returns
        -------
        observation
            The new observation (None if terminated).
        reward
            The step reward.
        terminated
            True if the solver finished.
        truncated
            True if ``max_steps`` was reached.
        info
            Dictionary with ``action_set`` and any information function output.
        """
        if self.action_set is None:
            raise RuntimeError("No action set available. Call reset() first.")

        # Map positional index to actual action
        if 0 <= action < len(self.action_set):
            gyozas_action = self.action_set[action]
        else:
            raise ValueError(f"Action {action} out of range [0, {len(self.action_set)})")

        obs, action_set, reward, done, info = self.env.step(gyozas_action)
        self._step_count += 1
        self.action_set = action_set

        if action_set is not None:
            self.action_space = spaces.Discrete(len(action_set))

        terminated = done
        truncated = False
        if self.max_steps is not None and self._step_count >= self.max_steps and not done:
            truncated = True
            self.env.close()
            self.action_set = None

        info_dict = {"action_set": action_set, "gyozas_info": info}
        return obs, float(reward), terminated, truncated, info_dict

    def close(self) -> None:
        """Close the underlying gyozas environment."""
        self.env.close()
