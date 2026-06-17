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

import warnings
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gyozas.dynamics.probing import ProbingDynamics
from gyozas.environment import Environment
from gyozas.instances import InstanceGenerator
from gyozas.observations.node_bipartite_probing import NodeBipartiteProbing


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


class ProbingGymnasiumWrapper(gym.Env):
    """Wraps a probing ``Environment`` as a standard ``gymnasium.Env``.

    ``ProbingDynamics`` asks the agent for a *subset* of the current branching candidates to
    strong-branch — a variable-length set decision that RL libraries (SB3, CleanRL) cannot
    consume directly. This wrapper exposes it as a fixed ``MultiBinary(K)`` action over a
    padded, masked candidate slate, so it trains with stock SB3. The observation is a ``Dict``
    of per-candidate features (gathered from a per-variable observation such as
    ``NodeBipartiteProbing``) plus a validity ``mask``. ``action[i] == 1`` means
    "strong-branch candidate ``i``"; padded/invalid slots are ignored, and an all-zero action
    is an empty subset (pure pseudocost branching).

    Pair this with ``NodeBipartiteProbing`` (the default) or another per-variable observation.
    Do **not** use ``StrongBranchingScores`` as the observation: it computes the full strong
    branching this dynamics exists to avoid. Use a probing-cost-aware reward (e.g.
    ``TotalLPIterations``) or the subset action is meaningless.

    Parameters
    ----------
    instance_generator
        Iterator yielding PySCIPOpt Model instances.
    observation_function
        A per-variable observation exposing ``.variable_features`` (e.g.
        ``NodeBipartiteProbing``). Defaults to ``NodeBipartiteProbing`` sharing the dynamics'
        ledger.
    reward_function
        Gyozas reward function. Use a probing-cost-aware reward (e.g. ``TotalLPIterations``).
    information_function
        Gyozas information function.
    dynamics
        Defaults to ``ProbingDynamics``. Custom dynamics require an explicit
        ``observation_function``.
    scip_params
        SCIP parameters applied at the start of each episode.
    max_candidates
        The fixed action-space size ``K``. Nodes with more candidates are handled per
        ``on_overflow``.
    on_overflow
        ``"truncate"`` (default) keeps the first ``K`` candidates and warns once;
        ``"error"`` raises.
    max_steps
        If set, truncate episodes after this many steps.

    Example
    -------
    >>> import gyozas
    >>> env = gyozas.ProbingGymnasiumWrapper(
    ...     instance_generator=gyozas.SetCoverGenerator(n_rows=200, n_cols=100, rng=0),
    ...     reward_function=-gyozas.NNodes() - 0.001 * gyozas.TotalLPIterations(),
    ... )
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
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
        max_candidates: int = 1000,
        on_overflow: str = "truncate",
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if on_overflow not in ("truncate", "error"):
            raise ValueError(f"on_overflow must be 'truncate' or 'error', got {on_overflow!r}.")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates}.")
        self.K = int(max_candidates)
        self.on_overflow = on_overflow
        self.max_steps = max_steps
        self.render_mode = render_mode

        if dynamics is None:
            dynamics = ProbingDynamics()
        if observation_function is None:
            if not isinstance(dynamics, ProbingDynamics):
                raise ValueError("Provide an observation_function when using a non-ProbingDynamics dynamics.")
            observation_function = NodeBipartiteProbing(ledger=dynamics.ledger)
        self.env = Environment(
            instance_generator=instance_generator,
            observation_function=observation_function,
            reward_function=reward_function,
            information_function=information_function,
            dynamics=dynamics,
            scip_params=scip_params,
            render_mode=render_mode,
        )

        self.action_space = spaces.MultiBinary(self.K)
        self._action_set: np.ndarray | None = None
        self._n_eff = 0
        self._step_count = 0
        self._overflow_warned = False
        self._cached: tuple[dict, dict] | None = None

        # Warm up one episode to infer the per-candidate feature dimension and set the
        # observation space (so RL libraries can read it before the first reset). The
        # warm-up observation is cached and returned by the first unseeded reset().
        obs, info = self._do_reset(seed=None)
        self._n_features = obs["candidates"].shape[1]
        self.observation_space = spaces.Dict(
            {
                "candidates": spaces.Box(low=-np.inf, high=np.inf, shape=(self.K, self._n_features), dtype=np.float32),
                "mask": spaces.MultiBinary(self.K),
            }
        )
        self._cached = (obs, info)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)  # seed Gymnasium's RNG (gym.Env contract)
        # Reuse the warm-up episode for the first unseeded reset; otherwise start fresh.
        if self._cached is not None and seed is None:
            cached, self._cached = self._cached, None
            return cached
        self._cached = None
        return self._do_reset(seed)

    def _do_reset(self, seed: int | None) -> tuple[dict, dict]:
        if seed is not None:
            self.env.seed(seed)
        bg, action_set, reward, done, info = self.env.reset()
        retries = 0
        while done:
            retries += 1
            if retries > 100:
                raise RuntimeError(
                    "ProbingGymnasiumWrapper.reset() got 100 consecutive instances solved without "
                    "any agent decisions (e.g. by presolving). Use a harder instance generator."
                )
            bg, action_set, reward, done, info = self.env.reset()
        self._step_count = 0
        obs, self._n_eff = self._build_obs(bg, action_set)
        self._action_set = action_set
        return obs, {"action_set": action_set, "gyozas_info": info, "reward_offset": float(reward)}

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        if self._action_set is None:
            raise RuntimeError("No action set available. Call reset() first.")
        action = np.asarray(action).reshape(-1)
        subset = np.array(
            [self._action_set[i] for i in range(self._n_eff) if action[i]],
            dtype=np.int64,
        )
        bg, action_set, reward, done, info = self.env.step(subset)
        self._step_count += 1

        terminated = bool(done)
        truncated = False
        if terminated:
            self._action_set = None
            obs = self._zero_obs()
        else:
            obs, self._n_eff = self._build_obs(bg, action_set)
            self._action_set = action_set
            if self.max_steps is not None and self._step_count >= self.max_steps:
                # Truncate: release the still-running episode so no SCIP thread lingers,
                # and require a reset() before the next step (consistent with GymnasiumWrapper).
                truncated = True
                self.env.close()
                self._action_set = None
        return obs, float(reward), terminated, truncated, {"action_set": action_set, "gyozas_info": info}

    def _build_obs(self, bg, action_set) -> tuple[dict, int]:
        if not hasattr(bg, "variable_features"):
            raise TypeError(
                "ProbingGymnasiumWrapper requires a per-variable observation exposing "
                "'variable_features' (e.g. NodeBipartiteProbing); got "
                f"{type(bg).__name__}."
            )
        var_feats = bg.variable_features
        candidates = np.zeros((self.K, var_feats.shape[1]), dtype=np.float32)
        mask = np.zeros(self.K, dtype=np.int8)

        n = len(action_set)
        if n > self.K:
            if self.on_overflow == "error":
                raise RuntimeError(f"Action set size ({n}) exceeds max_candidates ({self.K}).")
            if not self._overflow_warned:
                warnings.warn(
                    f"Action set size ({n}) exceeds max_candidates ({self.K}); truncating to the "
                    "first K candidates. Increase max_candidates to avoid this.",
                    stacklevel=2,
                )
                self._overflow_warned = True
        n_eff = min(n, self.K)

        # Map candidate LP positions -> variable_features rows (same getVars(transformed=True)
        # order the observation used), so mask[i] aligns with action_set[i].
        model = self.env.model
        assert model is not None  # always set while an episode is live
        pos_to_row = {v.getCol().getLPPos(): i for i, v in enumerate(model.getVars(transformed=True))}
        for i in range(n_eff):
            row = pos_to_row.get(int(action_set[i]))
            if row is not None:
                candidates[i] = var_feats[row]
            mask[i] = 1

        return {"candidates": candidates, "mask": mask}, n_eff

    def _zero_obs(self) -> dict:
        return {
            "candidates": np.zeros((self.K, self._n_features), dtype=np.float32),
            "mask": np.zeros(self.K, dtype=np.int8),
        }

    def close(self) -> None:
        self.env.close()
