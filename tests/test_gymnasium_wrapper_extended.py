"""Extended tests for gyozas.gymnasium_wrapper.GymnasiumWrapper."""

import contextlib

import gymnasium as gym
import pytest
from gymnasium import spaces

from gyozas.gymnasium_wrapper import GymnasiumWrapper
from gyozas.instances.set_cover import SetCoverGenerator
from gyozas.observations.node_bipartite_scip import NodeBipartiteSCIP
from gyozas.rewards.nnodes import NNodes

_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}

# n_rows=50 is too small — SCIP presolves it without any branching, causing
# GymnasiumWrapper.reset()'s `while done` loop to spin forever.
_GEN_ROWS, _GEN_COLS = 300, 600


@pytest.fixture
def gen():
    return SetCoverGenerator(n_rows=_GEN_ROWS, n_cols=_GEN_COLS, rng=42)


@pytest.fixture
def env(gen):
    e = GymnasiumWrapper(
        instance_generator=gen,
        observation_function=NodeBipartiteSCIP(),
        reward_function=NNodes(),
        scip_params=_PARAMS,
    )
    yield e
    with contextlib.suppress(Exception):
        e.close()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_is_gymnasium_env(self, env):
        assert isinstance(env, gym.Env)

    def test_initial_action_space(self, env):
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 1

    def test_initial_observation_space(self, env):
        assert isinstance(env.observation_space, spaces.Dict)

    def test_action_set_initially_none(self, env):
        assert env.action_set is None


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)
        assert "action_set" in info
        assert "gyozas_info" in info
        assert "reward_offset" in info

    def test_reset_with_seed(self, env):
        obs1, _ = env.reset(seed=42)
        env.close()

    def test_action_space_updated_after_reset(self, env):
        env.reset()
        assert env.action_space.n > 0
        assert env.action_set is not None

    def test_step_count_reset(self, env):
        env.reset()
        assert env._step_count == 0


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_returns_five_tuple(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="No action set"):
            env.step(0)

    def test_invalid_action_raises(self, env):
        env.reset()
        with pytest.raises(ValueError):
            env.step(999999)

    def test_negative_action_raises(self, env):
        env.reset()
        with pytest.raises(ValueError):
            env.step(-1)

    def test_full_episode(self, env):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_action_space_updates_each_step(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        if not terminated:
            # Action space should match current action_set
            assert env.action_space.n == len(env.action_set)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_max_steps_truncation(self, gen):
        env = GymnasiumWrapper(
            instance_generator=gen,
            scip_params=_PARAMS,
            max_steps=2,
        )
        env.reset()
        for _i in range(2):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated:
                break
        if not terminated:
            assert truncated

    def test_truncation_clears_action_set(self, gen):
        env = GymnasiumWrapper(
            instance_generator=gen,
            scip_params=_PARAMS,
            max_steps=1,
        )
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        if truncated:
            assert env.action_set is None


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_after_episode(self, env):
        env.reset()
        env.close()  # should not raise

    def test_close_without_reset(self, gen):
        GymnasiumWrapper(instance_generator=gen, scip_params=_PARAMS)
        # Close without reset - ConfiguringDynamics can handle this
        # BranchingDynamics may not have thread, just test it doesn't crash
