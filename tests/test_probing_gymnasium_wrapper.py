"""Tests for gyozas.probing_gymnasium_wrapper.ProbingGymnasiumWrapper."""

import warnings

import numpy as np
import pytest
from gymnasium import spaces
from pyscipopt import Model

from gyozas import NNodes, NodeBipartite
from gyozas.dynamics.branching import BranchingDynamics
from gyozas.dynamics.probing import ProbingDynamics
from gyozas.gymnasium_wrapper import ProbingGymnasiumWrapper
from gyozas.observations import Pseudocosts

_INSTANCE = "tests/instance.lp"
_PARAMS = {"limits/nodes": 20, "display/verblevel": 0}


def gen():
    while True:
        m = Model()
        m.readProblem(_INSTANCE)
        yield m


def make_wrapper(max_candidates=128, on_overflow="truncate", max_steps=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence root-node overflow warnings during warm-up
        return ProbingGymnasiumWrapper(
            gen(),
            reward_function=NNodes(),
            scip_params=_PARAMS,
            max_candidates=max_candidates,
            on_overflow=on_overflow,
            max_steps=max_steps,
        )


class TestSpaces:
    def test_action_space_is_multibinary(self):
        env = make_wrapper(max_candidates=64)
        assert env.action_space == spaces.MultiBinary(64)
        env.close()

    def test_observation_space_shape(self):
        env = make_wrapper(max_candidates=64)
        assert isinstance(env.observation_space, spaces.Dict)
        assert env.observation_space["candidates"].shape[0] == 64
        assert env.observation_space["mask"] == spaces.MultiBinary(64)
        env.close()


class TestConformance:
    def test_check_env(self):
        from gymnasium.utils.env_checker import check_env

        env = make_wrapper(max_candidates=128)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_env(env, skip_render_check=True)
        env.close()


class TestResetStep:
    def test_reset_obs_in_space(self):
        env = make_wrapper()
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert "action_set" in info
        env.close()

    def test_mask_matches_candidate_count(self):
        env = make_wrapper()
        obs, info = env.reset()
        n_cand = min(len(info["action_set"]), env.K)
        assert int(obs["mask"].sum()) == n_cand
        # Padding rows (beyond the real candidates) are zeroed out.
        assert np.all(obs["candidates"][n_cand:] == 0.0)
        env.close()

    def test_step_obs_in_space_and_reward_float(self):
        env = make_wrapper()
        obs, info = env.reset()
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        env.close()

    def test_selecting_candidates_triggers_probing(self):
        env = make_wrapper()
        obs, _ = env.reset()
        action = np.zeros(env.K, dtype=np.int8)
        action[np.where(obs["mask"] == 1)[0]] = 1  # probe all real candidates
        env.step(action)
        assert env.env.model.getNStrongbranchLPIterations() > 0
        env.close()

    def test_empty_action_does_no_probing(self):
        env = make_wrapper()
        env.reset()
        _obs, _r, term, _t, _i = env.step(np.zeros(env.K, dtype=np.int8))
        if not term:
            assert env.env.model.getNStrongbranchLPIterations() == 0
        env.close()

    def test_episode_terminates(self):
        env = make_wrapper()
        obs, _ = env.reset()
        term = False
        steps = 0
        while not term and steps < 200:
            obs, r, term, trunc, info = env.step(np.zeros(env.K, dtype=np.int8))
            steps += 1
        assert term
        env.close()


class TestOverflow:
    def test_truncate_warns(self):
        # Root node of this instance has far more candidates than K=8.
        with pytest.warns(UserWarning, match="exceeds max_candidates"):
            env = ProbingGymnasiumWrapper(gen(), reward_function=NNodes(), scip_params=_PARAMS, max_candidates=8)
        env.close()

    def test_error_raises(self):
        with pytest.raises(RuntimeError, match="exceeds max_candidates"):
            ProbingGymnasiumWrapper(
                gen(), reward_function=NNodes(), scip_params=_PARAMS, max_candidates=8, on_overflow="error"
            )


class TestValidation:
    def test_bad_on_overflow(self):
        with pytest.raises(ValueError, match="on_overflow"):
            make_wrapper(on_overflow="nonsense")

    def test_bad_max_candidates(self):
        with pytest.raises(ValueError, match="max_candidates"):
            make_wrapper(max_candidates=0)

    def test_non_probing_dynamics_requires_observation(self):
        with pytest.raises(ValueError, match="observation_function"):
            ProbingGymnasiumWrapper(gen(), dynamics=BranchingDynamics(), scip_params=_PARAMS)


class TestMaxSteps:
    def test_truncated_after_max_steps(self):
        env = make_wrapper(max_steps=1)
        env.reset()
        _obs, _r, term, trunc, _i = env.step(np.zeros(env.K, dtype=np.int8))
        assert trunc or term  # truncated at the step limit (unless it already terminated)
        env.close()

    def test_truncation_releases_episode_and_allows_reset(self):
        env = make_wrapper(max_steps=1)
        env.reset()
        _obs, _r, term, trunc, _i = env.step(np.zeros(env.K, dtype=np.int8))
        if term:
            env.close()
            pytest.skip("episode terminated at the first step")
        assert trunc
        # The episode is released: no action set, and stepping again is an error.
        assert env._action_set is None
        with pytest.raises(RuntimeError):
            env.step(np.zeros(env.K, dtype=np.int8))
        # ...but the env is still usable for a new episode.
        obs2, _ = env.reset()
        assert env.observation_space.contains(obs2)
        env.close()


class TestReproducibility:
    def test_seeded_reset_is_reproducible(self):
        e1 = make_wrapper()
        e2 = make_wrapper()
        o1, _ = e1.reset(seed=7)
        o2, _ = e2.reset(seed=7)
        assert np.array_equal(o1["mask"], o2["mask"])
        assert np.allclose(o1["candidates"], o2["candidates"])
        e1.close()
        e2.close()


class TestProbingVolume:
    def test_partial_mask_probes_fewer_iters_than_full(self):
        env_full = make_wrapper()
        env_half = make_wrapper()
        obs_f, _ = env_full.reset()
        obs_h, _ = env_half.reset()  # identical instance + cached warm-up => same candidates
        real = np.where(obs_f["mask"] == 1)[0]
        full = np.zeros(env_full.K, dtype=np.int8)
        full[real] = 1
        half = np.zeros(env_half.K, dtype=np.int8)
        half[real[: len(real) // 2]] = 1
        env_full.step(full)
        env_half.step(half)
        sb_full = env_full.env.model.getNStrongbranchLPIterations()
        sb_half = env_half.env.model.getNStrongbranchLPIterations()
        assert sb_full > sb_half >= 0
        env_full.close()
        env_half.close()


class TestObservationContract:
    def test_accepts_plain_node_bipartite(self):
        env = ProbingGymnasiumWrapper(
            gen(),
            observation_function=NodeBipartite(),
            dynamics=ProbingDynamics(),
            reward_function=NNodes(),
            scip_params=_PARAMS,
            max_candidates=256,
        )
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        env.close()

    def test_rejects_non_graph_observation(self):
        # Pseudocosts yields a per-variable score array, not a graph with .variable_features.
        with pytest.raises(TypeError, match="variable_features"):
            ProbingGymnasiumWrapper(
                gen(),
                observation_function=Pseudocosts(),
                dynamics=ProbingDynamics(),
                reward_function=NNodes(),
                scip_params=_PARAMS,
                max_candidates=64,
            )

    def test_reset_info_has_reward_offset(self):
        env = make_wrapper()
        _obs, info = env.reset()
        assert "reward_offset" in info
        env.close()
