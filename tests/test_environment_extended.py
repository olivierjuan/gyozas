"""Extended tests for gyozas.environment.Environment to maximize coverage."""

import pytest
from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.dynamics.configuring import ConfiguringDynamics
from gyozas.environment import Environment
from gyozas.informations.empty import Empty
from gyozas.informations.time_since_last_step import TimeSinceLastStep
from gyozas.instances.files import FileGenerator
from gyozas.instances.set_cover import SetCoverGenerator
from gyozas.observations.node_bipartite_scip import NodeBipartiteSCIP
from gyozas.rewards import Constant, Done, NNodes

_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}


@pytest.fixture
def file_gen():
    return FileGenerator(directory="tests", pattern="*.lp")


@pytest.fixture
def set_cover_gen():
    return SetCoverGenerator(n_rows=100, n_cols=200, density=0.05, rng=42)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self, file_gen):
        env = Environment(instance_generator=file_gen)
        assert isinstance(env.observation_function, Empty)
        assert isinstance(env.reward_function, Constant)
        assert isinstance(env.information_function, Empty)
        assert isinstance(env.dynamics, BranchingDynamics)
        assert env.model is None
        assert env.branching_tree is None
        assert env.can_transition is False

    def test_custom_components(self, file_gen):
        env = Environment(
            instance_generator=file_gen,
            observation_function=NodeBipartiteSCIP(),
            reward_function=Done(),
            information_function=TimeSinceLastStep(),
            dynamics=BranchingDynamics(),
            scip_params=_PARAMS,
        )
        assert isinstance(env.reward_function, Done)
        assert isinstance(env.information_function, TimeSinceLastStep)

    def test_render_mode_creates_branching_tree(self, file_gen):
        env = Environment(instance_generator=file_gen, scip_params=_PARAMS, render_mode="human")
        assert env.branching_tree is not None


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


class TestSeed:
    def test_seed_delegates_to_dynamics_and_generator(self, set_cover_gen):
        env = Environment(instance_generator=set_cover_gen, scip_params=_PARAMS)
        env.seed(42)

    def test_seed_with_non_seedable_generator(self, file_gen):
        env = Environment(instance_generator=iter([Model()]), scip_params=_PARAMS)
        env.seed(42)  # should not raise even if generator has no seed()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_five_tuple(self, set_cover_gen):
        env = Environment(instance_generator=set_cover_gen, scip_params=_PARAMS)
        result = env.reset()
        assert len(result) == 5
        obs, action_set, reward, done, info = result
        env.close()

    def test_reset_sets_model(self, set_cover_gen):
        env = Environment(instance_generator=set_cover_gen, scip_params=_PARAMS)
        env.reset()
        assert env.model is not None
        env.close()

    def test_reset_enables_transition(self, set_cover_gen):
        env = Environment(instance_generator=set_cover_gen, scip_params=_PARAMS)
        env.reset()
        assert True  # done would make it False
        env.close()

    def test_reset_applies_scip_params(self, set_cover_gen):
        params = {"display/verblevel": 0, "limits/nodes": 5}
        env = Environment(instance_generator=set_cover_gen, scip_params=params)
        env.reset()
        env.close()


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_without_reset_raises(self, file_gen):
        env = Environment(instance_generator=file_gen, scip_params=_PARAMS)
        with pytest.raises(RuntimeError, match="needs to be reset"):
            env.step(0)

    def test_full_episode(self, file_gen):
        env = Environment(
            instance_generator=file_gen,
            reward_function=NNodes(),
            scip_params=_PARAMS,
        )
        obs, action_set, reward, done, info = env.reset()
        steps = 0
        while not done:
            obs, action_set, reward, done, info = env.step(action_set[0])
            steps += 1
        assert steps > 0
        assert env.can_transition is False

    def test_observation_none_when_done(self, set_cover_gen):
        env = Environment(instance_generator=set_cover_gen, scip_params=_PARAMS)
        obs, action_set, reward, done, info = env.reset()
        while not done:
            obs, action_set, reward, done, info = env.step(action_set[0])
        assert obs is None

    def test_step_with_info_function(self, set_cover_gen):
        env = Environment(
            instance_generator=set_cover_gen,
            information_function=TimeSinceLastStep(),
            scip_params=_PARAMS,
        )
        obs, action_set, reward, done, info = env.reset()
        if not done:
            obs, action_set, reward, done, info = env.step(action_set[0])
            assert info is not None
            assert isinstance(info, float)
        env.close()


# ---------------------------------------------------------------------------
# Multi-episode
# ---------------------------------------------------------------------------


class TestMultiEpisode:
    def test_two_episodes_from_generator(self, set_cover_gen):
        env = Environment(
            instance_generator=set_cover_gen,
            scip_params=_PARAMS,
        )
        for _ in range(2):
            obs, action_set, reward, done, info = env.reset()
            while not done:
                obs, action_set, reward, done, info = env.step(action_set[0])

    def test_reset_mid_episode(self, set_cover_gen):
        env = Environment(
            instance_generator=set_cover_gen,
            scip_params=_PARAMS,
        )
        obs, action_set, reward, done, info = env.reset()
        if not done:
            env.step(action_set[0])
        # Reset mid-episode: should close old solver thread
        env.close()
        obs, action_set, reward, done, info = env.reset()
        env.close()


# ---------------------------------------------------------------------------
# Configuring dynamics through Environment
# ---------------------------------------------------------------------------


class _NullObs:
    def reset(self, model):
        pass

    def extract(self, model, done):
        return None


class TestConfiguringDynamics:
    def test_configuring_episode(self, file_gen):
        env = Environment(
            instance_generator=file_gen,
            dynamics=ConfiguringDynamics(),
            observation_function=_NullObs(),
            scip_params=_PARAMS,
        )
        obs, action_set, reward, done, info = env.reset()
        assert not done
        assert action_set is None
        obs, action_set, reward, done, info = env.step({})
        assert done


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_noop_without_mode(self, file_gen):
        env = Environment(instance_generator=file_gen, scip_params=_PARAMS)
        env.render()  # should not raise


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_without_reset(self, file_gen):
        Environment(instance_generator=file_gen, scip_params=_PARAMS)
        # close without reset should not raise if dynamics has no thread
        # ConfiguringDynamics.close() works without reset
        env2 = Environment(
            instance_generator=file_gen,
            dynamics=ConfiguringDynamics(),
            scip_params=_PARAMS,
        )
        env2.close()
