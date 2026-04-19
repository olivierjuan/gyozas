import random

import pytest

from gyozas.environment import Environment
from gyozas.instances.files import FileGenerator
from gyozas.rewards.done import Done
from gyozas.rewards.lp_iterations import LPIterations
from gyozas.rewards.nnodes import NNodes
from gyozas.rewards.solving_time import SolvingTime


@pytest.fixture
def file_instances():
    return FileGenerator(directory="tests", pattern="*.lp")


@pytest.fixture
def scip_params():
    return {
        "display/verblevel": 0,
        "limits/nodes": 50,
    }


class TestEnvironmentBranching:
    def test_reset_returns_five_tuple(self, file_instances, scip_params):
        env = Environment(instance_generator=file_instances, scip_params=scip_params)
        result = env.reset()
        assert len(result) == 5
        obs, action_set, reward, done, info = result
        if not done:
            assert action_set is not None
            assert len(action_set) > 0
        env.close()

    def test_full_episode_nnodes(self, file_instances, scip_params):
        env = Environment(
            instance_generator=file_instances,
            reward_function=NNodes(),
            scip_params=scip_params,
        )
        random.seed(42)
        obs, action_set, reward, done, info = env.reset()
        total_reward = reward
        steps = 0
        while not done:
            action = action_set[len(action_set) // 2]
            obs, action_set, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        assert steps > 0
        assert total_reward > 0
        env.close()

    def test_full_episode_solving_time(self, file_instances, scip_params):
        env = Environment(
            instance_generator=file_instances,
            reward_function=SolvingTime(),
            scip_params=scip_params,
        )
        obs, action_set, reward, done, info = env.reset()
        while not done:
            action = action_set[0]
            obs, action_set, reward, done, info = env.step(action)
            assert reward >= 0
        env.close()

    def test_full_episode_lp_iterations(self, file_instances, scip_params):
        env = Environment(
            instance_generator=file_instances,
            reward_function=LPIterations(),
            scip_params=scip_params,
        )
        obs, action_set, reward, done, info = env.reset()
        while not done:
            action = action_set[0]
            obs, action_set, reward, done, info = env.step(action)
            assert reward >= 0
        env.close()

    def test_done_reward(self, file_instances, scip_params):
        env = Environment(
            instance_generator=file_instances,
            reward_function=Done(),
            scip_params=scip_params,
        )
        obs, action_set, reward, done, info = env.reset()
        while not done:
            action = action_set[0]
            obs, action_set, reward, done, info = env.step(action)
        env.close()

    def test_step_without_reset_raises(self, file_instances, scip_params):
        env = Environment(instance_generator=file_instances, scip_params=scip_params)
        with pytest.raises(RuntimeError):
            env.step(0)

    def test_render_without_mode_is_noop(self, file_instances, scip_params):
        env = Environment(instance_generator=file_instances, scip_params=scip_params)
        env.render()  # should not raise
