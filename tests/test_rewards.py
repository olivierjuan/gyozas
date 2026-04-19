import random

import pytest

from gyozas.environment import Environment
from gyozas.instances.files import FileGenerator
from gyozas.rewards.integral_bound import DualIntegral, PrimalDualIntegral, PrimalIntegral


@pytest.fixture
def file_instances():
    return FileGenerator(directory="tests", pattern="*.lp", rng=42)


@pytest.fixture
def scip_params():
    return {"display/verblevel": 0, "limits/nodes": 50}


class TestDualIntegral:
    def test_full_episode(self, file_instances, scip_params):
        env = Environment(
            instance_generator=file_instances,
            reward_function=DualIntegral(),
            scip_params=scip_params,
        )
        random.seed(42)
        obs, action_set, reward, done, info = env.reset()
        while not done:
            action = action_set[0]
            obs, action_set, reward, done, info = env.step(action)
        env.close()


class TestPrimalIntegral:
    def test_full_episode(self, file_instances, scip_params):
        env = Environment(
            instance_generator=file_instances,
            reward_function=PrimalIntegral(),
            scip_params=scip_params,
        )
        random.seed(42)
        obs, action_set, reward, done, info = env.reset()
        while not done:
            action = action_set[0]
            obs, action_set, reward, done, info = env.step(action)
        env.close()


class TestPrimalDualIntegral:
    def test_full_episode(self, file_instances, scip_params):
        env = Environment(
            instance_generator=file_instances,
            reward_function=PrimalDualIntegral(),
            scip_params=scip_params,
        )
        random.seed(42)
        obs, action_set, reward, done, info = env.reset()
        while not done:
            action = action_set[0]
            obs, action_set, reward, done, info = env.step(action)
        env.close()
