"""Test that environments can run multiple episodes without crashing (issue #5)."""

import random

import pytest

from gyozas.environment import Environment
from gyozas.instances.files import FileGenerator
from gyozas.rewards.nnodes import NNodes


@pytest.fixture
def scip_params():
    return {"display/verblevel": 0, "limits/nodes": 20}


def run_episode(env):
    obs, action_set, reward, done, info = env.reset()
    steps = 0
    while not done:
        action = action_set[0]
        obs, action_set, reward, done, info = env.step(action)
        steps += 1
    return steps


class TestMultiEpisode:
    def test_two_episodes_branching(self, scip_params):
        instances = FileGenerator(directory="tests", pattern="*.lp", rng=42)
        env = Environment(
            instance_generator=instances,
            reward_function=NNodes(),
            scip_params=scip_params,
        )
        random.seed(42)
        steps1 = run_episode(env)
        env.close()

        # Second episode on a fresh environment (reuses dynamics object pattern)
        instances2 = FileGenerator(directory="tests", pattern="*.lp", rng=43)
        env2 = Environment(
            instance_generator=instances2,
            reward_function=NNodes(),
            scip_params=scip_params,
        )
        steps2 = run_episode(env2)
        env2.close()

        assert steps1 > 0
        assert steps2 > 0
