"""Tests for pluggable branching experts (R4.1)."""

import random

import pytest

from gyozas.environment import Environment
from gyozas.experts import PseudocostExpert, RelpscostExpert, StrongBranchingExpert
from gyozas.instances.files import FileGenerator

EXPERTS = [RelpscostExpert, StrongBranchingExpert, PseudocostExpert]


@pytest.fixture(params=EXPERTS, ids=lambda e: e.__name__)
def env(request):
    env = Environment(
        instance_generator=FileGenerator(directory="tests", pattern="*.lp"),
        scip_params={"display/verblevel": 0, "limits/nodes": 50},
    )
    env.expert = request.param()
    yield env
    env.close()


class TestExperts:
    def test_select_returns_action_from_action_set(self, env):
        obs, action_set, reward, done, info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        action = env.expert.select(env.model, action_set)
        assert action is not None
        assert action in set(action_set.tolist())

    def test_expert_drives_full_episode(self, env):
        obs, action_set, reward, done, info = env.reset()
        steps = 0
        while not done:
            action = env.expert.select(env.model, action_set)
            if action is None:
                action = action_set[len(action_set) // 2]
            obs, action_set, reward, done, info = env.step(action)
            steps += 1
        assert steps > 0
        assert env.model.getStatus() in {"optimal", "nodelimit"}

    def test_reset_is_callable_between_episodes(self, env):
        obs, action_set, reward, done, info = env.reset()
        if not done:
            action = env.expert.select(env.model, action_set)
            env.step(action)
        env.expert.reset(env.model)
        obs, action_set, reward, done, info = env.reset()
        if not done:
            assert env.expert.select(env.model, action_set) is not None

    def test_experts_are_deterministic_given_state(self, env):
        random.seed(42)
        obs, action_set, reward, done, info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        assert env.expert.select(env.model, action_set) == env.expert.select(env.model, action_set)
