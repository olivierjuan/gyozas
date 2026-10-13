"""Tests for the single policy interface (R1.1/R1.5)."""

import numpy as np

from gyozas.environment import Environment
from gyozas.instances import SetCoverGenerator
from gyozas.observations import NodeBipartite
from gyozas.policy import CallablePolicy, Policy
from gyozas.rewards import NNodes


class TestPolicyProtocol:
    def test_plain_class_satisfies_the_protocol(self):
        class MyPolicy:
            def select_action(self, observation, action_set):
                return int(action_set[0])

        assert isinstance(MyPolicy(), Policy)

    def test_callable_policy_wraps_a_function(self):
        policy = CallablePolicy(lambda obs, action_set: action_set[-1])
        action = policy.select_action("obs", np.array([3, 7, 11]))
        assert action == 11

    def test_callable_policy_runs_a_full_episode(self):
        env = Environment(
            instance_generator=SetCoverGenerator(n_rows=500, n_cols=1000, rng=0),
            observation_function=NodeBipartite(),
            reward_function=NNodes(),
            scip_params={"display/verblevel": 0, "limits/nodes": 50},
        )
        policy = CallablePolicy(
            lambda obs, action_set: action_set[int(np.argmax(obs.variable_features[action_set, 4]))]
        )
        try:
            obs, action_set, reward, done, info = env.reset()
            steps = 0
            while not done:
                obs, action_set, reward, done, info = env.step(policy.select_action(obs, action_set))
                steps += 1
            assert done
            assert steps > 0
        finally:
            env.close()
