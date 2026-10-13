"""Tests for the Zarpellon search-tree augmentation (seed s, k) (R4.2)."""

import random

import pytest

from gyozas.augmentation import ZarpellonAugmentation
from gyozas.dynamics.branching import BranchingDynamics
from gyozas.environment import Environment
from gyozas.instances.files import FileGenerator


def _run_episode(dynamics, seed=42):
    env = Environment(
        instance_generator=FileGenerator(directory="tests", pattern="*.lp"),
        dynamics=dynamics,
        scip_params={"display/verblevel": 0, "limits/nodes": 50},
    )
    try:
        random.seed(seed)
        obs, action_set, reward, done, info = env.reset()
        while not done:
            obs, action_set, reward, done, info = env.step(action_set[len(action_set) // 2])
        return env
    finally:
        env.close()


class TestZarpellonAugmentation:
    def test_apply_seeds_solver_randomization(self):
        augmentation = ZarpellonAugmentation(seed=7, k=3)
        from pyscipopt import Model

        model = Model()
        augmentation.apply(model)
        assert model.getParam("randomization/permutevars") is True
        assert model.getParam("randomization/permutationseed") == 7
        model.freeProb()

    def test_negative_k_raises(self):
        with pytest.raises(ValueError, match="k"):
            ZarpellonAugmentation(seed=0, k=-1)

    def test_dict_roundtrip(self):
        augmentation = ZarpellonAugmentation(seed=5, k=2)
        assert ZarpellonAugmentation.from_dict(augmentation.to_dict()) == augmentation

    def test_explorer_takes_the_first_k_decisions(self):
        k = 2
        dynamics = BranchingDynamics(augmentation=ZarpellonAugmentation(seed=3, k=k))
        _run_episode(dynamics)
        # Exactly k explorer branchings, and the policy still did its own work:
        # the explorer's nodes are neither consultations nor decisions.
        assert dynamics.augmentation_explored == k
        counters = dynamics.counters
        assert counters.decisions > 0
        assert counters.nodes >= counters.decisions

    def test_without_augmentation_nothing_is_explored(self):
        dynamics = BranchingDynamics()
        _run_episode(dynamics)
        assert dynamics.augmentation_explored == 0

    def test_same_seed_reproduces_the_same_search(self):
        counters = []
        for _ in range(2):
            dynamics = BranchingDynamics(augmentation=ZarpellonAugmentation(seed=11, k=4))
            dynamics.seed(123)
            _run_episode(dynamics)
            counters.append((dynamics.counters.decisions, dynamics.counters.nodes))
        assert counters[0] == counters[1]
