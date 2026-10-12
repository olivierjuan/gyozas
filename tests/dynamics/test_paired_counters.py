"""Tests for the typed branching decision and paired decision/node counters (R1.4)."""

import random

import pytest

from gyozas.dynamics.branching import BranchingDecision, BranchingDirection, PairedCounters
from gyozas.environment import Environment
from gyozas.instances.files import FileGenerator


@pytest.fixture
def env():
    env = Environment(
        instance_generator=FileGenerator(directory="tests", pattern="*.lp"),
        scip_params={"display/verblevel": 0, "limits/nodes": 50},
    )
    yield env
    env.close()


class TestPairedCountersUnit:
    def test_empty_counters_ratio_is_nan(self):
        counters = PairedCounters()
        assert counters.decisions == 0
        assert counters.nodes == 0
        assert counters.nodes_per_decision != counters.nodes_per_decision  # NaN

    def test_ratio(self):
        counters = PairedCounters(decisions=4, nodes=8)
        assert counters.nodes_per_decision == pytest.approx(2.0)

    def test_as_dict(self):
        counters = PairedCounters(decisions=2, nodes=3)
        d = counters.as_dict()
        assert d["decisions"] == 2
        assert d["nodes"] == 3
        assert d["nodes_per_decision"] == pytest.approx(1.5)


class TestPairedCountersOnSolver:
    def test_counters_reset_per_episode(self, env):
        random.seed(42)
        obs, action_set, reward, done, info = env.reset()
        while not done:
            obs, action_set, reward, done, info = env.step(action_set[len(action_set) // 2])
        dynamics = env.dynamics
        first = (dynamics.counters.decisions, dynamics.counters.nodes)
        assert first[0] > 0

        env.reset()
        # The root node has already been consulted by the time reset() returns,
        # but no decision has been taken yet.
        assert dynamics.counters.decisions == 0
        assert dynamics.counters.nodes == 1
        assert dynamics.last_branching_decision is None

    def test_counters_stay_paired_and_separate(self, env):
        """At least one consultation per decision, never mixed into one number."""
        random.seed(42)
        obs, action_set, reward, done, info = env.reset()
        while not done:
            obs, action_set, reward, done, info = env.step(action_set[len(action_set) // 2])
        counters = env.dynamics.counters
        assert counters.decisions > 0
        assert counters.nodes >= counters.decisions
        assert counters.nodes_per_decision > 0

    def test_last_branching_decision_typed_with_direction(self, env):
        random.seed(42)
        obs, action_set, reward, done, info = env.reset()
        env.step(action_set[len(action_set) // 2])
        decision = env.dynamics.last_branching_decision
        if decision is None:
            pytest.skip("Instance too small to have a non-root branching node")
        assert isinstance(decision, BranchingDecision)
        assert decision.direction in (BranchingDirection.UP, BranchingDirection.DOWN)
        assert isinstance(decision.variable_name, str) and len(decision.variable_name) > 0
        assert decision.variable_index >= 0
        assert 0.0 <= decision.lp_value <= 1.0 or decision.lp_value == int(decision.lp_value)
