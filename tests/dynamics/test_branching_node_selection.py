"""Tests for gyozas.dynamics.branching_node_selection.

Unlike NodeSelectionDynamics (construction-only: its Nodesel can hand SCIP a
NULL selnode on the die route, which crashes the C solving loop), the joint
dynamics' protocol is pinned here end-to-end: the die route always returns a
real node, both plugins re-check ``die_event`` after their action wait, and
an empty open set is answered inside the plugin. The integration tests below
solve small set covers with both decisions controlled — the empirically
validated regime (see the module docstring).
"""

import numpy as np
import pytest

from gyozas.dynamics import (
    BranchingNodeSelectionDynamics,
    Decision,
    DecisionKind,
    Dynamics,
)
from gyozas.dynamics.branching_node_selection import SelectedPairedCounters
from gyozas.environment import Environment
from gyozas.informations.solve_status import SolveStatus
from gyozas.instances import SetCoverGenerator


def _instances(n, rows=120, cols=150, density=0.10, seed=7):
    """Small set covers, one fresh model per draw. Most draws solve at the
    root; the branching ones exercise the full joint protocol."""
    generator = SetCoverGenerator(rng=np.random.default_rng(seed))
    for _ in range(n):
        model = generator.generate_instance(n_rows=rows, n_cols=cols, density=density)
        model.hideOutput()
        yield model


class TestConstruction:
    def test_is_dynamics_subclass(self):
        assert issubclass(BranchingNodeSelectionDynamics, Dynamics)

    def test_default_attributes(self):
        dynamics = BranchingNodeSelectionDynamics()
        assert dynamics.done is False
        assert dynamics.pending is None
        assert dynamics.counters.nodes_selected == 0
        assert dynamics.last_branching_decision is None

    def test_seed_does_not_raise(self):
        BranchingNodeSelectionDynamics().seed(123)


class TestDecision:
    def test_membership_and_length(self):
        decision = Decision(kind=DecisionKind.BRANCHING, actions=np.array([3, 7, 9]))
        assert 7 in decision
        assert 8 not in decision
        assert len(decision) == 3
        assert list(decision) == [3, 7, 9]

    def test_kinds_are_distinct(self):
        assert DecisionKind.NODE_SELECTION != DecisionKind.BRANCHING
        assert int(DecisionKind.NODE_SELECTION) == 0
        assert int(DecisionKind.BRANCHING) == 1


class TestCounters:
    def test_selected_paired_counters_extends_the_report(self):
        counters = SelectedPairedCounters()
        counters.nodes = 5
        counters.decisions = 4
        counters.nodes_selected = 11
        report = counters.as_dict()
        assert report["nodes"] == 5
        assert report["decisions"] == 4
        assert report["nodes_per_decision"] == 5 / 4
        assert report["nodes_selected"] == 11


def _run_joint_episode(dynamics, model, node_choice, branch_choice):
    """One joint episode under deterministic policies; returns the decision
    kinds in order plus the final status."""
    done, decision = dynamics.reset(model)
    kinds = []
    while not done:
        assert decision is not None
        kinds.append(decision.kind)
        if decision.kind == DecisionKind.NODE_SELECTION:
            numbers = decision.actions
            action = int(numbers[node_choice(len(numbers))])
        else:
            positions = decision.actions
            action = int(positions[branch_choice(len(positions))])
        done, decision = dynamics.step(action)
    return kinds, model.getStatus()


class TestJointEpisode:
    def test_first_decision_selects_a_node(self):
        dynamics = BranchingNodeSelectionDynamics()
        generator = SetCoverGenerator(rng=np.random.default_rng(7))
        model = generator.generate_instance(n_rows=120, n_cols=150, density=0.10)
        model.hideOutput()
        done, decision = dynamics.reset(model)
        assert decision is not None
        assert decision.kind == DecisionKind.NODE_SELECTION
        assert len(decision) >= 1
        assert (decision.actions > 0).all()  # node numbers
        dynamics.close()

    def test_nodesel_rule_exposes_the_open_node_objects(self):
        # The joint-regime policies read node tokens (depth, bounds, ancestor
        # branchings) at the pause; the rule keeps the Node objects that back
        # the action set's numbers.
        dynamics = BranchingNodeSelectionDynamics()
        generator = SetCoverGenerator(rng=np.random.default_rng(7))
        model = generator.generate_instance(n_rows=120, n_cols=150, density=0.10)
        model.hideOutput()
        _done, decision = dynamics.reset(model)
        nodes = dynamics.nodesel_rule.nodes
        assert nodes is not None and len(nodes) == len(decision)
        assert [node.getNumber() for node in nodes] == decision.actions.tolist()
        assert all(node.getDepth() == 0 for node in nodes)  # the root
        assert decision.actions[0] in decision
        done, _ = dynamics.step(int(decision.actions[0]))
        # A fresh node-selection decision refreshes the objects.
        if not done and dynamics.pending == DecisionKind.NODE_SELECTION:
            assert dynamics.nodesel_rule.nodes is not None
        dynamics.close()

    def test_joint_control_solves_optimal_with_both_decisions(self):
        # A branching instance: the policy drives BOTH the node selection
        # (always the first open node) and the branching (middle candidate).
        dynamics = BranchingNodeSelectionDynamics()
        generator = SetCoverGenerator(rng=np.random.default_rng(7))
        kinds, status = None, None
        for _draw in range(24):
            model = generator.generate_instance(n_rows=120, n_cols=150, density=0.10)
            model.hideOutput()
            kinds, status = _run_joint_episode(dynamics, model, node_choice=lambda n: 0, branch_choice=lambda n: n // 2)
            if DecisionKind.BRANCHING in kinds:
                break
        assert DecisionKind.BRANCHING in kinds, "no drawn instance branched"
        assert status == "optimal"
        # The node selection opens the episode; both decision kinds occur.
        assert kinds[0] == DecisionKind.NODE_SELECTION
        # Every branching consultation applies its decision (no extra actions).
        assert dynamics.counters.decisions == dynamics.counters.nodes
        assert dynamics.counters.nodes_selected >= dynamics.counters.nodes
        # The last-branching bookkeeping records a creating decision deeper
        # than the root on a branching instance.
        assert dynamics.last_branching_decision is not None
        assert dynamics.infeasible_nodes or dynamics.feasible_nodes
        dynamics.close()

    def test_consecutive_node_selections_when_no_branching(self):
        # An instance solved at the root still consults the node selection:
        # nodes are selected, processed, and fathomed without any branching.
        dynamics = BranchingNodeSelectionDynamics()
        generator = SetCoverGenerator(rng=np.random.default_rng(7))
        kinds, status = None, None
        for _draw in range(24):
            model = generator.generate_instance(n_rows=60, n_cols=150, density=0.10)
            model.hideOutput()
            kinds, status = _run_joint_episode(dynamics, model, node_choice=lambda n: 0, branch_choice=lambda n: 0)
            if DecisionKind.BRANCHING not in kinds and len(kinds) >= 1:
                break
        assert kinds is not None
        assert DecisionKind.BRANCHING not in kinds
        assert status == "optimal"
        assert dynamics.counters.nodes_selected >= 1
        assert dynamics.counters.decisions == 0
        dynamics.close()

    def test_rejects_actions_outside_the_set(self):
        dynamics = BranchingNodeSelectionDynamics()
        generator = SetCoverGenerator(rng=np.random.default_rng(11))
        model = generator.generate_instance(n_rows=120, n_cols=150, density=0.10)
        model.hideOutput()
        _done, decision = dynamics.reset(model)
        assert decision is not None
        bad = int(max(decision.actions)) + 1000
        with pytest.raises(ValueError, match="not in action set"):
            dynamics.step(bad)
        dynamics.close()

    def test_step_before_reset_raises(self):
        dynamics = BranchingNodeSelectionDynamics()
        with pytest.raises(RuntimeError, match="Call reset"):
            dynamics.step(0)

    def test_close_mid_solve_does_not_crash(self):
        # The die route: interrupt the solve between decisions. The Nodesel
        # must hand SCIP a real node (NULL after an interrupt segfaults) and
        # both plugins must ignore the stale action that _stop_thread()'s
        # action_event wakes them with.
        dynamics = BranchingNodeSelectionDynamics()
        generator = SetCoverGenerator(rng=np.random.default_rng(7))
        interrupted = 0
        for _draw in range(8):
            model = generator.generate_instance(n_rows=200, n_cols=250, density=0.10)
            model.hideOutput()
            done, decision = dynamics.reset(model)
            steps = 0
            while not done and steps < 5:
                assert decision is not None
                action = int(decision.actions[0])
                done, decision = dynamics.step(action)
                steps += 1
            if not done:
                dynamics.close()  # mid-solve interruption
                interrupted += 1
        assert interrupted >= 1, "no episode was long enough to interrupt"
        # The dynamics stays usable after the interruptions.
        model = generator.generate_instance(n_rows=60, n_cols=150, density=0.10)
        model.hideOutput()
        done, decision = dynamics.reset(model)
        assert decision is not None
        dynamics.close()


class TestEnvironmentIntegration:
    def test_environment_runs_the_joint_episode(self):
        # The Environment contract: the Decision reaches the policy as the
        # action set, the status information flows back, the runner's
        # counters access works.
        env = Environment(
            _instances(8),
            information_function=SolveStatus(),
            scip_params={"display/verblevel": 0},
            dynamics=BranchingNodeSelectionDynamics(),
        )
        env.dynamics.seed(0)
        obs, action_set, _reward, done, info = env.reset()
        kinds = []
        while not done:
            kinds.append(action_set.kind)
            obs, action_set, _reward, done, info = env.step(int(action_set.actions[0]))
        assert kinds[0] == DecisionKind.NODE_SELECTION
        assert info.get("solved") is True
        counters = env.dynamics.counters.as_dict()
        assert counters["nodes_selected"] == len(kinds)
        env.close()
