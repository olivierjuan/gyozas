"""Tests for the SolverStats observation (R1.2b — solver-stats view)."""

import pytest

from gyozas.environment import Environment
from gyozas.instances.files import FileGenerator
from gyozas.observations import SolverStats


@pytest.fixture
def env():
    env = Environment(
        instance_generator=FileGenerator(directory="tests", pattern="*.lp"),
        observation_function=SolverStats(),
        scip_params={"display/verblevel": 0, "limits/nodes": 50},
    )
    yield env
    env.close()


EXPECTED_TREE_KEYS = {
    "primal_bound",
    "dual_bound",
    "gap",
    "dualbound_root",
    "lowerbound_root",
    "avg_dualbound",
    "avg_lowerbound",
    "first_primal_bound",
    "cutoff_bound",
    "n_nodes",
    "n_total_nodes",
    "n_nodes_left",
    "max_depth",
    "max_total_depth",
    "plunge_depth",
    "n_lps",
    "n_lp_iterations",
    "n_node_lp_iterations",
    "n_sb_lp_iterations",
    "n_lp_cols",
    "n_lp_rows",
    "n_cuts",
    "n_cuts_applied",
    "n_sepa_rounds",
    "n_sols_found",
    "n_lim_sols_found",
    "n_counted_sols",
    "solving_time",
    "presolving_time",
    "deterministic_time",
}

EXPECTED_NODE_KEYS = {"node_number", "node_depth", "node_lowerbound"}


class TestSolverStats:
    def test_returns_dict_with_all_keys(self, env):
        obs, action_set, reward, done, info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        assert isinstance(obs, dict)
        assert set(obs.keys()) == EXPECTED_TREE_KEYS | EXPECTED_NODE_KEYS
        assert 25 <= len(obs) <= 100

    def test_values_are_finite_floats(self, env):
        obs, action_set, reward, done, info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        for key, value in obs.items():
            assert isinstance(value, float), key
            assert value == value, f"NaN in {key}"

    def test_node_and_tree_levels_present(self, env):
        obs, action_set, reward, done, info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        assert obs["node_depth"] == 0
        assert obs["node_number"] > 0
        assert obs["solving_time"] >= 0.0

    def test_step_updates_tree_counters(self, env):
        obs, action_set, reward, done, info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        n_nodes_before = obs["n_nodes"]
        obs2, action_set2, reward, done, info = env.step(action_set[len(action_set) // 2])
        if done:
            return
        assert obs2["n_nodes"] >= n_nodes_before

    def test_returns_none_when_done(self, env):
        obs, action_set, reward, done, info = env.reset()
        while not done:
            obs, action_set, reward, done, info = env.step(action_set[len(action_set) // 2])
        assert obs is None

    def test_reset_resets_state(self, env):
        obs, action_set, reward, done, info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        obs2, action_set2, reward, done, info = env.reset()
        if done:
            pytest.skip("Instance solved during presolve")
        assert obs2["node_depth"] == 0
        assert obs2["node_number"] > 0
