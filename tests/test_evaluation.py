"""Tests for the evaluation helper: policy vs classic baselines (R6.1)."""

import pytest

from gyozas.evaluation import evaluate
from gyozas.experts import RelpscostExpert
from gyozas.policy import CallablePolicy

INSTANCE = {"setcover300": "tests/instance.lp"}
BUDGET = {"limits/nodes": 20, "display/verblevel": 0}
SEEDS = [0, 1, 2, 3, 4]


def worst_policy(obs, action_set):
    return action_set[-1]


class TestEvaluate:
    def test_default_baseline_is_implicit_and_is_the_unit(self):
        result = evaluate({"default": None}, INSTANCE, SEEDS, scip_params=BUDGET)
        assert result["baseline"] == "default"
        assert result["sgm"]["default"] == 1.0
        assert result["ratios"]["default"] == {"setcover300": 1.0}

    def test_fewer_than_min_seeds_is_rejected(self):
        with pytest.raises(ValueError, match="minimum"):
            evaluate({"default": None}, INSTANCE, [1, 2, 3], scip_params=BUDGET)

    def test_explicit_baseline_must_be_the_default_solver(self):
        with pytest.raises(ValueError, match="baseline"):
            evaluate({"default": worst_policy}, INSTANCE, SEEDS, scip_params=BUDGET)

    def test_ratios_are_paired_per_instance_against_the_baseline(self):
        result = evaluate({"worst": CallablePolicy(worst_policy)}, INSTANCE, SEEDS, scip_params=BUDGET)
        assert result["sgm"]["default"] == 1.0
        ratios = result["ratios"]["worst"]
        assert list(ratios) == ["setcover300"]  # paired on the common instance
        nodes = result["nodes"]
        assert nodes["worst"]["setcover300"] == pytest.approx(
            ratios["setcover300"] * nodes["default"]["setcover300"]
        )
        assert result["per_seed"]["default"]["0"]["setcover300"]["nodes"] >= 0
        assert result["summary"]["default"]["n_instances"] == 1

    def test_expert_runs_as_a_method(self):
        result = evaluate(
            {"relpscost": RelpscostExpert(), "worst": CallablePolicy(worst_policy)},
            INSTANCE,
            SEEDS,
            scip_params=BUDGET,
        )
        assert set(result["nodes"]) == {"default", "relpscost", "worst"}
        for seed_runs in result["per_seed"]["relpscost"].values():
            run = seed_runs["setcover300"]
            assert set(run) >= {"nodes", "decisions", "status", "solved", "truncated", "limit"}
            assert run["status"] in {"optimal", "nodelimit"}

    def test_statuses_are_reported_per_seed_instance(self):
        result = evaluate({"default": None}, INSTANCE, SEEDS, scip_params=BUDGET)
        assert set(result["statuses"]["default"]) == {f"{s}/setcover300" for s in SEEDS}
        assert result["summary"]["default"]["solved_vs_truncated"]["n_solved"] >= 0
