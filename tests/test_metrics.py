"""Tests for benchmark metrics and statistics (R6.2, R6.3, R7.2-R7.5)."""

import csv
import json

import pytest

from gyozas.metrics import (
    catastrophic_losses,
    friedman_nemenyi,
    paired_per_instance_ratios,
    quantiles,
    save_table,
    sgm_of_ratios,
    shifted_geometric_mean,
    solved_vs_truncated,
    summary_table,
    wilcoxon_holm,
    wilcoxon_pair,
    win_rate,
)


class TestShiftedGeometricMean:
    def test_constant_values(self):
        assert shifted_geometric_mean([2, 2, 2]) == pytest.approx(2.0)

    def test_shift_absorbs_zeros_literature_formula(self):
        # Achterberg SGM_s = (prod(v + s))^(1/n) - s; a zero (solved in presolve) is absorbed.
        expected = (1.0 * 6.0 * 4.0) ** (1 / 3) - 1.0
        assert shifted_geometric_mean([0.0, 5.0, 3.0], shift=1.0) == pytest.approx(expected)

    def test_known_shifted_value(self):
        assert shifted_geometric_mean([1.0, 4.0], shift=1.0) == pytest.approx(10.0**0.5 - 1.0)

    def test_no_shift_by_default_preserves_ratios(self):
        assert shifted_geometric_mean([0.5, 2.0]) == pytest.approx(1.0)

    def test_known_value(self):
        assert shifted_geometric_mean([1, 4]) == pytest.approx(2.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            shifted_geometric_mean([])

    def test_non_positive_raises(self):
        with pytest.raises(ValueError):
            shifted_geometric_mean([1.0, 0.0])

    def test_too_negative_for_shift_raises(self):
        with pytest.raises(ValueError):
            shifted_geometric_mean([-2.0, 5.0], shift=1.0)


class TestPairedRatios:
    def test_paired_per_instance(self):
        method = {"a": 10.0, "b": 4.0}
        baseline = {"a": 20.0, "b": 8.0, "c": 1.0}
        ratios = paired_per_instance_ratios(method, baseline)
        assert ratios == {"a": 0.5, "b": 0.5}

    def test_zero_baseline_skipped(self):
        ratios = paired_per_instance_ratios({"a": 1.0}, {"a": 0.0})
        assert ratios == {}

    def test_sgm_of_ratios(self):
        method = {"a": 10.0, "b": 4.0}
        baseline = {"a": 20.0, "b": 8.0}
        assert sgm_of_ratios(method, baseline) == pytest.approx(0.5)


class TestQuantilesAndTails:
    def test_quantiles(self):
        values = list(range(1, 101))
        q = quantiles(values)
        assert q[0.5] == pytest.approx(50.5)
        assert q[0.95] == pytest.approx(95.05)

    def test_catastrophic_losses_named_and_sorted(self):
        per_instance = {"easy1": 10, "hard1": 1000, "hard2": 900, "mid": 100, "easy2": 5, "hard3": 800}
        worst = catastrophic_losses(per_instance, k=3)
        assert [name for name, _ in worst] == ["hard1", "hard2", "hard3"]
        assert worst[0][1] == 1000


class TestSolvedVsTruncated:
    def test_counts_and_limits(self):
        statuses = {"i1": "optimal", "i2": "timelimit", "i3": "nodelimit", "i4": "timelimit", "i5": "infeasible"}
        report = solved_vs_truncated(statuses)
        assert report["n_total"] == 5
        assert report["n_solved"] == 2
        assert report["n_truncated"] == 3
        assert report["truncated_by_limit"] == {"timelimit": 2, "nodelimit": 1}
        assert "i1" in report["solved_instances"]

    def test_gaplimit_counts_as_solved(self):
        report = solved_vs_truncated({"i1": "gaplimit"})
        assert report["n_solved"] == 1
        assert report["n_truncated"] == 0


class TestWinRate:
    def test_wins_losses_ties(self):
        method = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 5.0}
        baseline = {"a": 2.0, "b": 2.0, "c": 2.0, "d": 5.0 + 1e-15}
        report = win_rate(method, baseline)
        assert report["wins"] == 1  # a
        assert report["losses"] == 1  # c
        assert report["ties"] == 2  # b and d (relative tolerance)
        assert report["win_rate"] == pytest.approx(0.5)
        assert report["n_common"] == 4

    def test_no_common_instances(self):
        report = win_rate({"a": 1.0}, {"b": 1.0})
        assert report["n_common"] == 0
        assert report["win_rate"] != report["win_rate"]  # NaN


class TestFriedmanNemenyi:
    @staticmethod
    def _three_methods(n_instances=20, spread=1.0):
        instances = [f"i{t}" for t in range(n_instances)]
        return {
            "good": {i: 10.0 + 0.0 * t for t, i in enumerate(instances)},
            "mid": {i: 15.0 for i in instances},
            "bad": {i: 20.0 for i in instances},
        }, instances

    def test_clear_ordering_is_significant(self):
        per_method, _ = self._three_methods()
        result = friedman_nemenyi(per_method)
        assert result["friedman_p"] < 0.05
        assert result["average_ranks"]["good"] == pytest.approx(1.0)
        assert result["average_ranks"]["bad"] == pytest.approx(3.0)
        assert ("good", "bad") in [tuple(p) for p in result["significant_pairs"]]

    def test_identical_methods_not_significant(self):
        per_method, instances = self._three_methods()
        shared = {i: 10.0 + t for t, i in enumerate(instances)}
        per_method = {name: dict(shared) for name in per_method}
        result = friedman_nemenyi(per_method)
        assert result["friedman_p"] > 0.05
        assert result["significant_pairs"] == []

    def test_two_methods_raises(self):
        with pytest.raises(ValueError, match="three methods"):
            friedman_nemenyi({"a": {"i1": 1.0, "i2": 2.0}, "b": {"i1": 2.0, "i2": 1.0}})

    def test_nan_raises_instead_of_reading_as_no_difference(self):
        per_method = {
            "good": {"i1": 1.0, "i2": 1.0, "i3": 1.0},
            "mid": {"i1": 2.0, "i2": 2.0, "i3": 2.0},
            "bad": {"i1": 3.0, "i2": 3.0, "i3": float("nan")},
        }
        with pytest.raises(ValueError, match="finite"):
            friedman_nemenyi(per_method)

    def test_nemenyi_alpha_010_matches_exact_quantiles(self):
        # k=10, alpha=0.10: exact studentized-range quantile / sqrt(2) = 2.919889.
        from gyozas.metrics import _NEMENYI_Q

        assert _NEMENYI_Q[0.10][10] == pytest.approx(2.919889, abs=1e-5)
        assert _NEMENYI_Q[0.10][4] == pytest.approx(2.291341, abs=1e-5)


class TestWilcoxon:
    def test_one_sided_significant(self):
        baseline = {f"i{t}": 100.0 + t for t in range(20)}
        method = {i: v * 0.5 for i, v in baseline.items()}
        result = wilcoxon_pair(method, baseline)
        assert result["p_value"] < 0.05

    def test_identical_gives_p_one(self):
        values = {f"i{t}": float(t) for t in range(10)}
        assert wilcoxon_pair(values, dict(values))["p_value"] == 1.0

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            wilcoxon_pair({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": float("nan")})

    def test_holm_family(self):
        methods = {"ref": {f"i{t}": 100.0 for t in range(30)}}
        methods["better"] = {i: v * 0.5 for i, v in methods["ref"].items()}
        methods["worse"] = {i: v * 1.5 for i, v in methods["ref"].items()}
        details = wilcoxon_holm(methods, reference="ref")
        assert set(details) == {"better", "worse"}
        assert details["better"]["p_holm"] < 0.05
        assert details["better"]["significant_at_0.05"] is True
        # One-sided test: "worse" is not better than "ref", so not significant.
        assert details["worse"]["significant_at_0.05"] is False


class TestSummaryTable:
    def test_table_metrics(self):
        per_method = {
            "m1": {f"i{t}": 10.0 + t for t in range(10)},
            "m2": {f"i{t}": 20.0 + t for t in range(10)},
        }
        statuses = {"m1": {f"i{t}": "optimal" for t in range(10)}, "m2": {f"i{t}": "timelimit" for t in range(10)}}
        table = summary_table(per_method, statuses=statuses)
        entry = table["m1"]
        assert entry["n_instances"] == 10
        assert entry["sgm"] == pytest.approx(shifted_geometric_mean([10.0 + t for t in range(10)]))
        assert entry["p50"] < entry["p95"]
        assert set(entry["worst_instances"]) == {f"i{t}" for t in range(5, 10)}
        assert entry["solved_vs_truncated"]["n_solved"] == 10
        assert table["m2"]["solved_vs_truncated"]["n_truncated"] == 10

    def test_save_table_json_and_csv(self, tmp_path):
        per_method = {"m1": {"a": 1.0, "b": 2.0}, "m2": {"a": 3.0, "b": 4.0}}
        table = summary_table(per_method)
        path_json = str(tmp_path / "table.json")
        path_csv = str(tmp_path / "table.csv")
        save_table(table, path_json, path_csv)
        with open(path_json) as f:
            loaded = json.load(f)
        assert set(loaded) == {"m1", "m2"}
        with open(path_csv) as f:
            rows = list(csv.reader(f))
        assert rows[0][0] == "method"
        assert {row[0] for row in rows[1:]} == {"m1", "m2"}
