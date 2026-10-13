"""Tests for the config-driven runner (R5.1, R5.2, R5.3, R8.3)."""

import json
from pathlib import Path

import pytest

from gyozas.experts import RelpscostExpert
from gyozas.fidelity import FidelityProfile, get_fidelity_profile, register_fidelity_profile
from gyozas.policy import CallablePolicy
from gyozas.runner import MIN_SEEDS, Cell, ExperimentConfig, Method, Runner
from gyozas.strates import build_strate

MICRO_LP = """Minimize
 obj: x0 + x1
Subject To
 c0: x0 + x1 >= 1
Binary
 x0
 x1
End
"""


@pytest.fixture
def micro_strate(tmp_path: Path) -> Path:
    """A two-instance strate with its signed manifest."""
    root = tmp_path / "data" / "micro"
    split = root / "train"
    split.mkdir(parents=True)
    for name in ("tiny_a", "tiny_b"):
        (split / f"{name}.lp").write_text(MICRO_LP)
    build_strate(root, provenance={"source": "synthetic micro test strate"})
    return root


def make_runner(tmp_path: Path, micro_strate: Path, seeds=None, method_name="first", budget=None) -> Runner:
    seeds = list(range(5)) if seeds is None else seeds
    config = ExperimentConfig(
        cells=[
            Cell(
                method=method_name,
                strate="micro",
                split="train",
                seeds=seeds,
                profile="default",
                budget=budget or {"limits/nodes": 20, "display/verblevel": 0},
            )
        ]
    )
    methods = {
        method_name: Method(
            name=method_name,
            policy=lambda: CallablePolicy(lambda obs, action_set: action_set[0]),
            config={"kind": "test"},
        )
    }
    return Runner(config, methods, {"micro": micro_strate}, tmp_path / "results")


class TestRunner:
    def test_run_executes_every_cell_run_and_records_the_identity(self, tmp_path, micro_strate):
        runner = make_runner(tmp_path, micro_strate)
        report = runner.run()
        assert report == {"first/micro/train": {"executed": 10, "skipped": 0}}
        run_dirs = sorted((tmp_path / "results" / "first" / "micro" / "train").iterdir())
        assert [d.name for d in run_dirs] == [str(s) for s in range(5)]
        records = sorted(run_dirs[0].glob("*.json"))
        assert [r.stem for r in records] == ["tiny_a", "tiny_b"]
        record = json.loads(records[0].read_text())
        assert record["complete"] is True
        identity = record["run_identity"]
        assert identity["method"] == "first:micro/train"
        assert identity["scip_version"]
        assert identity["seeds"] == record["seed"]["triplet"]
        assert len(identity["manifest_hash"]) == 64

    def test_run_records_the_strate_manifest_hash(self, tmp_path, micro_strate):
        manifest_hash = json.loads((micro_strate / "manifest.json").read_text())["hash"]
        runner = make_runner(tmp_path, micro_strate)
        runner.run()
        record = json.loads(next((tmp_path / "results").rglob("tiny_a.json")).read_text())
        assert record["run_identity"]["manifest_hash"] == manifest_hash

    def test_rerun_is_a_no_op(self, tmp_path, micro_strate):
        runner = make_runner(tmp_path, micro_strate)
        runner.run()
        record_path = next((tmp_path / "results").rglob("tiny_a.json"))
        record_path.write_text(json.dumps({**json.loads(record_path.read_text()), "complete": True, "tampered": 1}))
        report = runner.run()
        assert report == {"first/micro/train": {"executed": 0, "skipped": 10}}
        assert json.loads(record_path.read_text())["tampered"] == 1

    def test_interrupted_run_resumes(self, tmp_path, micro_strate):
        runner = make_runner(tmp_path, micro_strate)
        runner.run()
        # A crashed run leaves an incomplete record: it is re-executed.
        record_path = next((tmp_path / "results").rglob("tiny_b.json"))
        record_path.write_text(json.dumps({"complete": False}))
        # A lost record (deleted mid-campaign) is re-executed too.
        (record_path.parent / "tiny_a.json").unlink()
        report = runner.run()
        assert report == {"first/micro/train": {"executed": 2, "skipped": 8}}
        assert json.loads(record_path.read_text())["complete"] is True

    def test_incomplete_record_is_not_counted_complete(self, tmp_path, micro_strate):
        runner = make_runner(tmp_path, micro_strate)
        runner.run()
        record_path = next((tmp_path / "results").rglob("tiny_b.json"))
        record_path.write_text(json.dumps({"complete": False}))
        status = runner.cell_status(runner.config.cells[0])
        assert status["complete"] is False
        assert status["seeds"]["0"] == 1

    def test_cell_results_are_stored_per_run(self, tmp_path, micro_strate):
        runner = make_runner(tmp_path, micro_strate)
        runner.run()
        record = json.loads(next((tmp_path / "results").rglob("tiny_a.json")).read_text())
        assert set(record["result"]) >= {"status", "solved", "truncated", "limit", "total_reward", "steps"}
        assert record["cell"]["budget"]["limits/nodes"] == 20
        assert record["cell"]["method_config"] == {"kind": "test"}

    def test_experiment_config_is_saved_with_the_results(self, tmp_path, micro_strate):
        runner = make_runner(tmp_path, micro_strate)
        runner.run()
        saved = json.loads((tmp_path / "results" / "experiment_config.json").read_text())
        assert saved == runner.config.to_dict()

    def test_a_different_config_cannot_reuse_the_results_dir(self, tmp_path, micro_strate):
        make_runner(tmp_path, micro_strate).run()
        other = ExperimentConfig(
            cells=[Cell(method="other", strate="micro", split="train", seeds=[1, 2, 3, 4, 5])]
        )
        with pytest.raises(ValueError, match="different experiment"):
            Runner(other, {}, {"micro": micro_strate}, tmp_path / "results")


class TestPublishGuard:
    def test_min_seeds_is_five(self):
        assert MIN_SEEDS == 5

    def test_cell_with_too_few_seeds_is_not_publishable(self, tmp_path, micro_strate):
        runner = make_runner(tmp_path, micro_strate, seeds=[1, 2])
        ok, reason = runner.is_publishable(runner.config.cells[0])
        assert ok is False
        assert "5" in reason
        with pytest.raises(ValueError, match="not publishable"):
            runner.collect_cell(runner.config.cells[0])

    def test_cell_with_missing_records_is_not_publishable(self, tmp_path, micro_strate):
        runner = make_runner(tmp_path, micro_strate)
        runner.run()
        for path in (tmp_path / "results").rglob("tiny_b.json"):
            path.unlink()
        ok, reason = runner.is_publishable(runner.config.cells[0])
        assert ok is False
        assert "incomplete" in reason
        with pytest.raises(ValueError, match="not publishable"):
            runner.collect_cell(runner.config.cells[0])

    def test_complete_cell_is_publishable_and_collectable(self, tmp_path, micro_strate):
        runner = make_runner(tmp_path, micro_strate)
        runner.run()
        ok, _ = runner.is_publishable(runner.config.cells[0])
        assert ok
        results = runner.collect_cell(runner.config.cells[0])
        assert set(results) == {"0", "1", "2", "3", "4"}
        assert set(results["0"]) == {"tiny_a", "tiny_b"}
        assert "solved" in results["0"]["tiny_a"]


class TestFidelityProfiles:
    def test_profile_recharges_observation_params_and_augmentation(self, tmp_path, micro_strate):
        calls = []

        def observation():
            calls.append("obs")
            return None

        profile = FidelityProfile(
            name="fake",
            observation=observation,
            scip_params={"display/verblevel": 0},
            training={"loss": "fake", "schedule": "none"},
        )
        register_fidelity_profile(profile)
        config = ExperimentConfig(
            cells=[Cell(method="f", strate="micro", split="train", seeds=list(range(5)), budget={"limits/nodes": 20})]
        )
        runner = Runner(
            config,
            {"f": Method(name="f", policy=lambda: CallablePolicy(lambda obs, a: a[0]), fidelity="fake")},
            {"micro": micro_strate},
            tmp_path / "results-fidelity",
        )
        runner.run()
        assert calls  # the profile's observation factory was used
        record = json.loads(next((tmp_path / "results-fidelity").rglob("tiny_a.json")).read_text())
        assert record["cell"]["fidelity"] == "fake"
        assert record["cell"]["budget"]["limits/nodes"] == 20

    def test_profile_expert_drives_the_episode(self, tmp_path, micro_strate):
        register_fidelity_profile(FidelityProfile(name="relpscost", expert=RelpscostExpert))
        config = ExperimentConfig(
            cells=[Cell(method="e", strate="micro", split="train", seeds=list(range(5)), budget={"limits/nodes": 20})]
        )
        runner = Runner(
            config,
            {"e": Method(name="e", fidelity="relpscost", config={"kind": "expert"})},
            {"micro": micro_strate},
            tmp_path / "results-expert",
        )
        runner.run()
        record = json.loads(next((tmp_path / "results-expert").rglob("tiny_a.json")).read_text())
        assert record["complete"] is True
        assert record["result"]["steps"] >= 0

    def test_method_without_policy_nor_expert_is_rejected(self, tmp_path, micro_strate):
        config = ExperimentConfig(
            cells=[Cell(method="x", strate="micro", split="train", seeds=list(range(5)))]
        )
        runner = Runner(
            config,
            {"x": Method(name="x")},
            {"micro": micro_strate},
            tmp_path / "results-none",
        )
        with pytest.raises(ValueError, match="neither a policy"):
            runner.run()

    def test_unknown_fidelity_profile_is_rejected(self):
        with pytest.raises(KeyError, match="nope"):
            get_fidelity_profile("nope")
