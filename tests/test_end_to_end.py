"""End-to-end micro-grid: the harness's regression detector (R9.2).

A tiny family, a one-line policy, five seeds, a few seconds — but through the
whole pipeline: strate with signed manifest → config-driven runner → records
with full run identity → publish guard → summary table exported as JSON+CSV.
"""

import json
from pathlib import Path

import pytest

from gyozas.fidelity import FidelityProfile, register_fidelity_profile
from gyozas.metrics import save_table, summary_table
from gyozas.policy import CallablePolicy
from gyozas.runner import Cell, ExperimentConfig, Method, Runner
from gyozas.strates import build_strate

MICRO_LP = """Minimize
 obj: x0 + x1 + x2
Subject To
 c0: x0 + x1 + x2 >= 1
Binary
 x0
 x1
 x2
End
"""


@pytest.fixture
def micro_strate(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "micro"
    split = root / "train"
    split.mkdir(parents=True)
    for name in ("tiny_a", "tiny_b"):
        (split / f"{name}.lp").write_text(MICRO_LP)
    build_strate(root, provenance={"source": "micro e2e"})
    return root


def test_micro_grid_end_to_end(tmp_path, micro_strate):
    register_fidelity_profile(
        FidelityProfile(
            name="argmax",
            scip_params={"display/verblevel": 0},
            training={"loss": "none", "note": "test policy"},
        )
    )
    config = ExperimentConfig(
        cells=[Cell(method="first", strate="micro", split="train", seeds=[0, 1, 2, 3, 4], budget={"limits/nodes": 20})]
    )
    runner = Runner(
        config,
        {"first": Method(name="first", policy=lambda: CallablePolicy(lambda obs, a: a[0]), fidelity="argmax")},
        {"micro": micro_strate},
        tmp_path / "results",
    )

    # One seed runs in seconds: the smoke part of the grid.
    report = runner.run()
    assert report == {"first/micro/train": {"executed": 10, "skipped": 0}}

    # Idempotence: relaunching the same grid is a no-op.
    assert runner.run() == {"first/micro/train": {"executed": 0, "skipped": 10}}

    # The 5-seed guard: satisfied here.
    cell = runner.config.cells[0]
    ok, _ = runner.is_publishable(cell)
    assert ok

    # Records carry the full run identity, tied to the strate's manifest.
    manifest_hash = json.loads((micro_strate / "manifest.json").read_text())["hash"]
    record = json.loads((tmp_path / "results" / "first" / "micro" / "train" / "0" / "tiny_a.json").read_text())
    assert record["run_identity"]["manifest_hash"] == manifest_hash
    assert record["cell"]["fidelity"] == "argmax"
    results = runner.collect_cell(cell)

    # The single output table (R7.5), exported as JSON + CSV.
    per_instance = {
        seed: {name: run["counters"]["nodes"] for name, run in runs.items()} for seed, runs in results.items()
    }
    flat: dict[str, float] = {}
    for seed, runs in per_instance.items():
        for name, nodes in runs.items():
            flat[f"{seed}/{name}"] = nodes
    table = summary_table({"first": flat}, shift=1.0)  # raw node counts: zeros (presolve) absorbed by the shift
    save_table(table, tmp_path / "table.json", tmp_path / "table.csv")
    assert (tmp_path / "table.json").exists()
    assert (tmp_path / "table.csv").read_text().startswith("method,")
