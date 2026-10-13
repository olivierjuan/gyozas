"""Tests for the signed protocol snapshot (R8.2)."""

import json
from pathlib import Path

import pytest

from gyozas.protocol import ProtocolSnapshot
from gyozas.run_identity import RunIdentity
from gyozas.seeds import SeedTriplet
from gyozas.solver_config import SolverConfig
from gyozas.strates import build_strate


@pytest.fixture
def strate_dir(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "micro"
    split = root / "train"
    split.mkdir(parents=True)
    (split / "tiny.lp").write_text("Minimize\n obj: x0\nSubject To\n c0: x0 >= 0\nBinary\n x0\nEnd\n")
    build_strate(root, provenance={"source": "test"})
    return root


@pytest.fixture
def snapshot(tmp_path: Path, strate_dir: Path) -> ProtocolSnapshot:
    snap = ProtocolSnapshot.capture("G2-v0", {"micro": strate_dir}, provenance={"date": "2026-10-13"})
    snap.save(tmp_path / "protocol.json")
    return ProtocolSnapshot.load(tmp_path / "protocol.json")


class TestProtocolSnapshot:
    def test_capture_records_the_frozen_content(self, tmp_path, strate_dir):
        snap = ProtocolSnapshot.capture("G2-v0", {"micro": strate_dir})
        manifest_hash = json.loads((strate_dir / "manifest.json").read_text())["hash"]
        assert snap.strates == {"micro": manifest_hash}
        assert snap.solver_profiles["default"] == {"name": "default", "params": {}}
        assert snap.name == "G2-v0"
        assert len(snap.hash) == 64

    def test_save_load_roundtrip_and_signature_check(self, snapshot):
        assert snapshot.name == "G2-v0"
        assert len(snapshot.hash) == 64

    def test_tampered_snapshot_is_rejected(self, tmp_path, snapshot):
        path = tmp_path / "protocol.json"
        data = json.loads(path.read_text())
        data["strates"]["micro"] = "0" * 64
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="modified after it was signed"):
            ProtocolSnapshot.load(path)

    def test_capture_requires_a_signed_manifest(self, tmp_path):
        empty = tmp_path / "data" / "empty"
        empty.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="manifest"):
            ProtocolSnapshot.capture("G2-v0", {"empty": empty})

    def test_check_run_accepts_a_frozen_run(self, snapshot, strate_dir):
        manifest_hash = json.loads((strate_dir / "manifest.json").read_text())["hash"]
        run = RunIdentity.capture(
            method="m:micro/train",
            solver_config=SolverConfig(name="default"),
            seeds=SeedTriplet.from_base_seed(0),
            manifest=None,
        )
        # scip/pyscipopt/gyozas versions are read live, hence identical to the snapshot's.
        assert run.scip_version == snapshot.scip_version
        run = RunIdentity(
            method=run.method,
            solver_config=run.solver_config,
            seeds=run.seeds,
            scip_version=snapshot.scip_version,
            pyscipopt_version=snapshot.pyscipopt_version,
            gyozas_version=snapshot.gyozas_version,
            manifest_hash=manifest_hash,
        )
        assert snapshot.check_run(run) == []

    def test_check_run_flags_a_frozen_dataset_mismatch(self, snapshot):
        run = RunIdentity(
            method="m:micro/train",
            solver_config={},
            seeds={},
            scip_version=snapshot.scip_version,
            pyscipopt_version=snapshot.pyscipopt_version,
            gyozas_version=snapshot.gyozas_version,
            manifest_hash="0" * 64,
        )
        problems = snapshot.check_run(run)
        assert any("not frozen" in p for p in problems)

    def test_check_run_flags_version_drift(self, snapshot, strate_dir):
        manifest_hash = json.loads((strate_dir / "manifest.json").read_text())["hash"]
        run = RunIdentity(
            method="m:micro/train",
            solver_config={},
            seeds={},
            scip_version="9.9.9",
            pyscipopt_version=snapshot.pyscipopt_version,
            gyozas_version=snapshot.gyozas_version,
            manifest_hash=manifest_hash,
        )
        problems = snapshot.check_run(run)
        assert any("scip_version" in p for p in problems)

    def test_check_run_flags_an_unanchored_run(self, snapshot):
        run = RunIdentity(
            method="m",
            solver_config={},
            seeds={},
            scip_version=snapshot.scip_version,
            pyscipopt_version=snapshot.pyscipopt_version,
            gyozas_version=snapshot.gyozas_version,
            manifest_hash="",
        )
        assert any("no manifest hash" in p for p in snapshot.check_run(run))
