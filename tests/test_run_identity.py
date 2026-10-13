import json

import pytest

from gyozas.manifest import InstanceManifest
from gyozas.run_identity import RunIdentity
from gyozas.seeds import SeedTriplet
from gyozas.solver_config import get_profile


@pytest.fixture
def manifest(tmp_path):
    instance = tmp_path / "i.lp"
    instance.write_text("Maximize\n x\nSubject To\n x <= 1\nEnd\n")
    return InstanceManifest.build({("synthetic", "train", "i"): instance})


def test_capture_reads_live_versions(manifest):
    identity = RunIdentity.capture("relpscost", get_profile("lin2022"), SeedTriplet.from_base_seed(5), manifest)
    import pyscipopt

    assert identity.scip_version
    assert identity.pyscipopt_version == pyscipopt.__version__
    assert identity.gyozas_version
    assert identity.manifest_hash == manifest.hash


def test_solver_config_and_seeds_are_captured(manifest):
    config = get_profile("lin2022")
    seeds = SeedTriplet.from_base_seed(5)
    identity = RunIdentity.capture("relpscost", config, seeds, manifest)
    assert identity.method == "relpscost"
    assert identity.solver_config == config.to_dict()
    assert identity.seeds == seeds.to_dict()


def test_dict_roundtrip(manifest):
    identity = RunIdentity.capture("relpscost", get_profile("default"), SeedTriplet.from_base_seed(0), manifest)
    assert RunIdentity.from_dict(identity.to_dict()) == identity


def test_save_load_roundtrip(tmp_path, manifest):
    identity = RunIdentity.capture("relpscost", get_profile("default"), SeedTriplet.from_base_seed(0), manifest)
    path = tmp_path / "identity.json"
    identity.save(path)
    assert json.loads(path.read_text())["manifest_hash"] == identity.manifest_hash
    assert RunIdentity.load(path) == identity


def test_identity_distinguishes_manifests(tmp_path):
    a = tmp_path / "a.lp"
    a.write_text("Maximize\n x\nSubject To\n x <= 1\nEnd\n")
    b = tmp_path / "b.lp"
    b.write_text("Maximize\n x\nSubject To\n x <= 2\nEnd\n")
    seeds = SeedTriplet.from_base_seed(0)
    i1 = RunIdentity.capture("m", get_profile("default"), seeds, InstanceManifest.build({("s", "t", "a"): a}))
    i2 = RunIdentity.capture("m", get_profile("default"), seeds, InstanceManifest.build({("s", "t", "a"): b}))
    assert i1.manifest_hash != i2.manifest_hash
