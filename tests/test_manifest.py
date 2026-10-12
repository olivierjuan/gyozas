import pytest

from gyozas.manifest import InstanceManifest, file_hash


@pytest.fixture
def files(tmp_path):
    d = tmp_path / "instances"
    d.mkdir()
    paths = {}
    for strate, split, name, content in [
        ("synthetic", "train", "setcover_0", "a" * 100),
        ("synthetic", "train", "setcover_1", "b" * 100),
        ("synthetic", "test", "setcover_0", "c" * 100),
        ("miplib-s", "test", "29iv", "d" * 100),
    ]:
        p = d / f"{name}_{split}.lp"
        p.write_text(content)
        paths[(strate, split, name)] = p
    return paths


def test_file_hash_is_sha256(files):
    import hashlib

    path = next(iter(files.values()))
    assert file_hash(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_counts_per_strate_split(files):
    manifest = InstanceManifest.build(files, provenance={"source": "unit test"})
    assert manifest.counts == {"synthetic/train": 2, "synthetic/test": 1, "miplib-s/test": 1}
    assert manifest.provenance == {"source": "unit test"}


def test_hash_is_deterministic(files):
    m1 = InstanceManifest.build(files)
    m2 = InstanceManifest.build(files)
    assert m1.hash == m2.hash
    assert m1 == m2


def test_hash_covers_provenance(files):
    with_prov = InstanceManifest.build(files, provenance={"version": 1})
    without = InstanceManifest.build(files)
    assert with_prov.hash != without.hash


def test_verify_detects_tampering(files, tmp_path):
    manifest = InstanceManifest.build(files)
    tampered = dict(files)
    key = ("synthetic", "test", "setcover_0")
    tampered[key].write_text("modified contents, same path")
    problems = manifest.verify(tampered)
    assert len(problems) == 1
    assert "hash mismatch" in problems[0]
    assert "setcover_0" in problems[0]


def test_verify_detects_unknown_instance(files):
    manifest = InstanceManifest.build({k: v for k, v in files.items() if k[0] == "synthetic"})
    extra = files[("miplib-s", "test", "29iv")]
    problems = manifest.verify({("miplib-s", "test", "29iv"): extra})
    assert problems == ["miplib-s/test/29iv: not in manifest"]


def test_verify_clean(files):
    manifest = InstanceManifest.build(files)
    assert manifest.verify(files) == []


def test_save_load_roundtrip(tmp_path, files):
    manifest = InstanceManifest.build(files, provenance={"origin": "eco"})
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = InstanceManifest.load(path)
    assert loaded == manifest
    assert loaded.hash == manifest.hash
