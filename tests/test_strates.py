"""Tests for strate directories with embedded, signed splits (R3.1/R3.2)."""

import pytest

from gyozas.strates import Strate, build_strate, discover_files


@pytest.fixture
def strate_root(tmp_path):
    # Two splits, one instance each, like data/miplib_s on disk.
    (tmp_path / "demo_strate" / "train").mkdir(parents=True)
    (tmp_path / "demo_strate" / "test").mkdir(parents=True)
    (tmp_path / "demo_strate" / "train" / "alpha.mps.gz").write_bytes(b"alpha instance")
    (tmp_path / "demo_strate" / "test" / "beta.mps").write_bytes(b"beta instance")
    return tmp_path / "demo_strate"


class TestDiscoverFiles:
    def test_keys_come_from_directory_layout(self, strate_root):
        files = discover_files(strate_root)
        assert set(files) == {
            ("demo_strate", "train", "alpha"),
            ("demo_strate", "test", "beta"),
        }

    def test_ignores_non_instance_files(self, strate_root):
        (strate_root / "train" / "notes.txt").write_text("not an instance")
        (strate_root / "manifest.json").write_text("{}")
        files = discover_files(strate_root)
        assert set(files) == {
            ("demo_strate", "train", "alpha"),
            ("demo_strate", "test", "beta"),
        }


class TestStrate:
    def test_build_signs_manifest_with_provenance(self, strate_root):
        manifest = build_strate(strate_root, provenance={"source": "somewhere"})
        assert manifest.counts == {"demo_strate/train": 1, "demo_strate/test": 1}
        assert manifest.provenance == {"source": "somewhere"}
        assert (strate_root / "manifest.json").exists()

    def test_load_roundtrip_and_split_access(self, strate_root):
        build_strate(strate_root, provenance={})
        strate = Strate.load(strate_root)
        assert strate.splits() == {"train": ["alpha"], "test": ["beta"]}
        assert len(strate.manifest.entries) == 2

    def test_verify_clean_then_detect_alteration(self, strate_root):
        build_strate(strate_root, provenance={})
        strate = Strate.load(strate_root)
        assert strate.verify() == []
        (strate_root / "test" / "beta.mps").write_bytes(b"tampered")
        problems = strate.verify()
        assert len(problems) == 1
        assert "hash mismatch" in problems[0]

    def test_verify_reports_missing_file(self, strate_root):
        build_strate(strate_root, provenance={})
        strate = Strate.load(strate_root)
        (strate_root / "test" / "beta.mps").unlink()
        problems = strate.verify()
        assert problems == ["demo_strate/test/beta: in manifest but no file provided"]

    def test_extra_file_on_disk_is_reported(self, strate_root):
        build_strate(strate_root, provenance={})
        strate = Strate.load(strate_root)
        (strate_root / "test" / "gamma.mps.gz").write_bytes(b"new instance")
        problems = strate.verify()
        assert problems == ["demo_strate/test/gamma: not in manifest"]

    def test_loaded_manifest_rejects_tampering(self, strate_root):
        build_strate(strate_root, provenance={})
        manifest = Strate.load(strate_root).manifest
        d = manifest.to_dict()
        d["entries"]["demo_strate/train/alpha"]["sha256"] = "0" * 64
        import json

        (strate_root / "manifest.json").write_text(json.dumps(d))
        with pytest.raises(ValueError, match="hash mismatch"):
            Strate.load(strate_root)
