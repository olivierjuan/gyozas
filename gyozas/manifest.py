import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Instance keys are (strate, split, name).
InstanceKey = tuple[str, str, str]


def file_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


@dataclass
class InstanceManifest:
    """The frozen unit of a benchmark protocol: instances, hashes, counts, provenance.

    Each instance is hashed (SHA-256) so a run's data can be identified
    byte-for-byte; the manifest itself carries a content hash, its *signature*:
    after the protocol freeze (G2), any change to any instance — or to the
    manifest — changes the hash and is thereby detected. The manifest is what
    makes a results table refer to a fixed dataset rather than to a dataset
    name.

    Instances are addressed by ``(strate, split, name)``; splits are embedded
    data, never rebuilt by the harness.
    """

    entries: dict[str, dict[str, Any]]
    counts: dict[str, int]
    provenance: dict[str, Any]
    hash: str

    @classmethod
    def build(
        cls,
        files: dict[InstanceKey, str | Path],
        provenance: dict[str, Any] | None = None,
    ) -> "InstanceManifest":
        """Hash each instance file and assemble the manifest."""
        entries: dict[str, dict[str, Any]] = {}
        for key in sorted(files):
            strate, split, name = key
            path = Path(files[key])
            entries[f"{strate}/{split}/{name}"] = {
                "strate": strate,
                "split": split,
                "name": name,
                "sha256": file_hash(path),
                "size": path.stat().st_size,
            }
        counts: dict[str, int] = {}
        for entry in entries.values():
            group = f"{entry['strate']}/{entry['split']}"
            counts[group] = counts.get(group, 0) + 1
        manifest = cls(entries=entries, counts=counts, provenance=provenance or {}, hash="")
        manifest.hash = _manifest_hash(manifest.to_dict())
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "InstanceManifest":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": self.entries, "counts": self.counts, "provenance": self.provenance, "hash": self.hash}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstanceManifest":
        return cls(
            entries=dict(data["entries"]),
            counts=dict(data["counts"]),
            provenance=dict(data.get("provenance", {})),
            hash=data.get("hash", ""),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, InstanceManifest) and self.to_dict() == other.to_dict()

    def verify(self, files: dict[InstanceKey, str | Path]) -> list[str]:
        """Re-hash the given files and report every mismatch or missing entry."""
        problems = []
        for key in sorted(files):
            strate, split, name = key
            entry_key = f"{strate}/{split}/{name}"
            entry = self.entries.get(entry_key)
            if entry is None:
                problems.append(f"{entry_key}: not in manifest")
                continue
            actual = file_hash(files[key])
            if actual != entry["sha256"]:
                problems.append(f"{entry_key}: hash mismatch (expected {entry['sha256']}, got {actual})")
        return problems


def _manifest_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(data).encode()).hexdigest()
