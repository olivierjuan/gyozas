"""Built-in benchmark strates: on-disk instance sets with frozen splits (R3.1/R3.2).

A strate is a directory laid out as::

    <root>/
        manifest.json      signed InstanceManifest (committed to the repo)
        <split>/           one subdirectory per split, e.g. train/ and test/
            <name>.mps.gz

The split assignment *is* the directory layout: splits come embedded with the
data and are never rebuilt by the harness (R3.2). The manifest pins every
instance byte-for-byte, so a strate is comparable across publications.
"""

from dataclasses import dataclass
from pathlib import Path

from .manifest import InstanceKey, InstanceManifest

# Instance files are compressed MPS files under each split directory.
_INSTANCE_SUFFIXES = {".mps", ".mps.gz"}


def _is_instance_file(path: Path) -> bool:
    if path.name.endswith(".mps.gz"):
        return True
    return path.suffix in _INSTANCE_SUFFIXES and not path.name.endswith(".gz")


def discover_files(root: str | Path) -> dict[InstanceKey, Path]:
    """Map every instance file under ``root`` to its ``(strate, split, name)`` key.

    The strate is the name of the root directory, the split the name of the
    immediate subdirectory, and the instance name the file name stripped of
    ``.mps.gz`` / ``.mps``.
    """
    root = Path(root)
    files: dict[InstanceKey, Path] = {}
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(split_dir.iterdir()):
            if not _is_instance_file(path):
                continue
            name = path.name
            for suffix in (".mps.gz", ".mps"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            files[(root.name, split_dir.name, name)] = path
    return files


def build_strate(root: str | Path, provenance: dict) -> InstanceManifest:
    """Hash the instances on disk and sign the strate's ``manifest.json``."""
    root = Path(root)
    manifest = InstanceManifest.build(discover_files(root), provenance=provenance)
    manifest.save(root / "manifest.json")
    return manifest


@dataclass
class Strate:
    """A strate directory: signed manifest plus the instance files it covers.

    Attributes
    ----------
    root:
        The strate directory (whose name is the strate name).
    manifest:
        The signed manifest loaded from ``root/manifest.json``.
    """

    root: Path
    manifest: InstanceManifest

    @classmethod
    def load(cls, root: str | Path) -> "Strate":
        root = Path(root)
        return cls(root=root, manifest=InstanceManifest.load(root / "manifest.json"))

    def files(self) -> dict[InstanceKey, Path]:
        """Every instance file on disk, keyed by ``(strate, split, name)``."""
        return discover_files(self.root)

    def verify(self) -> list[str]:
        """Re-hash every instance file and report any alteration of the strate."""
        return self.manifest.verify(self.files())

    def splits(self) -> dict[str, list[str]]:
        """Instance names grouped by split, in the manifest's (sorted) order."""
        out: dict[str, list[str]] = {}
        for entry in self.manifest.entries.values():
            out.setdefault(entry["split"], []).append(entry["name"])
        return {split: sorted(names) for split, names in out.items()}
