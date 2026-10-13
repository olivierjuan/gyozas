"""The signed protocol snapshot (R8.2) — the technical side of the G2 freeze.

The freeze itself is a *decision* (the project's); the snapshot is what makes
it enforceable: a single document capturing the protocol's content — strate
manifests, solver profiles, solver/harness versions — and signing it with a
content hash. Once captured, any drift — an instance altered (its manifest
hash changes), a profile re-parameterized, a solver or harness upgrade —
changes the hash or fails :meth:`ProtocolSnapshot.check_run`, so every result
published after the freeze demonstrably went through the same pipeline.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyscipopt

from gyozas import __version__ as _gyozas_version

from .manifest import InstanceManifest
from .solver_config import available_profiles, get_profile


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _snapshot_hash(data: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json(data).encode()).hexdigest()


@dataclass(frozen=True)
class ProtocolSnapshot:
    """The frozen protocol, signed by its content hash (R8.2 / G2).

    Attributes
    ----------
    name:
        The freeze's name (e.g. ``"G2-v0"``).
    strates:
        Strate name → instance-manifest hash: the datasets, byte-for-byte.
    solver_profiles:
        Solver profile name → parameters: how instances are solved.
    scip_version / pyscipopt_version / gyozas_version:
        The solver build and harness the protocol is frozen on.
    provenance:
        Free-form context of the freeze (date, decision reference, ...).
    hash:
        The signature: SHA-256 of the snapshot's content, itself.
    """

    name: str
    strates: dict[str, str]
    solver_profiles: dict[str, dict[str, Any]]
    scip_version: str
    pyscipopt_version: str
    gyozas_version: str
    provenance: dict[str, Any]
    hash: str

    @classmethod
    def capture(
        cls,
        name: str,
        strate_dirs: dict[str, str | Path],
        provenance: dict[str, Any] | None = None,
    ) -> "ProtocolSnapshot":
        """Snapshot the current protocol: strate manifests, profiles, versions."""
        strates: dict[str, str] = {}
        for strate_name, directory in strate_dirs.items():
            manifest_file = Path(directory) / "manifest.json"
            manifest = InstanceManifest.load(manifest_file) if manifest_file.exists() else None
            if manifest is None:
                raise FileNotFoundError(f"no manifest.json in {directory}: a strate without a signed manifest cannot be frozen")
            strates[strate_name] = manifest.hash
        solver_profiles = {profile: get_profile(profile).to_dict() for profile in available_profiles()}
        model = pyscipopt.Model()
        scip_version = model.version()
        model.freeProb()
        snapshot = cls(
            name=name,
            strates=strates,
            solver_profiles=solver_profiles,
            scip_version=scip_version,
            pyscipopt_version=pyscipopt.__version__,
            gyozas_version=_gyozas_version,
            provenance=provenance or {},
            hash="",
        )
        snapshot_hash = _snapshot_hash(snapshot._content())
        return cls(**{**snapshot.__dict__, "hash": snapshot_hash})

    def _content(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strates": self.strates,
            "solver_profiles": self.solver_profiles,
            "scip_version": self.scip_version,
            "pyscipopt_version": self.pyscipopt_version,
            "gyozas_version": self.gyozas_version,
            "provenance": self.provenance,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "hash": self.hash}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolSnapshot":
        snapshot = cls(
            name=data["name"],
            strates=dict(data["strates"]),
            solver_profiles=dict(data["solver_profiles"]),
            scip_version=data["scip_version"],
            pyscipopt_version=data["pyscipopt_version"],
            gyozas_version=data["gyozas_version"],
            provenance=dict(data.get("provenance", {})),
            hash=data.get("hash", ""),
        )
        recomputed = _snapshot_hash(snapshot._content())
        if snapshot.hash != recomputed:
            raise ValueError(
                f"Protocol snapshot hash mismatch: recorded {snapshot.hash}, content hashes to {recomputed}. "
                "The snapshot was modified after it was signed."
            )
        return snapshot

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "ProtocolSnapshot":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def check_run(self, run_identity: Any) -> list[str]:
        """Whether a run's identity belongs to this frozen protocol.

        Returns the list of problems (empty means the run matches: its
        instances are the frozen ones, its solver and harness versions are the
        frozen ones). A run against an altered dataset or a different build
        cannot silently pass as a result of the frozen protocol.
        """
        problems: list[str] = []
        manifest_hash = getattr(run_identity, "manifest_hash", None) or (run_identity.get("manifest_hash") if isinstance(run_identity, dict) else None)
        method = getattr(run_identity, "method", None) or (run_identity.get("method") if isinstance(run_identity, dict) else "?")
        if manifest_hash is None or manifest_hash == "":
            problems.append(f"{method}: no manifest hash — the run is not tied to a frozen dataset")
        elif manifest_hash not in self.strates.values():
            problems.append(f"{method}: manifest hash {manifest_hash} is not frozen in this protocol")
        for version, frozen in (
            ("scip_version", self.scip_version),
            ("pyscipopt_version", self.pyscipopt_version),
            ("gyozas_version", self.gyozas_version),
        ):
            actual = getattr(run_identity, version, None) or (run_identity.get(version) if isinstance(run_identity, dict) else None)
            if actual is not None and actual != frozen:
                problems.append(f"{method}: {version} {actual} != frozen {frozen}")
        return problems
