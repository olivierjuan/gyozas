import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyscipopt

from gyozas import __version__ as _gyozas_version
from gyozas.manifest import InstanceManifest
from gyozas.seeds import SeedTriplet
from gyozas.solver_config import SolverConfig


@dataclass(frozen=True)
class RunIdentity:
    """Everything needed to replay a run's number without ambiguity (R8.1).

    A results table row refers to a fixed dataset (manifest hash), a fixed
    solver build (SCIP version), a fixed configuration (solver profile and
    its parameters), fixed randomness (the seed triplet) and a fixed harness
    (library version). Two runs with the same identity are the same run.
    """

    method: str
    solver_config: dict[str, Any]
    seeds: dict[str, int]
    scip_version: str
    pyscipopt_version: str
    gyozas_version: str
    manifest_hash: str

    @classmethod
    def capture(
        cls,
        method: str,
        solver_config: SolverConfig,
        seeds: SeedTriplet,
        manifest: InstanceManifest,
    ) -> "RunIdentity":
        """Snapshot the identity of a run about to start; versions are read live."""
        model = pyscipopt.Model()
        scip_version = model.version()
        model.freeProb()
        return cls(
            method=method,
            solver_config=solver_config.to_dict(),
            seeds=seeds.to_dict(),
            scip_version=scip_version,
            pyscipopt_version=pyscipopt.__version__,
            gyozas_version=_gyozas_version,
            manifest_hash=manifest.hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "solver_config": self.solver_config,
            "seeds": self.seeds,
            "scip_version": self.scip_version,
            "pyscipopt_version": self.pyscipopt_version,
            "gyozas_version": self.gyozas_version,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunIdentity":
        return cls(**data)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "RunIdentity":
        with open(path) as f:
            return cls.from_dict(json.load(f))
