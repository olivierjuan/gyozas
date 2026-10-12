from dataclasses import dataclass, field
from typing import Any

from pyscipopt import Model

# Registry of named solver profiles, keyed by profile name.
_PROFILES: dict[str, "SolverConfig"] = {}


@dataclass(frozen=True)
class SolverConfig:
    """Declarative solver configuration for a benchmark protocol.

    A named, immutable set of SCIP parameters that defines how instances are
    solved (budgets, heuristics, cutoffs, randomization, ...). The config is
    serializable so it can be recorded alongside run results: a run that does
    not remember its solver configuration cannot be reproduced or compared.

    Use :func:`register_profile` to expose a protocol's profile by name, and
    :func:`get_profile` to retrieve it.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def apply(self, model: Model) -> None:
        """Set all profile parameters on the given model."""
        model.setParams(dict(self.params))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation, to be stored with run results."""
        return {"name": self.name, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolverConfig":
        return cls(name=data["name"], params=dict(data.get("params", {})))


def register_profile(config: SolverConfig) -> SolverConfig:
    """Register a named profile in the global registry and return it."""
    _PROFILES[config.name] = config
    return config


def get_profile(name: str) -> SolverConfig:
    """Retrieve a registered profile by name."""
    if name not in _PROFILES:
        raise KeyError(f"Unknown solver profile {name!r}. Registered profiles: {sorted(_PROFILES)}")
    return _PROFILES[name]


def available_profiles() -> list[str]:
    return sorted(_PROFILES)


# The neutral profile: SCIP's own defaults, no protocol restrictions.
register_profile(SolverConfig(name="default", params={}))
