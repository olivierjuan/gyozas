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

    def apply(self, model: Model, cutoff: float | None = None) -> None:
        """Set all profile parameters on the given model.

        Parameters
        ----------
        model:
            The model to configure.
        cutoff:
            Optional objective cutoff value (prune solutions worse than this
            bound), applied via ``setObjlimit``. Used by profiles whose protocol
            fixes the cutoff at the known optimum (e.g. the Lin 2022 profile).
        """
        model.setParams(dict(self.params))
        if cutoff is not None:
            model.setObjlimit(cutoff)

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

# Lin 2022 (T-BranT) sandbox profile, as used for expert data collection in
# https://github.com/linjc16/TBranT : 1 h budget, primal heuristics disabled,
# strong-branching auxiliaries off (conflict analysis, probing bounds, solution
# checks, reevaluation age), presolve and cuts left at SCIP defaults. The cutoff
# at the known optimum is instance-dependent: pass it to apply(cutoff=...).
_LIN_2022_HEURISTICS = (
    "actconsdiving",
    "alns",
    "bound",
    "clique",
    "coefdiving",
    "completesol",
    "conflictdiving",
    "crossover",
    "dins",
    "distributiondiving",
    "dualval",
    "farkasdiving",
    "feaspump",
    "fixandinfer",
    "fracdiving",
    "gins",
    "guideddiving",
    "indicator",
    "intdiving",
    "intshifting",
    "linesearchdiving",
    "localbranching",
    "locks",
    "lpface",
    "mutation",
    "multistart",
    "mpec",
    "nlpdiving",
    "objpscostdiving",
    "octane",
    "ofins",
    "oneopt",
    "proximity",
    "pscostdiving",
    "randrounding",
    "rens",
    "reoptsols",
    "repair",
    "rins",
    "rootsoldiving",
    "rounding",
    "shiftandpropagate",
    "shifting",
    "simplerounding",
    "subnlp",
    "trivial",
    "trivialnegation",
    "trysol",
    "twoopt",
    "undercover",
    "vbounds",
    "veclendiving",
    "zeroobj",
    "zirounding",
)

_LIN_2022_PARAMS: dict[str, Any] = {
    "display/verblevel": 0,
    "limits/time": 3600.0,
    "limits/nodes": -1,
    "presolving/maxrounds": -1,
    "separating/maxrounds": -1,
    "separating/maxroundsroot": -1,
    "reoptimization/enable": False,
    "conflict/usesb": False,
    "branching/fullstrong/probingbounds": False,
    "branching/relpscost/probingbounds": False,
    "branching/checksol": False,
    "branching/fullstrong/reevalage": 0,
}
_LIN_2022_PARAMS.update({f"heuristics/{name}/freq": -1 for name in _LIN_2022_HEURISTICS})

register_profile(SolverConfig(name="lin2022", params=_LIN_2022_PARAMS))
