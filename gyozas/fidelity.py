"""Native fidelity profiles (R5.4) — the activable mechanism, scaffolding only.

A fidelity profile is a method's *original regime*, declaratively: the
observation (features) it consumes, the expert it was designed against, the
augmentation it applies, its solver parameters — and, as recorded metadata,
its training specifics (loss, schedules, pre-training). Declaring one on a
:class:`gyozas.runner.Method` recharges the method's native regime for both
data collection and evaluation, without the method being rewritten.

The mechanism is complete; the documented method-specific regimes (Lookback's
second-choice smoothing + PAT, TGPPO's H3 + nested Optuna, Mamba's contrastive
pre-training, T-BranT's BiMatching, ...) are **not** provided here: they are
training-loop code, outside the harness's scope. A profile records them and a
method's own training code consumes them.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .augmentation import ZarpellonAugmentation
from .experts import BranchingExpert


@dataclass(frozen=True)
class FidelityProfile:
    """One method's native regime, declaratively activable.

    Attributes
    ----------
    name:
        The profile's identifier (usually the method's name).
    observation:
        Factory for the observation function the method consumes.
    expert:
        Factory for the expert the method was designed against; when a method
        declares no policy of its own, the expert drives the solver (falling
        back to the solver's default rule where it cannot decide).
    augmentation:
        Factory for the search-tree augmentation (seed s, k) applied on reset.
    scip_params:
        Solver parameters of the method's original setup.
    training:
        Training-loop specifics (loss, smoothing, schedules, hyperparameter
        search, ...), recorded for the method's own training code — the harness
        does not execute them.
    """

    name: str
    observation: Callable[[], Any] | None = None
    expert: Callable[[], BranchingExpert] | None = None
    augmentation: Callable[[], ZarpellonAugmentation] | None = None
    scip_params: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)


# Registry of fidelity profiles, keyed by name.
_PROFILES: dict[str, FidelityProfile] = {}


def register_fidelity_profile(profile: FidelityProfile) -> FidelityProfile:
    """Register a fidelity profile by name and return it."""
    _PROFILES[profile.name] = profile
    return profile


def get_fidelity_profile(name: str) -> FidelityProfile:
    """Retrieve a registered fidelity profile by name."""
    if name not in _PROFILES:
        raise KeyError(f"Unknown fidelity profile {name!r}. Registered profiles: {sorted(_PROFILES)}")
    return _PROFILES[name]


def available_fidelity_profiles() -> list[str]:
    """Names of the registered fidelity profiles."""
    return sorted(_PROFILES)
