"""Config-driven experiment runner (R5.1, R5.2, R5.3, R8.3).

A config file describes the experiment grid: method × strate × split × seed ×
solver profile. The runner executes every (cell, seed, instance) episode,
writing one JSON record per run under ``results_dir``. The layout is the
protocol::

    <results_dir>/<method>/<strate>/<split>/<seed>/<instance>.json

Properties the protocol requires:

- **Idempotent** (R5.1): a run whose record exists and is complete is never
  re-executed — relaunching the same config is a no-op.
- **Resumable** (R5.1/R5.3): an interrupted run leaves an incomplete record
  which the next launch re-executes.
- **5-seed guard** (R5.2): a cell with fewer than the protocol's five seeds —
  or missing records — yields no publishable summary.
- **Artifacts as outputs** (R8.3): every record embeds the run's full identity
  (config, seeds, SCIP/pyscipopt/gyozas versions, manifest hash), so the
  publishable artifacts are natural by-products of running.

Config files are JSON (``ExperimentConfig.load``); the runner itself is
scripted, since method factories are code.
"""

import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dynamics.branching import BranchingDynamics, ExtraBranchingActions
from .environment import Environment
from .fidelity import get_fidelity_profile
from .informations.solve_status import SolveStatus
from .instances.files import FileGenerator
from .manifest import InstanceManifest
from .policy import Policy
from .run_identity import RunIdentity
from .seeds import SeedTriplet
from .solver_config import SolverConfig, get_profile

# The protocol's minimum seed count (spec §2/§8): an incomplete cell produces
# no publishable result.
MIN_SEEDS = 5


class _ExpertDrivenObservation:
    """Observation function that also resets a fidelity expert with each new model.

    A fidelity expert decides on the live model (``expert.select(model, ...)``),
    so it must receive every new episode's model through the observation hook —
    the one place the harness touches the model before the solve starts.
    """

    def __init__(self, expert: Any, inner: Any | None) -> None:
        self._expert = expert
        self._inner = inner

    def reset(self, model: Any) -> None:
        if self._inner is not None:
            self._inner.reset(model)
        self._expert.reset(model)

    def extract(self, model: Any, done: bool) -> Any:
        return self._inner.extract(model, done) if self._inner is not None else None


@dataclass
class Method:
    """One benchmark method: a policy factory plus the observation it declares.

    Attributes
    ----------
    name:
        The method's identifier in the experiment grid.
    policy:
        Factory returning a fresh :class:`~gyozas.policy.Policy` per run. May
        be ``None`` when a fidelity profile with an expert is declared — the
        expert then drives the solver.
    observation:
        Factory returning the observation function the policy consumes, or
        ``None`` for the default (solver-stats-free) observation.
    config:
        Method configuration, embedded in every run record (R8.1/R8.3).
    fidelity:
        Optional name of a registered :class:`~gyozas.fidelity.FidelityProfile`
        (R5.4): the method's native regime — observation, expert, augmentation,
        solver parameters — is recharged for the run.
    """

    name: str
    policy: Callable[[], Policy] | None = None
    observation: Callable[[], Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    fidelity: str | None = None


@dataclass
class Cell:
    """One experiment cell: method × strate × split, with its seeds.

    Attributes
    ----------
    method:
        Key into the runner's method table.
    strate:
        Strate name (key into the runner's strate-directory table).
    split:
        Split name (subdirectory of the strate).
    seeds:
        Base seeds — one run per seed; each derives a deterministic
        :class:`~gyozas.seeds.SeedTriplet`. The protocol's minimum is
        ``MIN_SEEDS`` for a publishable cell.
    profile:
        Name of a registered :class:`~gyozas.solver_config.SolverConfig`
        profile, applied to every episode of the cell.
    budget:
        SCIP parameters bounding the solve (e.g. ``{"limits/time": 3600.0}``);
        the end state (solved vs truncated, and which budget) is recorded.
    """

    method: str
    strate: str
    split: str
    seeds: list[int]
    profile: str | None = None
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.method}/{self.strate}/{self.split}"


@dataclass
class ExperimentConfig:
    """The experiment grid: the file that drives a benchmark campaign (R5.1)."""

    cells: list[Cell]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": [
                {
                    "method": c.method,
                    "strate": c.strate,
                    "split": c.split,
                    "seeds": list(c.seeds),
                    "profile": c.profile,
                    "budget": dict(c.budget),
                }
                for c in self.cells
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            cells=[
                Cell(
                    method=c["method"],
                    strate=c["strate"],
                    split=c["split"],
                    seeds=list(c["seeds"]),
                    profile=c.get("profile"),
                    budget=dict(c.get("budget", {})),
                )
                for c in data["cells"]
            ]
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class Runner:
    """Executes an :class:`ExperimentConfig`, idempotently and resumably.

    Parameters
    ----------
    config:
        The experiment grid.
    methods:
        The method table, keyed by name.
    strate_dirs:
        Strate directories, keyed by strate name (a :class:`gyozas.strates.Strate`
        root, or any directory of instance files with split subdirectories).
    results_dir:
        Where run records are written (created on demand).
    manifest:
        Optional :class:`~gyozas.manifest.InstanceManifest` hashed into every
        run identity — the strate's signed manifest when instances come from a
        strate.
    reward_factory:
        Factory returning the reward function for an episode (defaults to the
        environment's own default).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        methods: Mapping[str, Method],
        strate_dirs: Mapping[str, str | Path],
        results_dir: str | Path,
        manifest: InstanceManifest | None = None,
        reward_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.methods = dict(methods)
        self.strate_dirs = {k: Path(v) for k, v in strate_dirs.items()}
        self.results_dir = Path(results_dir)
        self.manifest = manifest
        self._manifests: dict[str, InstanceManifest | None] = {}
        if manifest is not None:
            self._manifests = {name: manifest for name in self.strate_dirs}
        else:
            for name, directory in self.strate_dirs.items():
                manifest_file = directory / "manifest.json"
                self._manifests[name] = InstanceManifest.load(manifest_file) if manifest_file.exists() else None
        self.reward_factory = reward_factory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        saved_config = self.results_dir / "experiment_config.json"
        if saved_config.exists():
            if ExperimentConfig.load(saved_config) != self.config:
                raise ValueError(
                    f"{saved_config} describes a different experiment; use a new results directory."
                )
        else:
            self.config.save(saved_config)

    # -- paths ---------------------------------------------------------------

    def _run_path(self, cell: Cell, seed: int, instance: str) -> Path:
        return self.results_dir / cell.method / cell.strate / cell.split / str(seed) / f"{instance}.json"

    # -- execution -----------------------------------------------------------

    def run(self) -> dict[str, dict[str, int]]:
        """Execute every missing run of every cell; skip completed ones.

        Returns per-cell counters: ``{"skipped": …, "executed": …}`` — a
        relaunch of a finished experiment executes nothing (R5.1).
        """
        report: dict[str, dict[str, int]] = {}
        for cell in self.config.cells:
            executed, skipped = self.run_cell(cell)
            report[cell.key] = {"executed": executed, "skipped": skipped}
        return report

    def run_cell(self, cell: Cell) -> tuple[int, int]:
        """Execute the missing runs of one cell; returns (executed, skipped)."""
        executed = 0
        skipped = 0
        for seed in cell.seeds:
            for instance, path in self._instances(cell):
                run_path = self._run_path(cell, seed, instance)
                if run_path.exists() and json.loads(run_path.read_text()).get("complete"):
                    skipped += 1
                    continue
                record = self._run_one(cell, seed, instance, path)
                run_path.parent.mkdir(parents=True, exist_ok=True)
                with open(run_path, "w") as f:
                    json.dump(record, f, indent=2)
                executed += 1
        return executed, skipped

    def _instances(self, cell: Cell) -> list[tuple[str, Path]]:
        split_dir = self.strate_dirs[cell.strate] / cell.split
        files = sorted(p for p in split_dir.iterdir() if p.is_file() and not p.name.startswith("."))
        if not files:
            raise FileNotFoundError(f"no instance files in {split_dir}")
        return [(p.name.removesuffix(".mps.gz").removesuffix(".mps").removesuffix(".lp"), p) for p in files]

    def _scip_params(self, cell: Cell, method: Method | None = None) -> dict[str, Any]:
        """Fidelity and solver-profile parameters, overlaid with the cell's budget."""
        params: dict[str, Any] = {}
        if method is not None and method.fidelity is not None:
            params.update(get_fidelity_profile(method.fidelity).scip_params)
        if cell.profile is not None:
            params.update(get_profile(cell.profile).params)
        params.update(cell.budget)
        return params

    def _policy_for(self, method: Method, env: Environment, expert: Any | None) -> Callable[[Any, Any], int]:
        """Resolve the method's decision function for one episode."""
        if method.policy is not None:
            return method.policy().select_action
        if expert is not None:

            def decide(_obs, action_set):
                action = expert.select(env.model, action_set)
                return int(action) if action is not None else int(ExtraBranchingActions.SKIP)

            return decide
        raise ValueError(f"method {method.name!r} declares neither a policy nor a fidelity profile with an expert")

    def _run_one(self, cell: Cell, seed: int, instance: str, path: Path) -> dict[str, Any]:
        method = self.methods[cell.method]
        profile = get_fidelity_profile(method.fidelity) if method.fidelity is not None else None
        augmentation = profile.augmentation() if profile is not None and profile.augmentation is not None else None
        expert = profile.expert() if profile is not None and method.policy is None else None
        extra_actions = [ExtraBranchingActions.SKIP] if expert is not None else None
        observation = None
        if method.observation is not None:
            observation = method.observation()
        elif profile is not None and profile.observation is not None:
            observation = profile.observation()
        if expert is not None:
            # The expert is model-driven: reset it on each new model through the
            # observation hook, before the solve thread starts consuming decisions.
            observation = _ExpertDrivenObservation(expert, observation)
        triplet = SeedTriplet.from_base_seed(seed)
        triplet.seed_python()
        generator = FileGenerator(directory=path.parent, pattern=path.name, sampling_mode="replace", rng=triplet.python)
        env = Environment(
            instance_generator=generator,
            observation_function=observation,
            reward_function=self.reward_factory() if self.reward_factory else None,
            information_function=SolveStatus(),
            scip_params=self._scip_params(cell, method),
            dynamics=BranchingDynamics(augmentation=augmentation, with_extra_actions=extra_actions),
        )
        try:
            env.seed(triplet.solver)
            policy = self._policy_for(method, env, expert)
            total_reward = 0.0
            obs, action_set, reward, done, info = env.reset()
            total_reward += reward
            steps = 0
            while not done:
                action = policy(obs, action_set)
                obs, action_set, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
            counters = env.dynamics.counters.as_dict() if hasattr(env.dynamics, "counters") else {}
            return {
                "cell": {
                    "method": cell.method,
                    "strate": cell.strate,
                    "split": cell.split,
                    "profile": cell.profile,
                    "budget": dict(cell.budget),
                    "method_config": dict(method.config),
                    "fidelity": method.fidelity,
                },
                "seed": {"base": seed, "triplet": triplet.to_dict()},
                "instance": instance,
                "run_identity": self._identity(cell, method, triplet).to_dict(),
                "result": {
                    "status": info.get("status"),
                    "solved": info.get("solved"),
                    "truncated": info.get("truncated"),
                    "limit": info.get("limit"),
                    "total_reward": total_reward,
                    "steps": steps,
                    "counters": counters,
                },
                "complete": True,
            }
        finally:
            env.close()

    def _identity(self, cell: Cell, method: Method, triplet: SeedTriplet) -> RunIdentity:
        profile = get_profile(cell.profile) if cell.profile is not None else get_profile("default")
        solver_config = SolverConfig(
            name=f"{profile.name}+budget" if cell.budget else profile.name,
            params=self._scip_params(cell, method),
        )
        return RunIdentity.capture(
            method=f"{cell.method}:{cell.strate}/{cell.split}",
            solver_config=solver_config,
            seeds=triplet,
            manifest=self._manifests[cell.strate],
        )

    # -- publication guard (R5.2) --------------------------------------------

    def cell_status(self, cell: Cell) -> dict[str, Any]:
        """Completeness of a cell: records present and complete per seed."""
        instances = [name for name, _ in self._instances(cell)]
        per_seed = {}
        for seed in cell.seeds:
            records = [self._run_path(cell, seed, name) for name in instances]
            per_seed[str(seed)] = sum(1 for p in records if p.exists() and json.loads(p.read_text()).get("complete"))
        return {
            "cell": cell.key,
            "n_instances": len(instances),
            "seeds": per_seed,
            "n_seeds": len(cell.seeds),
            "complete": all(n == len(instances) for n in per_seed.values()),
        }

    def is_publishable(self, cell: Cell) -> tuple[bool, str]:
        """Whether a cell may produce a publishable result (R5.2).

        A cell needs at least ``MIN_SEEDS`` seeds — the protocol's minimum —
        and a complete record for every (seed, instance) pair.
        """
        if len(cell.seeds) < MIN_SEEDS:
            return False, f"{cell.key}: {len(cell.seeds)} seed(s) < the protocol's {MIN_SEEDS}"
        status = self.cell_status(cell)
        if not status["complete"]:
            return False, f"{cell.key}: incomplete records {status['seeds']}"
        return True, f"{cell.key}: {len(cell.seeds)} seeds × {status['n_instances']} instances, complete"

    def collect_cell(self, cell: Cell) -> dict[str, dict[str, Any]]:
        """Load the per-instance results of a complete, publishable cell (R5.2).

        Returns ``{seed: {instance: result}}``; raises if the cell is not
        publishable — an incomplete cell produces no result.
        """
        ok, reason = self.is_publishable(cell)
        if not ok:
            raise ValueError(f"cell not publishable: {reason}")
        instances = [name for name, _ in self._instances(cell)]
        out: dict[str, dict[str, Any]] = {}
        for seed in cell.seeds:
            out[str(seed)] = {name: json.loads(self._run_path(cell, seed, name).read_text())["result"] for name in instances}
        return out


def export_results(results_dir: str | Path, destination: str | Path) -> None:
    """Copy a results directory's records as publishable artifacts (R8.3).

    The artifacts are natural outputs of the pipeline: the experiment config
    and every run record (identity, seeds, result) — nothing assembled by hand
    at the end of the project.
    """
    results_dir = Path(results_dir)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(results_dir, destination)
