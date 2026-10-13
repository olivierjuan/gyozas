"""Full B&B evaluation of a branching policy against the classic baselines (R6.1).

Every method — the evaluated policy and the classic baselines — runs complete
branch-and-bound episodes on the *same* instances under the *same* seeds; the
default solver is the unit 1.000× and every other method is reported through
per-instance paired ratios (R6.2's unit, never transverse averages).

A "method" here is one of:

- ``None`` — the default solver: every branching decision is handed back to
  SCIP (its own default rule decides); this is the baseline.
- a :class:`~gyozas.policy.Policy` — used as is.
- a :class:`~gyozas.experts.BranchingExpert` (pscost, relpscost, strong
  branching, ...) — driven on the live model; when it cannot decide (``None``),
  the decision falls back to the solver's default rule.

The evaluation respects the protocol's minimum seed count: fewer seeds than
``MIN_SEEDS`` is an error, not a quieter number.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .dynamics.branching import BranchingDynamics, ExtraBranchingActions
from .environment import Environment
from .experts import BranchingExpert
from .informations.solve_status import SolveStatus
from .instances.files import FileGenerator
from .metrics import paired_per_instance_ratios, sgm_of_ratios, summary_table
from .policy import Policy
from .runner import MIN_SEEDS
from .seeds import SeedTriplet

DEFAULT_BASELINE = "default"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _select_for(method: Policy | BranchingExpert | None, env: Environment) -> Callable[[Any, Any], int]:
    """Resolve one method to a decision function bound to the episode's environment."""
    if method is None:
        return lambda _obs, _action_set: int(ExtraBranchingActions.SKIP)
    if hasattr(method, "select_action"):
        return method.select_action
    if hasattr(method, "select"):
        expert = method

        def decide(_obs, action_set):
            action = expert.select(env.model, action_set)
            return int(action) if action is not None else int(ExtraBranchingActions.SKIP)

        return decide
    raise TypeError(f"unsupported method {method!r}: expected a Policy, a BranchingExpert, or None")


def _run_episode(
    method: Policy | BranchingExpert | None,
    path: Any,
    triplet: SeedTriplet,
    scip_params: Mapping[str, Any],
) -> dict[str, Any]:
    """One full B&B episode; returns the node count and end status."""
    triplet.seed_python()
    path = Path(path)
    generator = FileGenerator(directory=path.parent, pattern=path.name, sampling_mode="replace", rng=triplet.python)
    env = Environment(
        instance_generator=generator,
        information_function=SolveStatus(),
        scip_params=dict(scip_params),
        # The default-solver baseline expresses "SCIP decides" as a SKIP action,
        # which the action set must carry.
        dynamics=BranchingDynamics(with_extra_actions=[ExtraBranchingActions.SKIP]),
    )
    try:
        env.seed(triplet.solver)
        policy = _select_for(method, env)
        obs, action_set, _reward, done, info = env.reset()
        steps = 0
        while not done:
            obs, action_set, _reward, done, info = env.step(policy(obs, action_set))
            steps += 1
        return {
            "nodes": env.dynamics.counters.nodes,
            "decisions": env.dynamics.counters.decisions,
            "steps": steps,
            "status": info.get("status"),
            "solved": info.get("solved"),
            "truncated": info.get("truncated"),
            "limit": info.get("limit"),
        }
    finally:
        env.close()


def evaluate(
    methods: Mapping[str, Policy | BranchingExpert | None],
    instance_files: Mapping[str, str],
    seeds: Sequence[int],
    scip_params: Mapping[str, Any] | None = None,
    baseline: str = DEFAULT_BASELINE,
    min_seeds: int = MIN_SEEDS,
) -> dict[str, Any]:
    """Evaluate methods on the same instances and seeds; the default solver is 1.000×.

    Parameters
    ----------
    methods:
        The methods to compare; the baseline (``"default"`` by default) may be
        omitted and is then run as the default solver. Passing it explicitly
        must map it to ``None``.
    instance_files:
        Instance name → file path. The *same* files are run under every method
        and every seed.
    seeds:
        Base seeds, one :class:`~gyozas.seeds.SeedTriplet` each; fewer than
        ``min_seeds`` (the protocol's five) is rejected.
    scip_params:
        SCIP parameters (budget, profile) applied identically to every episode.
    baseline:
        Name of the reference method; its per-instance ratios are 1.0 by
        definition, and every other method is divided by it.

    Returns
    -------
    A dict with, per method: per-seed raw results, per-instance mean node
    counts, per-instance paired ratios against the baseline, the SGM of those
    ratios, and the summary table (R7.5 double reporting: SGM, p50/p95, worst
    instances, solved vs truncated).
    """
    if len(seeds) < min_seeds:
        raise ValueError(f"{len(seeds)} seed(s) < the protocol's minimum of {min_seeds}")
    if not instance_files:
        raise ValueError("no instances to evaluate")
    if baseline in methods and methods[baseline] is not None:
        raise ValueError(f"the baseline {baseline!r} must be the default solver (None)")
    scip_params = dict(scip_params or {})
    all_methods: dict[str, Policy | BranchingExpert | None] = dict(methods)
    if baseline not in all_methods:
        all_methods[baseline] = None
    if list(instance_files) != sorted(instance_files):
        instance_files = dict(sorted(instance_files))

    per_seed: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    nodes: dict[str, dict[str, float]] = {}
    statuses: dict[str, dict[str, str]] = {}
    for method_name, method in all_methods.items():
        per_seed[method_name] = {}
        for seed in seeds:
            triplet = SeedTriplet.from_base_seed(seed)
            per_seed[method_name][str(seed)] = {
                name: _run_episode(method, path, triplet, scip_params) for name, path in instance_files.items()
            }
        nodes[method_name] = {
            name: _mean([per_seed[method_name][str(seed)][name]["nodes"] for seed in seeds])
            for name in instance_files
        }
        statuses[method_name] = {
            f"{seed}/{name}": per_seed[method_name][str(seed)][name]["status"]
            for seed in seeds
            for name in instance_files
        }

    baseline_nodes = nodes[baseline]
    ratios: dict[str, dict[str, float]] = {}
    sgms: dict[str, float] = {}
    for method_name in all_methods:
        if method_name == baseline:
            ratios[method_name] = {name: 1.0 for name in instance_files}
            sgms[method_name] = 1.0
        else:
            ratios[method_name] = paired_per_instance_ratios(nodes[method_name], baseline_nodes)
            sgms[method_name] = sgm_of_ratios(nodes[method_name], baseline_nodes)

    return {
        "baseline": baseline,
        "n_seeds": len(seeds),
        "n_instances": len(instance_files),
        "per_seed": per_seed,
        "nodes": nodes,
        "statuses": statuses,
        "ratios": ratios,
        "sgm": sgms,
        "summary": summary_table(nodes, statuses=statuses),
    }
