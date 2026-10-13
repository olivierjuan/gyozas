"""Benchmark metrics and statistics.

Everything the evaluation protocol prescribes, implemented once and applied
everywhere: per-instance paired units (never transverse averages of decisions),
double reporting median vs tail, rank tests, head-to-head win rates, and the
single output table that feeds the paper.
"""

import csv
import json
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

# Nemenyi critical values q_alpha (Demšar 2006, Journal of Statistical Software),
# indexed by number of methods k, for the two standard significance levels.
_NEMENYI_Q = {
    0.05: {
        2: 1.959964,
        3: 2.342657,
        4: 2.570582,
        5: 2.727679,
        6: 2.849669,
        7: 2.947342,
        8: 3.030149,
        9: 3.102490,
        10: 3.164135,
    },
    0.10: {
        2: 1.644854,
        3: 2.052281,
        4: 2.291341,
        5: 2.459516,
        6: 2.588521,
        7: 2.692732,
        8: 2.779884,
        9: 2.854606,
        10: 2.919889,
    },
}

# A run stopped by a budget is truncated; these map a SCIP status to the budget.
_TRUNCATED_STATUSES = {
    "timelimit",
    "nodelimit",
    "totalnodelimit",
    "stallnodelimit",
    "memlimit",
    "sollimit",
    "bestsollimit",
    "restartlimit",
    "primallimit",
    "duallimit",
}


# ---------------------------------------------------------------------------
# Per-instance aggregates
# ---------------------------------------------------------------------------


def shifted_geometric_mean(values, shift: float = 0.0) -> float:
    """Shifted geometric mean (SGM) of a per-instance metric.

    The SCIP/MIPLIB convention (Achterberg 2007):
    ``SGM = (prod(v + shift))^(1/n) - shift``. With the default ``shift=0``
    this is the plain geometric mean and values must be strictly positive;
    pass ``shift=1.0`` for raw node/time counts, where zeros (instances
    solved in presolve) are absorbed by the shift. Per-instance ratios must
    be aggregated *without* a positive shift, or improvements (ratio < 1)
    would be erased.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("shifted_geometric_mean needs at least one value.")
    if (arr + shift <= 0).any():
        raise ValueError("shifted_geometric_mean needs values + shift > 0.")
    return float(np.exp(np.mean(np.log(arr + shift))) - shift)


def accuracy_at_1(
    policy_actions,
    expert_scores,
    action_sets,
    *,
    collection,
    tie_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Top-1 accuracy of a policy against the expert scores — never alone (R7.1).

    A policy action is *correct* when its expert score ties the best score of
    the node's action set within ``tie_tolerance``: a different-but-equally-
    optimal action must not be penalized, which is exactly what raw
    label-matching accuracy gets wrong. The report always carries the
    collection's multi-optimal node distribution beside the accuracy — without
    it, acc@1 is unreadable — and must never be compared between collections
    made with different experts (the tie structure, hence the accuracy, is an
    expert property).

    Parameters
    ----------
    policy_actions:
        The action chosen by the policy at each collected node.
    expert_scores:
        Per node, the per-candidate expert scores (array indexed by action, or
        mapping action -> score), as fed to ``CollectionMetadata.record``.
    action_sets:
        Per node, the array of admissible actions.
    collection:
        The :class:`~gyozas.collection.CollectionMetadata` (or its ``to_dict``
        output) recorded on the same nodes — required, keyword-only: acc@1 is
        never reported without the multi-optimal distribution.

    Returns
    -------
    dict with ``accuracy_at_1``, ``n_nodes``, and the collection record
    (``multi_optimal_distribution``, ``frac_multi_optimal``, ``n_decisions``).
    """
    actions = list(policy_actions)
    if not (len(actions) == len(expert_scores) == len(action_sets)):
        raise ValueError("policy_actions, expert_scores and action_sets must cover the same nodes.")
    if isinstance(collection, dict):
        collection_record = dict(collection)
    else:
        collection_record = collection.to_dict()
    if len(actions) != collection_record.get("n_decisions"):
        raise ValueError(
            f"{len(actions)} policy nodes but the collection recorded "
            f"{collection_record.get('n_decisions')}: acc@1 and the multi-optimal "
            "distribution must cover the same collected nodes."
        )

    n_correct = 0
    for action, scores, action_set in zip(actions, expert_scores, action_sets, strict=False):
        action_scores = [
            float(scores[a])
            if isinstance(scores, dict) and a in scores
            else (float(scores[int(a)]) if not isinstance(scores, dict) and 0 <= int(a) < len(scores) else None)
            for a in np.asarray(action_set)
        ]
        finite = [s for s in action_scores if s is not None and np.isfinite(s)]
        if not finite:
            continue
        best = max(finite)
        tol = tie_tolerance * max(1.0, abs(best))
        own = (
            float(scores[action])
            if isinstance(scores, dict) and action in scores
            else (
                float(scores[int(action)]) if not isinstance(scores, dict) and 0 <= int(action) < len(scores) else None
            )
        )
        if own is not None and np.isfinite(own) and own >= best - tol:
            n_correct += 1
    accuracy = n_correct / len(actions) if actions else 0.0
    return {
        "accuracy_at_1": accuracy,
        "n_nodes": len(actions),
        **collection_record,
    }


def paired_per_instance_ratios(method: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    """Paired per-instance ratios ``method / baseline`` on the common instances.

    The paired unit of comparison: ratios are formed *per instance first*, and
    only then aggregated (by SGM). Transverse averages of per-decision or
    per-node counts are forbidden — instance scales are too heterogeneous.
    """
    common = sorted(set(method) & set(baseline))
    ratios = {}
    for instance in common:
        if baseline[instance] == 0:
            continue
        ratios[instance] = method[instance] / baseline[instance]
    return ratios


def sgm_of_ratios(method: dict[str, float], baseline: dict[str, float], shift: float = 0.0) -> float:
    """SGM of the per-instance ratios of ``method`` against ``baseline``."""
    return shifted_geometric_mean(paired_per_instance_ratios(method, baseline).values(), shift=shift)


def quantiles(values, qs: tuple[float, ...] = (0.5, 0.95)) -> dict[float, float]:
    """Empirical quantiles of a per-instance metric (the tail side of double reporting)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("quantiles needs at least one value.")
    return {q: float(np.quantile(arr, q)) for q in qs}


def catastrophic_losses(per_instance: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
    """The ``k`` worst instances by metric, named — catastrophic losses must be listed, not averaged away."""
    return sorted(per_instance.items(), key=lambda kv: kv[1], reverse=True)[:k]


def solved_vs_truncated(statuses: dict[str, str]) -> dict[str, Any]:
    """Count solved instances vs truncated ones, and by which budget (R6.3).

    The solved-at-limit lesson: on hard strata the ranking can invert once
    truncated runs are excluded, so both populations are reported.
    """
    solved = {name for name, status in statuses.items() if status not in _TRUNCATED_STATUSES}
    truncated = {name: status for name, status in statuses.items() if status in _TRUNCATED_STATUSES}
    by_limit: dict[str, int] = {}
    for status in truncated.values():
        by_limit[status] = by_limit.get(status, 0) + 1
    return {
        "n_total": len(statuses),
        "n_solved": len(solved),
        "n_truncated": len(truncated),
        "truncated_by_limit": by_limit,
        "solved_instances": sorted(solved),
    }


def win_rate(method: dict[str, float], baseline: dict[str, float], tie_tolerance: float = 1e-9) -> dict[str, float]:
    """Head-to-head win rate per instance between two methods (R7.4).

    A win is a strictly better metric on that instance; ties use a relative
    tolerance on the baseline value.
    """
    common = sorted(set(method) & set(baseline))
    wins = losses = ties = 0
    for instance in common:
        m, b = method[instance], baseline[instance]
        tol = tie_tolerance * max(1.0, abs(b))
        if m < b - tol:
            wins += 1
        elif m > b + tol:
            losses += 1
        else:
            ties += 1
    decided = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_common": len(common),
        "win_rate": wins / decided if decided else float("nan"),
    }


# ---------------------------------------------------------------------------
# Rank tests
# ---------------------------------------------------------------------------


def _align_methods(per_method: dict[str, dict[str, float]]) -> tuple[list[str], list[str], np.ndarray]:
    """Common instances x methods value matrix, raising if too few instances."""
    methods = list(per_method)
    if len(methods) < 2:
        raise ValueError("Rank tests need at least two methods.")
    instances = sorted(set.intersection(*(set(per_method[m]) for m in methods)))
    if len(instances) < 2:
        raise ValueError("Rank tests need at least two common instances.")
    values = np.array([[per_method[m][i] for m in methods] for i in instances], dtype=float)
    return methods, instances, values


def friedman_nemenyi(per_method: dict[str, dict[str, float]], alpha: float = 0.05) -> dict[str, Any]:
    """Friedman test plus Nemenyi post-hoc on intra-instance ranks (R7.3).

    Ranks are computed per instance (rank 1 = best, i.e. lowest metric value).
    Returns the Friedman statistic and p-value, the average rank of each method,
    the Nemenyi critical distance, and which method pairs differ significantly.
    """
    methods, _instances, values = _align_methods(per_method)
    if not np.isfinite(values).all():
        raise ValueError("Rank tests need finite values; a NaN or inf metric would silently read as 'no difference'.")
    k = len(methods)
    if k < 3:
        raise ValueError("Friedman needs at least three methods; use wilcoxon_pair for two.")
    try:
        chi2, p_value = scipy_stats.friedmanchisquare(*[values[:, j] for j in range(k)])
    except ValueError:
        chi2, p_value = 0.0, 1.0
    if not np.isfinite(chi2) or not np.isfinite(p_value):
        # Degenerate case: all ranks tied across every instance — no difference.
        chi2, p_value = 0.0, 1.0
    ranks = np.apply_along_axis(lambda row: scipy_stats.rankdata(row, method="average"), 1, values)
    avg_ranks = ranks.mean(axis=0)
    n = values.shape[0]
    q_table = _NEMENYI_Q.get(alpha)
    if q_table is None or k not in q_table:
        raise ValueError(f"No Nemenyi critical values for alpha={alpha}, k={k}.")
    critical_distance = q_table[k] * np.sqrt(k * (k + 1) / (6.0 * n))
    significant_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(avg_ranks[i] - avg_ranks[j]) > critical_distance:
                significant_pairs.append((methods[i], methods[j]))
    return {
        "methods": methods,
        "n_instances": n,
        "friedman_chi2": float(chi2),
        "friedman_p": float(p_value),
        "average_ranks": dict(zip(methods, avg_ranks.tolist(), strict=True)),
        "nemenyi_critical_distance": float(critical_distance),
        "significant_pairs": significant_pairs,
    }


def wilcoxon_pair(method: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    """One-sided paired Wilcoxon signed-rank test: method < baseline (lower is better)."""
    common = sorted(set(method) & set(baseline))
    x = np.array([method[i] for i in common], dtype=float)
    y = np.array([baseline[i] for i in common], dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Wilcoxon needs finite values; a NaN or inf metric would silently produce a NaN p-value.")
    if np.allclose(x, y):
        return {"statistic": 0.0, "p_value": 1.0, "n_common": len(common)}
    result = scipy_stats.wilcoxon(x, y, alternative="less")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n_common": len(common)}


def wilcoxon_holm(per_method: dict[str, dict[str, float]], reference: str) -> dict[str, dict[str, float]]:
    """One-sided paired Wilcoxon of every method against ``reference``, Holm-corrected (R7.3).

    Returns one entry per other method with its raw and Holm-adjusted p-value
    (the family is the set of pairwise comparisons of this run).
    """
    raw: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    for method_name in per_method:
        if method_name == reference:
            continue
        result = wilcoxon_pair(per_method[method_name], per_method[reference])
        details[method_name] = result
        raw[method_name] = result["p_value"]
    # Holm step-down over the family of comparisons.
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    m = len(ordered)
    for rank, method_name in enumerate(ordered):
        value = min(1.0, (m - rank) * raw[method_name])
        running_max = max(running_max, value)
        adjusted[method_name] = running_max
    for method_name in details:
        details[method_name]["p_holm"] = adjusted[method_name]
        details[method_name]["significant_at_0.05"] = adjusted[method_name] < 0.05
    return details


# ---------------------------------------------------------------------------
# Unified output table (R7.5)
# ---------------------------------------------------------------------------


def summary_table(
    per_method_per_instance: dict[str, dict[str, float]],
    statuses: dict[str, dict[str, str]] | None = None,
    shift: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """One table: method x metric, the single source for paper and result repos.

    Double reporting is built in: SGM and p50, p95 quantiles, the named worst
    instances, and — when per-run statuses are provided — solved vs truncated
    counts. ``statuses[method]`` maps instance -> end status.
    """
    table: dict[str, dict[str, Any]] = {}
    for method_name, per_instance in per_method_per_instance.items():
        values = list(per_instance.values())
        entry: dict[str, Any] = {
            "n_instances": len(values),
            "sgm": shifted_geometric_mean(values, shift=shift),
            **{f"p{int(q * 100)}": v for q, v in quantiles(values).items()},
            "worst_instances": dict(catastrophic_losses(per_instance, k=5)),
        }
        if statuses and method_name in statuses:
            entry["solved_vs_truncated"] = solved_vs_truncated(statuses[method_name])
        table[method_name] = entry
    return table


_NUMERIC_COLUMNS = ["n_instances", "sgm", "p50", "p95", "wins", "losses", "ties", "n_common", "win_rate"]


def save_table(table: dict[str, dict[str, Any]], path_json: str, path_csv: str | None = None) -> None:
    """Export the summary table: raw JSON always, plus a readable wide CSV."""
    with open(path_json, "w") as f:
        json.dump(table, f, indent=2, default=str)
    if path_csv is None:
        return
    with open(path_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *_NUMERIC_COLUMNS])
        for method_name, metrics in table.items():
            row = [method_name]
            for column in _NUMERIC_COLUMNS:
                value = metrics.get(column)
                if isinstance(value, float):
                    value = f"{value:.6g}"
                row.append("" if value is None else value)
            writer.writerow(row)
