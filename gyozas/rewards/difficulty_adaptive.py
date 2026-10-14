"""The difficulty-adaptive reward of TGPPO 2025 (Tree-Gate PPO).

Port of ``RewardH3`` from the official repository
(github.com/benmhamed-a/tgppo-aaai-26, ``project/reward/reward.py``): a
per-step shaped reward whose component weights adapt to the difficulty of the
instance, estimated from a *baseline* SCIP solve (run without the learned
brancher). The difficulty ``d`` is a sigmoid-free linear map of
``log1p(baseline_nodes)`` between log(2) (trivial) and log(1e6 + 1) (hard);
easy instances weight node efficiency, hard instances weight gap/PDI progress.

Two protocol facts the porter must know (both mirror the official code):

- The baseline statistics (nodes, gap, primal-dual integral, final status)
  come from a separate solve of the same instance, computed by the training
  loop and injected with :meth:`DifficultyAdaptive.configure` **before** the
  episode starts. Unconfigured, the reward falls back to the official
  defaults (a 1-node baseline that timed out).
- ``time_limit`` must match the solver's own time limit — the pace component
  reads the fraction of it already spent.
"""

import math

import numpy as np
from pyscipopt import Model

#: log1p bounds of the difficulty schedule: log(2) .. log(1e6 + 1).
_LO = math.log1p(1.0)
_HI = math.log1p(1e6)


def _tanh(x: float, s: float = 1.0) -> float:
    return float(np.tanh(s * x))


def _ratio(num: float, den: float, cap: float | None = None) -> float:
    val = float(num) / max(float(den), 1e-12)
    return min(val, cap) if cap is not None else val


class DifficultyAdaptive:
    """H3: per-step node efficiency, pace, gap, PDI and progress, weighted by
    instance difficulty; terminal bonus from the final status and speedup.

    Step rewards are clipped to ``[-1, 1]``; the terminal reward (``done``)
    is a bonus of ``1 + 3*speedup`` (optimal), ``0.8 + 2*speedup``
    (infeasible/unbounded) or ``0.5*speedup + 0.3*gap_gain + 0.2*pdi_gain``
    (timelimit), with speedup = baseline_nodes / nodes capped at 5.
    """

    def __init__(self, scale: float = 2.0) -> None:
        self.scale = scale
        self.baseline_nodes: float | None = None
        self.baseline_gap: float | None = None
        self.baseline_pdi: float | None = None
        self.solver_status: str | None = None
        self.time_limit: float | None = None
        self.reset(None)

    def configure(
        self,
        baseline_nodes: float,
        baseline_gap: float,
        baseline_pdi: float,
        solver_status: str,
        time_limit: float,
    ) -> None:
        """Inject the episode's baseline statistics (the official info dict)."""
        self.baseline_nodes = baseline_nodes
        self.baseline_gap = baseline_gap
        self.baseline_pdi = baseline_pdi
        self.solver_status = solver_status
        self.time_limit = time_limit

    def reset(self, model: Model) -> None:
        # The official reset()'s None-handling defaults.
        self.B = max(float(self.baseline_nodes or 1.0), 1.0)
        self.logB = math.log1p(self.B)
        self.baseline_gap = float(self.baseline_gap or 0.0)
        self.baseline_pdi = max(float(self.baseline_pdi or 1.0), 1.0)
        self.solver_status = self.solver_status or "timelimit"
        self.time_limit = max(float(self.time_limit or 3600.0), 1.0)

        # Difficulty weight schedule: easy -> emphasize speedup, hard -> gap/PDI.
        d = min(max((self.logB - _LO) / (_HI - _LO), 0.0), 1.0)
        w = {
            "nodes": 0.55 * (1 - d) + 0.25 * d,
            "gap": 0.10 * (1 - d) + 0.30 * d,
            "pdi": 0.05 * (1 - d) + 0.20 * d,
            "progress": 0.15 * (1 - d) + 0.15 * d,
            "pace": 0.15 * (1 - d) + 0.10 * d,
        }
        total = sum(w.values())
        self.w = {k: v / total for k, v in w.items()}

        # The progress component compares against the previous extraction.
        self.prev_gap = float("inf")
        self.prev_pdi = 0.0
        self.prev_pb = None
        self.prev_db = None
        # The official computes the first reward only at the *second* branching
        # decision (for the first one); the extraction before any decision has
        # no counterpart and is a no-op.
        self._first_extraction = True

    def _node_eff(self, nodes: int) -> float:
        eff = 1.0 - (math.log1p(nodes) / self.logB)
        return _tanh(eff, s=self.scale)

    def _pace(self, nodes: int, tfrac: float) -> float:
        # Piecewise expected curve: early aggressive, then conservative; the
        # exponent shrinks with difficulty (harder -> smaller exponent).
        exponent = 0.9 - 0.5 * min(max(self.logB / _HI, 0.0), 1.0)
        expected = self.B * (tfrac**exponent)
        return _tanh((expected - nodes) / (expected + 1.0), s=self.scale)

    def extract(self, model: Model, done: bool) -> float:
        if self._first_extraction:
            # Before the first decision: no transition to score (the official
            # starts remembering at the second branchexeclp call).
            self._first_extraction = False
            return 0.0

        nodes = int(model.getNNodes())
        tfrac = min(max(model.getSolvingTime() / self.time_limit, 0.0), 1.0)
        gap = float(model.getGap())
        if math.isinf(gap):
            gap = 1e6
        pdi = float(model.getPrimalDualIntegral())
        pb = model.getPrimalbound()
        db = model.getDualbound()

        r_nodes = self._node_eff(nodes)
        r_pace = self._pace(nodes, tfrac)
        r_progress = 0.0
        # Gap improvement.
        if self.prev_gap < float("inf"):
            r_progress += _tanh((self.prev_gap - gap) / max(abs(self.prev_gap), 1e-9), s=self.scale)
        # Bound improvements (minimization sense, as the official).
        if self.prev_pb is not None and pb < self.prev_pb:
            r_progress += _tanh((self.prev_pb - pb) / (abs(self.prev_pb) + 1e-9), s=self.scale)
        if self.prev_db is not None and db > self.prev_db:
            r_progress += _tanh((db - self.prev_db) / (abs(self.prev_db) + 1e-9), s=self.scale)
        # PDI decrease.
        if self.prev_pdi > 0:
            r_progress += _tanh((self.prev_pdi - pdi) / self.baseline_pdi, s=self.scale)

        self.prev_gap = gap
        self.prev_pdi = pdi
        self.prev_pb = pb
        self.prev_db = db

        if not done:
            reward = (
                self.w["nodes"] * r_nodes
                + self.w["pace"] * r_pace
                + self.w["gap"] * -_tanh(gap / (self.baseline_gap + 1e-9), s=0.5)
                + self.w["pdi"] * _tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
                + self.w["progress"] * r_progress
            )
            return float(np.clip(reward, -1.0, 1.0))

        # Terminal: blend speedup and quality gains, by final status.
        status = model.getStatus()
        speedup = _ratio(self.B, max(nodes, 1), cap=5.0)
        gap_gain = _tanh(self.baseline_gap - gap)
        pdi_gain = _tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
        if status == "optimal":
            return float(1.0 + 3.0 * speedup)
        if status in ("infeasible", "unbounded"):
            return float(0.8 + 2.0 * speedup)
        if status == "timelimit":
            return float(0.5 * speedup + 0.3 * gap_gain + 0.2 * pdi_gain)
        return float(0.3 * speedup)
