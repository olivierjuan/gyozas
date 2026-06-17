"""Correctness tests for gyozas.dynamics.probing.

Covers the gaps not exercised by test_probing.py: optimum correctness (incl. the
updateNodeLowerbound / reduction paths), maximization (gain-sign), reproducibility,
pseudocost seeding, and the pure infeasibility classifier (cutoff path)."""

import math

from pyscipopt import Model

from gyozas.dynamics.probing import ProbingDynamics, ProbingOracle
from gyozas.instances.independent_set import IndependentSetGenerator

_INSTANCE = "tests/instance.lp"


def make_minimization() -> Model:
    m = Model()
    m.setParams({"display/verblevel": 0})
    m.readProblem(_INSTANCE)
    return m


def make_maximization() -> Model:
    # Deterministic maximization instance (Independent Set is a maximization problem).
    g = IndependentSetGenerator(n_nodes=40, edge_probability=0.2, graph_type="erdos_renyi", rng=3)
    m = g.generate_instance(n_nodes=40, edge_probability=0.2, affinity=4, graph_type="erdos_renyi", rng=3)
    m.setParams({"display/verblevel": 0})
    return m


def reference_optimum(make_model) -> float:
    m = make_model()
    m.hideOutput()
    m.optimize()
    return m.getObjVal()


def drive_probe_all(d: ProbingDynamics, m: Model):
    """Run a full probe-all episode; return the per-step action-set lengths."""
    done, action_set = d.reset(m)
    lengths = []
    while not done:
        assert action_set is not None
        lengths.append(len(action_set))
        done, action_set = d.step(action_set)
    return lengths


# ---------------------------------------------------------------------------
# Optimum correctness (guards reductions + updateNodeLowerbound)
# ---------------------------------------------------------------------------


class TestOptimumCorrectness:
    def test_minimization_optimum_matches_reference(self):
        reference = reference_optimum(make_minimization)
        d = ProbingDynamics()
        m = make_minimization()
        drive_probe_all(d, m)
        assert math.isclose(m.getObjVal(), reference, rel_tol=1e-6, abs_tol=1e-6)
        assert m.getStatus() == "optimal"
        d.close()

    def test_maximization_optimum_matches_reference(self):
        # Validates the gain-sign math on a maximization problem.
        reference = reference_optimum(make_maximization)
        d = ProbingDynamics()
        m = make_maximization()
        drive_probe_all(d, m)
        assert math.isclose(m.getObjVal(), reference, rel_tol=1e-6, abs_tol=1e-6)
        assert m.getStatus() == "optimal"
        # Probing genuinely ran on the maximization instance.
        assert m.getNStrongbranchLPIterations() > 0
        d.close()


# ---------------------------------------------------------------------------
# Reduction path + pseudocost seeding on a real solve
# ---------------------------------------------------------------------------


class TestSideEffects:
    def test_reductions_applied(self):
        d = ProbingDynamics()
        m = make_minimization()
        drive_probe_all(d, m)
        # Strong-branch probes on this instance discover infeasible branches.
        assert d.oracle.n_reductions > 0
        d.close()

    def test_pseudocosts_seeded_when_probing(self):
        d = ProbingDynamics()
        m = make_minimization()
        drive_probe_all(d, m)
        assert d.oracle.n_pseudocost_updates > 0
        d.close()

    def test_no_pseudocost_updates_without_probing(self):
        d = ProbingDynamics()
        m = make_minimization()
        done, action_set = d.reset(m)
        while not done:
            assert action_set is not None
            done, action_set = d.step([])  # empty subset => no strong branching
        assert d.oracle.n_pseudocost_updates == 0
        d.close()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_identical_runs_match(self):
        d1, d2 = ProbingDynamics(), ProbingDynamics()
        m1, m2 = make_minimization(), make_minimization()
        lengths1 = drive_probe_all(d1, m1)
        lengths2 = drive_probe_all(d2, m2)
        assert lengths1 == lengths2
        assert m1.getNNodes() == m2.getNNodes()
        assert m1.getNStrongbranchLPIterations() == m2.getNStrongbranchLPIterations()
        assert (d1.oracle.n_branched, d1.oracle.n_reductions, d1.oracle.n_cutoffs) == (
            d2.oracle.n_branched,
            d2.oracle.n_reductions,
            d2.oracle.n_cutoffs,
        )
        assert math.isclose(m1.getObjVal(), m2.getObjVal(), rel_tol=1e-9)
        d1.close()
        d2.close()


# ---------------------------------------------------------------------------
# Pure infeasibility classifier (covers the cutoff path deterministically)
# ---------------------------------------------------------------------------


def _res(downinf: bool, upinf: bool) -> dict:
    return {"downinf": downinf, "upinf": upinf}


class TestClassifyInfeasibilities:
    def test_both_infeasible_is_cutoff(self):
        cutoff, tighten, forced = ProbingOracle._classify_infeasibilities({5: _res(True, True)}, can_change=True)
        assert cutoff and tighten == [] and forced is None

    def test_down_infeasible_tightens_lb(self):
        cutoff, tighten, forced = ProbingOracle._classify_infeasibilities({5: _res(True, False)}, can_change=True)
        assert not cutoff and tighten == [(5, "lb")] and forced is None

    def test_up_infeasible_tightens_ub(self):
        cutoff, tighten, forced = ProbingOracle._classify_infeasibilities({7: _res(False, True)}, can_change=True)
        assert not cutoff and tighten == [(7, "ub")] and forced is None

    def test_single_infeasible_without_permission_forces_branch(self):
        cutoff, tighten, forced = ProbingOracle._classify_infeasibilities({5: _res(True, False)}, can_change=False)
        assert not cutoff and tighten == [] and forced == 5

    def test_no_infeasibility(self):
        cutoff, tighten, forced = ProbingOracle._classify_infeasibilities({5: _res(False, False)}, can_change=True)
        assert not cutoff and tighten == [] and forced is None

    def test_both_infeasible_short_circuits_over_other_entries(self):
        results = {1: _res(True, False), 2: _res(True, True)}
        cutoff, tighten, forced = ProbingOracle._classify_infeasibilities(results, can_change=True)
        assert cutoff and tighten == [] and forced is None
