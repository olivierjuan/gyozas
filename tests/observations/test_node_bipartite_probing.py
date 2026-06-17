"""Tests for gyozas.observations.node_bipartite_probing.NodeBipartiteProbing."""

import numpy as np
import pytest
from pyscipopt import Model

from gyozas.dynamics.probing import ProbeLedger, ProbingDynamics
from gyozas.observations import ObservationFunction
from gyozas.observations.node_bipartite_ecole import NodeBipartiteEcole
from gyozas.observations.node_bipartite_probing import PROBING_FEATURES, NodeBipartiteProbing

_INSTANCE = "tests/instance.lp"


def make_model() -> Model:
    m = Model()
    m.setParams({"display/verblevel": 0, "limits/nodes": 40})
    m.readProblem(_INSTANCE)
    return m


def probe_steps(dyn: ProbingDynamics, model: Model, n: int):
    """Reset and take up to n probe-one-variable steps; return (done, action_set)."""
    done, action_set = dyn.reset(model)
    taken = 0
    while not done and taken < n:
        done, action_set = dyn.step(action_set[:1])
        taken += 1
    return done, action_set


class TestProtocolAndShape:
    def test_is_observation_function(self):
        assert isinstance(NodeBipartiteProbing(ProbeLedger()), ObservationFunction)

    def test_feature_names(self):
        assert NodeBipartiteProbing(ProbeLedger()).probing_feature_names == PROBING_FEATURES

    def test_appends_five_columns(self):
        ledger = ProbeLedger()
        dyn = ProbingDynamics(ledger=ledger)
        m = make_model()
        done, _ = dyn.reset(m)
        if done:
            dyn.close()
            pytest.skip("Instance solved at root")
        try:
            base = NodeBipartiteEcole().extract(m, done=False)
            probing = NodeBipartiteProbing(ledger).extract(m, done=False)
            assert probing.variable_features.shape[0] == base.variable_features.shape[0]
            assert probing.variable_features.shape[1] == base.variable_features.shape[1] + len(PROBING_FEATURES)
        finally:
            dyn.close()

    def test_returns_none_when_done(self):
        assert NodeBipartiteProbing(ProbeLedger()).extract(make_model(), done=True) is None


class TestProbingFeatures:
    def test_evaluations_recorded_in_observation(self):
        ledger = ProbeLedger()
        dyn = ProbingDynamics(ledger=ledger)
        m = make_model()
        done, _ = probe_steps(dyn, m, 3)
        if done:
            dyn.close()
            pytest.skip("Episode ended before features could be observed")
        obs = NodeBipartiteProbing(ledger).extract(m, done=False)
        dyn.close()
        n_eval = obs.variable_features[:, -len(PROBING_FEATURES)]  # n_evaluations is first appended
        assert (n_eval > 0).any()  # some probed variable shows up

    def test_empty_subset_leaves_evaluations_zero(self):
        ledger = ProbeLedger()
        dyn = ProbingDynamics(ledger=ledger)
        m = make_model()
        done, action_set = dyn.reset(m)
        if done:
            dyn.close()
            pytest.skip("Instance solved at root")
        # Take a few empty-subset steps (pure pseudocost branching, no probing).
        obs_obj = NodeBipartiteProbing(ledger)
        obs = obs_obj.extract(m, done=False)
        taken = 0
        while not done and taken < 3:
            done, action_set = dyn.step([])
            if not done:
                obs = obs_obj.extract(m, done=False)
            taken += 1
        dyn.close()
        n_eval = obs.variable_features[:, -len(PROBING_FEATURES)]
        assert float(n_eval.max()) == 0.0

    def test_normalize_true_vs_false(self):
        ledger = ProbeLedger()
        dyn = ProbingDynamics(ledger=ledger)
        m = make_model()
        done, _ = probe_steps(dyn, m, 3)
        if done:
            dyn.close()
            pytest.skip("Episode ended before features could be observed")
        raw = NodeBipartiteProbing(ledger, normalize=False).extract(m, done=False)
        norm = NodeBipartiteProbing(ledger, normalize=True).extract(m, done=False)
        dyn.close()
        raw_neval = raw.variable_features[:, -len(PROBING_FEATURES)]
        norm_neval = norm.variable_features[:, -len(PROBING_FEATURES)]
        assert raw_neval.max() >= 1.0  # raw counts are integers
        assert norm_neval.max() < 1.0  # saturating transform is bounded in [0, 1)
        # normalized == count / (count + CSTE), with CSTE = 5.0
        assert np.allclose(norm_neval, raw_neval / (raw_neval + 5.0))


class TestLedgerLifecycle:
    def test_ledger_shared_with_dynamics(self):
        dyn = ProbingDynamics()
        obs = NodeBipartiteProbing(ledger=dyn.ledger)
        assert obs.ledger is dyn.ledger

    def test_reset_clears_ledger(self):
        ledger = ProbeLedger()
        dyn = ProbingDynamics(ledger=ledger)
        done, _ = probe_steps(dyn, make_model(), 3)
        recorded = len(ledger._count)
        dyn.close()
        dyn.reset(make_model())  # new episode clears the ledger
        dyn.close()
        assert recorded >= 0  # sanity (may be 0 if solved early)
        assert len(ledger._count) == 0
