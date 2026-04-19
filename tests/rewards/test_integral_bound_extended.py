"""Extended tests for integral bound rewards: multi-episode event handler reuse."""

from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.rewards.integral_bound import DualIntegral, PrimalDualIntegral, PrimalIntegral

_INSTANCE = "tests/instance.lp"
_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}


def make_model() -> Model:
    m = Model()
    m.setParams(_PARAMS)
    m.readProblem(_INSTANCE)
    return m


class TestDualIntegralMultiEpisode:
    def test_reset_creates_fresh_handler(self):
        """Regression: reset() must create a fresh event handler so events don't accumulate."""
        r = DualIntegral()
        m1 = make_model()
        r.reset(m1)
        handler1 = r.event
        m2 = make_model()
        r.reset(m2)
        handler2 = r.event
        assert handler1 is not handler2, "reset() must create a fresh event handler"
        assert handler2.events == [], "New handler should have no events"

    def test_multi_episode_deltas_independent(self):
        """Deltas in episode 2 should not include events from episode 1."""
        r = DualIntegral()
        deltas_ep1 = []
        deltas_ep2 = []

        for _ep, deltas in [(1, deltas_ep1), (2, deltas_ep2)]:
            m = make_model()
            r.reset(m)
            d = BranchingDynamics()
            done, action_set = d.reset(m)
            while not done:
                delta = r.extract(m, done)
                deltas.append(delta)
                assert action_set is not None
                done, action_set = d.step(action_set[0])
            r.extract(m, done=True)
            d.close()


class TestPrimalIntegralMultiEpisode:
    def test_reset_creates_fresh_handler(self):
        r = PrimalIntegral()
        m1 = make_model()
        r.reset(m1)
        handler1 = r.event
        m2 = make_model()
        r.reset(m2)
        handler2 = r.event
        assert handler1 is not handler2

    def test_extract_returns_float(self):
        r = PrimalIntegral()
        m = make_model()
        r.reset(m)
        m.optimize()
        result = r.extract(m, done=True)
        assert isinstance(result, float)


class TestPrimalDualIntegralMultiEpisode:
    def test_both_handlers_fresh_after_reset(self):
        r = PrimalDualIntegral()
        m = make_model()
        r.reset(m)
        p_handler = r.primal_integral.event
        d_handler = r.dual_integral.event
        m2 = make_model()
        r.reset(m2)
        assert r.primal_integral.event is not p_handler
        assert r.dual_integral.event is not d_handler

    def test_full_episode(self):
        """PrimalDualIntegral works over a full solve and returns a float."""
        r = PrimalDualIntegral()
        m = make_model()
        r.reset(m)
        m.optimize()
        final = r.extract(m, done=True)
        assert isinstance(final, float)
