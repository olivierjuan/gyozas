"""Unit and integration tests for gyozas.rewards.integral_bound."""

from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.rewards import (
    DualIntegral,
    PrimalDualIntegral,
    PrimalIntegral,
    RewardFunction,
)
from gyozas.rewards.integral_bound import (
    DualBoundEventHandler,
    EventData,
    PrimalBoundEventHandler,
)

_INSTANCE = "tests/instance.lp"
_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}


def make_model() -> Model:
    m = Model()
    m.setParams(_PARAMS)
    m.readProblem(_INSTANCE)
    return m


# ---------------------------------------------------------------------------
# EventData
# ---------------------------------------------------------------------------


class TestEventData:
    def test_fields(self):
        e = EventData(data=1.5, time=0.1)
        assert e.data == 1.5
        assert e.time == 0.1


# ---------------------------------------------------------------------------
# BoundEventHandler compute_integral
# ---------------------------------------------------------------------------


class TestComputeIntegral:
    def test_empty_events(self):
        handler = DualBoundEventHandler()
        assert handler._compute_integral() == 0.0

    def test_single_event(self):
        handler = DualBoundEventHandler()
        handler.events.append(EventData(data=2.0, time=3.0))
        assert handler._compute_integral() == 6.0

    def test_two_events_trapezoidal(self):
        handler = PrimalBoundEventHandler()
        handler.events.append(EventData(data=1.0, time=0.0))
        handler.events.append(EventData(data=3.0, time=2.0))
        # trapezoidal: (2.0 - 0.0) * (1.0 + 3.0) / 2 = 4.0
        assert abs(handler._compute_integral() - 4.0) < 1e-12

    def test_three_events(self):
        handler = DualBoundEventHandler()
        handler.events.append(EventData(data=0.0, time=0.0))
        handler.events.append(EventData(data=2.0, time=1.0))
        handler.events.append(EventData(data=4.0, time=2.0))
        # segment 1: 1.0 * (0+2)/2 = 1.0
        # segment 2: 1.0 * (2+4)/2 = 3.0
        assert abs(handler._compute_integral() - 4.0) < 1e-12

    def test_constant_function(self):
        handler = PrimalBoundEventHandler()
        handler.events.append(EventData(data=5.0, time=0.0))
        handler.events.append(EventData(data=5.0, time=3.0))
        # 3.0 * 5.0 = 15.0
        assert abs(handler._compute_integral() - 15.0) < 1e-12


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_dual_is_reward_function(self):
        assert isinstance(DualIntegral(), RewardFunction)

    def test_primal_is_reward_function(self):
        assert isinstance(PrimalIntegral(), RewardFunction)

    def test_primal_dual_is_reward_function(self):
        assert isinstance(PrimalDualIntegral(), RewardFunction)


# ---------------------------------------------------------------------------
# DualIntegral
# ---------------------------------------------------------------------------


class TestDualIntegral:
    def test_reset_clears_integral(self):
        r = DualIntegral()
        r.dual_integral = 999.0
        m = make_model()
        r.reset(m)
        assert r.dual_integral == 0.0

    def test_extract_returns_numeric(self):
        r = DualIntegral()
        m = make_model()
        r.reset(m)
        m.optimize()
        delta = r.extract(m, done=True)
        assert isinstance(delta, float)

    def test_extract_nonnegative_delta(self):
        r = DualIntegral()
        m = make_model()
        r.reset(m)
        m.optimize()
        delta = r.extract(m, done=True)
        assert delta >= 0.0

    def test_multiple_extracts(self):
        r = DualIntegral()
        m = make_model()
        r.reset(m)
        d = BranchingDynamics()
        done, action_set = d.reset(m)
        while not done:
            assert action_set is not None
            done, action_set = d.step(action_set[0])
            r.extract(m, done)
        d.close()


# ---------------------------------------------------------------------------
# PrimalIntegral
# ---------------------------------------------------------------------------


class TestPrimalIntegral:
    def test_reset_clears_integral(self):
        r = PrimalIntegral()
        r.primal_integral = 999.0
        m = make_model()
        r.reset(m)
        assert r.primal_integral == 0.0

    def test_extract_returns_numeric(self):
        r = PrimalIntegral()
        m = make_model()
        r.reset(m)
        m.optimize()
        delta = r.extract(m, done=True)
        assert isinstance(delta, float)


# ---------------------------------------------------------------------------
# PrimalDualIntegral
# ---------------------------------------------------------------------------


class TestPrimalDualIntegral:
    def test_reset_clears_both(self):
        r = PrimalDualIntegral()
        m = make_model()
        r.reset(m)
        assert r.primal_integral.primal_integral == 0.0
        assert r.dual_integral.dual_integral == 0.0

    def test_extract_sums_both(self):
        r = PrimalDualIntegral()
        m = make_model()
        r.reset(m)
        m.optimize()
        delta = r.extract(m, done=True)
        assert isinstance(delta, float)


# ---------------------------------------------------------------------------
# Multi-episode
# ---------------------------------------------------------------------------


class TestMultiEpisode:
    def test_dual_multi_episode(self):
        r = DualIntegral()
        for _ in range(2):
            m = make_model()
            r.reset(m)
            m.optimize()
            r.extract(m, done=True)

    def test_primal_multi_episode(self):
        r = PrimalIntegral()
        for _ in range(2):
            m = make_model()
            r.reset(m)
            m.optimize()
            r.extract(m, done=True)
