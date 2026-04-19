"""Unit and integration tests for gyozas.rewards.lp_iterations.LPIterations."""

from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.rewards import LPIterations, RewardFunction

_INSTANCE = "tests/instance.lp"
_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}


def make_model() -> Model:
    m = Model()
    m.setParams(_PARAMS)
    m.readProblem(_INSTANCE)
    return m


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_is_reward_function(self):
        assert isinstance(LPIterations(), RewardFunction)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_counter(self):
        r = LPIterations()
        r.n_lp_iterations = 999
        r.reset(make_model())
        assert r.n_lp_iterations == 0

    def test_initial_state(self):
        r = LPIterations()
        assert r.n_lp_iterations == 0


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


class TestExtract:
    def test_returns_numeric(self):
        r = LPIterations()
        m = make_model()
        r.reset(m)
        result = r.extract(m, done=False)
        assert isinstance(result, int | float)

    def test_delta_nonnegative(self):
        r = LPIterations()
        m = make_model()
        r.reset(m)
        m.optimize()
        delta = r.extract(m, done=True)
        assert delta >= 0

    def test_consecutive_extracts_accumulate(self):
        r = LPIterations()
        m = make_model()
        r.reset(m)
        d = BranchingDynamics()
        done, action_set = d.reset(m)
        total = 0
        first = r.extract(m, done=False)
        total += first
        while not done:
            assert action_set is not None
            done, action_set = d.step(action_set[0])
            delta = r.extract(m, done)
            assert delta >= 0
            total += delta
        d.close()
        assert total >= 0


# ---------------------------------------------------------------------------
# Multi-episode
# ---------------------------------------------------------------------------


class TestMultiEpisode:
    def test_reset_between_episodes(self):
        r = LPIterations()
        for _ in range(2):
            m = make_model()
            r.reset(m)
            m.optimize()
            delta = r.extract(m, done=True)
            assert delta >= 0
