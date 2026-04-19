"""Unit and integration tests for gyozas.rewards.solving_time.SolvingTime."""

from pyscipopt import Model

from gyozas.rewards import RewardFunction, SolvingTime

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
        assert isinstance(SolvingTime(), RewardFunction)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_counter(self):
        r = SolvingTime()
        r.solving_time = 999.0
        r.reset(make_model())
        assert r.solving_time == 0

    def test_initial_state(self):
        r = SolvingTime()
        assert r.solving_time == 0


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


class TestExtract:
    def test_returns_numeric(self):
        r = SolvingTime()
        m = make_model()
        r.reset(m)
        result = r.extract(m, done=False)
        assert isinstance(result, int | float)

    def test_delta_nonnegative_after_solve(self):
        r = SolvingTime()
        m = make_model()
        r.reset(m)
        m.optimize()
        delta = r.extract(m, done=True)
        assert delta >= 0.0


# ---------------------------------------------------------------------------
# Multi-episode
# ---------------------------------------------------------------------------


class TestMultiEpisode:
    def test_reset_between_episodes(self):
        r = SolvingTime()
        for _ in range(2):
            m = make_model()
            r.reset(m)
            m.optimize()
            delta = r.extract(m, done=True)
            assert delta >= 0.0
