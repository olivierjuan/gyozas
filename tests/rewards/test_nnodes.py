"""Unit and integration tests for gyozas.rewards.nnodes.NNodes."""

from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.rewards import NNodes, RewardFunction

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
        assert isinstance(NNodes(), RewardFunction)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_counter(self):
        r = NNodes()
        r.last_n_nodes = 42
        r.reset(make_model())
        assert r.last_n_nodes == 0

    def test_initial_state(self):
        r = NNodes()
        assert r.last_n_nodes == 0


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


class TestExtract:
    def test_returns_numeric(self):
        r = NNodes()
        m = make_model()
        r.reset(m)
        result = r.extract(m, done=False)
        assert isinstance(result, int | float)

    def test_delta_nonnegative_after_solve(self):
        r = NNodes()
        m = make_model()
        r.reset(m)
        m.optimize()
        delta = r.extract(m, done=True)
        assert delta >= 0

    def test_delta_accumulates_over_steps(self):
        r = NNodes()
        m = make_model()
        r.reset(m)
        d = BranchingDynamics()
        done, action_set = d.reset(m)
        total = 0
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
        r = NNodes()
        for _ in range(2):
            m = make_model()
            r.reset(m)
            m.optimize()
            delta = r.extract(m, done=True)
            assert delta >= 0
