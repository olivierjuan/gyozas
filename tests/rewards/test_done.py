"""Unit and integration tests for gyozas.rewards.done.Done."""

from pyscipopt import Model

from gyozas.rewards import Done, RewardFunction

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
        assert isinstance(Done(), RewardFunction)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_does_not_raise(self):
        r = Done()
        m = make_model()
        r.reset(m)  # no-op, should not raise

    def test_reset_is_idempotent(self):
        r = Done()
        m = make_model()
        r.reset(m)
        r.reset(m)


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


class TestExtract:
    def test_returns_zero_before_solve(self):
        r = Done()
        m = make_model()
        r.reset(m)
        assert r.extract(m, done=False) == 0.0

    def test_returns_float(self):
        r = Done()
        m = make_model()
        r.reset(m)
        result = r.extract(m, done=False)
        assert isinstance(result, float)

    def test_returns_one_when_optimal(self):
        r = Done()
        m = make_model()
        m.setParam("limits/nodes", -1)  # no node limit
        r.reset(m)
        m.optimize()
        status = m.getStatus()
        result = r.extract(m, done=True)
        if status == "optimal":
            assert result == 1.0
        else:
            assert result == 0.0

    def test_returns_zero_on_node_limit(self):
        r = Done()
        m = make_model()
        m.setParam("limits/nodes", 1)
        r.reset(m)
        m.optimize()
        if m.getStatus() != "optimal":
            assert r.extract(m, done=True) == 0.0
