"""Extended tests for gyozas.rewards.done.Done."""

from pyscipopt import Model

from gyozas.rewards.done import Done

_INSTANCE = "tests/instance.lp"


def make_model(**extra_params) -> Model:
    m = Model()
    m.setParam("display/verblevel", 0)
    m.setParams(extra_params)
    m.readProblem(_INSTANCE)
    return m


class TestExtractStatus:
    def test_returns_one_when_optimal(self):
        r = Done()
        m = make_model()
        r.reset(m)
        m.optimize()
        if m.getStatus() == "optimal":
            assert r.extract(m, done=True) == 1.0

    def test_returns_zero_on_node_limit(self):
        r = Done()
        m = make_model(**{"limits/nodes": 1})
        r.reset(m)
        m.optimize()
        if m.getStatus() != "optimal":
            assert r.extract(m, done=True) == 0.0

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

    def test_multi_episode(self):
        r = Done()
        for _ in range(2):
            m = make_model()
            r.reset(m)
            m.optimize()
            result = r.extract(m, done=True)
            assert result in (0.0, 1.0)
