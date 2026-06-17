"""Unit and integration tests for TotalLPIterations."""

from pyscipopt import Model

from gyozas.rewards import RewardFunction, TotalLPIterations

_INSTANCE = "tests/instance.lp"


def make_model() -> Model:
    m = Model()
    m.setParams({"display/verblevel": 0, "limits/nodes": 50})
    m.readProblem(_INSTANCE)
    return m


def make_strong_branching_model() -> Model:
    """A model forced to use full strong branching, so SB LP iterations accrue."""
    m = Model()
    m.setParams(
        {
            "display/verblevel": 0,
            "limits/nodes": 100,
            "presolving/maxrounds": 0,
            "separating/maxrounds": 0,
            "separating/maxroundsroot": 0,
            "branching/fullstrong/priority": 536870911,
            "branching/fullstrong/maxdepth": -1,
        }
    )
    m.readProblem(_INSTANCE)
    return m


class TestProtocol:
    def test_is_reward_function(self):
        assert isinstance(TotalLPIterations(), RewardFunction)


class TestReset:
    def test_reset_clears_counter(self):
        r = TotalLPIterations()
        r.n_total_lp_iterations = 999
        r.reset(make_model())
        assert r.n_total_lp_iterations == 0

    def test_initial_state(self):
        assert TotalLPIterations().n_total_lp_iterations == 0


class TestExtract:
    def test_returns_numeric(self):
        r = TotalLPIterations()
        m = make_model()
        r.reset(m)
        assert isinstance(r.extract(m, done=False), int | float)

    def test_delta_nonnegative(self):
        r = TotalLPIterations()
        m = make_model()
        r.reset(m)
        m.optimize()
        assert r.extract(m, done=True) >= 0

    def test_equals_sum_of_both_counters(self):
        r = TotalLPIterations()
        m = make_model()
        r.reset(m)
        m.optimize()
        delta = r.extract(m, done=True)
        assert delta == m.getNLPIterations() + m.getNStrongbranchLPIterations()

    def test_includes_strong_branching(self):
        r = TotalLPIterations()
        m = make_strong_branching_model()
        r.reset(m)
        m.optimize()
        delta = r.extract(m, done=True)
        sb = m.getNStrongbranchLPIterations()
        assert sb > 0
        # Total covers both node LPs and strong-branching LPs.
        assert delta >= sb
        assert delta == m.getNLPIterations() + sb


class TestMultiEpisode:
    def test_reset_between_episodes(self):
        r = TotalLPIterations()
        for _ in range(2):
            m = make_model()
            r.reset(m)
            m.optimize()
            assert r.extract(m, done=True) >= 0
