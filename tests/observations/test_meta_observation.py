"""Unit and integration tests for gyozas.observations.meta_observation.MetaObservation."""

import pytest
from pyscipopt import Model

from gyozas.dynamics.branching import BranchingDynamics
from gyozas.observations import Pseudocosts, StrongBranchingScores
from gyozas.observations.meta_observation import MetaObservation

_INSTANCE = "tests/instance.lp"
_PARAMS = {"display/verblevel": 0, "limits/nodes": 20}


def make_model() -> Model:
    m = Model()
    m.setParams(_PARAMS)
    m.readProblem(_INSTANCE)
    return m


class _DummyObs:
    """Minimal observation function for unit testing."""

    def __init__(self, value):
        self._value = value
        self.reset_count = 0

    def reset(self, model):
        self.reset_count += 1

    def extract(self, model, done):
        return self._value


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_list_input(self):
        meta = MetaObservation([_DummyObs(1), _DummyObs(2)])
        assert isinstance(meta.observations, list)

    def test_tuple_input(self):
        meta = MetaObservation((_DummyObs(1), _DummyObs(2)))
        assert isinstance(meta.observations, tuple)

    def test_dict_input(self):
        meta = MetaObservation({"a": _DummyObs(1), "b": _DummyObs(2)})
        assert isinstance(meta.observations, dict)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_calls_all_list(self):
        obs_a = _DummyObs(1)
        obs_b = _DummyObs(2)
        meta = MetaObservation([obs_a, obs_b])
        meta.reset(Model())
        assert obs_a.reset_count == 1
        assert obs_b.reset_count == 1

    def test_reset_calls_all_dict(self):
        obs_a = _DummyObs(1)
        obs_b = _DummyObs(2)
        meta = MetaObservation({"x": obs_a, "y": obs_b})
        meta.reset(Model())
        assert obs_a.reset_count == 1
        assert obs_b.reset_count == 1

    def test_reset_calls_all_tuple(self):
        obs_a = _DummyObs(1)
        obs_b = _DummyObs(2)
        meta = MetaObservation((obs_a, obs_b))
        meta.reset(Model())
        assert obs_a.reset_count == 1
        assert obs_b.reset_count == 1


# ---------------------------------------------------------------------------
# Extract: list
# ---------------------------------------------------------------------------


class TestExtractList:
    def test_returns_list(self):
        meta = MetaObservation([_DummyObs(10), _DummyObs(20)])
        result = meta.extract(Model(), done=False)
        assert isinstance(result, list)
        assert result == [10, 20]

    def test_preserves_order(self):
        meta = MetaObservation([_DummyObs("a"), _DummyObs("b"), _DummyObs("c")])
        result = meta.extract(Model(), done=False)
        assert result == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Extract: tuple
# ---------------------------------------------------------------------------


class TestExtractTuple:
    def test_returns_tuple(self):
        meta = MetaObservation((_DummyObs(10), _DummyObs(20)))
        result = meta.extract(Model(), done=False)
        assert isinstance(result, tuple)

    def test_values_correct(self):
        meta = MetaObservation((_DummyObs(5),))
        result = meta.extract(Model(), done=False)
        assert result == (5,)


# ---------------------------------------------------------------------------
# Extract: dict
# ---------------------------------------------------------------------------


class TestExtractDict:
    def test_returns_dict(self):
        meta = MetaObservation({"foo": _DummyObs(1), "bar": _DummyObs(2)})
        result = meta.extract(Model(), done=False)
        assert isinstance(result, dict)
        assert result == {"foo": 1, "bar": 2}

    def test_preserves_keys(self):
        meta = MetaObservation({"alpha": _DummyObs(None)})
        result = meta.extract(Model(), done=False)
        assert "alpha" in result


# ---------------------------------------------------------------------------
# Extract: done flag forwarded
# ---------------------------------------------------------------------------


class TestExtractDone:
    def test_done_forwarded(self):
        class _CheckDone:
            def reset(self, model):
                pass

            def extract(self, model, done):
                return done

        meta = MetaObservation([_CheckDone()])
        assert meta.extract(Model(), done=True) == [True]
        assert meta.extract(Model(), done=False) == [False]


# ---------------------------------------------------------------------------
# Integration with real observation functions
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_with_pseudocosts_list(self):
        meta = MetaObservation([Pseudocosts()])
        m = make_model()
        d = BranchingDynamics()
        done, action_set = d.reset(m)
        if done:
            d.close()
            pytest.skip("Instance solved at root")
        meta.reset(m)
        result = meta.extract(m, done=False)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is not None
        d.close()

    def test_with_dict_of_observations(self):
        meta = MetaObservation(
            {
                "pseudo": Pseudocosts(),
                "strong": StrongBranchingScores(),
            }
        )
        m = make_model()
        d = BranchingDynamics()
        done, action_set = d.reset(m)
        if done:
            d.close()
            pytest.skip("Instance solved at root")
        meta.reset(m)
        result = meta.extract(m, done=False)
        assert isinstance(result, dict)
        assert "pseudo" in result
        assert "strong" in result
        d.close()
