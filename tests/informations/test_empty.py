"""Unit tests for gyozas.informations.empty.Empty."""

from pyscipopt import Model

from gyozas.informations import InformationFunction
from gyozas.informations.empty import Empty


class TestProtocol:
    def test_is_information_function(self):
        assert isinstance(Empty(), InformationFunction)


class TestReset:
    def test_reset_does_not_raise(self):
        e = Empty()
        m = Model()
        e.reset(m)  # no-op

    def test_reset_is_idempotent(self):
        e = Empty()
        m = Model()
        e.reset(m)
        e.reset(m)


class TestExtract:
    def test_returns_none(self):
        e = Empty()
        m = Model()
        assert e.extract(m, done=False) is None

    def test_returns_none_when_done(self):
        e = Empty()
        m = Model()
        assert e.extract(m, done=True) is None
