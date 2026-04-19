"""Unit tests for gyozas.informations.time_since_last_step.TimeSinceLastStep."""

import time

from pyscipopt import Model

from gyozas.informations import InformationFunction
from gyozas.informations.time_since_last_step import TimeSinceLastStep


class TestProtocol:
    def test_is_information_function(self):
        assert isinstance(TimeSinceLastStep(), InformationFunction)


class TestReset:
    def test_reset_updates_timestamp(self):
        t = TimeSinceLastStep()
        before = t.previous_timestamp
        time.sleep(0.01)
        t.reset(Model())
        assert t.previous_timestamp > before


class TestExtract:
    def test_returns_nonnegative(self):
        t = TimeSinceLastStep()
        t.reset(Model())
        delta = t.extract(Model(), done=False)
        assert delta >= 0.0

    def test_returns_float(self):
        t = TimeSinceLastStep()
        t.reset(Model())
        delta = t.extract(Model(), done=False)
        assert isinstance(delta, float)

    def test_consecutive_calls(self):
        t = TimeSinceLastStep()
        t.reset(Model())
        d1 = t.extract(Model(), done=False)
        d2 = t.extract(Model(), done=False)
        assert d1 >= 0.0
        assert d2 >= 0.0

    def test_done_flag_irrelevant(self):
        t = TimeSinceLastStep()
        t.reset(Model())
        d1 = t.extract(Model(), done=True)
        assert d1 >= 0.0
