import contextlib
from dataclasses import dataclass

from pyscipopt import SCIP_EVENTTYPE, Eventhdlr, Model

from .arithmetic import ArithmeticMixin


@dataclass
class EventData:
    data: float
    time: float


def BoundEventHandlerGenerator(event_type, func) -> type:
    """
    Generates a BoundEventHandler class that can be used to catch events in the SCIP solver.
    :param event_type: The type of event to catch.
    :param func: The function to call when the event is caught.
    :return: A BoundEventHandler class.
    """

    class _BoundEventHandler(Eventhdlr):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[EventData] = []
            self._registered = False

        def eventinit(self) -> None:
            self.model.catchEvent(event_type, self)
            self._registered = True

        def eventexit(self) -> None:
            # Guard against double-call: SCIPfree triggers SCIPexit which calls
            # eventexit again after our close() already called dropEvent.
            if self._registered:
                self._registered = False
                self.model.dropEvent(event_type, self)

        def eventexec(self, event) -> dict:
            time = self.model.getSolvingTime()
            data = func(self.model)
            self.events.append(EventData(data, time))
            return {}

        def _compute_integral(self) -> float:
            if len(self.events) < 1:
                return 0.0
            elif len(self.events) == 1:
                return self.events[0].data * self.events[0].time
            integral = 0.0
            for i in range(1, len(self.events)):
                delta_time = self.events[i].time - self.events[i - 1].time
                average_value = (self.events[i].data + self.events[i - 1].data) / 2
                integral += delta_time * average_value
            return integral

    return _BoundEventHandler


PrimalBoundEventHandler = BoundEventHandlerGenerator(SCIP_EVENTTYPE.BESTSOLFOUND, lambda model: model.getPrimalbound())
DualBoundEventHandler = BoundEventHandlerGenerator(SCIP_EVENTTYPE.LPEVENT, lambda model: model.getDualbound())


class DualIntegral(ArithmeticMixin):
    """Reward based on the change in the dual bound integral over solving time.

    Uses a SCIP event handler to track the dual bound at each LP event and
    computes the trapezoidal integral.
    """

    def __init__(self) -> None:
        self.dual_integral = 0.0
        self.event = DualBoundEventHandler()

    def close(self) -> None:
        """Drop caught events so the SCIP model can be GC'd.

        catchEvent() calls Py_INCREF(model) internally; without a matching
        dropEvent() the model's refcount never reaches zero.
        """
        if self.event.model is not None:
            with contextlib.suppress(Exception):
                self.event.eventexit()
            self.event.model = None

    def __del__(self) -> None:
        self.close()

    def reset(self, model: Model) -> None:
        self.close()
        self.dual_integral = 0.0
        self.event = DualBoundEventHandler()
        model.includeEventhdlr(self.event, "dual_bound_tracker", "tracks dual bound for integral computation")

    def extract(self, model: Model, done: bool) -> float:
        dual_integral = self.event._compute_integral()
        delta = dual_integral - self.dual_integral
        self.dual_integral = dual_integral
        return delta


class PrimalIntegral(ArithmeticMixin):
    """Reward based on the change in the primal bound integral over solving time.

    Uses a SCIP event handler to track the primal bound at each best-solution-found
    event and computes the trapezoidal integral.
    """

    def __init__(self) -> None:
        self.primal_integral = 0.0
        self.event = PrimalBoundEventHandler()

    def close(self) -> None:
        if self.event.model is not None:
            with contextlib.suppress(Exception):
                self.event.eventexit()
            self.event.model = None

    def __del__(self) -> None:
        self.close()

    def reset(self, model: Model) -> None:
        self.close()
        self.primal_integral = 0.0
        self.event = PrimalBoundEventHandler()
        model.includeEventhdlr(self.event, "primal_bound_tracker", "tracks primal bound for integral computation")

    def extract(self, model: Model, done: bool) -> float:
        primal_integral = self.event._compute_integral()
        delta = primal_integral - self.primal_integral
        self.primal_integral = primal_integral
        return delta


class PrimalDualIntegral(ArithmeticMixin):
    """Reward that sums the primal and dual bound integral changes."""

    def __init__(self) -> None:
        self.primal_integral = PrimalIntegral()
        self.dual_integral = DualIntegral()

    def reset(self, model: Model) -> None:
        self.primal_integral.reset(model)
        self.dual_integral.reset(model)

    def extract(self, model: Model, done: bool) -> float:
        primal_delta = self.primal_integral.extract(model, done)
        dual_delta = self.dual_integral.extract(model, done)
        return primal_delta + dual_delta
