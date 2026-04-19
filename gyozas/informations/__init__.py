from typing import Any, Protocol, runtime_checkable

from pyscipopt import Model

from .empty import Empty
from .time_since_last_step import TimeSinceLastStep


@runtime_checkable
class InformationFunction(Protocol):
    def reset(self, model: Model) -> None: ...
    def extract(self, model: Model, done: bool) -> Any: ...


__all__ = ["InformationFunction", "Empty", "TimeSinceLastStep"]
