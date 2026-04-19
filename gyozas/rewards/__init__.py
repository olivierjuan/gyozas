from typing import Protocol, runtime_checkable

from pyscipopt import Model

from .arithmetic import ArithmeticMixin
from .arithmetic import _Constant as Constant
from .done import Done
from .integral_bound import DualIntegral, PrimalDualIntegral, PrimalIntegral
from .lp_iterations import LPIterations
from .nnodes import NNodes
from .solving_time import SolvingTime


@runtime_checkable
class RewardFunction(Protocol):
    def reset(self, model: Model) -> None: ...
    def extract(self, model: Model, done: bool) -> float: ...


__all__ = [
    "RewardFunction",
    "ArithmeticMixin",
    "Constant",
    "Done",
    "PrimalIntegral",
    "DualIntegral",
    "PrimalDualIntegral",
    "LPIterations",
    "NNodes",
    "SolvingTime",
]
