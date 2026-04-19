"""Arithmetic composition for reward functions.

Any class inheriting ``ArithmeticMixin`` gains operator overloads so that
reward functions can be combined with plain Python arithmetic::

    reward = -NNodes() * 0.5 + SolvingTime().cumsum()

Every operator returns another ``ArithmeticMixin`` instance, so the result
satisfies the ``RewardFunction`` protocol and supports further composition.
"""

from __future__ import annotations

import math
import operator as _op
from abc import abstractmethod
from collections.abc import Callable

from pyscipopt import Model


class ArithmeticMixin:
    """Mixin that equips a reward function with arithmetic operators.

    Concrete reward classes should inherit from this mixin in addition to
    implementing ``reset`` / ``extract``.  The resulting objects satisfy the
    ``RewardFunction`` protocol and compose freely::

        reward = (NNodes() + LPIterations() * 0.1).cumsum()
    """

    # --- binary operators ---------------------------------------------------

    def __add__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.add)

    def __radd__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.add)

    def __sub__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.sub)

    def __rsub__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.sub)

    def __mul__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.mul)

    def __rmul__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.mul)

    def __matmul__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.matmul)

    def __rmatmul__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.matmul)

    def __truediv__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.truediv)

    def __rtruediv__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.truediv)

    def __floordiv__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.floordiv)

    def __rfloordiv__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.floordiv)

    def __mod__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.mod)

    def __rmod__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.mod)

    def __divmod__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, divmod)

    def __rdivmod__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, divmod)

    def __pow__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.pow)

    def __rpow__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.pow)

    def __lshift__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.lshift)

    def __rlshift__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.lshift)

    def __rshift__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.rshift)

    def __rrshift__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.rshift)

    def __and__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.and_)

    def __rand__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.and_)

    def __xor__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.xor)

    def __rxor__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.xor)

    def __or__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(self, other, _op.or_)

    def __ror__(self, other: object) -> ArithmeticMixin:
        return _BinaryOp(other, self, _op.or_)

    # --- unary operators ----------------------------------------------------

    def __neg__(self) -> ArithmeticMixin:
        return _UnaryOp(self, _op.neg)

    def __pos__(self) -> ArithmeticMixin:
        return _UnaryOp(self, _op.pos)

    def __abs__(self) -> ArithmeticMixin:
        return _UnaryOp(self, abs)

    def __invert__(self) -> ArithmeticMixin:
        return _UnaryOp(self, _op.invert)

    def __round__(self) -> ArithmeticMixin:
        return _UnaryOp(self, round)

    def __trunc__(self) -> ArithmeticMixin:
        return _UnaryOp(self, math.trunc)

    def __floor__(self) -> ArithmeticMixin:
        return _UnaryOp(self, math.floor)

    def __ceil__(self) -> ArithmeticMixin:
        return _UnaryOp(self, math.ceil)

    # --- functional combinators ---------------------------------------------

    # --- reward interface (must be implemented by concrete subclasses) ------

    @abstractmethod
    def reset(self, model: Model) -> None: ...

    @abstractmethod
    def extract(self, model: Model, done: bool) -> float: ...

    # --- functional combinators ---------------------------------------------

    def apply(self, fn: Callable[[float], float]) -> ArithmeticMixin:
        """Return a reward function whose value is ``fn(self)``."""
        return _UnaryOp(self, fn)

    def cumsum(self) -> ArithmeticMixin:
        """Return a reward function that accumulates the sum over the episode."""
        return _CumSum(self)

    # --- math module wrappers ------------------------------------------------

    def exp(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.exp(self)``."""
        return _UnaryOp(self, math.exp)

    def log(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.log(self)``."""
        return _UnaryOp(self, math.log)

    def log2(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.log2(self)``."""
        return _UnaryOp(self, math.log2)

    def log10(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.log10(self)``."""
        return _UnaryOp(self, math.log10)

    def sqrt(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.sqrt(self)``."""
        return _UnaryOp(self, math.sqrt)

    def sin(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.sin(self)``."""
        return _UnaryOp(self, math.sin)

    def cos(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.cos(self)``."""
        return _UnaryOp(self, math.cos)

    def tan(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.tan(self)``."""
        return _UnaryOp(self, math.tan)

    def asin(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.asin(self)``."""
        return _UnaryOp(self, math.asin)

    def acos(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.acos(self)``."""
        return _UnaryOp(self, math.acos)

    def atan(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.atan(self)``."""
        return _UnaryOp(self, math.atan)

    def sinh(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.sinh(self)``."""
        return _UnaryOp(self, math.sinh)

    def cosh(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.cosh(self)``."""
        return _UnaryOp(self, math.cosh)

    def tanh(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.tanh(self)``."""
        return _UnaryOp(self, math.tanh)

    def asinh(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.asinh(self)``."""
        return _UnaryOp(self, math.asinh)

    def acosh(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.acosh(self)``."""
        return _UnaryOp(self, math.acosh)

    def atanh(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.atanh(self)``."""
        return _UnaryOp(self, math.atanh)

    def isfinite(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.isfinite(self)``."""
        return _UnaryOp(self, math.isfinite)

    def isinf(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.isinf(self)``."""
        return _UnaryOp(self, math.isinf)

    def isnan(self) -> ArithmeticMixin:
        """Return a reward function whose value is ``math.isnan(self)``."""
        return _UnaryOp(self, math.isnan)


# ---------------------------------------------------------------------------
# Scalar wrapper
# ---------------------------------------------------------------------------


class _Constant(ArithmeticMixin):
    """Lifts a scalar into the reward-function interface."""

    __slots__ = ("_value",)

    def __init__(self, value: object) -> None:
        self._value = value

    def reset(self, model: Model) -> None:  # noqa: ARG002
        pass

    def extract(self, model: Model, done: bool) -> float:  # noqa: ARG002
        return float(self._value)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def _coerce(obj: object) -> ArithmeticMixin:
    """Return *obj* unchanged if it's an ``ArithmeticMixin``, else wrap as ``_Constant``."""
    if isinstance(obj, ArithmeticMixin):
        return obj
    return _Constant(obj)


# ---------------------------------------------------------------------------
# Expression-tree nodes (all inherit ArithmeticMixin so they compose further)
# ---------------------------------------------------------------------------


class _UnaryOp(ArithmeticMixin):
    __slots__ = ("_operand", "_fn")

    def __init__(self, operand: ArithmeticMixin, fn: Callable) -> None:
        self._operand = operand
        self._fn = fn

    def reset(self, model: Model) -> None:
        self._operand.reset(model)

    def extract(self, model: Model, done: bool) -> float:
        return self._fn(self._operand.extract(model, done))


class _BinaryOp(ArithmeticMixin):
    __slots__ = ("_left", "_right", "_fn")

    def __init__(self, left: object, right: object, fn: Callable) -> None:
        self._left = _coerce(left)
        self._right = _coerce(right)
        self._fn = fn

    def reset(self, model: Model) -> None:
        self._left.reset(model)
        self._right.reset(model)

    def extract(self, model: Model, done: bool) -> float:
        return self._fn(
            self._left.extract(model, done),
            self._right.extract(model, done),
        )


class _CumSum(ArithmeticMixin):
    __slots__ = ("_operand", "_acc")

    def __init__(self, operand: ArithmeticMixin) -> None:
        self._operand = operand
        self._acc = 0.0

    def reset(self, model: Model) -> None:
        self._operand.reset(model)
        self._acc = 0.0

    def extract(self, model: Model, done: bool) -> float:
        self._acc += self._operand.extract(model, done)
        return self._acc
