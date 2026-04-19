"""Unit tests for gyozas.rewards.arithmetic (ArithmeticMixin expression tree)."""

import math

from pyscipopt import Model

from gyozas.rewards.arithmetic import (
    ArithmeticMixin,
    _BinaryOp,
    _coerce,
    _Constant,
)

# ---------------------------------------------------------------------------
# Minimal concrete reward for testing (returns a configurable constant)
# ---------------------------------------------------------------------------


class _Fixed(ArithmeticMixin):
    """Trivial reward that always returns `value` (mutating allowed via .value)."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self._reset_count = 0

    def reset(self, model: Model) -> None:
        self._reset_count += 1

    def extract(self, model: Model, done: bool) -> float:
        return self.value


_MODEL = None  # tests don't need a real model for arithmetic


def _m():
    """Sentinel – arithmetic tests don't need a real SCIP model."""
    return None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# _Constant
# ---------------------------------------------------------------------------


class TestConstant:
    def test_extract_int(self):
        c = _Constant(3)
        assert c.extract(_m(), False) == 3.0

    def test_extract_float(self):
        c = _Constant(2.5)
        assert c.extract(_m(), False) == 2.5

    def test_reset_noop(self):
        c = _Constant(1)
        c.reset(_m())  # must not raise

    def test_is_arithmetic_mixin(self):
        assert isinstance(_Constant(0), ArithmeticMixin)


# ---------------------------------------------------------------------------
# _coerce
# ---------------------------------------------------------------------------


class TestCoerce:
    def test_passthrough_arithmetic_mixin(self):
        f = _Fixed(5.0)
        assert _coerce(f) is f

    def test_wraps_scalar(self):
        c = _coerce(7)
        assert isinstance(c, _Constant)
        assert c.extract(_m(), False) == 7.0

    def test_wraps_float(self):
        c = _coerce(3.14)
        assert isinstance(c, _Constant)
        assert abs(c.extract(_m(), False) - 3.14) < 1e-12


# ---------------------------------------------------------------------------
# _UnaryOp
# ---------------------------------------------------------------------------


class TestUnaryOp:
    def test_neg(self):
        op = _Fixed(4.0).__neg__()
        assert op.extract(_m(), False) == -4.0

    def test_pos(self):
        op = _Fixed(4.0).__pos__()
        assert op.extract(_m(), False) == 4.0

    def test_abs_negative(self):
        op = abs(_Fixed(-3.0))
        assert op.extract(_m(), False) == 3.0

    def test_reset_propagates(self):
        inner = _Fixed(1.0)
        op = -inner
        op.reset(_m())
        assert inner._reset_count == 1

    def test_apply(self):
        op = _Fixed(9.0).apply(math.sqrt)
        assert abs(op.extract(_m(), False) - 3.0) < 1e-12

    def test_is_arithmetic_mixin(self):
        assert isinstance(-_Fixed(1.0), ArithmeticMixin)


# ---------------------------------------------------------------------------
# _BinaryOp
# ---------------------------------------------------------------------------


class TestBinaryOp:
    def test_add(self):
        r = _Fixed(2.0) + _Fixed(3.0)
        assert r.extract(_m(), False) == 5.0

    def test_radd_scalar(self):
        r = 10.0 + _Fixed(4.0)
        assert r.extract(_m(), False) == 14.0

    def test_sub(self):
        r = _Fixed(7.0) - _Fixed(2.0)
        assert r.extract(_m(), False) == 5.0

    def test_rsub_scalar(self):
        r = 10.0 - _Fixed(3.0)
        assert r.extract(_m(), False) == 7.0

    def test_mul(self):
        r = _Fixed(3.0) * _Fixed(4.0)
        assert r.extract(_m(), False) == 12.0

    def test_rmul_scalar(self):
        r = 5.0 * _Fixed(2.0)
        assert r.extract(_m(), False) == 10.0

    def test_truediv(self):
        r = _Fixed(9.0) / _Fixed(3.0)
        assert r.extract(_m(), False) == 3.0

    def test_rtruediv_scalar(self):
        r = 12.0 / _Fixed(4.0)
        assert r.extract(_m(), False) == 3.0

    def test_floordiv(self):
        r = _Fixed(7.0) // _Fixed(2.0)
        assert r.extract(_m(), False) == 3.0

    def test_mod(self):
        r = _Fixed(7.0) % _Fixed(3.0)
        assert r.extract(_m(), False) == 1.0

    def test_pow(self):
        r = _Fixed(2.0) ** _Fixed(3.0)
        assert r.extract(_m(), False) == 8.0

    def test_rpow_scalar(self):
        r = 2.0 ** _Fixed(4.0)
        assert r.extract(_m(), False) == 16.0

    def test_reset_propagates_to_both_operands(self):
        left = _Fixed(1.0)
        right = _Fixed(2.0)
        op = left + right
        op.reset(_m())
        assert left._reset_count == 1
        assert right._reset_count == 1

    def test_scalar_lhs_wrapped(self):
        r = 3 + _Fixed(1.0)
        assert isinstance(r, _BinaryOp)
        assert r.extract(_m(), False) == 4.0

    def test_is_arithmetic_mixin(self):
        assert isinstance(_Fixed(1.0) + _Fixed(2.0), ArithmeticMixin)


# ---------------------------------------------------------------------------
# _CumSum
# ---------------------------------------------------------------------------


class TestCumSum:
    def test_accumulates(self):
        inner = _Fixed(2.0)
        cs = inner.cumsum()
        assert cs.extract(_m(), False) == 2.0
        assert cs.extract(_m(), False) == 4.0
        assert cs.extract(_m(), False) == 6.0

    def test_reset_clears_accumulator(self):
        inner = _Fixed(3.0)
        cs = inner.cumsum()
        cs.extract(_m(), False)  # acc = 3.0
        cs.reset(_m())
        assert cs.extract(_m(), False) == 3.0  # acc restarted

    def test_reset_propagates_to_inner(self):
        inner = _Fixed(1.0)
        cs = inner.cumsum()
        cs.reset(_m())
        assert inner._reset_count == 1

    def test_is_arithmetic_mixin(self):
        assert isinstance(_Fixed(1.0).cumsum(), ArithmeticMixin)

    def test_cumsum_of_expression(self):
        """cumsum can wrap a composed expression."""
        r = (_Fixed(1.0) + _Fixed(2.0)).cumsum()
        assert r.extract(_m(), False) == 3.0
        assert r.extract(_m(), False) == 6.0


# ---------------------------------------------------------------------------
# Math methods (programmatically added to ArithmeticMixin)
# ---------------------------------------------------------------------------


class TestMathMethods:
    def test_sqrt(self):
        r = _Fixed(16.0).sqrt()
        assert abs(r.extract(_m(), False) - 4.0) < 1e-12

    def test_log(self):
        r = _Fixed(math.e).log()
        assert abs(r.extract(_m(), False) - 1.0) < 1e-12

    def test_exp(self):
        r = _Fixed(0.0).exp()
        assert abs(r.extract(_m(), False) - 1.0) < 1e-12

    def test_isfinite(self):
        r = _Fixed(1.0).isfinite()
        assert r.extract(_m(), False) is True

    def test_isnan(self):
        r = _Fixed(float("nan")).isnan()
        assert r.extract(_m(), False) is True

    def test_isinf(self):
        r = _Fixed(float("inf")).isinf()
        assert r.extract(_m(), False) is True

    def test_abs_via_method(self):
        r = abs(_Fixed(-5.0))
        assert r.extract(_m(), False) == 5.0


# ---------------------------------------------------------------------------
# Composition: deeply nested expressions
# ---------------------------------------------------------------------------


class TestComposition:
    def test_chain_binary_ops(self):
        # (2 + 3) * 4 - 1 == 19
        r = (_Fixed(2.0) + _Fixed(3.0)) * _Fixed(4.0) - _Fixed(1.0)
        assert r.extract(_m(), False) == 19.0

    def test_neg_then_cumsum(self):
        r = (-_Fixed(1.0)).cumsum()
        assert r.extract(_m(), False) == -1.0
        assert r.extract(_m(), False) == -2.0

    def test_reset_propagates_deep(self):
        a = _Fixed(1.0)
        b = _Fixed(2.0)
        c = _Fixed(3.0)
        r = ((a + b) * c).cumsum()
        r.reset(_m())
        assert a._reset_count == 1
        assert b._reset_count == 1
        assert c._reset_count == 1

    def test_scalar_on_both_sides(self):
        # 2 * reward * 0.5 == reward
        r = 2 * _Fixed(5.0) * 0.5
        assert r.extract(_m(), False) == 5.0

    def test_further_compose_cumsum(self):
        """Result of cumsum() can be used in further arithmetic."""
        cs = _Fixed(2.0).cumsum()
        r = cs + _Fixed(10.0)
        # step 1: cs=2, r=12
        assert r.extract(_m(), False) == 12.0
        # step 2: cs=4, r=14
        assert r.extract(_m(), False) == 14.0
