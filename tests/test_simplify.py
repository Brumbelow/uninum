"""Tests for algebraic simplification."""

import pytest

from uninum import var, Const, sin, cos, exp, ln


class TestSimplify:
    # --- constant folding ---

    def test_fold_add(self):
        expr = Const(2) + Const(3)
        assert str(expr.simplify()) == "5"

    def test_fold_mul(self):
        expr = Const(4) * Const(5)
        assert str(expr.simplify()) == "20"

    def test_fold_exp(self):
        expr = exp(Const(0))
        s = expr.simplify()
        assert s.evaluate() == pytest.approx(1.0)

    # --- additive identities ---

    def test_add_zero_left(self):
        x = var("x")
        expr = Const(0) + x
        assert str(expr.simplify()) == "x"

    def test_add_zero_right(self):
        x = var("x")
        expr = x + Const(0)
        assert str(expr.simplify()) == "x"

    def test_sub_zero(self):
        x = var("x")
        expr = x - Const(0)
        assert str(expr.simplify()) == "x"

    def test_sub_self(self):
        x = var("x")
        expr = x - x
        assert str(expr.simplify()) == "0"

    def test_zero_minus_x(self):
        x = var("x")
        expr = Const(0) - x
        assert str(expr.simplify()) == "-x"

    # --- multiplicative identities ---

    def test_mul_one_left(self):
        x = var("x")
        expr = Const(1) * x
        assert str(expr.simplify()) == "x"

    def test_mul_one_right(self):
        x = var("x")
        expr = x * Const(1)
        assert str(expr.simplify()) == "x"

    def test_mul_zero(self):
        x = var("x")
        expr = x * Const(0)
        assert str(expr.simplify()) == "0"

    def test_mul_neg_one(self):
        x = var("x")
        expr = Const(-1) * x
        assert str(expr.simplify()) == "-x"

    def test_div_one(self):
        x = var("x")
        expr = x / Const(1)
        assert str(expr.simplify()) == "x"

    def test_div_self(self):
        x = var("x")
        expr = x / x
        assert str(expr.simplify()) == "1"

    # --- power identities ---

    def test_pow_zero(self):
        x = var("x")
        expr = x ** Const(0)
        assert str(expr.simplify()) == "1"

    def test_pow_one(self):
        x = var("x")
        expr = x ** Const(1)
        assert str(expr.simplify()) == "x"

    def test_one_pow(self):
        x = var("x")
        expr = Const(1) ** x
        assert str(expr.simplify()) == "1"

    # --- double negation ---

    def test_double_neg(self):
        x = var("x")
        expr = -(-x)
        assert str(expr.simplify()) == "x"

    # --- exp/ln cancellation ---

    def test_exp_ln(self):
        x = var("x")
        expr = exp(ln(x))
        assert str(expr.simplify()) == "x"

    def test_ln_exp(self):
        x = var("x")
        expr = ln(exp(x))
        assert str(expr.simplify()) == "x"

    # --- add/sub with neg ---

    def test_add_neg(self):
        x = var("x")
        y = var("y")
        expr = x + (-y)
        assert str(expr.simplify()) == "x - y"

    def test_sub_neg(self):
        x = var("x")
        y = var("y")
        expr = x - (-y)
        assert str(expr.simplify()) == "x + y"

    # --- compound ---

    def test_compound(self):
        x = var("x")
        # (x * 1 + 0) ** 1 -> x
        expr = (x * Const(1) + Const(0)) ** Const(1)
        assert str(expr.simplify()) == "x"
