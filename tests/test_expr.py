"""Tests for expression construction, printing, and evaluation."""

import math
import cmath
import pytest

from uninum import var, Const, Var, sin, cos, exp, ln, sqrt, eml, e, pi, I


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_var(self):
        x = var("x")
        assert isinstance(x, Var)
        assert x.name == "x"

    def test_const(self):
        c = Const(42)
        assert c.value == 42

    def test_add(self):
        x = var("x")
        expr = x + 1
        assert str(expr) == "x + 1"

    def test_radd(self):
        x = var("x")
        expr = 1 + x
        assert str(expr) == "1 + x"

    def test_sub(self):
        x = var("x")
        assert str(x - 2) == "x - 2"

    def test_mul(self):
        x = var("x")
        assert str(x * 3) == "x * 3"

    def test_div(self):
        x = var("x")
        assert str(x / 2) == "x / 2"

    def test_pow(self):
        x = var("x")
        assert str(x ** 2) == "x ** 2"

    def test_neg(self):
        x = var("x")
        assert str(-x) == "-x"

    def test_compound(self):
        x = var("x")
        y = var("y")
        expr = (x + y) * sin(x)
        assert str(expr) == "(x + y) * sin(x)"


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

class TestPrinting:
    def test_precedence_add_mul(self):
        x, y, z = var("x"), var("y"), var("z")
        expr = x + y * z
        assert str(expr) == "x + y * z"

    def test_precedence_mul_add(self):
        x, y, z = var("x"), var("y"), var("z")
        expr = (x + y) * z
        assert str(expr) == "(x + y) * z"

    def test_sub_right_assoc(self):
        x, y, z = var("x"), var("y"), var("z")
        expr = x - (y - z)
        assert str(expr) == "x - (y - z)"

    def test_div_right_assoc(self):
        x, y, z = var("x"), var("y"), var("z")
        expr = x / (y / z)
        assert str(expr) == "x / (y / z)"

    def test_pow_right_assoc(self):
        x, y = var("x"), var("y")
        # x ** (y ** 2) should NOT parenthesise right child (right-assoc)
        expr = x ** (y ** 2)
        assert str(expr) == "x ** y ** 2"

    def test_pow_left_assoc_needs_parens(self):
        x, y = var("x"), var("y")
        from uninum.expr import BinOp
        # (x ** y) ** 2 — left child same prec -> needs parens
        expr = BinOp("pow", BinOp("pow", x, y), Const(2))
        assert str(expr) == "(x ** y) ** 2"

    def test_neg_binop(self):
        x, y = var("x"), var("y")
        expr = -(x + y)
        assert str(expr) == "-(x + y)"

    def test_neg_func(self):
        x = var("x")
        expr = -sin(x)
        assert str(expr) == "-sin(x)"

    def test_eml_display(self):
        x, y = var("x"), var("y")
        expr = eml(x, y)
        assert str(expr) == "eml(x, y)"

    def test_named_constants(self):
        assert str(e) == "e"
        assert str(pi) == "pi"
        assert str(I) == "i"

    def test_const_int_display(self):
        assert str(Const(3)) == "3"
        assert str(Const(3.0)) == "3"
        assert str(Const(-1)) == "-1"

    def test_const_complex_display(self):
        assert str(Const(1j)) == "i"
        assert str(Const(-1j)) == "-i"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_const(self):
        assert Const(5).evaluate() == 5

    def test_var(self):
        x = var("x")
        assert x.evaluate(x=3.0) == 3.0

    def test_var_missing(self):
        x = var("x")
        with pytest.raises(ValueError, match="Missing variable"):
            x.evaluate()

    def test_add(self):
        x = var("x")
        assert (x + 2).evaluate(x=3) == 5

    def test_sub(self):
        x = var("x")
        assert (x - 1).evaluate(x=5) == 4

    def test_mul(self):
        x = var("x")
        assert (x * 3).evaluate(x=4) == 12

    def test_div(self):
        x = var("x")
        assert (x / 2).evaluate(x=10) == 5.0

    def test_pow(self):
        x = var("x")
        assert (x ** 2).evaluate(x=3) == 9

    def test_neg(self):
        x = var("x")
        assert (-x).evaluate(x=7) == -7

    def test_sin(self):
        x = var("x")
        assert sin(x).evaluate(x=0.0) == pytest.approx(0.0, abs=1e-15)

    def test_cos(self):
        x = var("x")
        assert cos(x).evaluate(x=0.0) == pytest.approx(1.0)

    def test_exp(self):
        x = var("x")
        assert exp(x).evaluate(x=0.0) == pytest.approx(1.0)

    def test_ln(self):
        x = var("x")
        assert ln(x).evaluate(x=math.e) == pytest.approx(1.0)

    def test_sqrt(self):
        x = var("x")
        assert sqrt(x).evaluate(x=9.0) == pytest.approx(3.0)

    def test_eml(self):
        # eml(x, y) = exp(x) - ln(y)
        x, y = var("x"), var("y")
        expr = eml(x, y)
        expected = cmath.exp(1.0) - cmath.log(2.0)
        assert expr.evaluate(x=1.0, y=2.0) == pytest.approx(expected.real)

    def test_compound(self):
        x = var("x")
        y = var("y")
        expr = (x + y) * sin(x) / ln(y)
        xv, yv = 1.2, 3.4
        expected = (xv + yv) * math.sin(xv) / math.log(yv)
        assert expr.evaluate(x=xv, y=yv) == pytest.approx(expected)

    def test_eml_constant_e(self):
        # eml(1, 1) = exp(1) - ln(1) = e
        expr = eml(1, 1)
        assert expr.evaluate() == pytest.approx(math.e)
