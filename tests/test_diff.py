"""Tests for symbolic differentiation."""

import math
import pytest

from uninum import var, Const, sin, cos, exp, ln, sqrt


def _numerical_diff(expr, wrt_name, point, h=1e-7):
    """Central-difference numerical derivative for verification."""
    kwargs_plus = dict(point)
    kwargs_minus = dict(point)
    kwargs_plus[wrt_name] = point[wrt_name] + h
    kwargs_minus[wrt_name] = point[wrt_name] - h
    fp = expr.evaluate(**kwargs_plus)
    fm = expr.evaluate(**kwargs_minus)
    return (fp - fm) / (2 * h)


class TestDiff:
    def test_const(self):
        x = var("x")
        assert Const(5).diff(x).evaluate(x=1.0) == 0

    def test_var_self(self):
        x = var("x")
        assert x.diff(x).evaluate(x=1.0) == 1

    def test_var_other(self):
        x = var("x")
        y = var("y")
        assert x.diff(y).evaluate(x=1.0, y=2.0) == 0

    def test_add(self):
        x = var("x")
        expr = x + Const(3)
        d = expr.diff(x)
        assert d.evaluate(x=2.0) == pytest.approx(1.0)

    def test_mul(self):
        x = var("x")
        # d/dx (x * x) = 2x
        expr = x * x
        d = expr.diff(x)
        assert d.evaluate(x=5.0) == pytest.approx(10.0)

    def test_sin(self):
        x = var("x")
        expr = sin(x)
        d = expr.diff(x)
        pt = {"x": 1.0}
        assert d.evaluate(**pt) == pytest.approx(math.cos(1.0), rel=1e-10)

    def test_cos(self):
        x = var("x")
        expr = cos(x)
        d = expr.diff(x)
        pt = {"x": 1.0}
        assert d.evaluate(**pt) == pytest.approx(-math.sin(1.0), rel=1e-10)

    def test_exp(self):
        x = var("x")
        expr = exp(x)
        d = expr.diff(x)
        pt = {"x": 2.0}
        assert d.evaluate(**pt) == pytest.approx(math.exp(2.0), rel=1e-10)

    def test_ln(self):
        x = var("x")
        expr = ln(x)
        d = expr.diff(x)
        pt = {"x": 3.0}
        assert d.evaluate(**pt) == pytest.approx(1.0 / 3.0, rel=1e-10)

    def test_sqrt(self):
        x = var("x")
        expr = sqrt(x)
        d = expr.diff(x)
        pt = {"x": 4.0}
        # d/dx sqrt(x) = 1 / (2*sqrt(x)) = 1/4
        assert d.evaluate(**pt) == pytest.approx(0.25, rel=1e-10)

    def test_pow(self):
        x = var("x")
        expr = x ** 3
        d = expr.diff(x)
        pt = {"x": 2.0}
        # d/dx x^3 = 3x^2 = 12
        assert d.evaluate(**pt) == pytest.approx(12.0, rel=1e-10)

    def test_div(self):
        x = var("x")
        expr = Const(1) / x
        d = expr.diff(x)
        pt = {"x": 2.0}
        # d/dx (1/x) = -1/x^2 = -0.25
        assert d.evaluate(**pt) == pytest.approx(-0.25, rel=1e-10)

    def test_compound_vs_numerical(self):
        x = var("x")
        expr = sin(x ** 2) * exp(x)
        d = expr.diff(x)
        pt = {"x": 0.5}
        symbolic = d.evaluate(**pt)
        numerical = _numerical_diff(expr, "x", pt)
        assert symbolic == pytest.approx(numerical, rel=1e-6)

    def test_multivar(self):
        x = var("x")
        y = var("y")
        expr = x * y + sin(x)
        dx = expr.diff(x)
        pt = {"x": 1.0, "y": 2.0}
        # d/dx (xy + sin(x)) = y + cos(x)
        expected = 2.0 + math.cos(1.0)
        assert dx.evaluate(**pt) == pytest.approx(expected, rel=1e-10)

    def test_diff_string_arg(self):
        x = var("x")
        expr = x ** 2
        d = expr.diff("x")
        assert d.evaluate(x=3.0) == pytest.approx(6.0)
