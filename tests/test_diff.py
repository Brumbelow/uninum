"""Tests for symbolic differentiation."""

import math
import pytest

from uninum import var, Const, sin, cos, exp, ln, sqrt, tan, asin, acos, atan, sinh, cosh, tanh


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

    # --- trig functions ---

    def test_tan(self):
        x = var("x")
        d = tan(x).diff(x)
        pt = {"x": 0.5}
        # d/dx tan(x) = 1/cos(x)^2
        expected = 1.0 / math.cos(0.5) ** 2
        assert d.evaluate(**pt) == pytest.approx(expected, rel=1e-10)

    def test_tan_vs_numerical(self):
        x = var("x")
        expr = tan(x)
        pt = {"x": 0.8}
        symbolic = expr.diff(x).evaluate(**pt)
        numerical = _numerical_diff(expr, "x", pt)
        assert symbolic == pytest.approx(numerical, rel=1e-6)

    # --- inverse trig ---

    def test_asin(self):
        x = var("x")
        d = asin(x).diff(x)
        pt = {"x": 0.5}
        # d/dx asin(x) = 1/sqrt(1-x^2)
        expected = 1.0 / math.sqrt(1 - 0.5**2)
        assert d.evaluate(**pt) == pytest.approx(expected, rel=1e-10)

    def test_asin_vs_numerical(self):
        x = var("x")
        expr = asin(x)
        pt = {"x": 0.3}
        symbolic = expr.diff(x).evaluate(**pt)
        numerical = _numerical_diff(expr, "x", pt)
        assert symbolic == pytest.approx(numerical, rel=1e-6)

    def test_acos(self):
        x = var("x")
        d = acos(x).diff(x)
        pt = {"x": 0.5}
        # d/dx acos(x) = -1/sqrt(1-x^2)
        expected = -1.0 / math.sqrt(1 - 0.5**2)
        assert d.evaluate(**pt) == pytest.approx(expected, rel=1e-10)

    def test_acos_vs_numerical(self):
        x = var("x")
        expr = acos(x)
        pt = {"x": 0.3}
        symbolic = expr.diff(x).evaluate(**pt)
        numerical = _numerical_diff(expr, "x", pt)
        assert symbolic == pytest.approx(numerical, rel=1e-6)

    def test_atan(self):
        x = var("x")
        d = atan(x).diff(x)
        pt = {"x": 1.0}
        # d/dx atan(x) = 1/(1+x^2)
        expected = 1.0 / (1 + 1.0**2)
        assert d.evaluate(**pt) == pytest.approx(expected, rel=1e-10)

    def test_atan_vs_numerical(self):
        x = var("x")
        expr = atan(x)
        pt = {"x": 2.0}
        symbolic = expr.diff(x).evaluate(**pt)
        numerical = _numerical_diff(expr, "x", pt)
        assert symbolic == pytest.approx(numerical, rel=1e-6)

    # --- hyperbolic ---

    def test_sinh(self):
        x = var("x")
        d = sinh(x).diff(x)
        pt = {"x": 1.0}
        # d/dx sinh(x) = cosh(x)
        expected = math.cosh(1.0)
        assert d.evaluate(**pt) == pytest.approx(expected, rel=1e-10)

    def test_cosh(self):
        x = var("x")
        d = cosh(x).diff(x)
        pt = {"x": 1.0}
        # d/dx cosh(x) = sinh(x)
        expected = math.sinh(1.0)
        assert d.evaluate(**pt) == pytest.approx(expected, rel=1e-10)

    def test_tanh(self):
        x = var("x")
        d = tanh(x).diff(x)
        pt = {"x": 1.0}
        # d/dx tanh(x) = 1 - tanh(x)^2
        expected = 1 - math.tanh(1.0) ** 2
        assert d.evaluate(**pt) == pytest.approx(expected, rel=1e-10)

    def test_tanh_vs_numerical(self):
        x = var("x")
        expr = tanh(x)
        pt = {"x": 0.5}
        symbolic = expr.diff(x).evaluate(**pt)
        numerical = _numerical_diff(expr, "x", pt)
        assert symbolic == pytest.approx(numerical, rel=1e-6)

    # --- higher-order derivatives ---

    def test_second_derivative(self):
        x = var("x")
        # d^2/dx^2 sin(x) = -sin(x)
        expr = sin(x)
        d2 = expr.diff(x).diff(x)
        pt = {"x": 1.0}
        assert d2.evaluate(**pt) == pytest.approx(-math.sin(1.0), rel=1e-10)

    def test_third_derivative(self):
        x = var("x")
        # d^3/dx^3 x^4 = 24x
        expr = x ** 4
        d3 = expr.diff(x).diff(x).diff(x)
        assert d3.evaluate(x=2.0) == pytest.approx(48.0, rel=1e-8)

    def test_second_derivative_exp(self):
        x = var("x")
        # d^2/dx^2 exp(x) = exp(x)
        d2 = exp(x).diff(x).diff(x)
        assert d2.evaluate(x=1.0) == pytest.approx(math.exp(1.0), rel=1e-10)

    # --- chain rule with additional functions ---

    def test_chain_sin_squared(self):
        x = var("x")
        # d/dx sin(x)^2 = 2*sin(x)*cos(x)
        expr = sin(x) ** 2
        d = expr.diff(x)
        pt = {"x": 0.7}
        expected = 2 * math.sin(0.7) * math.cos(0.7)
        assert d.evaluate(**pt) == pytest.approx(expected, rel=1e-8)

    def test_chain_exp_neg_x_sq(self):
        x = var("x")
        # d/dx exp(-x^2) = -2x * exp(-x^2)
        expr = exp(-(x ** 2))
        d = expr.diff(x)
        pt = {"x": 1.0}
        expected = -2 * 1.0 * math.exp(-1.0)
        symbolic = d.evaluate(**pt)
        if isinstance(symbolic, complex):
            symbolic = symbolic.real
        assert symbolic == pytest.approx(expected, rel=1e-8)
