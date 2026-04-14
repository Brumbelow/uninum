"""Tests for EML lowering."""

import math
import cmath
import pytest

from uninum import var, Const, Var, exp, ln, sin, cos, eml, tan, sqrt
from uninum import asin, acos, atan, sinh, cosh, tanh, e, pi
from uninum.expr import UnaryOp, BinOp


def _only_eml(node):
    """Return True if *node* contains only Const(1), Var, and BinOp('eml')."""
    if isinstance(node, Const):
        return node.value == 1
    if isinstance(node, Var):
        return True
    if isinstance(node, BinOp) and node.op == "eml":
        return _only_eml(node.left) and _only_eml(node.right)
    return False


class TestEmlLowering:
    def test_exp(self):
        x = var("x")
        lowered = exp(x).to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=1.5) == pytest.approx(math.exp(1.5), rel=1e-10)

    def test_ln(self):
        x = var("x")
        lowered = ln(x).to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=2.5) == pytest.approx(math.log(2.5), rel=1e-10)

    def test_eml_11_is_e(self):
        lowered = eml(1, 1).to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate() == pytest.approx(math.e, rel=1e-10)

    def test_add(self):
        x = var("x")
        y = var("y")
        expr = x + y
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=2.0, y=3.0) == pytest.approx(5.0, rel=1e-10)

    def test_sub(self):
        x = var("x")
        y = var("y")
        expr = x - y
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=5.0, y=2.0) == pytest.approx(3.0, rel=1e-10)

    def test_mul(self):
        x = var("x")
        y = var("y")
        expr = x * y
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=3.0, y=4.0) == pytest.approx(12.0, rel=1e-10)

    def test_div(self):
        x = var("x")
        y = var("y")
        expr = x / y
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=10.0, y=4.0) == pytest.approx(2.5, rel=1e-10)

    def test_pow(self):
        x = var("x")
        expr = x ** 2
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=3.0) == pytest.approx(9.0, rel=1e-10)

    def test_neg(self):
        x = var("x")
        expr = -x
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=7.0) == pytest.approx(-7.0, rel=1e-10)

    def test_const_zero(self):
        lowered = Const(0).to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate() == pytest.approx(0.0, abs=1e-14)

    def test_const_two(self):
        lowered = Const(2).to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate() == pytest.approx(2.0, rel=1e-10)

    def test_compound_roundtrip(self):
        """Original and EML-lowered expressions should agree numerically."""
        x = var("x")
        y = var("y")
        expr = (x + y) * exp(x) / ln(y)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        pt = dict(x=1.2, y=3.4)
        assert lowered.evaluate(**pt) == pytest.approx(
            expr.evaluate(**pt), rel=1e-8
        )

    def test_sin_roundtrip(self):
        x = var("x")
        expr = sin(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=0.7) == pytest.approx(math.sin(0.7), rel=1e-8)

    def test_cos_roundtrip(self):
        x = var("x")
        expr = cos(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=0.7) == pytest.approx(math.cos(0.7), rel=1e-8)

    # --- trig ---

    def test_tan_roundtrip(self):
        x = var("x")
        expr = tan(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=0.5) == pytest.approx(math.tan(0.5), rel=1e-8)

    # --- inverse trig ---

    def test_asin_roundtrip(self):
        x = var("x")
        expr = asin(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=0.5) == pytest.approx(math.asin(0.5), rel=1e-6)

    def test_acos_roundtrip(self):
        x = var("x")
        expr = acos(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=0.5) == pytest.approx(math.acos(0.5), rel=1e-6)

    def test_atan_roundtrip(self):
        x = var("x")
        expr = atan(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=0.5) == pytest.approx(math.atan(0.5), rel=1e-6)

    # --- hyperbolic ---

    def test_sinh_roundtrip(self):
        x = var("x")
        expr = sinh(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=1.0) == pytest.approx(math.sinh(1.0), rel=1e-8)

    def test_cosh_roundtrip(self):
        x = var("x")
        expr = cosh(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=1.0) == pytest.approx(math.cosh(1.0), rel=1e-8)

    def test_tanh_roundtrip(self):
        x = var("x")
        expr = tanh(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=1.0) == pytest.approx(math.tanh(1.0), rel=1e-8)

    # --- sqrt ---

    def test_sqrt_roundtrip(self):
        x = var("x")
        expr = sqrt(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate(x=4.0) == pytest.approx(2.0, rel=1e-10)

    # --- named constants ---

    def test_const_e(self):
        lowered = e.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate() == pytest.approx(math.e, rel=1e-10)

    def test_const_pi(self):
        lowered = pi.to_eml()
        assert _only_eml(lowered)
        assert lowered.evaluate() == pytest.approx(math.pi, rel=1e-8)

    # --- compound expressions ---

    def test_sin_plus_cos(self):
        x = var("x")
        expr = sin(x) + cos(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        expected = math.sin(0.7) + math.cos(0.7)
        assert lowered.evaluate(x=0.7) == pytest.approx(expected, rel=1e-8)

    def test_exp_times_sin(self):
        x = var("x")
        expr = exp(x) * sin(x)
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        expected = math.exp(0.5) * math.sin(0.5)
        assert lowered.evaluate(x=0.5) == pytest.approx(expected, rel=1e-8)

    def test_nested_trig(self):
        x = var("x")
        expr = sin(cos(x))
        lowered = expr.to_eml()
        assert _only_eml(lowered)
        expected = math.sin(math.cos(0.3))
        assert lowered.evaluate(x=0.3) == pytest.approx(expected, rel=1e-8)
