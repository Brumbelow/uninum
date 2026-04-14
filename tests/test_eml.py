"""Tests for EML lowering."""

import math
import cmath
import pytest

from uninum import var, Const, Var, exp, ln, sin, cos, eml
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
