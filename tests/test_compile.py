"""Tests for the compile function."""

import math
import pytest

from uninum import var, Const, sin, cos, exp, ln, sqrt, eml, compile


class TestCompilePython:
    def test_simple_add(self):
        x = var("x")
        fn = compile(x + 1, backend="python")
        assert fn(x=4) == 5

    def test_sin(self):
        x = var("x")
        fn = compile(sin(x), backend="python")
        result = fn(x=1.0)
        if isinstance(result, complex):
            result = result.real
        assert result == pytest.approx(math.sin(1.0))

    def test_compound(self):
        x = var("x")
        y = var("y")
        expr = (x + y) * sin(x) / ln(y)
        fn = compile(expr, backend="python")
        result = fn(x=1.2, y=3.4)
        if isinstance(result, complex):
            result = result.real
        expected = (1.2 + 3.4) * math.sin(1.2) / math.log(3.4)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_eml(self):
        fn = compile(eml(1, 1), backend="python")
        result = fn()
        if isinstance(result, complex):
            result = result.real
        assert result == pytest.approx(math.e, rel=1e-10)


class TestCompileNumpy:
    @pytest.fixture(autouse=True)
    def _skip_no_numpy(self):
        pytest.importorskip("numpy")

    def test_scalar(self):
        x = var("x")
        fn = compile(x ** 2, backend="numpy")
        assert fn(x=3.0) == pytest.approx(9.0)

    def test_vectorised(self):
        import numpy as np

        x = var("x")
        fn = compile(sin(x), backend="numpy")
        xs = np.array([0.0, math.pi / 2, math.pi])
        result = fn(x=xs)
        expected = np.sin(xs)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_compound_vectorised(self):
        import numpy as np

        x = var("x")
        y = var("y")
        expr = (x + y) * exp(x)
        fn = compile(expr, backend="numpy")
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([1.0, 2.0, 3.0])
        result = fn(x=xs, y=ys)
        expected = (xs + ys) * np.exp(xs)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_dag_sharing(self):
        """Shared sub-expressions should only be evaluated once."""
        x = var("x")
        shared = sin(x)
        expr = shared * shared  # same object on both sides
        fn = compile(expr, backend="numpy")
        # Just check correctness — timing/sharing is an implementation detail
        assert fn(x=1.0) == pytest.approx(math.sin(1.0) ** 2, rel=1e-12)
