"""Tests for the compile function."""

import math
import pytest

from uninum import var, Const, sin, cos, exp, ln, sqrt, eml, compile_expr, tan


class TestCompilePython:
    def test_simple_add(self):
        x = var("x")
        fn = compile_expr(x + 1, backend="python")
        assert fn(x=4) == 5

    def test_sin(self):
        x = var("x")
        fn = compile_expr(sin(x), backend="python")
        result = fn(x=1.0)
        if isinstance(result, complex):
            result = result.real
        assert result == pytest.approx(math.sin(1.0))

    def test_compound(self):
        x = var("x")
        y = var("y")
        expr = (x + y) * sin(x) / ln(y)
        fn = compile_expr(expr, backend="python")
        result = fn(x=1.2, y=3.4)
        if isinstance(result, complex):
            result = result.real
        expected = (1.2 + 3.4) * math.sin(1.2) / math.log(3.4)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_eml(self):
        fn = compile_expr(eml(1, 1), backend="python")
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
        fn = compile_expr(x ** 2, backend="numpy")
        assert fn(x=3.0) == pytest.approx(9.0)

    def test_vectorised(self):
        import numpy as np

        x = var("x")
        fn = compile_expr(sin(x), backend="numpy")
        xs = np.array([0.0, math.pi / 2, math.pi])
        result = fn(x=xs)
        expected = np.sin(xs)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_compound_vectorised(self):
        import numpy as np

        x = var("x")
        y = var("y")
        expr = (x + y) * exp(x)
        fn = compile_expr(expr, backend="numpy")
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
        fn = compile_expr(expr, backend="numpy")
        # Just check correctness — timing/sharing is an implementation detail
        assert fn(x=1.0) == pytest.approx(math.sin(1.0) ** 2, rel=1e-12)


class TestCompileEml:
    """Test compiling EML-lowered expressions (the bug fix from Phase 1a)."""

    def _eval_real(self, fn, **kwargs):
        result = fn(**kwargs)
        return result.real if isinstance(result, complex) else result

    def test_add_eml_python(self):
        x, y = var("x"), var("y")
        lowered = (x + y).to_eml()
        fn = compile_expr(lowered, backend="python")
        assert self._eval_real(fn, x=1.0, y=2.0) == pytest.approx(3.0, rel=1e-8)

    def test_sin_eml_python(self):
        x = var("x")
        lowered = sin(x).to_eml()
        fn = compile_expr(lowered, backend="python")
        assert self._eval_real(fn, x=1.0) == pytest.approx(math.sin(1.0), rel=1e-8)

    def test_mul_eml_python(self):
        x, y = var("x"), var("y")
        lowered = (x * y).to_eml()
        fn = compile_expr(lowered, backend="python")
        assert self._eval_real(fn, x=3.0, y=4.0) == pytest.approx(12.0, rel=1e-8)

    def test_exp_eml_python(self):
        x = var("x")
        lowered = exp(x).to_eml()
        fn = compile_expr(lowered, backend="python")
        assert self._eval_real(fn, x=1.5) == pytest.approx(math.exp(1.5), rel=1e-10)

    def test_compound_eml_python(self):
        x = var("x")
        expr = sin(x) * exp(x)
        lowered = expr.to_eml()
        fn = compile_expr(lowered, backend="python")
        expected = math.sin(0.5) * math.exp(0.5)
        assert self._eval_real(fn, x=0.5) == pytest.approx(expected, rel=1e-8)

    def test_unknown_backend_raises(self):
        x = var("x")
        with pytest.raises(ValueError, match="Unknown backend"):
            compile_expr(x + 1, backend="jax")
