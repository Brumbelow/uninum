"""Algebraic simplification via bottom-up rewriting."""

import cmath
import math

from .expr import Expr, Const, Var, UnaryOp, BinOp, _UNARY_FNS, _BINARY_FNS


def simplify(expr, max_passes=10):
    """Simplify *expr* by repeated bottom-up rewriting until a fixpoint."""
    prev = None
    cur = expr
    for _ in range(max_passes):
        nxt = _simplify_once(cur)
        if nxt == cur:
            return nxt
        prev = cur
        cur = nxt
    return cur


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_const(node, value=None):
    if not isinstance(node, Const):
        return False
    if value is None:
        return True
    if isinstance(value, float):
        return isinstance(node.value, (int, float)) and math.isclose(
            float(node.value), value, rel_tol=1e-15, abs_tol=1e-300
        )
    return node.value == value


def _const_val(node):
    """Return numeric value if Const, else None."""
    return node.value if isinstance(node, Const) else None


def _make_const(value):
    """Create a Const, preserving int/float type."""
    if isinstance(value, complex) and value.imag == 0:
        value = value.real
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        value = int(value)
    return Const(value)


def _extract_coeff_base(node):
    """Decompose a term into (coefficient, base).

    Returns (numeric_coeff, base_expr) where base_expr is None for pure constants.
    """
    if isinstance(node, Const):
        return (node.value, None)
    if isinstance(node, UnaryOp) and node.op == "neg":
        c, b = _extract_coeff_base(node.arg)
        return (-c, b)
    if isinstance(node, BinOp) and node.op == "mul":
        if isinstance(node.left, Const):
            return (node.left.value, node.right)
        if isinstance(node.right, Const):
            return (node.right.value, node.left)
    return (1, node)


def _make_term(coeff, base):
    """Reconstruct an expression from (coefficient, base)."""
    if base is None:
        return _make_const(coeff)
    if coeff == 0:
        return Const(0)
    if coeff == 1:
        return base
    if coeff == -1:
        return UnaryOp("neg", base)
    return BinOp("mul", _make_const(coeff), base)


# ---------------------------------------------------------------------------
# Single-pass bottom-up rewrite
# ---------------------------------------------------------------------------

def _simplify_once(node):
    if isinstance(node, (Const, Var)):
        return node

    if isinstance(node, UnaryOp):
        arg = _simplify_once(node.arg)
        return _simplify_unary(node.op, arg)

    if isinstance(node, BinOp):
        left = _simplify_once(node.left)
        right = _simplify_once(node.right)
        return _simplify_binary(node.op, left, right)

    return node


def _simplify_unary(op, arg):
    # Constant folding
    if isinstance(arg, Const):
        try:
            val = _UNARY_FNS[op](arg.value)
            return _make_const(val)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass

    # Double negation: --x -> x
    if op == "neg" and isinstance(arg, UnaryOp) and arg.op == "neg":
        return arg.arg

    # exp(ln(x)) -> x
    if op == "exp" and isinstance(arg, UnaryOp) and arg.op == "ln":
        return arg.arg

    # ln(exp(x)) -> x
    if op == "ln" and isinstance(arg, UnaryOp) and arg.op == "exp":
        return arg.arg

    # neg(0) -> 0
    if op == "neg" and _is_const(arg, 0):
        return Const(0)

    # --- negation distribution ---
    # -(x + y) -> (-x) - y
    if op == "neg" and isinstance(arg, BinOp) and arg.op == "add":
        return BinOp("sub", UnaryOp("neg", arg.left), arg.right)
    # -(x - y) -> y - x
    if op == "neg" and isinstance(arg, BinOp) and arg.op == "sub":
        return BinOp("sub", arg.right, arg.left)
    # -(x * y) -> (-x) * y
    if op == "neg" and isinstance(arg, BinOp) and arg.op == "mul":
        return BinOp("mul", UnaryOp("neg", arg.left), arg.right)

    # --- trig parity ---
    # sin(-x) -> -sin(x)
    if op == "sin" and isinstance(arg, UnaryOp) and arg.op == "neg":
        return UnaryOp("neg", UnaryOp("sin", arg.arg))
    # cos(-x) -> cos(x)
    if op == "cos" and isinstance(arg, UnaryOp) and arg.op == "neg":
        return UnaryOp("cos", arg.arg)
    # tan(-x) -> -tan(x)
    if op == "tan" and isinstance(arg, UnaryOp) and arg.op == "neg":
        return UnaryOp("neg", UnaryOp("tan", arg.arg))

    return UnaryOp(op, arg)


def _simplify_binary(op, left, right):
    # Constant folding
    if isinstance(left, Const) and isinstance(right, Const):
        try:
            val = _BINARY_FNS[op](left.value, right.value)
            return _make_const(val)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass

    # --- additive identities ---
    if op == "add":
        if _is_const(left, 0):
            return right
        if _is_const(right, 0):
            return left
        # x + (-y) -> x - y
        if isinstance(right, UnaryOp) and right.op == "neg":
            return BinOp("sub", left, right.arg)
        # (-x) + y -> y - x
        if isinstance(left, UnaryOp) and left.op == "neg":
            return BinOp("sub", right, left.arg)
        # Like-term collection: a*x + b*x -> (a+b)*x
        lc, lb = _extract_coeff_base(left)
        rc, rb = _extract_coeff_base(right)
        if lb is not None and rb is not None and lb == rb:
            return _make_term(lc + rc, lb)

    if op == "sub":
        if _is_const(right, 0):
            return left
        if _is_const(left, 0):
            return UnaryOp("neg", right)
        # x - x -> 0
        if left == right:
            return Const(0)
        # x - (-y) -> x + y
        if isinstance(right, UnaryOp) and right.op == "neg":
            return BinOp("add", left, right.arg)
        # Like-term collection: a*x - b*x -> (a-b)*x
        lc, lb = _extract_coeff_base(left)
        rc, rb = _extract_coeff_base(right)
        if lb is not None and rb is not None and lb == rb:
            return _make_term(lc - rc, lb)

    # --- multiplicative identities ---
    if op == "mul":
        if _is_const(left, 1):
            return right
        if _is_const(right, 1):
            return left
        if _is_const(left, 0) or _is_const(right, 0):
            return Const(0)
        if _is_const(left, -1):
            return UnaryOp("neg", right)
        if _is_const(right, -1):
            return UnaryOp("neg", left)
        # x * x -> x ^ 2
        if left == right:
            return BinOp("pow", left, Const(2))
        # x^a * x^b -> x^(a+b)
        if (isinstance(left, BinOp) and left.op == "pow" and
                isinstance(right, BinOp) and right.op == "pow" and
                left.left == right.left):
            new_exp = _simplify_binary("add", left.right, right.right)
            return BinOp("pow", left.left, new_exp)
        # x * x^a -> x^(a+1)
        if isinstance(right, BinOp) and right.op == "pow" and left == right.left:
            new_exp = _simplify_binary("add", right.right, Const(1))
            return BinOp("pow", left, new_exp)
        # x^a * x -> x^(a+1)
        if isinstance(left, BinOp) and left.op == "pow" and right == left.left:
            new_exp = _simplify_binary("add", left.right, Const(1))
            return BinOp("pow", right, new_exp)

    if op == "div":
        if _is_const(right, 1):
            return left
        if _is_const(left, 0):
            return Const(0)
        # x / x -> 1
        if left == right:
            return Const(1)

    # --- power identities ---
    if op == "pow":
        if _is_const(right, 0):
            return Const(1)
        if _is_const(right, 1):
            return left
        if _is_const(left, 1):
            return Const(1)
        # (x^a)^b -> x^(a*b)
        if isinstance(left, BinOp) and left.op == "pow":
            new_exp = _simplify_binary("mul", left.right, right)
            return BinOp("pow", left.left, new_exp)

    return BinOp(op, left, right)
