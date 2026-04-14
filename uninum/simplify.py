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
        if _struct_eq(nxt, cur):
            return nxt
        prev = cur
        cur = nxt
    return cur


# ---------------------------------------------------------------------------
# Structural equality
# ---------------------------------------------------------------------------

def _struct_eq(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, Const):
        return a.value == b.value
    if isinstance(a, Var):
        return a.name == b.name
    if isinstance(a, UnaryOp):
        return a.op == b.op and _struct_eq(a.arg, b.arg)
    if isinstance(a, BinOp):
        return a.op == b.op and _struct_eq(a.left, b.left) and _struct_eq(a.right, b.right)
    return False


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

    if op == "sub":
        if _is_const(right, 0):
            return left
        if _is_const(left, 0):
            return UnaryOp("neg", right)
        # x - x -> 0
        if _struct_eq(left, right):
            return Const(0)
        # x - (-y) -> x + y
        if isinstance(right, UnaryOp) and right.op == "neg":
            return BinOp("add", left, right.arg)

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

    if op == "div":
        if _is_const(right, 1):
            return left
        if _is_const(left, 0):
            return Const(0)
        # x / x -> 1
        if _struct_eq(left, right):
            return Const(1)

    # --- power identities ---
    if op == "pow":
        if _is_const(right, 0):
            return Const(1)
        if _is_const(right, 1):
            return left
        if _is_const(left, 1):
            return Const(1)

    return BinOp(op, left, right)
