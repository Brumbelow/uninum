"""EML lowering: convert high-level expressions to pure eml(S, S) trees.

Two-phase approach:
  Phase A — rewrite to {exp, ln, sub, Const(1), Var}
  Phase B — rewrite to {eml, Const(1), Var}
"""

import math

from .expr import Const, Var, UnaryOp, BinOp


# ===================================================================
# Public entry point
# ===================================================================

def to_eml(expr):
    """Lower *expr* to a pure EML tree: only Const(1), Var, and BinOp('eml')."""
    intermediate = _phase_a(expr)
    return _phase_b(intermediate)


# ===================================================================
# Phase A — rewrite everything to {Const(1), Var, exp, ln, sub}
# ===================================================================

# --- helper builders (all return expressions in {Const(1), Var, exp, ln, sub}) ---

_ONE = Const(1)


def _zero():
    """0 = ln(1)."""
    return UnaryOp("ln", _ONE)


def _neg_via_sub(a):
    """-a = 0 - a."""
    return BinOp("sub", _zero(), a)


def _add_via_sub(a, b):
    """a + b = a - (0 - b)."""
    return BinOp("sub", a, BinOp("sub", _zero(), b))


def _mul_via_exp_ln(a, b):
    """a * b = exp(ln(a) + ln(b)) = exp(ln(a) - (0 - ln(b)))."""
    ln_a = UnaryOp("ln", a)
    ln_b = UnaryOp("ln", b)
    return UnaryOp("exp", BinOp("sub", ln_a, BinOp("sub", _zero(), ln_b)))


def _div_via_exp_ln(a, b):
    """a / b = exp(ln(a) - ln(b))."""
    return UnaryOp("exp", BinOp("sub", UnaryOp("ln", a), UnaryOp("ln", b)))


def _make_e():
    """e = exp(1)."""
    return UnaryOp("exp", _ONE)


def _make_neg1():
    """-1 = 0 - 1."""
    return BinOp("sub", _zero(), _ONE)


def _make_two():
    """2 = 1 + 1."""
    return _add_via_sub(_ONE, _ONE)


def _make_i():
    """i = exp(ln(-1) / 2).

    Since ln(-1) = i*pi (principal branch), ln(-1)/2 = i*pi/2, exp(i*pi/2) = i.
    """
    ln_neg1 = UnaryOp("ln", _make_neg1())
    two = _make_two()
    half_ln_neg1 = _div_via_exp_ln(ln_neg1, two)
    return UnaryOp("exp", half_ln_neg1)


def _make_pi():
    """pi = ln(-1) / i."""
    ln_neg1 = UnaryOp("ln", _make_neg1())
    i_val = _make_i()
    return _div_via_exp_ln(ln_neg1, i_val)


def _lower_const_a(value):
    """Express a numeric constant using {Const(1), exp, ln, sub}."""
    if value == 1:
        return _ONE
    if value == 0:
        return _zero()
    if isinstance(value, int):
        if value > 0:
            result = _ONE
            for _ in range(value - 1):
                result = _add_via_sub(result, _ONE)
            return result
        # negative integer
        return _neg_via_sub(_lower_const_a(-value))
    if isinstance(value, float):
        if abs(value - math.e) < 1e-15:
            return _make_e()
        if abs(value - math.pi) < 1e-15:
            return _make_pi()
    if value == 1j:
        return _make_i()
    # Fallback: leave as Const (partial lowering)
    return Const(value)


def _phase_a(node):
    """Rewrite *node* so only {Const(1), Var, UnaryOp(exp/ln), BinOp(sub)} remain."""
    if isinstance(node, Var):
        return node

    if isinstance(node, Const):
        return _lower_const_a(node.value)

    if isinstance(node, UnaryOp):
        a = _phase_a(node.arg)
        op = node.op

        if op in ("exp", "ln"):
            return UnaryOp(op, a)

        if op == "neg":
            return _neg_via_sub(a)

        if op == "inv":
            # 1/a = exp(-ln(a)) = exp(0 - ln(a))
            return UnaryOp("exp", BinOp("sub", _zero(), UnaryOp("ln", a)))

        if op == "sqrt":
            # sqrt(a) = exp(ln(a) / 2) = exp(ln(a) - ln(2))
            return UnaryOp("exp", _div_via_exp_ln(UnaryOp("ln", a), _make_two()))

        if op == "abs":
            # |a| = sqrt(a * conj(a)) — but for real a, |a| = exp(ln(a^2)/2)
            # Use: |a| = sqrt(a^2) which works for positive reals
            a_sq = _mul_via_exp_ln(a, a)
            return UnaryOp(
                "exp",
                _div_via_exp_ln(UnaryOp("ln", a_sq), _make_two()),
            )

        if op == "sin":
            # sin(a) = (exp(ia) - exp(-ia)) / (2i)
            i_val = _make_i()
            ia = _mul_via_exp_ln(i_val, a)
            neg_ia = _neg_via_sub(ia)
            numerator = BinOp("sub", UnaryOp("exp", ia), UnaryOp("exp", neg_ia))
            two_i = _mul_via_exp_ln(_make_two(), i_val)
            return _div_via_exp_ln(numerator, two_i)

        if op == "cos":
            # cos(a) = (exp(ia) + exp(-ia)) / 2
            i_val = _make_i()
            ia = _mul_via_exp_ln(i_val, a)
            neg_ia = _neg_via_sub(ia)
            numerator = _add_via_sub(UnaryOp("exp", ia), UnaryOp("exp", neg_ia))
            return _div_via_exp_ln(numerator, _make_two())

        if op == "tan":
            # tan(a) = sin(a) / cos(a)
            sin_a = _phase_a(UnaryOp("sin", a))
            cos_a = _phase_a(UnaryOp("cos", a))
            return _div_via_exp_ln(sin_a, cos_a)

        if op == "sinh":
            # sinh(a) = (exp(a) - exp(-a)) / 2
            ea = UnaryOp("exp", a)
            e_neg_a = UnaryOp("exp", _neg_via_sub(a))
            return _div_via_exp_ln(BinOp("sub", ea, e_neg_a), _make_two())

        if op == "cosh":
            # cosh(a) = (exp(a) + exp(-a)) / 2
            ea = UnaryOp("exp", a)
            e_neg_a = UnaryOp("exp", _neg_via_sub(a))
            return _div_via_exp_ln(_add_via_sub(ea, e_neg_a), _make_two())

        if op == "tanh":
            # tanh(a) = sinh(a) / cosh(a)
            sinh_a = _phase_a(UnaryOp("sinh", a))
            cosh_a = _phase_a(UnaryOp("cosh", a))
            return _div_via_exp_ln(sinh_a, cosh_a)

        if op == "asin":
            # asin(a) = -i * ln(ia + sqrt(1 - a^2))
            i_val = _make_i()
            neg_i = _neg_via_sub(i_val)
            a_sq = _mul_via_exp_ln(a, a)
            inner = _add_via_sub(
                _mul_via_exp_ln(i_val, a),
                UnaryOp(
                    "exp",
                    _div_via_exp_ln(
                        UnaryOp("ln", BinOp("sub", _ONE, a_sq)),
                        _make_two(),
                    ),
                ),
            )
            return _mul_via_exp_ln(neg_i, UnaryOp("ln", inner))

        if op == "acos":
            # acos(a) = -i * ln(a + i*sqrt(1 - a^2))
            i_val = _make_i()
            neg_i = _neg_via_sub(i_val)
            a_sq = _mul_via_exp_ln(a, a)
            sqrt_part = UnaryOp(
                "exp",
                _div_via_exp_ln(
                    UnaryOp("ln", BinOp("sub", _ONE, a_sq)),
                    _make_two(),
                ),
            )
            inner = _add_via_sub(a, _mul_via_exp_ln(i_val, sqrt_part))
            return _mul_via_exp_ln(neg_i, UnaryOp("ln", inner))

        if op == "atan":
            # atan(a) = (i/2) * ln((1 - ia) / (1 + ia))
            i_val = _make_i()
            ia = _mul_via_exp_ln(i_val, a)
            num = BinOp("sub", _ONE, ia)
            den = _add_via_sub(_ONE, ia)
            ratio = _div_via_exp_ln(num, den)
            half_i = _div_via_exp_ln(i_val, _make_two())
            return _mul_via_exp_ln(half_i, UnaryOp("ln", ratio))

        raise ValueError(f"EML lowering not implemented for unary op {op!r}")

    if isinstance(node, BinOp):
        l = _phase_a(node.left)
        r = _phase_a(node.right)
        op = node.op

        if op == "sub":
            return BinOp("sub", l, r)

        if op == "add":
            return _add_via_sub(l, r)

        if op == "mul":
            return _mul_via_exp_ln(l, r)

        if op == "div":
            return _div_via_exp_ln(l, r)

        if op == "pow":
            # a^b = exp(b * ln(a))
            return UnaryOp("exp", _mul_via_exp_ln(r, UnaryOp("ln", l)))

        if op == "eml":
            # Already eml — but children need lowering; will be handled in phase B.
            # Rewrite as exp(l) - ln(r) then lower.
            return BinOp("sub", UnaryOp("exp", l), UnaryOp("ln", r))

        raise ValueError(f"EML lowering not implemented for binary op {op!r}")

    raise TypeError(f"Unknown node type: {type(node)}")


# ===================================================================
# Phase B — rewrite {Const(1), Var, exp, ln, sub} to {Const(1), Var, eml}
# ===================================================================

def _phase_b(node):
    """Single bottom-up pass converting exp/ln/sub to eml."""
    if isinstance(node, (Const, Var)):
        return node

    if isinstance(node, UnaryOp):
        a = _phase_b(node.arg)
        if node.op == "exp":
            # exp(a) = eml(a, 1)
            return BinOp("eml", a, _ONE)
        if node.op == "ln":
            # ln(a) = eml(1, eml(eml(1, a), 1))
            return BinOp("eml", _ONE, BinOp("eml", BinOp("eml", _ONE, a), _ONE))
        raise ValueError(
            f"Phase B encountered unexpected unary op {node.op!r} "
            "(Phase A incomplete)"
        )

    if isinstance(node, BinOp):
        l = _phase_b(node.left)
        r = _phase_b(node.right)
        if node.op == "sub":
            # a - b = eml(ln(a), exp(b))
            # ln(a) in eml form:
            ln_l = BinOp("eml", _ONE, BinOp("eml", BinOp("eml", _ONE, l), _ONE))
            # exp(b) in eml form:
            exp_r = BinOp("eml", r, _ONE)
            return BinOp("eml", ln_l, exp_r)
        if node.op == "eml":
            return BinOp("eml", l, r)
        raise ValueError(
            f"Phase B encountered unexpected binary op {node.op!r} "
            "(Phase A incomplete)"
        )

    raise TypeError(f"Unknown node type: {type(node)}")
