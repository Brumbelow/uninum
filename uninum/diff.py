"""Symbolic differentiation via standard derivative rules."""

from .expr import Expr, Const, Var, UnaryOp, BinOp, ln


def differentiate(expr, wrt):
    """Return the symbolic derivative of *expr* with respect to variable *wrt*."""
    if isinstance(wrt, Var):
        wrt_name = wrt.name
    elif isinstance(wrt, str):
        wrt_name = wrt
    else:
        raise TypeError("wrt must be a Var or variable name string")
    return _diff(expr, wrt_name)


def _diff(node, name):
    if isinstance(node, Const):
        return Const(0)

    if isinstance(node, Var):
        return Const(1) if node.name == name else Const(0)

    if isinstance(node, UnaryOp):
        return _diff_unary(node, name)

    if isinstance(node, BinOp):
        return _diff_binary(node, name)

    raise TypeError(f"Unknown node type: {type(node)}")


# ---------------------------------------------------------------------------
# Unary rules
# ---------------------------------------------------------------------------

def _diff_unary(node, name):
    f = node.arg
    fp = _diff(f, name)
    op = node.op

    if op == "neg":
        # d/dx (-f) = -f'
        return UnaryOp("neg", fp)

    if op == "exp":
        # d/dx exp(f) = exp(f) * f'
        return BinOp("mul", UnaryOp("exp", f), fp)

    if op == "ln":
        # d/dx ln(f) = f' / f
        return BinOp("div", fp, f)

    if op == "sin":
        # d/dx sin(f) = cos(f) * f'
        return BinOp("mul", UnaryOp("cos", f), fp)

    if op == "cos":
        # d/dx cos(f) = -sin(f) * f'
        return BinOp("mul", UnaryOp("neg", UnaryOp("sin", f)), fp)

    if op == "tan":
        # d/dx tan(f) = f' / cos(f)^2
        return BinOp("div", fp, BinOp("pow", UnaryOp("cos", f), Const(2)))

    if op == "asin":
        # d/dx asin(f) = f' / sqrt(1 - f^2)
        return BinOp(
            "div", fp,
            UnaryOp("sqrt", BinOp("sub", Const(1), BinOp("pow", f, Const(2)))),
        )

    if op == "acos":
        # d/dx acos(f) = -f' / sqrt(1 - f^2)
        return UnaryOp(
            "neg",
            BinOp(
                "div", fp,
                UnaryOp("sqrt", BinOp("sub", Const(1), BinOp("pow", f, Const(2)))),
            ),
        )

    if op == "atan":
        # d/dx atan(f) = f' / (1 + f^2)
        return BinOp("div", fp, BinOp("add", Const(1), BinOp("pow", f, Const(2))))

    if op == "sinh":
        # d/dx sinh(f) = cosh(f) * f'
        return BinOp("mul", UnaryOp("cosh", f), fp)

    if op == "cosh":
        # d/dx cosh(f) = sinh(f) * f'
        return BinOp("mul", UnaryOp("sinh", f), fp)

    if op == "tanh":
        # d/dx tanh(f) = f' / cosh(f)^2
        return BinOp("div", fp, BinOp("pow", UnaryOp("cosh", f), Const(2)))

    if op == "sqrt":
        # d/dx sqrt(f) = f' / (2 * sqrt(f))
        return BinOp("div", fp, BinOp("mul", Const(2), UnaryOp("sqrt", f)))

    if op == "inv":
        # d/dx (1/f) = -f' / f^2
        return UnaryOp("neg", BinOp("div", fp, BinOp("pow", f, Const(2))))

    if op == "abs":
        # d/dx |f| = f' * f / |f|  (undefined at 0)
        return BinOp("mul", fp, BinOp("div", f, UnaryOp("abs", f)))

    raise ValueError(f"No derivative rule for unary op {op!r}")


# ---------------------------------------------------------------------------
# Binary rules
# ---------------------------------------------------------------------------

def _diff_binary(node, name):
    f, g = node.left, node.right
    fp, gp = _diff(f, name), _diff(g, name)
    op = node.op

    if op == "add":
        # (f + g)' = f' + g'
        return BinOp("add", fp, gp)

    if op == "sub":
        # (f - g)' = f' - g'
        return BinOp("sub", fp, gp)

    if op == "mul":
        # (f * g)' = f'g + fg'
        return BinOp("add", BinOp("mul", fp, g), BinOp("mul", f, gp))

    if op == "div":
        # (f / g)' = (f'g - fg') / g^2
        return BinOp(
            "div",
            BinOp("sub", BinOp("mul", fp, g), BinOp("mul", f, gp)),
            BinOp("pow", g, Const(2)),
        )

    if op == "pow":
        # (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        return BinOp(
            "mul",
            BinOp("pow", f, g),
            BinOp(
                "add",
                BinOp("mul", gp, ln(f)),
                BinOp("mul", g, BinOp("div", fp, f)),
            ),
        )

    if op == "eml":
        # eml(f,g) = exp(f) - ln(g)
        # d/dx eml(f,g) = exp(f)*f' - g'/g
        return BinOp(
            "sub",
            BinOp("mul", UnaryOp("exp", f), fp),
            BinOp("div", gp, g),
        )

    raise ValueError(f"No derivative rule for binary op {op!r}")
