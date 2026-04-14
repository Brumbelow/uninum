"""LaTeX rendering for expression trees."""

from .expr import Expr, Const, Var, UnaryOp, BinOp

# ---------------------------------------------------------------------------
# Precedence (matches expr.py _PREC)
# ---------------------------------------------------------------------------

_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2, "neg": 3, "pow": 4}

# ---------------------------------------------------------------------------
# Greek letter detection
# ---------------------------------------------------------------------------

_GREEK = frozenset({
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
    "Phi", "Psi", "Omega",
})

# ---------------------------------------------------------------------------
# Unary op -> LaTeX name
# ---------------------------------------------------------------------------

_UNARY_LATEX = {
    "exp": r"\exp",
    "ln": r"\ln",
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "asin": r"\arcsin",
    "acos": r"\arccos",
    "atan": r"\arctan",
    "sinh": r"\sinh",
    "cosh": r"\cosh",
    "tanh": r"\tanh",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_latex(expr):
    """Convert an expression tree to a LaTeX math string."""
    return _render(expr)


# ---------------------------------------------------------------------------
# Internal rendering
# ---------------------------------------------------------------------------


def _render(node):
    if isinstance(node, Const):
        return _render_const(node)
    if isinstance(node, Var):
        return _render_var(node)
    if isinstance(node, UnaryOp):
        return _render_unary(node)
    if isinstance(node, BinOp):
        return _render_binary(node)
    raise TypeError(f"Unknown node type: {type(node)}")


def _render_const(node):
    if node.name:
        if node.name == "pi":
            return r"\pi"
        if node.name == "e":
            return r"\mathrm{e}"
        if node.name == "i":
            return r"\mathrm{i}"
        return node.name
    v = node.value
    if isinstance(v, complex):
        if v.imag == 0:
            v = v.real
        elif v.real == 0:
            if v.imag == 1:
                return r"\mathrm{i}"
            if v.imag == -1:
                return r"-\mathrm{i}"
            return rf"{_fmt_num(v.imag)}\mathrm{{i}}"
        else:
            return rf"{_fmt_num(v.real)} + {_fmt_num(v.imag)}\mathrm{{i}}"
    return _fmt_num(v)


def _fmt_num(v):
    """Format a real number for LaTeX."""
    if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def _render_var(node):
    if node.name in _GREEK:
        return rf"\{node.name}"
    return node.name


def _render_unary(node):
    op = node.op
    inner = _render(node.arg)

    if op == "neg":
        if isinstance(node.arg, BinOp) or (
            isinstance(node.arg, UnaryOp) and node.arg.op == "neg"
        ):
            return rf"-\left({inner}\right)"
        return f"-{inner}"

    if op == "sqrt":
        return rf"\sqrt{{{inner}}}"

    if op == "inv":
        return rf"\frac{{1}}{{{inner}}}"

    if op == "abs":
        return rf"\left\lvert {inner} \right\rvert"

    latex_name = _UNARY_LATEX.get(op, rf"\operatorname{{{op}}}")
    return rf"{latex_name}\left({inner}\right)"


def _render_binary(node):
    op = node.op

    if op == "eml":
        l = _render(node.left)
        r = _render(node.right)
        return rf"\operatorname{{eml}}\left({l}, {r}\right)"

    if op == "div":
        l = _render(node.left)
        r = _render(node.right)
        return rf"\frac{{{l}}}{{{r}}}"

    if op == "pow":
        l = _render(node.left)
        r = _render(node.right)
        # Base needs parens if it's a binop or negation
        if isinstance(node.left, BinOp) or (
            isinstance(node.left, UnaryOp) and node.left.op == "neg"
        ):
            l = rf"\left({l}\right)"
        return rf"{l}^{{{r}}}"

    if op == "add":
        l = _render_with_parens(node.left, op, "left")
        r = _render_with_parens(node.right, op, "right")
        return f"{l} + {r}"

    if op == "sub":
        l = _render_with_parens(node.left, op, "left")
        r = _render_with_parens(node.right, op, "right")
        return f"{l} - {r}"

    if op == "mul":
        l = _render_with_parens(node.left, op, "left")
        r = _render_with_parens(node.right, op, "right")
        return rf"{l} \cdot {r}"

    # Fallback
    l = _render(node.left)
    r = _render(node.right)
    return rf"\operatorname{{{op}}}\left({l}, {r}\right)"


def _render_with_parens(child, parent_op, side):
    """Render a child node, wrapping in parens if needed by precedence."""
    inner = _render(child)
    if not isinstance(child, BinOp):
        return inner

    child_prec = _PREC.get(child.op, 10)
    parent_prec = _PREC.get(parent_op, 10)

    needs = False
    if child_prec < parent_prec:
        needs = True
    elif child_prec == parent_prec and side == "right" and parent_op in ("sub", "div"):
        needs = True

    if needs:
        return rf"\left({inner}\right)"
    return inner
