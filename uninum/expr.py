"""Expression node types, operator overloading, printing, and evaluation."""

import cmath
import math
import operator

# ---------------------------------------------------------------------------
# Safe wrappers for extended-real semantics (needed by EML expressions)
# Python's cmath raises on log(0) and exp(inf); IEEE754 / C's math.h do not.
# ---------------------------------------------------------------------------

_INF = float("inf")
_NEG_INF = float("-inf")


def _safe_exp(x):
    if isinstance(x, complex):
        if math.isinf(x.real) and x.imag == 0:
            return _INF if x.real > 0 else 0.0
    elif isinstance(x, float) and math.isinf(x):
        return _INF if x > 0 else 0.0
    try:
        return cmath.exp(x)
    except OverflowError:
        r = x.real if isinstance(x, complex) else x
        return complex(_INF, 0) if r > 0 else complex(0, 0)


def _safe_log(x):
    if x == 0 or x == 0j:
        return complex(_NEG_INF, 0)
    return cmath.log(x)


# ---------------------------------------------------------------------------
# Dispatch tables for evaluation
# ---------------------------------------------------------------------------

_UNARY_FNS = {
    "exp": _safe_exp,
    "ln": _safe_log,
    "sin": cmath.sin,
    "cos": cmath.cos,
    "tan": cmath.tan,
    "asin": cmath.asin,
    "acos": cmath.acos,
    "atan": cmath.atan,
    "sinh": cmath.sinh,
    "cosh": cmath.cosh,
    "tanh": cmath.tanh,
    "sqrt": cmath.sqrt,
    "neg": operator.neg,
    "inv": lambda x: 1 / x,
    "abs": abs,
}

_BINARY_FNS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "pow": operator.pow,
    "eml": lambda x, y: _safe_exp(x) - _safe_log(y),
}

# ---------------------------------------------------------------------------
# Precedence for pretty-printing
# ---------------------------------------------------------------------------

_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2, "pow": 4}
_NEG_PREC = 3
_SYM = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}


def _child_prec(node):
    if isinstance(node, (Const, Var)):
        return 100
    if isinstance(node, UnaryOp):
        return _NEG_PREC if node.op == "neg" else 100
    if isinstance(node, BinOp):
        return _PREC.get(node.op, 100)
    return 100


def _needs_parens_left(child, parent_op):
    p = _PREC[parent_op]
    cp = _child_prec(child)
    if cp < p:
        return True
    if cp == p and parent_op == "pow":
        return True
    return False


def _needs_parens_right(child, parent_op):
    p = _PREC[parent_op]
    cp = _child_prec(child)
    if cp < p:
        return True
    if cp == p and parent_op in ("sub", "div"):
        return True
    return False


# ---------------------------------------------------------------------------
# Wrap helper
# ---------------------------------------------------------------------------


def _wrap(val):
    if isinstance(val, Expr):
        return val
    if isinstance(val, (int, float, complex)):
        return Const(val)
    raise TypeError(f"Cannot convert {type(val).__name__} to Expr")


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


class Expr:
    """Base class for all expression nodes."""

    # -- arithmetic operator overloads --

    def __add__(self, other):
        return BinOp("add", self, _wrap(other))

    def __radd__(self, other):
        return BinOp("add", _wrap(other), self)

    def __sub__(self, other):
        return BinOp("sub", self, _wrap(other))

    def __rsub__(self, other):
        return BinOp("sub", _wrap(other), self)

    def __mul__(self, other):
        return BinOp("mul", self, _wrap(other))

    def __rmul__(self, other):
        return BinOp("mul", _wrap(other), self)

    def __truediv__(self, other):
        return BinOp("div", self, _wrap(other))

    def __rtruediv__(self, other):
        return BinOp("div", _wrap(other), self)

    def __pow__(self, other):
        return BinOp("pow", self, _wrap(other))

    def __rpow__(self, other):
        return BinOp("pow", _wrap(other), self)

    def __neg__(self):
        return UnaryOp("neg", self)

    # -- delegated methods --

    def diff(self, wrt):
        from .diff import differentiate

        return differentiate(self, wrt)

    def simplify(self):
        from .simplify import simplify

        return simplify(self)

    def to_eml(self):
        from .eml import to_eml

        return to_eml(self)

    # -- inspection --

    def free_vars(self):
        """Return the set of variable names in this expression."""
        result = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if isinstance(node, Var):
                result.add(node.name)
            elif isinstance(node, UnaryOp):
                stack.append(node.arg)
            elif isinstance(node, BinOp):
                stack.append(node.left)
                stack.append(node.right)
        return result

    # -- evaluation --

    def evaluate(self, **kwargs):
        val = self._eval_impl(kwargs)
        if isinstance(val, complex):
            if abs(val.imag) <= 1e-15 * max(abs(val.real), 1e-300):
                return val.real
        return val

    def _eval_impl(self, env):
        raise NotImplementedError


class Const(Expr):
    __slots__ = ("value", "name")

    def __init__(self, value, name=None):
        self.value = value
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, Const):
            return NotImplemented
        return self.value == other.value

    def __hash__(self):
        return hash(("Const", self.value))

    def __str__(self):
        if self.name:
            return self.name
        v = self.value
        if isinstance(v, complex):
            if v.imag == 0:
                v = v.real
            elif v.real == 0:
                if v.imag == 1:
                    return "i"
                if v.imag == -1:
                    return "-i"
                return f"{v.imag}i"
            else:
                return repr(v)
        if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v)

    def __repr__(self):
        return f"Const({self.value!r})"

    def _eval_impl(self, env):
        return self.value


class Var(Expr):
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, Var):
            return NotImplemented
        return self.name == other.name

    def __hash__(self):
        return hash(("Var", self.name))

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Var({self.name!r})"

    def _eval_impl(self, env):
        if self.name not in env:
            raise ValueError(f"Missing variable: {self.name!r}")
        return env[self.name]


class UnaryOp(Expr):
    __slots__ = ("op", "arg")

    def __init__(self, op, arg):
        self.op = op
        self.arg = arg

    def __eq__(self, other):
        if not isinstance(other, UnaryOp):
            return NotImplemented
        return self.op == other.op and self.arg == other.arg

    def __hash__(self):
        return hash(("UnaryOp", self.op, self.arg))

    def __str__(self):
        if self.op == "neg":
            s = str(self.arg)
            if isinstance(self.arg, BinOp) or (
                isinstance(self.arg, UnaryOp) and self.arg.op == "neg"
            ):
                s = f"({s})"
            return f"-{s}"
        return f"{self.op}({self.arg})"

    def __repr__(self):
        return f"UnaryOp({self.op!r}, {self.arg!r})"

    def _eval_impl(self, env):
        val = self.arg._eval_impl(env)
        return _UNARY_FNS[self.op](val)


class BinOp(Expr):
    __slots__ = ("op", "left", "right")

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __eq__(self, other):
        if not isinstance(other, BinOp):
            return NotImplemented
        return self.op == other.op and self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(("BinOp", self.op, self.left, self.right))

    def __str__(self):
        if self.op == "eml":
            return f"eml({self.left}, {self.right})"
        sym = _SYM[self.op]
        ls = str(self.left)
        rs = str(self.right)
        if _needs_parens_left(self.left, self.op):
            ls = f"({ls})"
        if _needs_parens_right(self.right, self.op):
            rs = f"({rs})"
        return f"{ls} {sym} {rs}"

    def __repr__(self):
        return f"BinOp({self.op!r}, {self.left!r}, {self.right!r})"

    def _eval_impl(self, env):
        l = self.left._eval_impl(env)
        r = self.right._eval_impl(env)
        return _BINARY_FNS[self.op](l, r)


# ---------------------------------------------------------------------------
# Constructor functions
# ---------------------------------------------------------------------------


def var(name):
    return Var(name)


def exp(x):
    return UnaryOp("exp", _wrap(x))


def ln(x):
    return UnaryOp("ln", _wrap(x))


log = ln


def sin(x):
    return UnaryOp("sin", _wrap(x))


def cos(x):
    return UnaryOp("cos", _wrap(x))


def tan(x):
    return UnaryOp("tan", _wrap(x))


def asin(x):
    return UnaryOp("asin", _wrap(x))


def acos(x):
    return UnaryOp("acos", _wrap(x))


def atan(x):
    return UnaryOp("atan", _wrap(x))


def sinh(x):
    return UnaryOp("sinh", _wrap(x))


def cosh(x):
    return UnaryOp("cosh", _wrap(x))


def tanh(x):
    return UnaryOp("tanh", _wrap(x))


def sqrt(x):
    return UnaryOp("sqrt", _wrap(x))


def eml(x, y):
    return BinOp("eml", _wrap(x), _wrap(y))


# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

e = Const(math.e, "e")
pi = Const(math.pi, "pi")
I = Const(1j, "i")
