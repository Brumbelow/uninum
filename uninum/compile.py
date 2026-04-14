"""Compile an expression tree into a fast callable function."""

import cmath
import operator

from .expr import Expr, Const, Var, UnaryOp, BinOp


def compile(expr, backend="numpy"):
    """Compile *expr* into a callable ``f(**kwargs) -> value``.

    Parameters
    ----------
    expr : Expr
        The expression to compile.
    backend : str
        ``"numpy"`` for vectorised evaluation or ``"python"`` for scalar
        evaluation using *cmath*.
    """
    # --- flatten the expression DAG into a linear op list ---
    ops_list = []
    seen = {}

    def _flatten(node):
        nid = id(node)
        if nid in seen:
            return seen[nid]

        if isinstance(node, Const):
            idx = len(ops_list)
            ops_list.append(("const", node.value))
        elif isinstance(node, Var):
            idx = len(ops_list)
            ops_list.append(("var", node.name))
        elif isinstance(node, UnaryOp):
            ai = _flatten(node.arg)
            idx = len(ops_list)
            ops_list.append(("unary", node.op, ai))
        elif isinstance(node, BinOp):
            li = _flatten(node.left)
            ri = _flatten(node.right)
            idx = len(ops_list)
            ops_list.append(("binary", node.op, li, ri))
        else:
            raise TypeError(f"Unknown node type: {type(node)}")

        seen[nid] = idx
        return idx

    result_idx = _flatten(expr)
    frozen = tuple(ops_list)
    n = len(frozen)

    # --- build dispatch tables ---
    if backend == "numpy":
        import numpy as np

        ufns = {
            "exp": np.exp,
            "ln": np.log,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "sqrt": np.sqrt,
            "neg": np.negative,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
            "asin": np.arcsin,
            "acos": np.arccos,
            "atan": np.arctan,
            "abs": np.abs,
            "inv": np.reciprocal,
        }
        bfns = {
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "div": np.divide,
            "pow": np.power,
            "eml": lambda x, y: np.exp(x) - np.log(y),
        }
    elif backend == "python":
        ufns = {
            "exp": cmath.exp,
            "ln": cmath.log,
            "sin": cmath.sin,
            "cos": cmath.cos,
            "tan": cmath.tan,
            "sqrt": cmath.sqrt,
            "neg": operator.neg,
            "sinh": cmath.sinh,
            "cosh": cmath.cosh,
            "tanh": cmath.tanh,
            "asin": cmath.asin,
            "acos": cmath.acos,
            "atan": cmath.atan,
            "abs": abs,
            "inv": lambda x: 1 / x,
        }
        bfns = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.truediv,
            "pow": operator.pow,
            "eml": lambda x, y: cmath.exp(x) - cmath.log(y),
        }
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    # --- the compiled callable ---
    def run(**kwargs):
        results = [None] * n
        for i in range(n):
            entry = frozen[i]
            kind = entry[0]
            if kind == "const":
                results[i] = entry[1]
            elif kind == "var":
                results[i] = kwargs[entry[1]]
            elif kind == "unary":
                results[i] = ufns[entry[1]](results[entry[2]])
            else:  # binary
                results[i] = bfns[entry[1]](results[entry[2]], results[entry[3]])
        return results[result_idx]

    return run
