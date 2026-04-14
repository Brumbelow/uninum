# API Reference

## Node Types

### `Expr`

Base class for all expression nodes. Not instantiated directly.

**Operator overloading:**

| Operator | Python syntax | Internal op |
|----------|---------------|-------------|
| Addition | `a + b` | `"add"` |
| Subtraction | `a - b` | `"sub"` |
| Multiplication | `a * b` | `"mul"` |
| Division | `a / b` | `"div"` |
| Power | `a ** b` | `"pow"` |
| Negation | `-a` | `"neg"` |

All operators accept `Expr`, `int`, `float`, or `complex` on either side. Python numbers are automatically wrapped as `Const`.

**Methods:**

- **`.evaluate(**kwargs)`** -- Evaluate the expression numerically. Pass variable values as keyword arguments (e.g., `expr.evaluate(x=1.5, y=2.0)`). Raises `ValueError` if a variable is missing.

- **`.diff(wrt)`** -- Return the symbolic derivative with respect to `wrt`. Accepts a `Var` object or a variable name string. Returns an unsimplified `Expr` -- chain `.simplify()` for cleaner output.

- **`.simplify()`** -- Apply algebraic simplification rules. Returns a new `Expr`.

- **`.to_eml()`** -- Lower to a pure EML representation. Returns an `Expr` tree containing only `Const(1)`, `Var`, and `BinOp("eml", ...)`.

### `Const(value, name=None)`

A numeric constant.

- `value` -- `int`, `float`, or `complex`
- `name` -- optional display name (used by `e`, `pi`, `I`)

```python
Const(3)          # displays as "3"
Const(3.14)       # displays as "3.14"
Const(1j)         # displays as "i"
```

### `Var(name)`

A named variable. Usually created via the `var()` constructor.

```python
x = Var("x")     # equivalent to var("x")
```

### `UnaryOp(op, arg)`

A unary function application. Usually created via constructor functions.

Supported `op` values: `"exp"`, `"ln"`, `"sin"`, `"cos"`, `"tan"`, `"asin"`, `"acos"`, `"atan"`, `"sinh"`, `"cosh"`, `"tanh"`, `"sqrt"`, `"neg"`, `"inv"`, `"abs"`.

### `BinOp(op, left, right)`

A binary operation. Usually created via operators or `eml()`.

Supported `op` values: `"add"`, `"sub"`, `"mul"`, `"div"`, `"pow"`, `"eml"`.

## Constructor Functions

All constructors accept `Expr`, `int`, `float`, or `complex`.

| Function | Returns | Description |
|----------|---------|-------------|
| `var(name)` | `Var` | Create a named variable |
| `exp(x)` | `UnaryOp("exp", x)` | Exponential |
| `ln(x)` | `UnaryOp("ln", x)` | Natural logarithm |
| `log(x)` | `UnaryOp("ln", x)` | Alias for `ln` |
| `sin(x)` | `UnaryOp("sin", x)` | Sine |
| `cos(x)` | `UnaryOp("cos", x)` | Cosine |
| `tan(x)` | `UnaryOp("tan", x)` | Tangent |
| `asin(x)` | `UnaryOp("asin", x)` | Inverse sine |
| `acos(x)` | `UnaryOp("acos", x)` | Inverse cosine |
| `atan(x)` | `UnaryOp("atan", x)` | Inverse tangent |
| `sinh(x)` | `UnaryOp("sinh", x)` | Hyperbolic sine |
| `cosh(x)` | `UnaryOp("cosh", x)` | Hyperbolic cosine |
| `tanh(x)` | `UnaryOp("tanh", x)` | Hyperbolic tangent |
| `sqrt(x)` | `UnaryOp("sqrt", x)` | Square root |
| `eml(x, y)` | `BinOp("eml", x, y)` | EML operator: `exp(x) - ln(y)` |

## Named Constants

| Name | Value | Display |
|------|-------|---------|
| `e` | `math.e` (2.718...) | `e` |
| `pi` | `math.pi` (3.141...) | `pi` |
| `I` | `1j` | `i` |

These are `Const` instances and can be used in expressions like any other node:

```python
from uninum import e, pi, I, var
x = var("x")
expr = e ** (I * x)     # Euler's formula
```

## `compile(expr, backend="numpy")`

Compile an expression tree into a fast callable function.

**Parameters:**
- `expr` -- an `Expr` to compile
- `backend` -- `"numpy"` for vectorized evaluation or `"python"` for scalar evaluation

**Returns:** a callable `f(**kwargs) -> value`

```python
from uninum import var, sin, compile

x = var("x")
fn = compile(sin(x) + x ** 2, backend="python")
fn(x=1.5)    # scalar result
```

**Note:** `compile` shadows Python's builtin `compile`. If you need both, use `from uninum import compile as ucompile`.

**Backend differences:**
- **numpy** -- Uses numpy ufuncs. Accepts and returns numpy arrays for vectorized computation. Requires numpy to be installed.
- **python** -- Uses `cmath` functions. Scalar only. May return `complex` even for real-valued inputs (extract `.real` if needed). Always available.

**DAG sharing:** If the same `Expr` object appears multiple times in the tree, the compiled function evaluates it only once. This is based on Python object identity (`id()`), not structural equality.

## Evaluation Semantics

**Extended-real behavior:** The evaluator follows IEEE 754 / C conventions for edge cases:
- `exp(inf)` = `inf`, `exp(-inf)` = `0`
- `ln(0)` = `-inf`

**Complex stripping:** If the imaginary part of a result is negligibly small (|imag| <= 1e-15 * max(|real|, 1e-300)), the result is returned as a real float.

## Differentiation Rules

The `.diff()` method implements standard calculus rules:

| Expression | Derivative |
|-----------|------------|
| `c` (constant) | `0` |
| `x` (variable, same as wrt) | `1` |
| `x` (different variable) | `0` |
| `f + g` | `f' + g'` |
| `f - g` | `f' - g'` |
| `f * g` | `f'g + fg'` |
| `f / g` | `(f'g - fg') / g^2` |
| `f ^ g` | `f^g * (g' * ln(f) + g * f'/f)` |
| `exp(f)` | `exp(f) * f'` |
| `ln(f)` | `f' / f` |
| `sin(f)` | `cos(f) * f'` |
| `cos(f)` | `-sin(f) * f'` |
| `sqrt(f)` | `f' / (2 * sqrt(f))` |
| `eml(f, g)` | `exp(f) * f' - g' / g` |

All trig, inverse trig, and hyperbolic derivatives are supported. Results are unsimplified -- call `.simplify()` afterward.

## Simplification Rules

The simplifier performs bottom-up rewriting with up to 10 passes until a fixpoint.

**Constant folding:** Evaluates operations on constants directly.

**Identity rules:**
- `0 + x -> x`, `x + 0 -> x`, `x - 0 -> x`, `x - x -> 0`
- `1 * x -> x`, `0 * x -> 0`, `-1 * x -> -x`
- `x / 1 -> x`, `x / x -> 1`
- `x^0 -> 1`, `x^1 -> x`, `1^x -> 1`
- `--x -> x`, `-0 -> 0`

**Inverse cancellation:** `exp(ln(x)) -> x`, `ln(exp(x)) -> x`

**Sign normalization:** `x + (-y) -> x - y`, `x - (-y) -> x + y`
