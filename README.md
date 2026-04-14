# uninum

A symbolic-numeric expression runtime for Python with EML lowering.

Build mathematical expressions as Python objects, then evaluate, differentiate, simplify, compile to fast callables, or lower to a universal representation where every expression becomes a tree of a single binary operator.

Based off of the work and research presented in the paper "All elementary functions from a single operator" - (Odrzywolek, 2026) https://arxiv.org/pdf/2603.21852

## Features

- **Expression building** with natural Python syntax (`+`, `-`, `*`, `/`, `**`) and standard math functions (`sin`, `cos`, `exp`, `ln`, `sqrt`, ...)
- **Symbolic differentiation** with automatic chain rule, product rule, and all standard derivatives
- **Algebraic simplification** via identity elimination, constant folding, and inverse cancellation
- **Compilation** to fast callables with a numpy (vectorized) or pure-Python backend
- **EML lowering** -- rewrite any expression to a tree of a single operator `eml(x, y) = exp(x) - ln(y)` plus the constant 1

## Install

```bash
pip install .                  # core
pip install .[numpy]           # with numpy backend support
pip install .[dev]             # with pytest for running tests
```

Requires Python >= 3.10.

## Quickstart

```python
from uninum import var, sin, ln, compile_expr

x = var("x")
y = var("y")

# Build an expression
expr = (x + y) * sin(x) / ln(y)
print(expr)                              # (x + y) * sin(x) / ln(y)

# Evaluate numerically
print(expr.evaluate(x=1.2, y=3.4))      # 3.503404...

# Differentiate and simplify
df = expr.diff(x).simplify()
print(df)

# Compile to a fast callable
fn = compile_expr(expr, backend="python")
print(fn(x=1.2, y=3.4))

# Lower to pure EML form
lowered = expr.to_eml()
print(lowered.evaluate(x=1.2, y=3.4))   # same result
```

## API Overview

| Category | Name | Description |
|----------|------|-------------|
| **Node types** | `Expr` | Base class for all expression nodes |
| | `Const(value, name=None)` | Numeric constant |
| | `Var(name)` | Variable |
| | `UnaryOp(op, arg)` | Unary function application |
| | `BinOp(op, left, right)` | Binary operation |
| **Constructors** | `var(name)` | Create a variable |
| | `exp`, `ln`, `log` | Exponential and natural log (`log` is alias for `ln`) |
| | `sin`, `cos`, `tan` | Trigonometric functions |
| | `asin`, `acos`, `atan` | Inverse trigonometric functions |
| | `sinh`, `cosh`, `tanh` | Hyperbolic functions |
| | `sqrt` | Square root |
| | `eml(x, y)` | The EML operator: `exp(x) - ln(y)` |
| **Constants** | `e`, `pi`, `I` | Euler's number, pi, imaginary unit |
| **Methods** | `.evaluate(**kwargs)` | Numeric evaluation with variable bindings |
| | `.diff(wrt)` | Symbolic derivative (accepts `Var` or `str`) |
| | `.simplify()` | Algebraic simplification |
| | `.to_eml()` | Lower to pure EML representation |
| **Compilation** | `compile_expr(expr, backend)` | Compile to callable; `"numpy"` or `"python"` |

## EML in 30 Seconds

The EML operator `eml(x, y) = exp(x) - ln(y)` is a single binary function that, combined with the constant 1, can express all elementary mathematics. This was shown by Odrzywolek (2024).

```python
from uninum import var, sin, exp

x = var("x")
f = sin(x) + exp(x)

lowered = f.to_eml()                     # pure eml(S, S) tree with leaves = 1 or x
print(lowered.evaluate(x=1.0))           # matches f.evaluate(x=1.0)
```

The grammar of every lowered expression is simply `S -> 1 | x | eml(S, S)`. See [docs/eml_explained.md](docs/eml_explained.md) for the full theory.

## Examples

| Script | What it covers |
|--------|----------------|
| [01_expressions.py](examples/01_expressions.py) | Building expressions, evaluation, named constants |
| [02_differentiation.py](examples/02_differentiation.py) | Symbolic derivatives, chain rule, numerical verification |
| [03_simplification.py](examples/03_simplification.py) | Simplification rules, capabilities, and limitations |
| [04_compilation.py](examples/04_compilation.py) | Python and numpy backends, vectorized evaluation |
| [05_eml_lowering.py](examples/05_eml_lowering.py) | EML operator, lowering process, correctness verification |
| [06_full_workflow.py](examples/06_full_workflow.py) | End-to-end: build, diff, simplify, compile, lower |

## Documentation

- [API Reference](docs/api_reference.md) -- complete reference for all public symbols
- [EML Explained](docs/eml_explained.md) -- theory and implementation of EML lowering
- [Cookbook](docs/cookbook.md) -- copy-paste recipes for common tasks

## Running Tests

```bash
pip install .[dev]
pytest tests/ -v
```
