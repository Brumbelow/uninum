# Cookbook

Copy-paste recipes for common tasks with uninum.

## Gradient of a Multivariate Function

```python
from uninum import var, sin, exp

x = var("x")
y = var("y")

f = sin(x * y) + exp(x)
grad = [f.diff(x).simplify(), f.diff(y).simplify()]

print(f"df/dx = {grad[0]}")
print(f"df/dy = {grad[1]}")

# Evaluate at a point
point = {"x": 1.0, "y": 2.0}
grad_val = [g.evaluate(**point) for g in grad]
print(f"gradient at {point}: {grad_val}")
```

## Second Derivative

```python
from uninum import var

x = var("x")
f = x ** 3

f_prime = f.diff(x).simplify()
f_double_prime = f_prime.diff(x).simplify()

print(f"f   = {f}")
print(f"f'  = {f_prime}")
print(f"f'' = {f_double_prime}")
```

## Compile and Evaluate Over a NumPy Grid

```python
import numpy as np
from uninum import var, sin, compile

x = var("x")
y = var("y")
f = sin(x) * y ** 2

fn = compile(f, backend="numpy")

xs = np.linspace(0, 2 * np.pi, 100)
ys = np.linspace(0, 3, 100)
X, Y = np.meshgrid(xs, ys)

Z = fn(x=X, y=Y)       # 100x100 array
print(f"Z shape: {Z.shape}, min: {Z.min():.4f}, max: {Z.max():.4f}")
```

## Verify a Derivative Numerically

```python
from uninum import var, sin, exp

x = var("x")
f = sin(x ** 2) * exp(x)
df = f.diff(x).simplify()

def numerical_deriv(expr, var_name, x0, h=1e-7):
    return (expr.evaluate(**{var_name: x0 + h})
          - expr.evaluate(**{var_name: x0 - h})) / (2 * h)

x0 = 1.5
symbolic = df.evaluate(x=x0)
numerical = numerical_deriv(f, "x", x0)
print(f"symbolic:  {symbolic:.12f}")
print(f"numerical: {numerical:.12f}")
print(f"error:     {abs(symbolic - numerical):.2e}")
```

## Check EML Lowering Correctness

```python
from uninum import var, sin, exp, ln
from uninum.expr import Const, Var, BinOp

def only_eml(node):
    """Return True if tree contains only Const(1), Var, and eml."""
    if isinstance(node, Const):
        return node.value == 1
    if isinstance(node, Var):
        return True
    if isinstance(node, BinOp) and node.op == "eml":
        return only_eml(node.left) and only_eml(node.right)
    return False

x = var("x")
f = sin(x) + exp(x)
lowered = f.to_eml()

assert only_eml(lowered), "Tree is not pure EML"

# Verify numerical agreement at several points
for val in [0.5, 1.0, 1.5, 2.0]:
    orig = f.evaluate(x=val)
    low = lowered.evaluate(x=val)
    if isinstance(low, complex):
        low = low.real
    assert abs(orig - low) < 1e-8, f"Mismatch at x={val}"

print("EML lowering verified.")
```

## Build Expressions Dynamically

```python
from functools import reduce
from uninum import var, Const

x = var("x")

# Sum: x + x^2 + x^3 + ... + x^10
terms = [x ** k for k in range(1, 11)]
polynomial = reduce(lambda a, b: a + b, terms)
print(f"polynomial = {polynomial}")
print(f"p(2.0) = {polynomial.evaluate(x=2.0)}")

# Product: x * (x+1) * (x+2) * ... * (x+5)
factors = [x + k for k in range(6)]
product = reduce(lambda a, b: a * b, factors)
print(f"product(1.0) = {product.evaluate(x=1.0)}")  # 1*2*3*4*5*6 = 720
```

## Complex Numbers

```python
from uninum import var, exp, sin, I, pi

x = var("x")

# Euler's formula: e^(ix) = cos(x) + i*sin(x)
euler = exp(I * x)
result = euler.evaluate(x=3.14159)
print(f"exp(i*pi) = {result}")  # approximately -1 + 0i

# sin and cos of complex arguments
z = var("z")
print(f"sin(1+2i) = {sin(z).evaluate(z=1+2j)}")
```

## Taylor Series Approximation

Build an N-term Taylor expansion of a function around a point using repeated differentiation.

```python
from uninum import var, Const, sin
from math import factorial

x = var("x")
f = sin(x)
a = 0.0       # expand around x = a
N = 7         # number of terms

taylor_terms = []
current = f
for n in range(N):
    coeff = current.evaluate(x=a) / factorial(n)
    if abs(coeff) > 1e-15:
        taylor_terms.append(Const(coeff) * (x - a) ** n)
    current = current.diff(x).simplify()

taylor = taylor_terms[0]
for t in taylor_terms[1:]:
    taylor = taylor + t

# Compare: sin(0.5) vs Taylor approximation
print(f"sin(0.5)    = {f.evaluate(x=0.5):.12f}")
print(f"taylor(0.5) = {taylor.evaluate(x=0.5):.12f}")
```
