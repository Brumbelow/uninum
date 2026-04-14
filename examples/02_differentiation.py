"""Symbolic differentiation with uninum."""

from uninum import var, Const, sin, cos, exp, ln, sqrt

x = var("x")
y = var("y")

# --- Basic derivatives ---

# .diff(x) returns the symbolic derivative with respect to x.
# The result is an unsimplified expression tree, so we chain .simplify()
# for readable output.

print("=== Basic derivatives ===")

# Power rule
f = x ** 2
print(f"f       = {f}")
print(f"f'      = {f.diff(x)}")                    # raw (unsimplified)
print(f"f' simp = {f.diff(x).simplify()}")          # simplified
print()

# Trigonometric
print(f"d/dx sin(x) = {sin(x).diff(x).simplify()}")
print(f"d/dx cos(x) = {cos(x).diff(x).simplify()}")
print()

# Exponential and logarithmic
print(f"d/dx exp(x) = {exp(x).diff(x).simplify()}")
print(f"d/dx ln(x)  = {ln(x).diff(x).simplify()}")
print()

# Square root
print(f"d/dx sqrt(x) = {sqrt(x).diff(x).simplify()}")

# --- Chain rule ---

print("\n=== Chain rule ===")

g = sin(x ** 2)
print(f"g       = {g}")
print(f"g'      = {g.diff(x).simplify()}")

h = exp(sin(x))
print(f"h       = {h}")
print(f"h'      = {h.diff(x).simplify()}")

# --- Product rule ---

print("\n=== Product rule ===")

p = x * sin(x)
print(f"p       = {p}")
print(f"p'      = {p.diff(x).simplify()}")

# --- Multivariate: partial derivatives ---

print("\n=== Partial derivatives ===")

f2 = x * y + sin(x)
print(f"f       = {f2}")
print(f"df/dx   = {f2.diff(x).simplify()}")
print(f"df/dy   = {f2.diff(y).simplify()}")

# .diff() also accepts a variable name as a string
print(f"df/dx   = {f2.diff('x').simplify()}")

# --- Higher-order derivatives ---

print("\n=== Higher-order derivatives ===")

f3 = x ** 3
f3_prime = f3.diff(x).simplify()
f3_double_prime = f3_prime.diff(x).simplify()
print(f"f       = {f3}")
print(f"f'      = {f3_prime}")
print(f"f''     = {f3_double_prime}")

# --- Numerical verification ---

print("\n=== Numerical verification (central differences) ===")


def numerical_derivative(expr, var_name, point, h=1e-7):
    """Approximate df/dx using central differences."""
    kwargs_plus = {var_name: point + h}
    kwargs_minus = {var_name: point - h}
    return (expr.evaluate(**kwargs_plus) - expr.evaluate(**kwargs_minus)) / (2 * h)


f4 = sin(x ** 2) * exp(x)
df4 = f4.diff(x).simplify()

x_val = 1.5
symbolic_val = df4.evaluate(x=x_val)
numerical_val = numerical_derivative(f4, "x", x_val)

print(f"f           = {f4}")
print(f"f'          = {df4}")
print(f"f'(1.5) sym = {symbolic_val:.10f}")
print(f"f'(1.5) num = {numerical_val:.10f}")
print(f"difference  = {abs(symbolic_val - numerical_val):.2e}")
