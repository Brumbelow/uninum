"""Building and evaluating symbolic expressions with uninum."""

from uninum import var, Const, sin, cos, exp, ln, log, sqrt, eml, e, pi, I

# --- Creating variables ---

x = var("x")
y = var("y")

# --- Building expressions with operators ---

# Arithmetic operators (+, -, *, /, **) work naturally between
# expressions and between expressions and Python numbers.

print("=== Expression building ===")
print(f"x + 2       = {x + 2}")
print(f"x * y       = {x * y}")
print(f"x ** 3      = {x ** 3}")
print(f"2 * x + 1   = {2 * x + 1}")      # Python number on the left works too
print(f"-(x + y)    = {-(x + y)}")

# Compound expressions
expr = (x + y) * sin(x) / ln(y)
print(f"\nexpr = {expr}")

# --- Function constructors ---

# All standard mathematical functions are available:
print("\n=== Functions ===")
print(f"sin(x)      = {sin(x)}")
print(f"cos(x)      = {cos(x)}")
print(f"exp(x)      = {exp(x)}")
print(f"ln(x)       = {ln(x)}")
print(f"sqrt(x)     = {sqrt(x)}")

# log is an alias for ln (natural logarithm)
print(f"log(x)      = {log(x)}")

# The EML operator itself is also a constructor
print(f"eml(x, y)   = {eml(x, y)}")

# --- Named constants ---

print("\n=== Named constants ===")
print(f"e           = {e}  (value: {e.evaluate()})")
print(f"pi          = {pi}  (value: {pi.evaluate()})")
print(f"I           = {I}  (imaginary unit)")

# Constants can be used in expressions like any other node
print(f"e ** (I * pi) = {e ** (I * pi)}")

# --- Evaluation ---

print("\n=== Evaluation ===")

# Call .evaluate() with keyword arguments for each variable
result = expr.evaluate(x=1.2, y=3.4)
print(f"expr.evaluate(x=1.2, y=3.4) = {result:.6f}")

# Simpler examples
print(f"(x + 2).evaluate(x=3)       = {(x + 2).evaluate(x=3)}")
print(f"(x * y).evaluate(x=2, y=5)  = {(x * y).evaluate(x=2, y=5)}")
print(f"sin(x).evaluate(x=0)        = {sin(x).evaluate(x=0)}")
print(f"exp(x).evaluate(x=1)        = {exp(x).evaluate(x=1):.6f}")

# The EML operator: eml(x, y) = exp(x) - ln(y)
print(f"eml(x, y).evaluate(x=1, y=1) = {eml(x, y).evaluate(x=1, y=1):.6f}")

# --- Error handling ---

print("\n=== Error handling ===")
try:
    (x + y).evaluate(x=1)  # missing y
except ValueError as err:
    print(f"Missing variable error: {err}")
