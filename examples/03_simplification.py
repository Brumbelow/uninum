"""Algebraic simplification with uninum."""

from uninum import var, Const, exp, ln, sin

x = var("x")
y = var("y")

# uninum's simplifier performs bottom-up rewriting of local algebraic
# identities.  It runs multiple passes until the expression stops changing
# (up to 10 passes by default).

# --- Constant folding ---

print("=== Constant folding ===")
print(f"Const(2) + Const(3) -> {(Const(2) + Const(3)).simplify()}")
print(f"Const(6) / Const(2) -> {(Const(6) / Const(2)).simplify()}")
print(f"Const(2) ** Const(3) -> {(Const(2) ** Const(3)).simplify()}")

# --- Additive identities ---

print("\n=== Additive identities ===")
print(f"0 + x       -> {(Const(0) + x).simplify()}")
print(f"x + 0       -> {(x + Const(0)).simplify()}")
print(f"x - 0       -> {(x - Const(0)).simplify()}")
print(f"0 - x       -> {(Const(0) - x).simplify()}")
print(f"x - x       -> {(x - x).simplify()}")

# --- Multiplicative identities ---

print("\n=== Multiplicative identities ===")
print(f"1 * x       -> {(Const(1) * x).simplify()}")
print(f"x * 1       -> {(x * Const(1)).simplify()}")
print(f"0 * x       -> {(Const(0) * x).simplify()}")
print(f"x * 0       -> {(x * Const(0)).simplify()}")
print(f"-1 * x      -> {(Const(-1) * x).simplify()}")

# --- Power identities ---

print("\n=== Power identities ===")
print(f"x ** 0      -> {(x ** Const(0)).simplify()}")
print(f"x ** 1      -> {(x ** Const(1)).simplify()}")
print(f"1 ** x      -> {(Const(1) ** x).simplify()}")

# --- Division identities ---

print("\n=== Division identities ===")
print(f"x / 1       -> {(x / Const(1)).simplify()}")
print(f"0 / x       -> {(Const(0) / x).simplify()}")
print(f"x / x       -> {(x / x).simplify()}")

# --- Double negation ---

print("\n=== Double negation ===")
print(f"--x         -> {(-(-x)).simplify()}")

# --- Inverse function cancellation ---

print("\n=== Exp-ln cancellation ===")
print(f"exp(ln(x))  -> {exp(ln(x)).simplify()}")
print(f"ln(exp(x))  -> {ln(exp(x)).simplify()}")

# --- Sign normalization ---

print("\n=== Sign normalization ===")
print(f"x + (-y)    -> {(x + (-y)).simplify()}")
print(f"x - (-y)    -> {(x - (-y)).simplify()}")

# --- Compound simplification ---

print("\n=== Compound simplification ===")
print(f"(x * 1 + 0) ** 1 -> {((x * Const(1) + Const(0)) ** Const(1)).simplify()}")
print(f"(x + 0) * 1 / 1  -> {((x + Const(0)) * Const(1) / Const(1)).simplify()}")

# --- Simplification after differentiation ---

print("\n=== Diff then simplify ===")
f = x ** 2 + sin(x)
df = f.diff(x)
print(f"f       = {f}")
print(f"f' raw  = {df}")
print(f"f' simp = {df.simplify()}")

# --- Limitations ---

print("\n=== Limitations ===")
print("The simplifier handles local algebraic identities.")
print("It does NOT perform:")
print("  - Algebraic expansion or factoring")
print("  - Trigonometric identities (sin^2 + cos^2 = 1)")
print("  - Canonicalization of equivalent expressions")
print("  - Symbolic polynomial arithmetic")
