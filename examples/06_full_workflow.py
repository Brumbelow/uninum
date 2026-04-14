"""Full workflow: build, differentiate, simplify, compile, and lower to EML.

This example puts all of uninum's features together on two concrete
mathematical problems.
"""

from uninum import var, sin, cos, exp, ln, compile_expr

x = var("x")
y = var("y")

# =====================================================================
# Example 1: Gaussian-like function  f(x) = x * exp(-x^2)
# =====================================================================

print("=" * 60)
print("Example 1: f(x) = x * exp(-x^2)")
print("=" * 60)

f = x * exp(-(x ** 2))
print(f"\nf = {f}")

# Evaluate at several points
print("\n--- Evaluation ---")
for val in [0.0, 0.5, 1.0, 1.5, 2.0]:
    print(f"  f({val}) = {f.evaluate(x=val):.8f}")

# Differentiate
df = f.diff(x).simplify()
print(f"\n--- Derivative ---")
print(f"f' = {df}")

print("\n--- Derivative values ---")
# Note: the unsimplified derivative contains 1/x from the power rule,
# so we avoid evaluating at x=0.
for val in [0.5, 1.0, 1.5, 2.0]:
    result = df.evaluate(x=val)
    if isinstance(result, complex):
        result = result.real
    print(f"  f'({val}) = {result:.8f}")

# Compile both
fn_f = compile_expr(f, backend="python")
fn_df = compile_expr(df, backend="python")

print("\n--- Compiled evaluation ---")
r1 = fn_f(x=1.0)
r2 = fn_df(x=1.0)
if isinstance(r1, complex):
    r1 = r1.real
if isinstance(r2, complex):
    r2 = r2.real
print(f"  compiled f(1.0)  = {r1:.8f}")
print(f"  compiled f'(1.0) = {r2:.8f}")

# Lower to EML and verify
lowered = f.to_eml()
print(f"\n--- EML lowering ---")
print(f"  EML tree length: {len(str(lowered))} characters")
for val in [0.5, 1.0, 1.5]:
    orig = f.evaluate(x=val)
    low = lowered.evaluate(x=val)
    if isinstance(low, complex):
        low = low.real
    print(f"  f({val}): original={orig:.8f}  lowered={low:.8f}  match={abs(orig - low) < 1e-8}")


# =====================================================================
# Example 2: Multivariate  g(x, y) = sin(x*y) + ln(x + y)
# =====================================================================

print()
print("=" * 60)
print("Example 2: g(x, y) = sin(x*y) + ln(x + y)")
print("=" * 60)

g = sin(x * y) + ln(x + y)
print(f"\ng = {g}")

# Partial derivatives
dg_dx = g.diff(x).simplify()
dg_dy = g.diff(y).simplify()
print(f"\n--- Partial derivatives ---")
print(f"dg/dx = {dg_dx}")
print(f"dg/dy = {dg_dy}")

# Evaluate at a point
x0, y0 = 1.0, 2.0
print(f"\n--- Values at (x={x0}, y={y0}) ---")
print(f"  g       = {g.evaluate(x=x0, y=y0):.8f}")

dg_dx_val = dg_dx.evaluate(x=x0, y=y0)
dg_dy_val = dg_dy.evaluate(x=x0, y=y0)
if isinstance(dg_dx_val, complex):
    dg_dx_val = dg_dx_val.real
if isinstance(dg_dy_val, complex):
    dg_dy_val = dg_dy_val.real
print(f"  dg/dx   = {dg_dx_val:.8f}")
print(f"  dg/dy   = {dg_dy_val:.8f}")

# Compile all three
fn_g = compile_expr(g, backend="python")
fn_dg_dx = compile_expr(dg_dx, backend="python")
fn_dg_dy = compile_expr(dg_dy, backend="python")

print(f"\n--- Compiled evaluation ---")
r_g = fn_g(x=x0, y=y0)
r_dx = fn_dg_dx(x=x0, y=y0)
r_dy = fn_dg_dy(x=x0, y=y0)
for r in [r_g, r_dx, r_dy]:
    if isinstance(r, complex):
        r = r.real
print(f"  g({x0},{y0})     = {r_g.real if isinstance(r_g, complex) else r_g:.8f}")
print(f"  dg/dx({x0},{y0}) = {r_dx.real if isinstance(r_dx, complex) else r_dx:.8f}")
print(f"  dg/dy({x0},{y0}) = {r_dy.real if isinstance(r_dy, complex) else r_dy:.8f}")

# Vectorized with numpy
print(f"\n--- NumPy vectorized ---")
try:
    import numpy as np

    fn_g_np = compile_expr(g, backend="numpy")
    xs = np.linspace(0.5, 2.0, 4)
    ys = np.full_like(xs, 2.0)
    results = fn_g_np(x=xs, y=ys)
    for xi, ri in zip(xs, results):
        print(f"  g({xi:.2f}, 2.0) = {ri:.8f}")
except ImportError:
    print("  (numpy not installed -- skipping)")
    print("  Install with: pip install uninum[numpy]")
