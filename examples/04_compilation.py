"""Compiling expressions to fast callables with uninum."""

from uninum import var, sin, cos, exp, ln, compile  # note: shadows builtin compile

x = var("x")
y = var("y")

expr = (x + y) * sin(x) / ln(y)

# --- Python backend (scalar, always available) ---

print("=== Python backend ===")

fn = compile(expr, backend="python")
result = fn(x=1.2, y=3.4)

# The python backend uses cmath internally, so results may be complex.
# For real-valued inputs where the result is real, extract the real part.
if isinstance(result, complex):
    result = result.real

print(f"expr            = {expr}")
print(f"compiled(1.2, 3.4) = {result:.10f}")
print(f"evaluate(1.2, 3.4) = {expr.evaluate(x=1.2, y=3.4):.10f}")

# --- Compiling a derivative ---

print("\n=== Compiling a derivative ===")

df = expr.diff(x).simplify()
fn_deriv = compile(df, backend="python")
result_d = fn_deriv(x=1.2, y=3.4)
if isinstance(result_d, complex):
    result_d = result_d.real
print(f"expr'           = {df}")
print(f"compiled deriv  = {result_d:.10f}")

# --- NumPy backend (vectorized) ---

print("\n=== NumPy backend ===")

try:
    import numpy as np

    fn_np = compile(expr, backend="numpy")

    # Scalar evaluation
    print(f"scalar: {fn_np(x=1.2, y=3.4):.10f}")

    # Vectorized evaluation: pass numpy arrays for element-wise computation
    xs = np.array([0.5, 1.0, 1.5, 2.0])
    ys = np.array([2.0, 2.5, 3.0, 3.5])
    results = fn_np(x=xs, y=ys)
    print(f"x values:  {xs}")
    print(f"y values:  {ys}")
    print(f"results:   {np.round(results, 6)}")

except ImportError:
    print("(numpy not installed -- skipping numpy backend examples)")
    print("Install with: pip install uninum[numpy]")

# --- DAG sharing ---

print("\n=== DAG sharing ===")
print("When the same sub-expression object appears multiple times,")
print("the compiler evaluates it only once.")

shared = sin(x) + cos(x)
expr2 = shared * shared  # shared is the *same* object in both operands

fn2 = compile(expr2, backend="python")
r = fn2(x=1.0)
if isinstance(r, complex):
    r = r.real
print(f"(sin(x) + cos(x))^2 at x=1: {r:.10f}")

# --- Performance note ---

print("\n=== How compilation works ===")
print("compile() flattens the expression tree into a linear operation list,")
print("then builds a closure that evaluates the list sequentially.")
print("This avoids the overhead of recursive tree-walking on every call.")
