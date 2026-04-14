"""EML lowering: the universal expression representation.

The EML operator is defined as:

    eml(x, y) = exp(x) - ln(y)

Odrzywolek (2024) showed that this single binary operator, paired with
the constant 1, is sufficient to express ALL elementary mathematical
functions.  Every expression can be lowered to a binary tree where:

    - every leaf is either the constant 1 or a variable
    - every internal node is an eml application

uninum implements this lowering via .to_eml().
"""

from uninum import var, Const, sin, cos, exp, ln, eml
from uninum.expr import BinOp, Var

x = var("x")
y = var("y")

# --- What is eml? ---

print("=== The EML operator ===")
print(f"eml(x, y) = exp(x) - ln(y)")
print(f"eml(1, 1) = exp(1) - ln(1) = e - 0 = {eml(Const(1), Const(1)).evaluate():.6f}")
print(f"eml(0, 1) = exp(0) - ln(1) = 1 - 0 = {eml(Const(0), Const(1)).evaluate():.6f}")
print()

# --- Simple lowerings ---

print("=== Simple lowerings ===")

# exp(x) lowers to eml(x, 1) because exp(x) - ln(1) = exp(x) - 0 = exp(x)
lowered_exp = exp(x).to_eml()
print(f"exp(x)        -> {lowered_exp}")
print(f"  original:  exp(1.5) = {exp(x).evaluate(x=1.5):.10f}")
print(f"  lowered:            = {lowered_exp.evaluate(x=1.5):.10f}")
print()

# ln(x) lowers to a nested eml expression
lowered_ln = ln(x).to_eml()
print(f"ln(x)         -> {lowered_ln}")
print(f"  original:  ln(2.0) = {ln(x).evaluate(x=2.0):.10f}")
print(f"  lowered:           = {lowered_ln.evaluate(x=2.0):.10f}")
print()

# --- Verifying correctness ---


def only_eml(node):
    """Check that a lowered tree contains only Const(1), Var, and eml."""
    if isinstance(node, Const):
        return node.value == 1
    if isinstance(node, Var):
        return True
    if isinstance(node, BinOp) and node.op == "eml":
        return only_eml(node.left) and only_eml(node.right)
    return False


print("=== Verifying pure EML form ===")

exprs = {
    "exp(x)":     exp(x),
    "ln(x)":      ln(x),
    "x + y":      x + y,
    "x * y":      x * y,
    "sin(x)":     sin(x),
    "x ** 2":     x ** 2,
}

test_vals = {"x": 1.5, "y": 2.3}

for name, e in exprs.items():
    lowered = e.to_eml()
    is_pure = only_eml(lowered)
    original_val = e.evaluate(**test_vals)
    lowered_val = lowered.evaluate(**test_vals)

    # Handle complex results from EML evaluation
    if isinstance(lowered_val, complex):
        lowered_val = lowered_val.real
    if isinstance(original_val, complex):
        original_val = original_val.real

    match = abs(original_val - lowered_val) < 1e-8
    print(f"  {name:10s}  pure_eml={is_pure}  values_match={match}")

# --- The two-phase process ---

print("\n=== How EML lowering works ===")
print("""
Phase A: Rewrite to {Const(1), Var, exp, ln, sub}
  - 0 = ln(1)
  - -a = 0 - a
  - a + b = a - (0 - b)
  - a * b = exp(ln(a) + ln(b))
  - a / b = exp(ln(a) - ln(b))
  - a^b = exp(b * ln(a))
  - sin(a) = (exp(ia) - exp(-ia)) / 2i    (Euler's formula)
  - Constants: e = exp(1), i = exp(ln(-1)/2), pi = ln(-1)/i

Phase B: Rewrite to {Const(1), Var, eml}
  - exp(a) = eml(a, 1)         because exp(a) - ln(1) = exp(a)
  - ln(a) = eml(1, eml(eml(1, a), 1))
  - a - b = eml(ln(a), exp(b))
""")

# --- Tree size trade-off ---

print("=== Tree size ===")
print("EML trees are universal but verbose.  Even simple expressions")
print("produce large trees.  The value is in the uniform representation,")
print("not in human readability.\n")

for name, e in [("exp(x)", exp(x)), ("x + y", x + y), ("sin(x)", sin(x))]:
    lowered = e.to_eml()
    tree_str = str(lowered)
    print(f"  {name:10s} -> {len(tree_str):>5d} characters in EML form")
