# EML Explained

## What is EML?

EML stands for **Exp-Minus-Log**. It is the binary operator:

```
eml(x, y) = exp(x) - ln(y)
```

Odrzywolek (2024) showed that this single operator, combined with the constant 1, is sufficient to express **all elementary mathematical functions** -- arithmetic, powers, roots, exponentials, logarithms, trigonometry, and their inverses.

Every expression can be rewritten as a binary tree where:
- Every **leaf** is either the constant `1` or a variable
- Every **internal node** is an `eml` application

The grammar is: `S -> 1 | x | eml(S, S)`

## Why is This Interesting?

**Uniform representation.** Every node in an EML tree has the same semantics. There are no special cases for different operators -- just `eml` everywhere.

**Uniform differentiation.** The derivative of `eml` has a single rule:

```
d/dx eml(f, g) = exp(f) * f' - g' / g
```

This one rule, applied recursively, differentiates any expression.

**A single computational primitive.** EML trees could be mapped directly to uniform hardware (circuits where every gate computes `eml`) or used as a search space for symbolic regression.

**Caveat: structural equality is not semantic equality.** Two EML trees that look different can compute the same function. Canonicalization requires additional work beyond raw lowering.

## How uninum Implements EML Lowering

The `.to_eml()` method converts any expression to pure EML form via two phases.

### Phase A: Reduce to {1, Var, exp, ln, sub}

The first phase rewrites all operations in terms of just `exp`, `ln`, and subtraction, using these identities:

**Constants:**
- `0 = ln(1)`
- `e = exp(1)`
- `i = exp(ln(-1) / 2)` (since `ln(-1) = i*pi`, so `exp(i*pi/2) = i`)
- `pi = ln(-1) / i`
- Integers by repeated addition of 1

**Arithmetic:**
- `-a = 0 - a`
- `a + b = a - (0 - b)`
- `a * b = exp(ln(a) + ln(b))` = `exp(ln(a) - (0 - ln(b)))`
- `a / b = exp(ln(a) - ln(b))`
- `a ^ b = exp(b * ln(a))`

**Trigonometry** (via Euler's formula `e^(ix) = cos(x) + i*sin(x)`):
- `sin(a) = (exp(ia) - exp(-ia)) / 2i`
- `cos(a) = (exp(ia) + exp(-ia)) / 2`
- `tan(a) = sin(a) / cos(a)`

**Hyperbolic:**
- `sinh(a) = (exp(a) - exp(-a)) / 2`
- `cosh(a) = (exp(a) + exp(-a)) / 2`
- `tanh(a) = sinh(a) / cosh(a)`

**Inverse trig** (via logarithmic forms):
- `asin(a) = -i * ln(ia + sqrt(1 - a^2))`
- `acos(a) = -i * ln(a + i*sqrt(1 - a^2))`
- `atan(a) = (i/2) * ln((1 - ia) / (1 + ia))`

**Other:**
- `sqrt(a) = exp(ln(a) / 2)`
- `|a| = exp(ln(a^2) / 2)` (for positive reals)

### Phase B: Reduce to {1, Var, eml}

A single bottom-up pass converts the three remaining operations to `eml`:

| Operation | EML form | Why |
|-----------|----------|-----|
| `exp(a)` | `eml(a, 1)` | `exp(a) - ln(1) = exp(a) - 0 = exp(a)` |
| `ln(a)` | `eml(1, eml(eml(1, a), 1))` | Derived from `eml(1, y) = e - ln(y)`, solving for `ln` |
| `a - b` | `eml(ln(a), exp(b))` | `exp(ln(a)) - ln(exp(b)) = a - b` |

## Worked Example: `ln(x)`

Phase A leaves `ln(x)` unchanged (it's already in the target set).

Phase B applies the `ln` rule:

```
ln(x)
  = eml(1, eml(eml(1, x), 1))
```

Verification: let `x = 2.0`:
```python
>>> from uninum import var, ln
>>> x = var("x")
>>> lowered = ln(x).to_eml()
>>> print(lowered)
eml(1, eml(eml(1, x), 1))
>>> lowered.evaluate(x=2.0)   # 0.6931471805599453
>>> ln(x).evaluate(x=2.0)     # 0.6931471805599453
```

## Worked Example: `x + y`

Phase A rewrites addition:
```
x + y
  = x - (0 - y)
  = x - (ln(1) - y)
```

Phase B converts `sub`, `ln`, and `exp` to `eml`:
```
x - (ln(1) - y)
  -> eml(ln(x), exp(ln(1) - y))
  -> eml(eml(1, eml(eml(1, x), 1)), eml(eml(1, eml(eml(1, eml(1, eml(eml(1, 1), 1))), 1)) - ..., 1))
```

The full tree is large, but evaluates to the same result as `x + y`.

## Tree Sizes

EML universality comes at the cost of verbosity:

| Expression | EML string length |
|-----------|-------------------|
| `exp(x)` | ~9 characters |
| `x + y` | ~105 characters |
| `sin(x)` | ~1,700 characters |

This is expected. The value of EML is in the **uniform structure**, not human readability. Typical uses operate on the tree programmatically, not by reading the string.

## Limitations (v0.1.0)

- **Tree size grows quickly** for complex expressions. No optimization pass to reduce size yet.
- **Arbitrary float constants** (like `Const(3.14)`) are not fully lowered -- they remain as `Const` nodes in the EML tree. Only integers, `e`, `pi`, and `i` are fully decomposed.
- **Trig lowering produces complex intermediate values** (via Euler's formula), even when the original expression is real-valued. The final result is real, but intermediate nodes involve complex arithmetic.
- **No canonicalization.** Structurally different EML trees may represent the same function. Building a canonical form is a hard problem that would require additional compiler passes.

## References

Odrzywolek, A. (2024). *Exp-minus-log: a single binary operator for all of elementary mathematics.* See `researchpaper.pdf` in the repository root.
