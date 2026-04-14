# Changelog

## [0.2.0] - 2026-04-14

### Added
- Structural equality (`__eq__`) and hashing (`__hash__`) for all expression nodes
- Expressions are now usable in sets and dicts
- Like-term collection: `x + x -> 2*x`, `2*x + 3*x -> 5*x`
- Power combination: `x*x -> x^2`, `x^a * x^b -> x^(a+b)`, `(x^a)^b -> x^(a*b)`
- Negation distribution: `-(x+y) -> -x-y`, `-(x-y) -> y-x`
- Trig parity rules: `sin(-x) -> -sin(x)`, `cos(-x) -> cos(x)`, `tan(-x) -> -tan(x)`
- LaTeX output via `.to_latex()` method and `to_latex()` function
- `__version__` attribute on the package

### Fixed
- `compile_expr` now handles EML-lowered expressions with the Python backend

### Changed
- `compile_expr()` is now the primary function name; `compile()` is a deprecated alias

## [0.1.0] - 2026-04-13

### Added
- Expression building with natural Python syntax (`+`, `-`, `*`, `/`, `**`)
- Standard math functions: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `sqrt`
- Named constants: `e`, `pi`, `I`
- Numeric evaluation via `.evaluate(**kwargs)`
- Symbolic differentiation via `.diff(wrt)`
- Algebraic simplification via `.simplify()`
- EML lowering via `.to_eml()`
- Compilation to fast callables via `compile_expr()` (numpy and python backends)
