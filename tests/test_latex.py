"""Tests for LaTeX rendering."""

from uninum import var, Const, exp, ln, sin, cos, tan, asin, sqrt, eml, e, pi, I


class TestLatexConst:
    def test_int(self):
        assert Const(3).to_latex() == "3"

    def test_float(self):
        assert Const(3.14).to_latex() == "3.14"

    def test_negative_int(self):
        assert Const(-5).to_latex() == "-5"

    def test_named_e(self):
        assert e.to_latex() == r"\mathrm{e}"

    def test_named_pi(self):
        assert pi.to_latex() == r"\pi"

    def test_named_i(self):
        assert I.to_latex() == r"\mathrm{i}"


class TestLatexVar:
    def test_simple(self):
        assert var("x").to_latex() == "x"

    def test_greek(self):
        assert var("alpha").to_latex() == r"\alpha"

    def test_greek_upper(self):
        assert var("Omega").to_latex() == r"\Omega"

    def test_non_greek(self):
        assert var("foo").to_latex() == "foo"


class TestLatexUnary:
    def test_neg_var(self):
        x = var("x")
        assert (-x).to_latex() == "-x"

    def test_neg_binop(self):
        x, y = var("x"), var("y")
        assert (-(x + y)).to_latex() == r"-\left(x + y\right)"

    def test_sin(self):
        x = var("x")
        assert sin(x).to_latex() == r"\sin\left(x\right)"

    def test_cos(self):
        x = var("x")
        assert cos(x).to_latex() == r"\cos\left(x\right)"

    def test_tan(self):
        x = var("x")
        assert tan(x).to_latex() == r"\tan\left(x\right)"

    def test_exp(self):
        x = var("x")
        assert exp(x).to_latex() == r"\exp\left(x\right)"

    def test_ln(self):
        x = var("x")
        assert ln(x).to_latex() == r"\ln\left(x\right)"

    def test_arcsin(self):
        x = var("x")
        assert asin(x).to_latex() == r"\arcsin\left(x\right)"

    def test_sqrt(self):
        x = var("x")
        assert sqrt(x).to_latex() == r"\sqrt{x}"


class TestLatexBinary:
    def test_add(self):
        x, y = var("x"), var("y")
        assert (x + y).to_latex() == "x + y"

    def test_sub(self):
        x, y = var("x"), var("y")
        assert (x - y).to_latex() == "x - y"

    def test_mul(self):
        x, y = var("x"), var("y")
        assert (x * y).to_latex() == r"x \cdot y"

    def test_div(self):
        x, y = var("x"), var("y")
        assert (x / y).to_latex() == r"\frac{x}{y}"

    def test_pow_simple(self):
        x = var("x")
        assert (x ** Const(2)).to_latex() == "x^{2}"

    def test_pow_expr_exponent(self):
        x, y = var("x"), var("y")
        assert (x ** (y + Const(1))).to_latex() == "x^{y + 1}"

    def test_eml(self):
        x, y = var("x"), var("y")
        assert eml(x, y).to_latex() == r"\operatorname{eml}\left(x, y\right)"


class TestLatexPrecedence:
    def test_add_in_mul(self):
        x, y, z = var("x"), var("y"), var("z")
        assert ((x + y) * z).to_latex() == r"\left(x + y\right) \cdot z"

    def test_mul_in_add(self):
        x, y, z = var("x"), var("y"), var("z")
        assert (x * y + z).to_latex() == r"x \cdot y + z"

    def test_no_parens_in_frac(self):
        x, y, z = var("x"), var("y"), var("z")
        assert ((x + y) / z).to_latex() == r"\frac{x + y}{z}"

    def test_pow_base_binop(self):
        x, y = var("x"), var("y")
        assert ((x + y) ** Const(2)).to_latex() == r"\left(x + y\right)^{2}"

    def test_sub_right_assoc(self):
        x, y, z = var("x"), var("y"), var("z")
        # x - (y - z) needs parens on the right
        from uninum.expr import BinOp
        expr = BinOp("sub", x, BinOp("sub", y, z))
        assert expr.to_latex() == r"x - \left(y - z\right)"


class TestLatexCompound:
    def test_derivative_display(self):
        x = var("x")
        expr = sin(x) ** Const(2)
        assert expr.to_latex() == r"\sin\left(x\right)^{2}"

    def test_fraction_with_functions(self):
        x = var("x")
        expr = sin(x) / cos(x)
        assert expr.to_latex() == r"\frac{\sin\left(x\right)}{\cos\left(x\right)}"

    def test_greek_in_expression(self):
        theta = var("theta")
        expr = sin(theta) + cos(theta)
        assert expr.to_latex() == r"\sin\left(\theta\right) + \cos\left(\theta\right)"
