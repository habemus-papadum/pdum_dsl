"""The jvp linearization table, verified NUMERICALLY (stage-1 audit).

Line coverage lies about a rule table: the lambdas execute at import, so a
sign error in a derivative ships "covered". This battery pushes every rule
through `jvp` and compares against central finite differences — one test,
the whole dispatch table, checked against calculus rather than executed.
"""

import pytest

import pdum.dsl  # noqa: F401
from pdum.dsl.kernel.api import jit
from pdum.dsl.stdlib.transforms import jvp


def k_mul(a, b):
    return a * b


def k_div(a, b):
    return a / b


def k_div_denominator_only(a, b):
    return 2.5 / b + a * 0.0


def k_sub_neg(a, b):
    return -a - b


def k_pow_const(a, b):
    return a**3.0 + b


def k_sqrt(a, b):
    return sqrt(a * b)  # noqa: F821


def k_exp(a, b):
    return exp(a - b)  # noqa: F821


def k_sin(a, b):
    return sin(a * b)  # noqa: F821


def k_cos(a, b):
    return cos(a + b)  # noqa: F821


def k_abs_pos(a, b):
    return abs(a * b)  # noqa: F821


def k_abs_neg(a, b):
    return abs(a * b - 10.0)  # noqa: F821 — inner value negative at the probe point


def k_min(a, b):
    return min(a * 2.0, b)  # noqa: F821


def k_max(a, b):
    return max(a * 2.0, b)  # noqa: F821


def k_floor_dead(a, b):
    return floor(b) + a * a  # noqa: F821 — floor's tangent is zero (piecewise constant)


def k_tuple_extract(a, b):
    p = (a * b, a + b)
    return p[0] * 2.0 + p[1]


def k_int_cast_kills(a, b):
    return float(int(b)) * a  # d/db = 0 away from integers; d/da = float(int(b))


def k_branch(a, b):
    y = a * b
    if a > b:
        y = a * a
    return y


CASES = [
    k_mul, k_div, k_div_denominator_only, k_sub_neg, k_pow_const, k_sqrt, k_exp,
    k_sin, k_cos, k_abs_pos, k_abs_neg, k_min, k_max, k_floor_dead,
    k_tuple_extract, k_int_cast_kills, k_branch,
]  # fmt: skip

POINT = (1.7, 2.3)  # away from kinks (abs/min/max/floor/branch switch elsewhere)


@pytest.mark.parametrize("fn", CASES, ids=lambda f: f.__name__)
def test_jvp_matches_central_differences(fn):
    k = jit()(fn)
    jf = jvp(k)
    eps = 1e-6
    for arg in (0, 1):
        seed = [0.0, 0.0]
        seed[arg] = 1.0
        primal, tangent = jf(*POINT, *seed)
        hi = list(POINT)
        lo = list(POINT)
        hi[arg] += eps
        lo[arg] -= eps
        fd = (k(*hi) - k(*lo)) / (2 * eps)
        assert primal == k(*POINT)
        assert abs(tangent - fd) < 1e-4, f"d/d{'ab'[arg]}: jvp {tangent} vs FD {fd}"


def test_varying_exponent_refuses():
    @jit()
    def k(a, b):
        return a**b

    with pytest.raises(Exception, match="varying exponent"):
        jvp(k)(2.0, 3.0, 1.0, 0.0)
