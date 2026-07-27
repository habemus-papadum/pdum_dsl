"""The value-tier derivative engine (200 §S.3 amendment, P8): one forward-
seeding engine over THE one table — with_respect_to as a lowering macro,
value_and_grad as a Derived transform, refusals where the doctrine draws
lines (branches unvetted, the table never guesses)."""

import pytest
from pdum.dsl import jit, reference, value_and_grad
from pdum.dsl.ir import VerifyError


@jit()
def _dist_ddy(y, x):
    d = sqrt(y * y + x * x)  # noqa: F821
    return with_respect_to(d, y)  # noqa: F821


def test_with_respect_to_a_local_value_on_the_oracle():
    assert reference(_dist_ddy)(3.0, 4.0) == pytest.approx(0.6)  # y / sqrt(y² + x²)


def test_with_respect_to_an_independent_param_is_exact_zero():
    @jit()
    def go(y, x):
        v = x * x
        return with_respect_to(v, y)  # noqa: F821

    assert reference(go)(7.0, 3.0) == 0.0


def test_at_kink_first_wins_at_the_value_tier():
    """THE at-kink law through the value tier: a tie at max sends the whole
    derivative LEFT (ge) — the same frozen rows as the marker tier."""

    @jit()
    def dleft(y, x):
        m = maximum(y, x)  # noqa: F821 — numpy names the primitive
        return with_respect_to(m, y)  # noqa: F821

    @jit()
    def dright(y, x):
        m = maximum(y, x)  # noqa: F821
        return with_respect_to(m, x)  # noqa: F821

    assert reference(dleft)(2.0, 2.0) == 1.0  # the tie goes left...
    assert reference(dright)(2.0, 2.0) == 0.0  # ...and ONLY left (partition)


def test_value_and_grad_is_the_function_space_operator():
    def circle(cy, cx, r):
        @jit()
        def go(y, x):
            d = sqrt((y - cy) * (y - cy) + (x - cx) * (x - cx))  # noqa: F821
            return d - r

        return go

    g = value_and_grad(circle(0.0, 0.0, 5.0), wrt=("y", "x"))
    v, (dy, dx) = reference(g)(3.0, 4.0)
    assert v == pytest.approx(0.0) and (dy, dx) == (pytest.approx(0.6), pytest.approx(0.8))


def test_value_and_grad_chains_through_the_battery_kinks():
    """smoothstep inlines to clamp = min(max(...)) — the derivative chains
    through DSL-written batteries and the kink rows without new machinery."""

    @jit()
    def go(x):
        return smoothstep(0.0, 1.0, x)  # noqa: F821

    v, (dx,) = reference(value_and_grad(go, wrt=("x",)))(0.5)
    assert v == pytest.approx(0.5) and dx == pytest.approx(1.5)  # 6t(1-t) at t=1/2


def test_value_and_grad_identity_keys_on_the_wrt_set():
    @jit()
    def go(y, x):
        return y * x

    a, b = value_and_grad(go, wrt=("y",)), value_and_grad(go, wrt=("y", "x"))
    assert a.fp != b.fp  # a different wrt set is a different artifact


def test_branch_derivatives_refuse_as_unvetted():
    @jit()
    def go(y, x):
        v = 1.0 if y > x else 0.0
        return with_respect_to(v, y)  # noqa: F821

    with pytest.raises(VerifyError, match="not yet vetted"):
        reference(go)(2.0, 1.0)


def test_a_primitive_off_the_table_refuses():
    @jit()
    def go(y, x):
        v = y**2.0
        return with_respect_to(v, y)  # noqa: F821

    with pytest.raises(VerifyError, match="no entry in the derivative table"):
        reference(go)(2.0, 1.0)


def test_value_and_grad_refuses_an_unknown_wrt_name():
    @jit()
    def go(y, x):
        return y * x

    with pytest.raises(VerifyError, match="wrt name 'z' is not a parameter"):
        reference(value_and_grad(go, wrt=("z",)))(1.0, 2.0)


def test_bare_marker_calls_and_operators_share_one_op():
    """The clarity-review catch: value_op sends core-owned arithmetic to
    core.* — a bare ``mul(y, x)`` and ``y * x`` are the SAME op, so the
    tangent engine differentiates both (one vocabulary, one table)."""

    @jit()
    def go(y, x):
        s = mul(y, x)  # noqa: F821 — the marker call IS core.mul
        return with_respect_to(s, y)  # noqa: F821

    assert reference(go)(2.0, 3.0) == 3.0


# --- the numpy-authority amendment (200 §S.2) --------------------------------


def test_markers_are_ordinary_math_on_host_scalars():
    """The marker OBJECT is the identity at every tier — and on plain
    numbers it just computes (np.sqrt on a scalar IS a float)."""
    from pdum.dsl.markers import maximum, sqrt

    assert sqrt(4.0) == 2.0 and isinstance(sqrt(4.0), float)
    assert maximum(2.0, 3.0) == 3.0
    with pytest.raises(TypeError, match="spelled\\s+pointwise"):
        sqrt([1.0, 2.0])  # anything non-scalar refuses toward the tensor tier


def test_the_oracle_is_ieee_non_trapping():
    """Floats compute on numpy scalars: 0/0 flows as nan and sqrt(-1) is
    nan — like a device, never a Python exception (210 amendment)."""
    import numpy as np

    @jit()
    def go(y, x):
        return sqrt(y) + x / (x - 1.0)  # noqa: F821

    with np.errstate(invalid="ignore", divide="ignore"):
        assert np.isnan(reference(go)(-1.0, 0.5))  # sqrt(-1) -> nan
        assert np.isinf(reference(go)(4.0, 1.0))  # 1/0 -> inf


def test_tanh_and_log_arrived_with_the_vocabulary():
    """The drift is closed: the value language speaks every table row."""

    @jit()
    def go(y, x):
        v = tanh(y) * log(x)  # noqa: F821
        return with_respect_to(v, y)  # noqa: F821 — (1 - tanh²y)·log x

    import math

    assert reference(go)(0.5, 2.0) == pytest.approx((1 - math.tanh(0.5) ** 2) * math.log(2.0))


def test_abs_row_ties_to_plus_one_and_floor_is_gradient_free():
    @jit()
    def dabs(y, x):
        return with_respect_to(abs(y), y)  # noqa: F821

    @jit()
    def dfloor(y, x):
        return with_respect_to(floor(y), y)  # noqa: F821

    assert reference(dabs)(0.0, 0.0) == 1.0  # the tie at 0 goes +1 (first-wins)
    assert reference(dabs)(-2.0, 0.0) == -1.0
    assert reference(dfloor)(2.5, 0.0) == 0.0  # gradient-free BY DECLARATION
