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
        m = max(y, x)
        return with_respect_to(m, y)  # noqa: F821

    @jit()
    def dright(y, x):
        m = max(y, x)
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
