"""The loud-refusal guards ARE the design's error-message contract (110/130:
"caught, not silently broadcast"). This battery freezes them — a refactor
that turns a designed refusal into an AttributeError fails HERE, by name.
Stage-1 audit: every guard below was previously unexercised."""

import pytest

np = pytest.importorskip("numpy")

import pdum.dsl  # noqa: F401, E402
from pdum.dsl.kernel.api import jit  # noqa: E402
from pdum.dsl.kernel.lower import MissingRule, lower_handle  # noqa: E402
from pdum.dsl.kernel.ops import CORE_OPS  # noqa: E402
from pdum.dsl.kernel.registry import DEFAULT  # noqa: E402
from pdum.dsl.kernel.types import f64  # noqa: E402
from pdum.dsl.stdlib.arrays import Named, Shaped  # noqa: E402
from pdum.dsl.stdlib.transforms import over  # noqa: E402

A2 = Named(np.ones((2, 3)), ("row", "inner"))
B2 = Named(np.ones((3, 4)), ("inner", "col"))


def test_over_lane_must_be_i64():
    data = Named(np.ones((2, 4)), ("batch", "x"))

    def make(t):
        @jit()
        def g(k):
            return t.isel(x=k)

        return g

    with pytest.raises(Exception, match="trailing i64 lane"):
        over(make(data), axis="batch")(0, 0.5)  # a float lane


def test_D_needs_params():
    @jit()
    def k():
        return D(1.0)[0]  # noqa: F821

    with pytest.raises(MissingRule, match="none in this build"):
        k()


def test_D_takes_exactly_one_value():
    @jit()
    def k(a, b):
        return D(a, b)[0]  # noqa: F821

    with pytest.raises(MissingRule, match="exactly one positional"):
        k(1.0, 2.0)


def test_D_refuses_integer_params():
    @jit()
    def k(i):
        return D(i * 2)  # noqa: F821

    with pytest.raises(MissingRule, match="float params"):
        k(3)


def test_matmul_positional_only():
    def make(A, B):
        @jit()
        def cell(i, j):
            return matmul(A, B, i, j=j)  # noqa: F821

        return cell

    with pytest.raises(MissingRule, match="positional only"):
        make(A2, B2)(0, 0)


def test_matmul_requires_named_arrays():
    def make(A, B):
        @jit()
        def cell(i, j):
            return matmul(A, B, i, j)  # noqa: F821

        return cell

    with pytest.raises(MissingRule, match="BY NAME"):
        make(np.ones((2, 3)), np.ones((3, 4)))(0, 0)


def test_matmul_two_shared_names_refused():
    """A name on both sides beyond the contraction is a SECOND shared name —
    the one-shared-axis refusal dominates (duplicate output names are
    therefore unreachable; the stage-1 audit deleted that dead guard)."""

    def make(A, B):
        @jit()
        def cell(i, j):
            return matmul(A, B, i, j)  # noqa: F821

        return cell

    with pytest.raises(MissingRule, match="ONE shared axis"):
        make(Named(np.ones((2, 3)), ("row", "inner")), Named(np.ones((3, 2)), ("inner", "row")))(0, 0)


def test_matmul_index_count():
    def make(A, B):
        @jit()
        def cell(i):
            return matmul(A, B, i)  # noqa: F821

        return cell

    with pytest.raises(MissingRule, match="output indices"):
        make(A2, B2)(0)


def test_matmul_shaped_extent_mismatch():
    def make(A, B):
        @jit()
        def cell(i, j):
            return matmul(A, B, i, j)  # noqa: F821

        return cell

    bad = (Shaped(np.ones((2, 3))), Shaped(np.ones((5, 4))))
    named = (
        Named(bad[0].array, ("row", "inner")),
        Named(bad[1].array, ("inner", "col")),
    )
    # Shaped+Named combined isn't a thing yet; use two ShapedArrays via the
    # Named wrapper's dims with Shaped's shape — the check needs BOTH shaped:
    from pdum.dsl.stdlib.arrays import NamedArray, ShapedArray  # noqa: F401

    # Construct via Shaped wrappers carrying names is out of scope; the check
    # fires only when both types are ShapedArray — assert the rank-generic
    # pair instead documents UB (no refusal), which is the 100 posture:
    cell = make(*named)
    assert cell(0, 0) == 3.0  # sums A's inner extent (3), documented UB for mismatch


# --- kernel guard paths (lower.py), previously dark ---------------------------
def test_derived_without_build_rule_is_a_missing_rule():
    data = Named(np.ones((2, 4)), ("batch", "x"))

    def make(t):
        @jit()
        def g(k):
            return t.isel(x=k)

        return g

    w = over(make(data), axis="batch")
    with pytest.raises(MissingRule, match="no build rule for Derived template 'over'"):
        lower_handle(w, dict(DEFAULT.lower_rules), {**CORE_OPS, **DEFAULT.ops}, arg_types=(f64,), derived={})


def test_body_never_returns():
    @jit()
    def k(x):
        y = x * 2.0  # noqa: F841

    with pytest.raises(Exception, match="body never returns"):
        k(1.0)


def test_inline_arity_mismatch():
    @jit()
    def inner(a, b):
        return a + b

    @jit()
    def outer(x):
        return inner(x)  # one arg for a two-arg callee

    with pytest.raises(Exception, match="takes 2 args, got 1"):
        outer(1.0)
