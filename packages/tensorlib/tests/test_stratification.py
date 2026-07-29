"""The tier battery (290, owner-ratified 2026-07-28).

Which syntax lives where, PINNED: the one stratification table
(families → tiers, dialect.py), the seam gate (check_tier — wrong-tier
ops refuse in the rulebook voice with the offending source line), the
control-flow amendment (values may branch, effects may not), the
distinct unknown-op voice, and tier preservation through transforms.
"""

import numpy as np
import pytest

from pdum.dsl.ir import Builder, VerifyError
from pdum.dsl.ops import CORE_OPS
from pdum.tl import Tensor, red, reduce
from pdum.tl.autodiff import grad
from pdum.tl.dialect import TL_OPS, check_tier, run_named, tier_admits
from pdum.tl.kernel import _compile, f32, thread_idx
from pdum.tl.lifting import lift_step
from pdum.tl.transforms import checkpoint


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def test_the_ratified_table_rows():
    """The §5 table, spot-pinned row by row — a table edit is a conscious act."""
    # effects are kernel privileges; layout/compute are tensor-tier vocabulary
    assert tier_admits("tl.store", "compute") and not tier_admits("tl.store", "tensor")
    assert tier_admits("tl.read", "compute") and not tier_admits("tl.read", "unit")
    assert tier_admits("tl.flip", "tensor") and not tier_admits("tl.flip", "compute")
    assert tier_admits("tl.reduce", "step") and not tier_admits("tl.reduce", "compute")
    # abi is a launch channel; coord is a thread fact; iota is also ambient
    assert tier_admits("abi.slot", "fragment") and not tier_admits("abi.slot", "unit")
    assert tier_admits("tl.coord", "compute") and not tier_admits("tl.coord", "tensor")
    assert tier_admits("tl.iota", "tensor") and tier_admits("tl.iota", "vertex")
    # graphics: sample is fragment-only; select is the vertex pulled read
    assert tier_admits("tl.sample", "fragment") and not tier_admits("tl.sample", "vertex")
    assert tier_admits("tl.select", "vertex") and not tier_admits("tl.select", "fragment")
    # machinery emissions stay in the compute definition (ledgered, 290 §4.1)
    assert tier_admits("tl.split", "compute") and tier_admits("tl.merge", "compute")
    # constants are every tier's vocabulary; the open marker families admit as SCALAR
    assert tier_admits("tl.const", "compute") and tier_admits("tl.const", "vertex")
    assert tier_admits("pw.anything", "step") and tier_admits("math.tanh", "unit")
    # dsl program ops are quoted data, never tl author vocabulary; unregistered admits nowhere
    assert not tier_admits("core.for", "compute") and not tier_admits("core.if", "tensor")
    assert not tier_admits("toy.unregistered", "tensor")


def test_layout_smuggle_refuses_in_the_rulebook_voice():
    """The 290 §1 probe, inverted into law: flip in a kernel body refuses at
    the seam, quoting the offending line — no more lowering-and-running what
    no device will ever admit."""

    def k(img, out):
        y, x = thread_idx("y", "x")
        ghost = img.flip("x")
        out[y, x] = f32(y) + 0.0 * ghost[y, x]

    img, out = T(np.zeros((2, 3)), ("y", "x")), T(np.zeros((2, 3)), ("y", "x"))
    with pytest.raises(ValueError, match=r"tl\.flip is a host citizen here.*stage it on the parameter"):
        _compile(k, (img, out))


def test_reduce_smuggle_refuses():
    def k(img, out):
        y, x = thread_idx("y", "x")
        s = reduce(red.sum, img, ("x",))
        s2 = s.repeat("x", (0, 3))
        out[y, x] = f32(y) + 0.0 * s2[y, x]

    img, out = T(np.zeros((2, 3)), ("y", "x")), T(np.zeros((2, 3)), ("y", "x"))
    with pytest.raises(ValueError, match="is a host citizen here"):
        _compile(k, (img, out))


def test_statement_if_store_free_joins_by_where():
    """211 §1.2, owner-amended: values may branch. The store-free `if`
    lowers to a where-join and agrees with numpy."""

    def k(img, out):
        y, x = thread_idx("y", "x")
        v = f32(y) + f32(x)
        if v > 3.0:
            v = v * 2.0
        out[y, x] = v

    img, out = T(np.zeros((3, 4)), ("y", "x")), T(np.zeros((3, 4)), ("y", "x"))
    _compile(k, (img, out)).launch((img, out))
    want = np.add.outer(np.arange(3.0), np.arange(4.0))
    np.testing.assert_allclose(out.to_numpy(order=("y", "x")), np.where(want > 3.0, want * 2.0, want))


def test_statement_if_with_store_refuses():
    """...effects may not."""

    def k(img, out):
        y, x = thread_idx("y", "x")
        if f32(y) > 1.0:
            out[y, x] = 2.0
        else:
            out[y, x] = 1.0

    img, out = T(np.zeros((2, 3)), ("y", "x")), T(np.zeros((2, 3)), ("y", "x"))
    with pytest.raises(ValueError, match="values may branch, effects may not"):
        _compile(k, (img, out))


def test_dead_statement_if_refuses():
    def k(img, out):
        y, x = thread_idx("y", "x")
        if f32(y) > 1.0:
            tmp = f32(y)  # noqa: F841 — born in one suite: it dies with it
        out[y, x] = f32(y)

    img, out = T(np.zeros((2, 3)), ("y", "x")), T(np.zeros((2, 3)), ("y", "x"))
    with pytest.raises(ValueError, match="binds nothing that survives"):
        _compile(k, (img, out))


def test_expression_if_and_boolop_in_compute():
    """The vertex spelling, promoted to all kernel bodies (211 §1.2)."""

    def k(img, out):
        y, x = thread_idx("y", "x")
        v = f32(y) + f32(x)
        m = 1.0 if (v > 1.0 and v < 4.0) else 0.0
        out[y, x] = v * m

    img, out = T(np.zeros((3, 4)), ("y", "x")), T(np.zeros((3, 4)), ("y", "x"))
    _compile(k, (img, out)).launch((img, out))
    want = np.add.outer(np.arange(3.0), np.arange(4.0))
    np.testing.assert_allclose(out.to_numpy(order=("y", "x")), want * ((want > 1.0) & (want < 4.0)))


def test_wrong_tier_and_unknown_op_are_distinct_voices():
    """Two doors, two refusals: an op that exists nowhere keeps the
    registry's frozen voice; an op in the wrong tier gets the rulebook's."""
    with pytest.raises(VerifyError, match="unknown op 'tl.astype'"):
        Builder({**CORE_OPS, **TL_OPS}).emit("tl.astype")
    ls = _scalar_loss()
    with pytest.raises(ValueError, match=r"tl\.(token|store) is a host citizen here — the tensor tier"):
        check_tier(_with_effect(ls.region), "tensor")


def _scalar_loss():
    def f(x, w):
        return reduce(red.sum, x * w, ("t", "d"))

    return lift_step(f, x=T(np.zeros((2, 3)), ("t", "d")).layout, w=T(np.zeros((2, 3)), ("t", "d")).layout)


def _with_effect(region):
    """A hand-built region smuggling an EFFECT op past capture — check_tier
    is the one gate for captured AND hand-authored regions alike."""
    from pdum.dsl.ir import Region
    from pdum.tl.dialect import ABI_OPS

    b = Builder({**CORE_OPS, **TL_OPS, **ABI_OPS})
    tok = b.emit("tl.token")
    store = b.emit("tl.store", tok, region.params[0], region.params[0])
    yielded = b.emit("core.yield", store)
    return Region(params=region.params, body=(yielded,))


def test_transforms_preserve_tier():
    """dce (at the assemblage seam), grad, and checkpoint outputs stay
    within their source tier — the invariant the gate makes checkable."""
    ls = _scalar_loss()
    check_tier(ls.region, "step")  # the capture seam already enforced this; the pin
    X, W = T(np.arange(6.0).reshape(2, 3), ("t", "d")), T(np.ones((2, 3)), ("t", "d"))
    rg = grad(ls.region, ls.outputs[0], {"x": X, "w": W}, names=ls.names)
    check_tier(rg.region, "tensor")
    got = run_named(rg.region, {"x": X, "w": W}, rg.names)[rg.grads["x"]]
    np.testing.assert_allclose(got.to_numpy(order=("t", "d")), np.ones((2, 3)))
    cp = checkpoint(ls.region, ls.outputs[0], names=ls.names)
    check_tier(cp.region, "tensor")
