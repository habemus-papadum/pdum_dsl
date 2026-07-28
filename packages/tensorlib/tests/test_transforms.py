"""L1 transformations: requested-gradients DCE + min-cut checkpointing."""

import numpy as np
import pytest
from pdum.tl import Tensor, peak_memory, pointwise, red, reduce, scan
from pdum.tl.autodiff import grad
from pdum.tl.dialect import run_named, walk_region
from pdum.tl.lifting import lift_step
from pdum.tl.markers import exp
from pdum.tl.transforms import checkpoint, dce
from pdum.tl.zoo import gpt2


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _with_loss(m, dims):
    """Extend a zoo region with zloss = sum(out², dims) — the scalar target,
    authored by hand on the region face (Builder + a names extension)."""
    from pdum.dsl.ir import Builder, Region
    from pdum.dsl.ops import CORE_OPS
    from pdum.tl.dialect import TL_OPS

    out = next(n for n in walk_region(m.region) if m.names.get(id(n)) == m.out)
    b = Builder({**CORE_OPS, **TL_OPS})
    zsq = b.emit("tl.pointwise", out, out, f="mul")
    zloss = b.emit("tl.reduce", zsq, f="sum", dims=tuple(dims))
    region = Region(params=m.region.params, body=(b.emit("core.yield", zloss),))
    names = {**m.names, id(zsq): "zsq", id(zloss): "zloss"}
    return region, names


def _nodes(region):
    return sum(1 for _ in walk_region(region))


def test_dce_keeps_exactly_the_requested_slice():
    def step(x):
        a = x * x
        b = x + x  # dead wrt 'ra'
        ra = reduce(red.sum, a, "i")
        rb = reduce(red.sum, b, "i")
        return ra, rb

    x = {"x": T(np.arange(4.0), ("i",))}
    ls = lift_step(step, x=x["x"].layout)
    pruned = dce(ls.region, ("ra",), names=ls.names)
    surviving = {ls.names[id(n)] for n in walk_region(pruned) if id(n) in ls.names}
    assert surviving == {"x", "a", "ra"}
    assert run_named(pruned, x, ls.names)["ra"].item() == run_named(ls.region, x, ls.names)["ra"].item()
    with pytest.raises(KeyError):
        dce(ls.region, ("nope",), names=ls.names)


def test_dce_prunes_unrequested_gradient_work():
    m = gpt2()
    region, names = _with_loss(m, ("t", "v"))
    rg = grad(region, "zloss", m.inputs, names=names)
    keep_x = dce(rg.region, (rg.grads["wte"], "zloss"), names=rg.names)
    assert _nodes(keep_x) < _nodes(rg.region)  # the other weight-grad work is gone
    ex, ej = run_named(keep_x, m.inputs, rg.names), run_named(rg.region, m.inputs, rg.names)
    np.testing.assert_allclose(
        ex[rg.grads["wte"]].to_numpy(order=("v", "d")), ej[rg.grads["wte"]].to_numpy(order=("v", "d")), rtol=1e-12
    )
    assert (
        peak_memory(keep_x, m.inputs, names=rg.names).peak_bytes
        <= peak_memory(rg.region, m.inputs, names=rg.names).peak_bytes
    )


def _chain():
    # x -> sq -> e (exp, big) -> r (reduce, banned) ; loss = r*r
    def step(x):
        sq = x * x
        e = pointwise(exp, sq)
        r = reduce(red.sum, e, "i")
        loss = r * r
        return loss

    inputs = {"x": T(0.1 * np.arange(8.0), ("i",))}
    return lift_step(step, x=inputs["x"].layout), inputs


def test_checkpoint_recomputes_the_cheap_chain():
    ls, inputs = _chain()
    rg = grad(ls.region, "loss", dict(inputs), names=ls.names)
    ck = checkpoint(rg.region, "loss", names=rg.names)
    # pointwise sq/e recompute from the (free-to-keep) input; only the
    # banned reduce output r must be saved — and it is 8 bytes
    assert [v for v, _ in ck.saved] == ["r"]
    assert ck.bytes_after == 8
    assert ck.bytes_before > ck.bytes_after
    assert any(v in ck.recomputed for v in ("sq", "e"))
    ej, ec = run_named(rg.region, inputs, rg.names), run_named(ck.region, inputs, ck.names)
    np.testing.assert_allclose(ec[rg.grads["x"]].to_numpy(), ej[rg.grads["x"]].to_numpy(), rtol=1e-12)


def test_checkpoint_recompute_everything_mode():
    ls, inputs = _chain()
    rg = grad(ls.region, "loss", dict(inputs), names=ls.names)
    ck = checkpoint(rg.region, "loss", ban=(), names=rg.names)
    assert ck.bytes_after == 0  # everything re-derives from the input
    ej, ec = run_named(rg.region, inputs, rg.names), run_named(ck.region, inputs, ck.names)
    np.testing.assert_allclose(ec[rg.grads["x"]].to_numpy(), ej[rg.grads["x"]].to_numpy(), rtol=1e-12)


def test_banned_ops_read_by_backward_are_saved():
    def step(x):
        s = scan(red.sum, x, "i")
        ss = s * s
        loss = reduce(red.sum, ss, "i")
        return loss

    inputs = {"x": T(np.arange(5.0), ("i",))}
    ls = lift_step(step, x=inputs["x"].layout)
    rg = grad(ls.region, "loss", dict(inputs), names=ls.names)
    ck = checkpoint(rg.region, "loss", names=rg.names)
    assert "s" in [v for v, _ in ck.saved]  # scan is banned from recompute
    ej, ec = run_named(rg.region, inputs, rg.names), run_named(ck.region, inputs, ck.names)
    np.testing.assert_allclose(ec[rg.grads["x"]].to_numpy(), ej[rg.grads["x"]].to_numpy(), rtol=1e-12)


def test_checkpoint_gpt2_shrinks_the_boundary_and_the_peak():
    m = gpt2()
    region, names = _with_loss(m, ("t", "v"))
    rg = grad(region, "zloss", m.inputs, names=names)
    ck = checkpoint(rg.region, "zloss", names=rg.names)
    assert ck.bytes_after < ck.bytes_before
    before = peak_memory(rg.region, m.inputs, names=rg.names).peak_bytes
    after = peak_memory(ck.region, m.inputs, names=ck.names).peak_bytes
    assert after <= before
    ej, ec = run_named(rg.region, m.inputs, rg.names), run_named(ck.region, m.inputs, ck.names)
    for v in ("wte", "h.0.attn.wq"):
        np.testing.assert_allclose(
            ec[rg.grads[v]].to_numpy(order=m.inputs[v].names),
            ej[rg.grads[v]].to_numpy(order=m.inputs[v].names),
            rtol=1e-10,
        )


def test_dce_then_checkpoint_compose():
    m = gpt2()
    region, names = _with_loss(m, ("t", "v"))
    rg = grad(region, "zloss", m.inputs, names=names)
    pruned = dce(rg.region, (rg.grads["wte"], "zloss"), names=rg.names)
    ck = checkpoint(pruned, "zloss", names=rg.names)
    ej, ec = run_named(rg.region, m.inputs, rg.names), run_named(ck.region, m.inputs, ck.names)
    np.testing.assert_allclose(
        ec[rg.grads["wte"]].to_numpy(order=("v", "d")), ej[rg.grads["wte"]].to_numpy(order=("v", "d")), rtol=1e-10
    )
    # the composition is the point: prune first, then plan what remains
    assert ck.bytes_before <= checkpoint(rg.region, "zloss", names=rg.names).bytes_before


# --- the Region face (the excavation, LEVELS) -------------------------------


def test_region_dce_keeps_one_output_of_a_multi_output_step():
    """Multi-output step; keep one output: only the kept slice survives (the
    surviving op set matches authoring the pruned computation directly), and
    the param feeding only the dropped output leaves the signature."""

    def step(x, w):
        a = reduce(red.sum, x * x, "i")
        b = reduce(red.sum, x * w, "i")
        return a, b

    x = Tensor.from_numpy(np.zeros(8), ("i",))
    ls = lift_step(step, x=x.layout, w=x.layout)
    rd = dce(ls.region, ("a",), names=ls.names)
    surviving = {ls.names[id(n)] for n in walk_region(rd) if id(n) in ls.names}

    def pruned_step(x):
        a = reduce(red.sum, x * x, "i")
        return a

    ls2 = lift_step(pruned_step, x=x.layout)
    direct = {ls2.names[id(n)] for n in walk_region(ls2.region) if id(n) in ls2.names}
    assert surviving == direct
    assert [ls.names[id(p)] for p in rd.params] == ["x"]  # w fed only b
    # the names map keeps serving the pruned region: analyses run unchanged
    assert peak_memory(rd, names=ls.names).peak_bytes == peak_memory(ls2.region, names=ls2.names).peak_bytes


def test_region_dce_refuses_unknown_keeps():
    def step(x):
        y = x * x
        return y

    x = Tensor.from_numpy(np.zeros(4), ("i",))
    ls = lift_step(step, x=x.layout)
    with pytest.raises(KeyError, match="kept vars not defined"):
        dce(ls.region, ("nope",), names=ls.names)
