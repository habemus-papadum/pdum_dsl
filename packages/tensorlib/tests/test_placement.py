"""L3-lite: machine-bound dims, placement alignment, the traffic pass."""

import numpy as np
import pytest
from pdum.tl import Machine, Tensor, mesh, peak_memory, pointwise, pw, red, reduce, traffic
from pdum.tl.autodiff import grad, numeric_grad
from pdum.tl.dialect import run_named
from pdum.tl.layout import Dim
from pdum.tl.lifting import lift_step
from pdum.tl.zoo import megatron_block


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def test_bind_is_metadata_and_survives_the_algebra():
    x = T(np.arange(6.0).reshape(2, 3), ("g", "i")).bind(g="gpu")
    assert x.layout.dim("g").level == "gpu"
    assert x.layout.dim("i").level is None
    y = x.slice(i=(0, 2)).repeat("j", 2)
    assert y.layout.dim("g").level == "gpu"  # binding rides through views
    z = pointwise(pw.mul, x, x)
    assert z.layout.dim("g").level == "gpu"  # and through compute results
    np.testing.assert_allclose(x.to_numpy(), np.arange(6.0).reshape(2, 3))  # values = erasure


def test_machine_bound_dims_are_chartless():
    with pytest.raises(ValueError, match="chartless"):
        T(np.zeros(3), ("x",)).with_charts(x=(0, 1)).bind(x="gpu")
    with pytest.raises(ValueError, match="addresses"):
        Dim("x", 8, 0, 3, labels=("a", "b", "c"), level="gpu")


def test_alignment_reports_placement_mismatch():
    x = T(np.zeros((2, 3)), ("g", "i")).bind(g="gpu")
    y = T(np.zeros((2, 3)), ("g", "i"))
    with pytest.raises(ValueError, match="placement differs"):
        pointwise(pw.add, x, y)
    try:
        pointwise(pw.add, x, y)
    except ValueError as err:
        assert "bind(g='gpu')" in str(err)  # the fix recipe is a collective


def test_megatron_block_matches_numpy_and_its_erasure():
    placed = megatron_block()
    erased = megatron_block(level=None)
    ref = placed.ref(placed.numpy_inputs())
    got_p = run_named(placed.region, placed.inputs, placed.names)[placed.out].to_numpy(order=placed.order)
    got_e = run_named(erased.region, erased.inputs, erased.names)[erased.out].to_numpy(order=erased.order)
    np.testing.assert_allclose(got_p, ref, rtol=1e-9)
    np.testing.assert_allclose(got_p, got_e, rtol=0, atol=0)  # placement never changes meaning


def test_megatron_traffic_is_exactly_two_all_reduces():
    m = megatron_block()
    rep = traffic(m.region, None, mesh(2), names=m.names)
    kinds = [(c.kind, c.level) for c in rep.collectives]
    assert kinds == [("all_reduce", "gpu"), ("all_reduce", "gpu")]
    # each all-reduce moves 2(p-1)/p x (t,d)-local bytes = 1 x 4*6*8 = 192
    assert [c.bytes for c in rep.collectives] == [192, 192]
    assert rep.per_level["gpu"] == 384
    # the erasure communicates nothing
    e = megatron_block(level=None)
    assert traffic(e.region, None, mesh(2), names=e.names).collectives == ()


def test_per_device_peak_is_below_replicated_peak():
    m = megatron_block()
    full = peak_memory(m.region, m.inputs, names=m.names)
    local = peak_memory(m.region, m.inputs, local=True, names=m.names)
    assert local.peak_bytes < full.peak_bytes


def test_merge_of_a_bound_part_is_an_all_gather():
    x = T(np.arange(6.0).reshape(2, 3), ("g", "i")).bind(g="gpu")

    def step(x):
        mi = x.merge(("g", "i"), "mi")
        return mi

    ls = lift_step(step, x=x.layout)
    rep = traffic(ls.region, None, mesh(2), names=ls.names)
    assert [(c.kind, c.bytes) for c in rep.collectives] == [("all_gather", 24)]  # (p-1)/p x 48


def test_free_distribution_costs_nothing():
    def step(w):
        wr = w.repeat("g", (0, 2))
        wb = wr.bind(g="gpu")
        return wb

    ls = lift_step(step, w=T(np.zeros(3), ("i",)).layout)
    rep = traffic(ls.region, None, mesh(2), names=ls.names)
    assert rep.collectives == ()


def test_traffic_refusals_are_loud():
    x = T(np.zeros((2, 3)), ("g", "i")).bind(g="gpu")

    def sliced(x):
        s = x.slice(g=(0, 1))
        return s

    lss = lift_step(sliced, x=x.layout)
    with pytest.raises(NotImplementedError, match="machine-bound"):
        traffic(lss.region, None, mesh(2), names=lss.names)

    def ident(x):
        return x

    lsi = lift_step(ident, x=x.layout)
    with pytest.raises(KeyError, match="no level"):
        traffic(lsi.region, None, Machine(()), names=lsi.names)
    with pytest.raises(ValueError, match="exceeds"):
        traffic(lsi.region, None, mesh(1), names=lsi.names)


def test_alpha_beta_time_estimate():
    m = megatron_block()
    rep = traffic(m.region, None, mesh(2), names=m.names)
    machine = mesh(2, link_bandwidth=1e9, link_latency=1e-6)
    expected = 2 * 1e-6 + 384 / 1e9
    assert rep.time(machine) == pytest.approx(expected)


# ----------------------------------------------------------------------
# placed backward: gradients carry placement; training-step traffic
# ----------------------------------------------------------------------


def _with_loss(m):
    """Extend the block's region with zloss = sum(out², (t, d)) — the scalar
    target, authored by hand on the region face (Builder + names extension)."""
    from pdum.dsl.ir import Builder, Region
    from pdum.dsl.ops import CORE_OPS
    from pdum.tl.dialect import TL_OPS, walk_region

    out = next(n for n in walk_region(m.region) if m.names.get(id(n)) == m.out)
    b = Builder({**CORE_OPS, **TL_OPS})
    zsq = b.emit("tl.pointwise", out, out, f="mul")
    zloss = b.emit("tl.reduce", zsq, f="sum", dims=("t", "d"))
    region = Region(params=m.region.params, body=(b.emit("core.yield", zloss),))
    names = {**m.names, id(zsq): "zsq", id(zloss): "zloss"}
    return region, names


def test_placed_gradients_equal_erased_gradients_bit_exact():
    p, e = megatron_block(), megatron_block(level=None)
    rp, np_p = _with_loss(p)
    re, np_e = _with_loss(e)
    rg_p = grad(rp, "zloss", p.inputs, names=np_p)
    rg_e = grad(re, "zloss", e.inputs, names=np_e)
    ep, ee = run_named(rg_p.region, p.inputs, rg_p.names), run_named(rg_e.region, e.inputs, rg_e.names)
    for v in ("x", "wq", "w2", "b1"):
        order = p.inputs[v].names
        np.testing.assert_allclose(
            ep[rg_p.grads[v]].to_numpy(order=order), ee[rg_e.grads[v]].to_numpy(order=order), rtol=0, atol=0
        )


def test_gradients_carry_their_primals_placement():
    p = megatron_block()
    region, names = _with_loss(p)
    rg = grad(region, "zloss", p.inputs, names=names)
    env = run_named(rg.region, p.inputs, rg.names)
    assert env[rg.grads["wq"]].layout.dim("g").level == "gpu"  # sharded weight, sharded grad
    assert all(d.level is None for d in env[rg.grads["x"]].layout.dims)  # replicated stays replicated


def test_training_step_traffic_counts_backward_collectives():
    p = megatron_block()
    region, names = _with_loss(p)
    fwd_rep = traffic(region, None, mesh(2), names=names)
    rg = grad(region, "zloss", p.inputs, names=names)
    joint_rep = traffic(rg.region, None, mesh(2), names=rg.names)
    assert len(fwd_rep.collectives) == 2  # Megatron's forward pair
    kinds = {c.kind for c in joint_rep.collectives}
    assert kinds == {"all_reduce"}
    # backward adds input-gradient all-reduces: one per broadcast chain
    # (q, k, v, mlp-up) — the reference is UNFUSED, so 4 where Megatron's
    # f/g operators fuse attention's three into one; collective fusion is
    # a recorded later optimization
    assert len(joint_rep.collectives) == 6


def test_data_parallel_gradient_sync_falls_out():
    def step(x, w):
        wr = w.repeat("n", (0, 4))
        wb = wr.bind(n="gpu")
        p = x * wb
        zloss = reduce(red.sum, p, ("n", "i"))
        return zloss

    inputs = {
        "x": T(np.arange(12.0).reshape(4, 3), ("n", "i")).bind(n="gpu"),
        "w": T(np.array([1.0, 2.0, 3.0]), ("i",)),
    }
    ls = lift_step(step, x=inputs["x"].layout, w=inputs["w"].layout)
    rg = grad(ls.region, "zloss", dict(inputs), names=ls.names)
    rep = traffic(rg.region, None, mesh(4), names=rg.names)
    # forward: loss aggregation over the bound batch; backward: THE
    # data-parallel gradient all-reduce for the replicated weight
    assert len(rep.collectives) == 2
    assert all(c.kind == "all_reduce" for c in rep.collectives)
    env = run_named(rg.region, inputs, rg.names)
    np.testing.assert_allclose(env[rg.grads["w"]].to_numpy(), inputs["x"].to_numpy().sum(axis=0))
    assert all(d.level is None for d in env[rg.grads["w"]].layout.dims)  # replicated grad


def test_fd_rebuild_preserves_placement():
    p = megatron_block()
    region, names = _with_loss(p)
    rg = grad(region, "zloss", p.inputs, names=names)
    env = run_named(rg.region, p.inputs, rg.names)
    fd = numeric_grad(region, "zloss", "x", p.inputs, names)  # would misalign without the rebind
    np.testing.assert_allclose(env[rg.grads["x"]].to_numpy(order=("t", "d")), fd, rtol=3e-4, atol=1e-6)


# --- the Region face (the excavation, LEVELS) -------------------------------


def test_region_face_traffic_reads_the_all_reduce_off_the_algebra():
    x = T(np.zeros((4, 8)), ("g", "e")).bind(g="gpu")
    w = T(np.zeros((4, 8)), ("g", "e")).bind(g="gpu")

    def step(x, w):
        p = x * w
        s = reduce(red.sum, p, "g")  # bound g: the all-reduce, read off
        return s

    ls = lift_step(step, x=x.layout, w=w.layout)
    ro = traffic(ls.region, None, mesh(4), names=ls.names)
    assert len(ro.collectives) == 1 and ro.collectives[0].kind == "all_reduce"
    assert ro.collectives[0].var == "s"


def test_region_face_refuses_lattice_surgery_on_bound_dims():
    x = T(np.zeros((4, 8)), ("g", "e")).bind(g="gpu")

    def step(x):
        return x.slice(g=(0, 2))

    ls = lift_step(step, x=x.layout)
    with pytest.raises(NotImplementedError, match="slice on machine-bound dim 'g'"):
        traffic(ls.region, None, mesh(4), names=ls.names)
