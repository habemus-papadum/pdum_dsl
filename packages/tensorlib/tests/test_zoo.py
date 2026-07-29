"""The model zoo: every entry's region matches its numpy denotation."""

import numpy as np
import pytest

from pdum.tl.autodiff import grad, numeric_grad
from pdum.tl.dialect import run_named, walk_region
from pdum.tl.zoo import (
    fdtd1d_staggered,
    flash_attention,
    gated_attention,
    gpt2,
    heat2d,
    llama_block,
    moe,
    qknorm_attention,
    sliding_attention,
    tiled_matmul,
    unrolled_trainer,
)

ENTRIES = {
    "gpt2": gpt2,
    "llama": llama_block,
    "sliding": sliding_attention,
    "gated": gated_attention,
    "qknorm": qknorm_attention,
    "flash": flash_attention,
    "flash_naive": lambda: flash_attention(naive=True),
    "heat2d": heat2d,
    "fdtd": fdtd1d_staggered,
    "moe": moe,
    "trainer": unrolled_trainer,
    "gemm": tiled_matmul,
}


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_zoo_forward_matches_numpy(name):
    m = ENTRIES[name]()
    vals = run_named(m.region, m.inputs, m.names)
    got = vals[m.out].to_numpy(order=m.order)
    np.testing.assert_allclose(got, m.ref(m.numpy_inputs()), rtol=1e-9, atol=1e-12)


def _with_loss(m):
    """Extend the entry's region with zloss = sum(out^2) — the scalar target
    the gradient and FD harnesses want. Shares every node with m.region, so
    m.names extends to serve the loss region."""
    from pdum.dsl.ir import Builder, Region
    from pdum.dsl.ops import CORE_OPS
    from pdum.tl.dialect import TL_OPS

    out_node = next(nd for nd in walk_region(m.region) if m.names.get(id(nd)) == m.out)
    dims = tuple(d.name for d in out_node.type.layout.dims)
    b = Builder({**CORE_OPS, **TL_OPS})
    zsq = b.emit("tl.pointwise", out_node, out_node, f="mul")
    zloss = b.emit("tl.reduce", zsq, f="sum", dims=dims)
    region = Region(params=m.region.params, body=(b.emit("core.yield", zloss),))
    return region, {**m.names, id(zsq): "zsq", id(zloss): "zloss"}


@pytest.mark.parametrize(
    ("name", "wrt"),
    [
        ("gpt2", ("wte", "h.0.attn.wq")),
        ("llama", ("x",)),
        ("heat2d", ("u0",)),
        ("fdtd", ("E0",)),
        ("moe", ("x", "wr", "w1")),  # routing gradient-free; gates/values by composition
    ],
)
def test_zoo_gradients_match_fd(name, wrt):
    m = ENTRIES[name]()
    region, names = _with_loss(m)
    rg = grad(region, "zloss", m.inputs, names=names)
    vals = run_named(rg.region, m.inputs, rg.names)
    for v in wrt:
        fd = numeric_grad(region, "zloss", v, m.inputs, names)
        got = vals[rg.grads[v]].to_numpy(order=m.inputs[v].names)
        np.testing.assert_allclose(got, fd, rtol=3e-4, atol=1e-6)


def test_flash_equals_naive_forward_and_backward():
    # the online-softmax reducer IS softmax-then-contract — including its
    # DERIVED backward pass (composite-reducer BPTT, no hand rule anywhere)
    fl, nv = flash_attention(), flash_attention(naive=True)
    outs, gs = [], []
    for m in (fl, nv):
        region, names = _with_loss(m)
        rg = grad(region, "zloss", m.inputs, names=names)
        vals = run_named(rg.region, m.inputs, rg.names)
        outs.append(run_named(m.region, m.inputs, m.names)[m.out].to_numpy(order=m.order))
        gs.append({v: vals[rg.grads[v]].to_numpy(order=m.inputs[v].names) for v in ("q", "k", "v")})
    np.testing.assert_allclose(outs[0], outs[1], rtol=1e-9)
    for v in ("q", "k", "v"):
        np.testing.assert_allclose(gs[0][v], gs[1][v], rtol=1e-6, atol=1e-10)


def test_fdtd_gradient_carries_the_staggered_chart():
    m = fdtd1d_staggered()
    region, names = _with_loss(m)
    rg = grad(region, "zloss", m.inputs, names=names)
    vals = run_named(rg.region, m.inputs, rg.names)
    gE = vals[rg.grads["E0"]]
    (xd,) = gE.layout.dims
    assert xd.chart == m.inputs["E0"].layout.dim("x").chart  # integer grid
    gH = vals[rg.grads["H0"]]
    assert gH.layout.dim("x").chart == m.inputs["H0"].layout.dim("x").chart  # half grid


def test_the_unrolled_trainer_is_the_p9_end_to_end_gate():
    """200 §6.7: K sampled decode chunks in ONE region — indexing (§1.9),
    sampling (§1.8 straight-through; objective-FD is blind to the declared
    estimator BY DESIGN, so the gates are the declared facts), KV reuse as
    ordinary dataflow, and checkpointing-with-replay together."""
    from pdum.tl.transforms import checkpoint

    m = unrolled_trainer()
    rg = grad(m.region, m.out, m.inputs, names=m.names)
    vals = run_named(rg.region, m.inputs, rg.names)
    # integer inputs are gradient-free; noise participates through the soft path
    assert rg.grads["ids"] is None and rg.grads["tgt"] is None
    assert rg.grads["gum1"] is not None
    # the temperature genuinely trains (through the straight-through soft path)
    d_tau = vals[rg.grads["tau"]].to_numpy()
    assert np.isfinite(d_tau).all() and float(np.abs(d_tau)) > 0
    # the tied wte accumulates from every chunk (embedding gathers + heads)
    d_wte = vals[rg.grads["wte"]].to_numpy()
    assert np.isfinite(d_wte).all() and (np.abs(d_wte).sum(axis=1) > 0).any()
    # THE REPLAY GATE: checkpointed recompute reproduces every gradient
    # exactly (deterministic replay — argtopk/take included), while the
    # fwd/bwd boundary shrinks
    ck = checkpoint(rg.region, m.out, names=rg.names)
    assert ck.bytes_after < ck.bytes_before
    ec = run_named(ck.region, m.inputs, ck.names)
    for v in ("tau", "wte", "wq", "wk", "wv", "wo", "wpe"):
        np.testing.assert_allclose(ec[rg.grads[v]].to_numpy(), vals[rg.grads[v]].to_numpy(), rtol=1e-10, err_msg=v)


def test_tiled_matmul_counts_the_standard_macs_and_trains():
    """Tiling is layout, not semantics: the tiled region IS the plain
    matmul (forward pinned by the generic zoo test); under fuse_mac the
    count is the standard m·n·k figure; the gradient flows through the
    tile splits/merges and matches FD."""
    from pdum.tl.opcount import ops_count

    m = tiled_matmul()
    c = ops_count(m.region, fuse_mac=True, names=m.names)
    assert c.total["mac"] == 8 * 6 * 4  # m·n·k, exactly
    region, names = _with_loss(m)
    rg = grad(region, "zloss", m.inputs, names=names)
    vals = run_named(rg.region, m.inputs, rg.names)
    for v in ("a", "b"):
        got = vals[rg.grads[v]].to_numpy(order=m.inputs[v].names)
        fd = numeric_grad(region, "zloss", v, m.inputs, names)
        np.testing.assert_allclose(got, fd, rtol=1e-5, atol=1e-8)


# --- the Region face (the excavation, LEVELS) -------------------------------


def test_region_analyses_serve_the_zoo():
    """EVERY entry carries its region + naming-law assignment, and the
    region-face analyses serve the whole corpus. (The incumbent Program
    face is deleted; the old differential collapsed to direct pins.)"""
    from pdum.tl import infer_signatures, ops_count, peak_memory

    for name in sorted(ENTRIES):
        m = ENTRIES[name]()
        assert m.region is not None and m.names is not None, name
        ro = ops_count(m.region, fuse_mac=True, names=m.names)
        assert sum(ro.total.values()) > 0, name
        rm = peak_memory(m.region, m.inputs, names=m.names)
        assert rm.peak_bytes > 0, name
        assert rm.input_bytes == sum(v.to_numpy().size * 8 for v in m.inputs.values()), name
        rs = infer_signatures(m.region, m.inputs, names=m.names)
        assert m.out in rs, name


def _stamped_ones(t):
    from pdum.tl import Tensor

    out = Tensor.from_numpy(np.ones_like(t.to_numpy()), t.names)
    charts = {d.name: d.chart for d in t.layout.dims if d.chart is not None}
    labels = {d.name: d.labels for d in t.layout.dims if d.labels is not None}
    levels = {d.name: d.level for d in t.layout.dims if d.level is not None}
    if charts:
        out = out.with_charts(**charts)
    if labels:
        out = out.with_labels(**labels)
    if levels:
        out = out.bind(**levels)
    return out


# test_region_grad_agrees_with_program_grad: DELETED — the differential's
# incumbent side (Program-face grad) is gone; its value coverage lives in the
# FD pins above and the schedule/chart/replay pins below.


def test_region_grad_fold_schedules_bit_identical():
    """The recompute theorem on the region face: store-all, uniform
    segments, and binomial revolve produce bit-identical fold gradients."""
    m = ENTRIES["fdtd"]()
    seed_val = _stamped_ones(run_named(m.region, m.inputs, m.names)[m.out])
    base = {}
    for kw in ({}, {"fold_segments": 3}, {"fold_slots": 2}):
        rg = grad(m.region, m.out, m.inputs, seed="seed_in", wrt=("E0", "H0"), names=m.names, **kw)
        vals = run_named(rg.region, {**m.inputs, "seed_in": seed_val}, rg.names)
        for v in ("E0", "H0"):
            got = vals[rg.grads[v]].to_numpy(order=("x",))
            base.setdefault(v, got)
            np.testing.assert_array_equal(got, base[v], err_msg=str(kw))


def test_region_grad_carries_the_staggered_charts():
    m = ENTRIES["fdtd"]()
    seed_val = _stamped_ones(run_named(m.region, m.inputs, m.names)[m.out])
    rg = grad(m.region, m.out, m.inputs, seed="seed_in", wrt=("E0", "H0"), names=m.names)
    vals = run_named(rg.region, {**m.inputs, "seed_in": seed_val}, rg.names)
    gE = vals[rg.grads["E0"]]
    gH = vals[rg.grads["H0"]]
    assert gE.layout.dim("x").chart == m.inputs["E0"].layout.dim("x").chart
    assert gH.layout.dim("x").chart == m.inputs["H0"].layout.dim("x").chart


def test_region_checkpoint_shrinks_and_replays():
    """The trainer replay gate on the Region face: the boundary shrinks,
    something genuinely recomputes, and every yielded value — loss and
    both gradients — replays bit-exactly through the recompute clones."""
    from pdum.tl.transforms import checkpoint

    m = ENTRIES["trainer"]()
    rg = grad(m.region, m.out, m.inputs, wrt=("tau", "wte"), names=m.names)
    rc = checkpoint(rg.region, m.out, names=rg.names)
    assert rc.bytes_after < rc.bytes_before
    assert rc.recomputed  # something genuinely recomputes
    before = run_named(rg.region, m.inputs, rg.names)
    after = run_named(rc.region, m.inputs, rc.names)
    for slot in rg.outputs:
        np.testing.assert_array_equal(after[slot].to_numpy(), before[slot].to_numpy(), err_msg=slot)
    # the analyses keep serving the transformed region through rc.names
    from pdum.tl import peak_memory

    assert peak_memory(rc.region, m.inputs, names=rc.names).peak_bytes > 0
