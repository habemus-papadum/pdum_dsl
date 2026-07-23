"""L3-lite: machine-bound dims, placement alignment, the traffic pass."""

import numpy as np
import pytest
from tensorlib import Machine, Tensor, mesh, peak_memory, pointwise, pw, traffic
from tensorlib.ir import Instr, Program, run
from tensorlib.layout import Dim
from tensorlib.zoo import megatron_block


def I(var, op, operands=(), **params):  # noqa: E743
    return Instr(var, op, tuple(operands), params)


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
    got_p = run(placed.program, placed.inputs)[placed.out].to_numpy(order=placed.order)
    got_e = run(erased.program, erased.inputs)[erased.out].to_numpy(order=erased.order)
    np.testing.assert_allclose(got_p, ref, rtol=1e-9)
    np.testing.assert_allclose(got_p, got_e, rtol=0, atol=0)  # placement never changes meaning


def test_megatron_traffic_is_exactly_two_all_reduces():
    m = megatron_block()
    rep = traffic(m.program, m.inputs, mesh(2))
    kinds = [(c.kind, c.level) for c in rep.collectives]
    assert kinds == [("all_reduce", "gpu"), ("all_reduce", "gpu")]
    # each all-reduce moves 2(p-1)/p x (t,d)-local bytes = 1 x 4*6*8 = 192
    assert [c.bytes for c in rep.collectives] == [192, 192]
    assert rep.per_level["gpu"] == 384
    # the erasure communicates nothing
    e = megatron_block(level=None)
    assert traffic(e.program, e.inputs, mesh(2)).collectives == ()


def test_per_device_peak_is_below_replicated_peak():
    m = megatron_block()
    full = peak_memory(m.program, m.inputs)
    local = peak_memory(m.program, m.inputs, local=True)
    assert local.peak_bytes < full.peak_bytes


def test_merge_of_a_bound_part_is_an_all_gather():
    prog = Program(
        (
            I("x", "input"),
            I("m", "merge", ["x"], parts=("g", "i"), name="mi"),
        )
    )
    x = T(np.arange(6.0).reshape(2, 3), ("g", "i")).bind(g="gpu")
    rep = traffic(prog, {"x": x}, mesh(2))
    assert [(c.kind, c.bytes) for c in rep.collectives] == [("all_gather", 24)]  # (p-1)/p x 48


def test_free_distribution_costs_nothing():
    prog = Program(
        (
            I("w", "input"),
            I("wr", "repeat", ["w"], name="g", extent=(0, 2)),
            I("wb", "bind", ["wr"], levels={"g": "gpu"}),
        )
    )
    rep = traffic(prog, {"w": T(np.zeros(3), ("i",))}, mesh(2))
    assert rep.collectives == ()


def test_traffic_refusals_are_loud():
    x = T(np.zeros((2, 3)), ("g", "i")).bind(g="gpu")
    sliced = Program((I("x", "input"), I("s", "slice", ["x"], ranges={"g": (0, 1)})))
    with pytest.raises(NotImplementedError, match="machine-bound"):
        traffic(sliced, {"x": x}, mesh(2))
    ident = Program((I("x", "input"),))
    with pytest.raises(KeyError, match="no level"):
        traffic(ident, {"x": x}, Machine(()))
    with pytest.raises(ValueError, match="exceeds"):
        traffic(ident, {"x": x}, mesh(1))


def test_alpha_beta_time_estimate():
    m = megatron_block()
    rep = traffic(m.program, m.inputs, mesh(2))
    machine = mesh(2, link_bandwidth=1e9, link_latency=1e-6)
    expected = 2 * 1e-6 + 384 / 1e9
    assert rep.time(machine) == pytest.approx(expected)


# ----------------------------------------------------------------------
# placed backward: gradients carry placement; training-step traffic
# ----------------------------------------------------------------------


def _with_loss(m):
    from tensorlib.ir import Program

    return Program(
        m.program.instrs
        + (
            I("zsq", "pointwise", (m.out, m.out), f="mul"),
            I("zloss", "reduce", ("zsq",), f="sum", dims=("t", "d")),
        )
    )


def test_placed_gradients_equal_erased_gradients_bit_exact():
    from tensorlib.autodiff import grad

    p, e = megatron_block(), megatron_block(level=None)
    jp_p, g_p = grad(_with_loss(p), "zloss", p.inputs)
    jp_e, g_e = grad(_with_loss(e), "zloss", e.inputs)
    ep, ee = run(jp_p, p.inputs), run(jp_e, e.inputs)
    for v in ("x", "wq", "w2", "b1"):
        order = p.inputs[v].names
        np.testing.assert_allclose(ep[g_p[v]].to_numpy(order=order), ee[g_e[v]].to_numpy(order=order), rtol=0, atol=0)


def test_gradients_carry_their_primals_placement():
    from tensorlib.autodiff import grad

    p = megatron_block()
    jp, g = grad(_with_loss(p), "zloss", p.inputs)
    env = run(jp, p.inputs)
    assert env[g["wq"]].layout.dim("g").level == "gpu"  # sharded weight, sharded grad
    assert all(d.level is None for d in env[g["x"]].layout.dims)  # replicated stays replicated


def test_training_step_traffic_counts_backward_collectives():
    from tensorlib import mesh, traffic
    from tensorlib.autodiff import grad

    p = megatron_block()
    prog = _with_loss(p)
    fwd_rep = traffic(prog, p.inputs, mesh(2))
    jp, _ = grad(prog, "zloss", p.inputs)
    joint_rep = traffic(jp, p.inputs, mesh(2))
    assert len(fwd_rep.collectives) == 2  # Megatron's forward pair
    kinds = {c.kind for c in joint_rep.collectives}
    assert kinds == {"all_reduce"}
    # backward adds input-gradient all-reduces: one per broadcast chain
    # (q, k, v, mlp-up) — the reference is UNFUSED, so 4 where Megatron's
    # f/g operators fuse attention's three into one; collective fusion is
    # a recorded later optimization
    assert len(joint_rep.collectives) == 6


def test_data_parallel_gradient_sync_falls_out():
    from tensorlib import mesh, traffic
    from tensorlib.autodiff import grad

    prog = Program(
        (
            I("x", "input"),
            I("w", "input"),
            I("wr", "repeat", ["w"], name="n", extent=(0, 4)),
            I("wb", "bind", ["wr"], levels={"n": "gpu"}),
            I("p", "pointwise", ["x", "wb"], f="mul"),
            I("zloss", "reduce", ["p"], f="sum", dims=("n", "i")),
        )
    )
    inputs = {
        "x": T(np.arange(12.0).reshape(4, 3), ("n", "i")).bind(n="gpu"),
        "w": T(np.array([1.0, 2.0, 3.0]), ("i",)),
    }
    jp, g = grad(prog, "zloss", inputs)
    rep = traffic(jp, inputs, mesh(4))
    # forward: loss aggregation over the bound batch; backward: THE
    # data-parallel gradient all-reduce for the replicated weight
    assert len(rep.collectives) == 2
    assert all(c.kind == "all_reduce" for c in rep.collectives)
    env = run(jp, inputs)
    np.testing.assert_allclose(env[g["w"]].to_numpy(), inputs["x"].to_numpy().sum(axis=0))
    assert all(d.level is None for d in env[g["w"]].layout.dims)  # replicated grad


def test_fd_rebuild_preserves_placement():
    from tensorlib.autodiff import grad, numeric_grad

    p = megatron_block()
    prog = _with_loss(p)
    jp, g = grad(prog, "zloss", p.inputs)
    env = run(jp, p.inputs)
    fd = numeric_grad(prog, "zloss", "x", p.inputs)  # would misalign without the rebind
    np.testing.assert_allclose(env[g["x"]].to_numpy(order=("t", "d")), fd, rtol=3e-4, atol=1e-6)
