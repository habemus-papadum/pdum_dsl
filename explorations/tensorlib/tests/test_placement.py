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
