"""The indexing family (200 §1.9): take/scatter_add — the adjoint pair.

`take` is a computation, not a view; indices are the VALUE door (250 §8):
integer-carrier, unitless data, bounds refused at RUN time — the data
complement of the coordinate law's in-bounds-by-construction. Structural
facts refuse at BUILD time. Duplicates sum through the adjoint (the
embedding gradient); indices are gradient-free.
"""

import numpy as np
import pytest
from pdum.tl import Tensor, mesh, ops_count, peak_memory, scatter_add, take, traffic
from pdum.tl.autodiff import grad, numeric_grad
from pdum.tl.ir import Instr, Program, infer, run
from pdum.tl.lifting import lift_step


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr), names)


def I(var, op, operands=(), **params):  # noqa: E743
    return Instr(var, op, tuple(operands), params)


TABLE = T(np.arange(12.0).reshape(4, 3), ("v", "d"))  # 4 rows of width 3
IDS = T(np.array([2, 0, 2]), ("t",))  # row 2 twice: duplicates are the point


# ----------------------------------------------------------------------
# take — forward
# ----------------------------------------------------------------------


def test_take_gathers_rows_in_place():
    """out[t, d] = table[ids[t], d]: the taken dim is replaced IN PLACE by
    the index tensor's dims; a fresh, plainly-laid-out tensor."""
    out = take(TABLE, IDS, dim="v")
    assert out.names == ("t", "d")
    want = TABLE.to_numpy()[[2, 0, 2]]
    np.testing.assert_array_equal(out.to_numpy(), want)


def test_take_is_a_computation_not_a_view():
    """Mutating the table afterward must not change the taken result —
    take materializes; zero-cost data-dependent views do not exist."""
    table = T(np.arange(4.0), ("v",))
    out = take(table, T(np.array([1, 1]), ("t",)), dim="v")
    np.frombuffer(table.buffer.data, dtype=np.float64)[:] = 0.0
    np.testing.assert_array_equal(out.to_numpy(), [1.0, 1.0])


def test_take_with_multi_dim_indices():
    ids = T(np.array([[0, 3], [1, 2]]), ("b", "t"))
    out = take(TABLE, ids, dim="v")
    assert out.names == ("b", "t", "d")
    np.testing.assert_array_equal(out.to_numpy(), TABLE.to_numpy()[[[0, 3], [1, 2]]])


def test_take_with_rank0_index_drops_the_dim():
    """A rank-0 index tensor is the data-dependent select: the dim drops,
    tensor-land is never left (the no-promotion law's value-door face)."""
    i = Tensor.from_numpy(np.asarray(np.int64(2)), ())
    out = take(TABLE, i, dim="v")
    assert out.names == ("d",)
    np.testing.assert_array_equal(out.to_numpy(), TABLE.to_numpy()[2])


def test_take_replaces_mid_position_in_place():
    cube = T(np.arange(24.0).reshape(2, 4, 3), ("a", "v", "d"))
    out = take(cube, IDS, dim="v")
    assert out.names == ("a", "t", "d")
    np.testing.assert_array_equal(out.to_numpy(), cube.to_numpy()[:, [2, 0, 2], :])


def test_take_frames_ride_and_the_consumed_frame_disappears():
    """Surviving dims keep chart/labels verbatim; the taken dim's frame is
    consumed — its lattice went data."""
    table = T(np.arange(12.0).reshape(4, 3), ("v", "d")).with_labels(d=("r", "g", "b"))
    out = take(table, IDS, dim="v")
    assert out.layout.dim("d").labels == ("r", "g", "b")
    assert "v" not in out.names


def test_take_reorders_by_sorted_indices():
    """Any differentiable reordering is take by indices (200 §1.9): the
    self-name case — idx over the SAME name as the taken dim — is the
    permutation door."""
    x = T(np.array([3.0, 1.0, 2.0]), ("t",))
    order = T(np.array([1, 2, 0]), ("t",))
    out = take(x, order, dim="t")
    np.testing.assert_array_equal(out.to_numpy(), [1.0, 2.0, 3.0])


# ----------------------------------------------------------------------
# take — refusals
# ----------------------------------------------------------------------


def test_take_refuses_out_of_range_loudly():
    with pytest.raises(IndexError, match=r"out of range.*\[0, 4\).*refuses out-of-range indices loudly"):
        take(TABLE, T(np.array([0, 4]), ("t",)), dim="v")
    with pytest.raises(IndexError, match=r"min -1"):
        take(TABLE, T(np.array([-1, 0]), ("t",)), dim="v")


def test_take_refuses_float_indices():
    """Indices are the VALUE door at integer carrier — a float never rounds
    itself into an address (no magic; exactness doctrine)."""
    with pytest.raises(TypeError, match="integer-carrier"):
        take(TABLE, T(np.array([0.0, 1.0]), ("t",)), dim="v")


def test_take_refuses_united_indices():
    idx = T(np.array([0, 1]), ("t",))
    from pdum.tl import u

    with pytest.raises(TypeError, match="not a measurement"):
        take(TABLE, idx.with_value_units(u.um), dim="v")


def test_take_refuses_colliding_index_dims():
    with pytest.raises(ValueError, match="collide.*rename is the adapter"):
        take(TABLE, T(np.array([0, 1]), ("d",)), dim="v")


def test_take_refuses_unknown_dim():
    with pytest.raises(KeyError, match="no dimension named 'w'"):
        take(TABLE, IDS, dim="w")


# ----------------------------------------------------------------------
# scatter_add — forward
# ----------------------------------------------------------------------


def test_scatter_add_sums_duplicates():
    """Duplicates SUM — deterministic by addition, the declared law."""
    vals = T(np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]]), ("t", "d"))
    out = scatter_add(vals, IDS, dim="v", extent=4)
    assert out.names == ("v", "d")
    want = np.zeros((4, 3))
    want[2] = [101.0, 202.0, 303.0]  # rows 0 and 2 of vals both land at v=2
    want[0] = [10.0, 20.0, 30.0]
    np.testing.assert_array_equal(out.to_numpy(), want)


def test_scatter_add_declares_its_domain():
    """The output frame is DECLARED (dim + extent) — never max(idx)+1; a
    (start, stop) pair is repeat's convention."""
    vals = T(np.array([1.0, 2.0]), ("t",))
    out = scatter_add(vals, T(np.array([5, 7]), ("t",)), dim="v", extent=(5, 8))
    d = out.layout.dim("v")
    assert (d.start, d.stop) == (5, 8)
    np.testing.assert_array_equal(out.to_numpy(), [1.0, 0.0, 2.0])


def test_scatter_add_riding_dims_ride():
    vals = T(np.arange(6.0).reshape(3, 2), ("t", "d"))
    idx = T(np.array([1, 0, 1]), ("t",))
    out = scatter_add(vals, idx, dim="s", extent=2)
    assert out.names == ("s", "d")
    np.testing.assert_array_equal(out.to_numpy(), [[2.0, 3.0], [4.0, 6.0]])


def test_scatter_add_refusals():
    vals = T(np.array([1.0, 2.0, 3.0]), ("t",))
    with pytest.raises(IndexError, match="refuses out-of-range"):
        scatter_add(vals, T(np.array([0, 1, 9]), ("t",)), dim="v", extent=4)
    with pytest.raises(ValueError, match="not values dims"):
        scatter_add(vals, T(np.array([0, 1]), ("q",)), dim="v", extent=4)
    with pytest.raises(ValueError, match="disagree on consumed dim"):
        scatter_add(vals, T(np.array([0, 1]), ("t",)), dim="v", extent=4)
    with pytest.raises(ValueError, match="collides with a surviving"):
        vals2 = T(np.arange(6.0).reshape(3, 2), ("t", "d"))
        scatter_add(vals2, T(np.array([0, 1, 0]), ("t",)), dim="d", extent=4)
    with pytest.raises(TypeError, match="bool values have no sum"):
        scatter_add(T(np.array([True, False]), ("t",)), T(np.array([0, 1]), ("t",)), dim="v", extent=2)


# ----------------------------------------------------------------------
# the Program tier: run / infer / opcount / memory / traffic
# ----------------------------------------------------------------------


def _prog():
    return Program(
        (
            I("table", "input"),
            I("ids", "input"),
            I("tok", "take", ["table", "ids"], dim="v"),
            I("y", "reduce", ["tok"], f="sum", dims=("t", "d")),
        )
    )


def test_take_runs_and_infers_as_a_program_op():
    prog = _prog()
    env = run(prog, {"table": TABLE, "ids": IDS})
    np.testing.assert_array_equal(env["tok"].to_numpy(), TABLE.to_numpy()[[2, 0, 2]])
    shadows = infer(prog, {"table": TABLE, "ids": IDS})
    assert tuple(d.name for d in shadows["tok"].dims) == ("t", "d")
    assert shadows["tok"].dim("t").size == 3


def test_take_and_scatter_cost_entries():
    """take: one read+write per OUTPUT element, its own bucket (random
    access is not a sequential copy — the cost model prices the difference);
    scatter_add: movement + accumulate per INPUT element."""
    prog = _prog()
    c = ops_count(prog, {"table": TABLE, "ids": IDS})
    assert c.per_var["tok"]["take"] == 9  # 3 tokens x width 3
    sc = Program(
        (
            I("vals", "input"),
            I("ids", "input"),
            I("g", "scatter_add", ["vals", "ids"], dim="v", extent=(0, 4)),
        )
    )
    vals = T(np.ones((3, 3)), ("t", "d"))
    cs = ops_count(sc, {"vals": vals, "ids": IDS})
    assert cs.per_var["g"]["scatter"] == 9 and cs.per_var["g"]["add"] == 9


def test_take_allocates_its_output():
    """A real node, never a free view (200 §1.9): the gather output owns
    numel x 8 bytes in the peak-memory model."""
    prog = _prog()
    r = peak_memory(prog, {"table": TABLE, "ids": IDS})
    assert r.alloc_bytes["tok"] == 9 * 8


def test_take_along_a_bound_dim_refuses_toward_the_fix():
    prog = _prog()
    bound = TABLE.bind(v="gpu")
    with pytest.raises(NotImplementedError, match="all-to-all.*all-gather the table"):
        traffic(prog, {"table": bound, "ids": IDS}, mesh(4))
    # a bound RIDING dim is a sharded batch: the take itself rides for free
    # (the final reduce over bound t still all-reduces — that is reduce's law)
    r = traffic(prog, {"table": TABLE, "ids": IDS.bind(t="gpu")}, mesh(4))
    assert not any(c.var == "tok" for c in r.collectives)


def test_scatter_add_over_a_bound_consumed_dim_refuses():
    sc = Program(
        (
            I("vals", "input"),
            I("ids", "input"),
            I("g", "scatter_add", ["vals", "ids"], dim="v", extent=(0, 4)),
        )
    )
    vals = T(np.ones(3), ("t",)).bind(t="gpu")
    with pytest.raises(NotImplementedError, match="partial sums.*all-reduce"):
        traffic(sc, {"vals": vals, "ids": IDS.bind(t="gpu")}, mesh(4))
    with pytest.raises(NotImplementedError, match="colocate"):
        traffic(sc, {"vals": vals, "ids": IDS}, mesh(4))


# ----------------------------------------------------------------------
# the step tier: take lowers through the one engine
# ----------------------------------------------------------------------


def test_take_lowers_in_a_step_body():
    def embed(table, ids):
        return take(table, ids, dim="v")

    ls = lift_step(embed, table=TABLE.layout, ids=IDS.layout)
    assert any(i.op == "take" for i in ls.program.instrs)
    env = run(ls.program, {"table": TABLE, "ids": IDS})
    np.testing.assert_array_equal(env[ls.outputs[0]].to_numpy(), TABLE.to_numpy()[[2, 0, 2]])


def test_scatter_add_lowers_in_a_step_body():
    def route(vals, ids):
        return scatter_add(vals, ids, dim="v", extent=4)

    vals = T(np.ones((3, 3)), ("t", "d"))
    ls = lift_step(route, vals=vals.layout, ids=IDS.layout)
    env = run(ls.program, {"vals": vals, "ids": IDS})
    want = np.zeros((4, 3))
    want[2], want[0] = 2.0, 1.0
    np.testing.assert_array_equal(env[ls.outputs[0]].to_numpy(), want)


# ----------------------------------------------------------------------
# the adjoint pair
# ----------------------------------------------------------------------


def test_take_gradient_is_the_embedding_gradient():
    """d_table counts occurrences through scatter_add: a token appearing
    twice accumulates BOTH contributions; indices are gradient-free."""
    prog = _prog()
    joint, grads = grad(prog, "y", {"table": TABLE, "ids": IDS})
    assert grads["ids"] is None  # d_idx = None (200 §1.9)
    env = run(joint, {"table": TABLE, "ids": IDS})
    g = env[grads["table"]].to_numpy(order=("v", "d"))
    want = np.zeros((4, 3))
    want[2], want[0] = 2.0, 1.0  # row 2 taken twice, row 0 once
    np.testing.assert_array_equal(g, want)
    # and the same fact by finite differences
    fd = numeric_grad(prog, "y", "table", {"table": TABLE, "ids": IDS})
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)


def test_scatter_add_gradient_is_take():
    """The self-dual pair: d_values gathers the cotangent back at the same
    indices — checked against finite differences."""
    vals = T(np.arange(6.0).reshape(3, 2), ("t", "d"))
    prog = Program(
        (
            I("vals", "input"),
            I("ids", "input"),
            I("s", "scatter_add", ["vals", "ids"], dim="v", extent=(0, 4)),
            I("sq", "pointwise", ["s", "s"], f="mul"),
            I("y", "reduce", ["sq"], f="sum", dims=("v", "d")),
        )
    )
    inputs = {"vals": vals, "ids": IDS}
    joint, grads = grad(prog, "y", inputs)
    assert grads["ids"] is None
    env = run(joint, inputs)
    g = env[grads["vals"]].to_numpy(order=("t", "d"))
    fd = numeric_grad(prog, "y", "vals", inputs)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


def test_units_ride_through_take():
    from pdum.tl import infer_signatures, u

    prog = _prog()
    sigs = infer_signatures(prog, {"table": TABLE.with_value_units(u.um), "ids": IDS})
    assert sigs["tok"].unit == u.um


# ----------------------------------------------------------------------
# the index producers: argtopk / argsort (gradient-free)
# ----------------------------------------------------------------------


def test_argtopk_descends_and_ties_go_first():
    """Descending; ties FIRST-WINS (the partition law's stable choice): a
    duplicated maximum yields the lower lattice position first."""
    from pdum.tl import argtopk

    x = T(np.array([1.0, 5.0, 3.0, 5.0]), ("t",))
    top = argtopk(x, dim="t", k=3, k_name="r")
    assert top.names == ("r",)
    assert top.layout.dim("r").size == 3
    np.testing.assert_array_equal(top.to_numpy(), [1, 3, 2])  # first 5.0 wins


def test_argmax_is_argtopk_k1():
    from pdum.tl import argtopk

    x = T(np.array([[1.0, 9.0, 2.0], [7.0, 0.0, 3.0]]), ("b", "e"))
    am = argtopk(x, dim="e", k=1, k_name="m")
    assert am.names == ("b", "m")
    np.testing.assert_array_equal(am.to_numpy(), [[1], [0]])


def test_argsort_is_ascending_stable_and_composes_with_take():
    """Any differentiable reordering is take by sorted indices."""
    from pdum.tl import argsort

    x = T(np.array([3.0, 1.0, 2.0, 1.0]), ("t",))
    order = argsort(x, dim="t")
    np.testing.assert_array_equal(order.to_numpy(), [1, 3, 2, 0])  # stable at the tie
    sorted_x = take(x, order, dim="t")
    np.testing.assert_array_equal(sorted_x.to_numpy(), [1.0, 1.0, 2.0, 3.0])


def test_producers_read_lattice_positions_not_zero_based_offsets():
    from pdum.tl import argsort

    x = T(np.array([3.0, 1.0, 2.0]), ("t",)).shift(t=5)
    order = argsort(x, dim="t")
    np.testing.assert_array_equal(order.to_numpy(), [6, 7, 5])
    np.testing.assert_array_equal(take(x, order, dim="t").to_numpy(), [1.0, 2.0, 3.0])


def test_producer_refusals():
    from pdum.tl import argtopk

    x = T(np.arange(6.0).reshape(2, 3), ("b", "e"))
    with pytest.raises(ValueError, match=r"k=4 outside \[1, 3\]"):
        argtopk(x, dim="e", k=4, k_name="r")
    with pytest.raises(ValueError, match="collides with an existing dim"):
        argtopk(x, dim="e", k=2, k_name="b")


def test_topk_gradient_is_correct_by_composition():
    """The factoring pays: grad of sum(take(x, argtopk(x))) flows through
    TAKE alone — ones at the winning positions, zero elsewhere — matching
    finite differences with no adjoint rule for the producer."""
    x = T(np.array([4.0, 9.0, 1.0, 7.0]), ("t",))
    prog = Program(
        (
            I("x", "input"),
            I("top", "argtopk", ["x"], dim="t", k=2, k_name="r"),
            I("vals", "take", ["x", "top"], dim="t"),
            I("y", "reduce", ["vals"], f="sum", dims=("r",)),
        )
    )
    joint, grads = grad(prog, "y", {"x": x})
    env = run(joint, {"x": x})
    g = env[grads["x"]].to_numpy(order=("t",))
    np.testing.assert_array_equal(g, [0.0, 1.0, 0.0, 1.0])
    fd = numeric_grad(prog, "y", "x", {"x": x})
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)


def test_producers_run_as_program_ops_with_int_signature():
    from pdum.tl import infer_signatures

    prog = Program(
        (
            I("x", "input"),
            I("s", "argsort", ["x"], dim="t"),
        )
    )
    x = T(np.array([2.0, 1.0]), ("t",))
    env = run(prog, {"x": x})
    np.testing.assert_array_equal(env["s"].to_numpy(), [1, 0])
    assert infer_signatures(prog, {"x": x})["s"].carrier == "int"
    shadows = infer(prog, {"x": x})
    assert shadows["s"].dim("t").chart is None


def test_producer_cost_and_bound_dim_refusal():
    from pdum.tl import argtopk  # noqa: F401 — the eager face is exercised above

    prog = Program(
        (
            I("x", "input"),
            I("top", "argtopk", ["x"], dim="e", k=2, k_name="r"),
        )
    )
    x = T(np.arange(6.0).reshape(2, 3), ("b", "e"))
    c = ops_count(prog, {"x": x})
    assert c.per_var["top"]["argtopk"] == 6  # one bucket entry per element examined
    with pytest.raises(NotImplementedError, match="distributed sort"):
        traffic(prog, {"x": x.bind(e="gpu")}, mesh(4))


def test_producers_lower_in_a_step_body():
    """The MoE router shape: argtopk in a step body, gates by take."""
    from pdum.tl import argtopk

    def router(logits):
        choice = argtopk(logits, dim="e", k=1, k_name="m")
        return choice

    logits = T(np.array([[0.1, 0.9], [0.8, 0.2]]), ("t", "e"))
    ls = lift_step(router, logits=logits.layout)
    assert any(i.op == "argtopk" for i in ls.program.instrs)
    env = run(ls.program, {"logits": logits})
    np.testing.assert_array_equal(env[ls.outputs[0]].to_numpy(), [[1], [0]])
