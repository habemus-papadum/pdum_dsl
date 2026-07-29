"""The indexing family (200 §1.9): take/scatter_add — the adjoint pair.

`take` is a computation, not a view; indices are the VALUE door (250 §8):
integer-carrier, unitless data, bounds refused at RUN time — the data
complement of the coordinate law's in-bounds-by-construction. Structural
facts refuse at BUILD time. Duplicates sum through the adjoint (the
embedding gradient); indices are gradient-free.
"""

import numpy as np
import pytest

from pdum.tl import (
    Tensor,
    argsort,
    argtopk,
    mesh,
    ops_count,
    peak_memory,
    pointwise,
    scatter_add,
    take,
    traffic,
)
from pdum.tl.autodiff import grad, numeric_grad
from pdum.tl.compute import red, reduce, repeat_like
from pdum.tl.dialect import run_named, walk_region
from pdum.tl.lifting import lift_step
from pdum.tl.markers import exp, stop_gradient


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr), names)


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


def test_take_aligns_shared_dims_by_the_naming_law():
    """An idx dim the table already carries ALIGNS (same name = same
    lattice — the tier's one naming law): out[d] = table[idx[d], d], the
    data-dependent diagonal. A domain mismatch refuses."""
    picks = T(np.array([3, 0, 2]), ("d",))  # one row choice PER column
    out = take(TABLE, picks, dim="v")
    assert out.names == ("d",)
    tn = TABLE.to_numpy()
    np.testing.assert_array_equal(out.to_numpy(), [tn[3, 0], tn[0, 1], tn[2, 2]])
    with pytest.raises(ValueError, match="disagrees in domain.*naming law"):
        take(TABLE, T(np.array([0, 1]), ("d",)), dim="v")  # d is [0,3), idx d is [0,2)


def test_take_batched_reorder_is_the_same_spelling():
    """The spec's 'any differentiable reordering is take by sorted indices'
    in BATCHED form: argsort over (b, t) aligns b, splices t — each line
    reorders independently."""
    x = T(np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]), ("b", "t"))
    out = take(x, argsort(x, dim="t"), dim="t")
    assert out.names == ("b", "t")
    np.testing.assert_array_equal(out.to_numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_take_gate_gather_is_the_moe_shape():
    """The router's gate gather: per-token choices over (t, c) against
    logits (t, e) — t aligns, c splices in e's place."""
    logits = T(np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.3]]), ("t", "e"))
    choice = T(np.array([[1], [0]]), ("t", "c"))
    gates = take(logits, choice, dim="e")
    assert gates.names == ("t", "c")
    np.testing.assert_array_equal(gates.to_numpy(), [[0.9], [0.8]])


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


def test_scatter_add_over_declares_the_per_line_form():
    """`over=` names the consumed dims (reduce's precedent); idx dims not
    named align and ride — the per-line histogram."""
    vals = T(np.ones((2, 3)), ("b", "t"))
    idx = T(np.array([[0, 1, 0], [2, 2, 2]]), ("b", "t"))
    out = scatter_add(vals, idx, dim="v", extent=3, over=("t",))
    assert out.names == ("b", "v")
    np.testing.assert_array_equal(out.to_numpy(), [[2.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    with pytest.raises(ValueError, match=r"over=\['q'\] are not idx dims"):
        scatter_add(vals, idx, dim="v", extent=3, over=("q",))


def test_scatter_add_refusals():
    vals = T(np.array([1.0, 2.0, 3.0]), ("t",))
    with pytest.raises(IndexError, match="refuses out-of-range"):
        scatter_add(vals, T(np.array([0, 1, 9]), ("t",)), dim="v", extent=4)
    with pytest.raises(ValueError, match="not values dims"):
        scatter_add(vals, T(np.array([0, 1]), ("q",)), dim="v", extent=4)
    with pytest.raises(ValueError, match="disagrees in domain.*naming law"):
        scatter_add(vals, T(np.array([0, 1]), ("t",)), dim="v", extent=4)
    with pytest.raises(ValueError, match="collides with a surviving"):
        vals2 = T(np.arange(6.0).reshape(3, 2), ("t", "d"))
        scatter_add(vals2, T(np.array([0, 1, 0]), ("t",)), dim="d", extent=4)
    with pytest.raises(TypeError, match="bool values have no sum"):
        scatter_add(T(np.array([True, False]), ("t",)), T(np.array([0, 1]), ("t",)), dim="v", extent=2)


# ----------------------------------------------------------------------
# the region tier: run / infer / opcount / memory / traffic
# ----------------------------------------------------------------------


def _embed_and_sum(table, ids):
    tok = take(table, ids, dim="v")
    y = reduce(red.sum, tok, ("t", "d"))
    return tok, y


def _embed_ls(table=TABLE, ids=IDS):
    return lift_step(_embed_and_sum, table=table.layout, ids=ids.layout)


def _route4(vals, ids):
    g = scatter_add(vals, ids, dim="v", extent=(0, 4))
    return g


def test_take_runs_and_infers_as_a_region_op():
    ls = _embed_ls()
    vals = run_named(ls.region, {"table": TABLE, "ids": IDS}, ls.names)
    np.testing.assert_array_equal(vals["tok"].to_numpy(), TABLE.to_numpy()[[2, 0, 2]])
    tok = next(n for n in walk_region(ls.region) if ls.names.get(id(n)) == "tok")
    assert tuple(d.name for d in tok.type.layout.dims) == ("t", "d")
    assert tok.type.layout.dim("t").size == 3


def test_take_and_scatter_cost_entries():
    """take: one read+write per OUTPUT element, its own bucket (random
    access is not a sequential copy — the cost model prices the difference);
    scatter_add: movement + accumulate per INPUT element."""
    ls = _embed_ls()
    c = ops_count(ls.region, names=ls.names)
    assert c.per_var["tok"]["take"] == 9  # 3 tokens x width 3
    vals = T(np.ones((3, 3)), ("t", "d"))
    ls2 = lift_step(_route4, vals=vals.layout, ids=IDS.layout)
    cs = ops_count(ls2.region, names=ls2.names)
    assert cs.per_var["g"]["scatter"] == 9 and cs.per_var["g"]["add"] == 9


def test_take_allocates_its_output():
    """A real node, never a free view (200 §1.9): the gather output owns
    numel x 8 bytes in the peak-memory model."""
    ls = _embed_ls()
    r = peak_memory(ls.region, {"table": TABLE, "ids": IDS}, names=ls.names)
    assert r.alloc_bytes["tok"] == 9 * 8


def test_take_along_a_bound_dim_refuses_toward_the_fix():
    ls = _embed_ls(table=TABLE.bind(v="gpu"))
    with pytest.raises(NotImplementedError, match="all-to-all.*all-gather the table"):
        traffic(ls.region, None, mesh(4), names=ls.names)
    # a bound RIDING dim is a sharded batch: the take itself rides for free
    # (the final reduce over bound t still all-reduces — that is reduce's law)
    ls2 = _embed_ls(ids=IDS.bind(t="gpu"))
    r = traffic(ls2.region, None, mesh(4), names=ls2.names)
    assert not any(c.var == "tok" for c in r.collectives)


def test_scatter_add_over_a_bound_consumed_dim_refuses():
    vals = T(np.ones(3), ("t",)).bind(t="gpu")
    ls = lift_step(_route4, vals=vals.layout, ids=IDS.bind(t="gpu").layout)
    with pytest.raises(NotImplementedError, match="partial sums.*all-reduce"):
        traffic(ls.region, None, mesh(4), names=ls.names)
    ls2 = lift_step(_route4, vals=vals.layout, ids=IDS.layout)
    with pytest.raises(NotImplementedError, match="colocate"):
        traffic(ls2.region, None, mesh(4), names=ls2.names)


# ----------------------------------------------------------------------
# the step tier: take lowers through the one engine
# ----------------------------------------------------------------------


def test_take_lowers_in_a_step_body():
    def embed(table, ids):
        return take(table, ids, dim="v")

    ls = lift_step(embed, table=TABLE.layout, ids=IDS.layout)
    assert any(n.op == "tl.take" for n in walk_region(ls.region))
    vals = run_named(ls.region, {"table": TABLE, "ids": IDS}, ls.names)
    np.testing.assert_array_equal(vals[ls.outputs[0]].to_numpy(), TABLE.to_numpy()[[2, 0, 2]])


def test_scatter_add_lowers_in_a_step_body():
    def route(vals, ids):
        return scatter_add(vals, ids, dim="v", extent=4)

    vals = T(np.ones((3, 3)), ("t", "d"))
    ls = lift_step(route, vals=vals.layout, ids=IDS.layout)
    out = run_named(ls.region, {"vals": vals, "ids": IDS}, ls.names)
    want = np.zeros((4, 3))
    want[2], want[0] = 2.0, 1.0
    np.testing.assert_array_equal(out[ls.outputs[0]].to_numpy(), want)


# ----------------------------------------------------------------------
# the adjoint pair
# ----------------------------------------------------------------------


def test_take_gradient_is_the_embedding_gradient():
    """d_table counts occurrences through scatter_add: a token appearing
    twice accumulates BOTH contributions; indices are gradient-free."""
    ls = _embed_ls()
    inputs = {"table": TABLE, "ids": IDS}
    rg = grad(ls.region, "y", inputs, names=ls.names)
    assert rg.grads["ids"] is None  # d_idx = None (200 §1.9)
    vals = run_named(rg.region, inputs, rg.names)
    g = vals[rg.grads["table"]].to_numpy(order=("v", "d"))
    want = np.zeros((4, 3))
    want[2], want[0] = 2.0, 1.0  # row 2 taken twice, row 0 once
    np.testing.assert_array_equal(g, want)
    # and the same fact by finite differences
    fd = numeric_grad(ls.region, "y", "table", inputs, ls.names)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)


def test_scatter_add_gradient_is_take():
    """The self-dual pair: d_values gathers the cotangent back at the same
    indices — checked against finite differences."""

    def body(vals, ids):
        s = scatter_add(vals, ids, dim="v", extent=(0, 4))
        sq = s * s
        y = reduce(red.sum, sq, ("v", "d"))
        return y

    vals = T(np.arange(6.0).reshape(3, 2), ("t", "d"))
    ls = lift_step(body, vals=vals.layout, ids=IDS.layout)
    inputs = {"vals": vals, "ids": IDS}
    rg = grad(ls.region, "y", inputs, names=ls.names)
    assert rg.grads["ids"] is None
    out = run_named(rg.region, inputs, rg.names)
    g = out[rg.grads["vals"]].to_numpy(order=("t", "d"))
    fd = numeric_grad(ls.region, "y", "vals", inputs, ls.names)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


def test_units_ride_through_take():
    from pdum.tl import infer_signatures, u

    ls = _embed_ls()
    sigs = infer_signatures(ls.region, {"table": TABLE.with_value_units(u.um), "ids": IDS}, names=ls.names)
    assert sigs["tok"].unit == u.um


def test_aligned_take_gradient_matches_finite_differences():
    """The batched gather's adjoint is the per-line scatter (over= the
    spliced dims) — pinned by FD, and by the self-duality round trip."""

    def body(table, idx):
        g = take(table, idx, dim="v")
        sq = g * g
        y = reduce(red.sum, sq, ("b", "t"))
        return y

    table = T(np.arange(8.0).reshape(2, 4), ("b", "v"))
    idx = T(np.array([[1, 1, 3], [0, 2, 2]]), ("b", "t"))
    ls = lift_step(body, table=table.layout, idx=idx.layout)
    inputs = {"table": table, "idx": idx}
    rg = grad(ls.region, "y", inputs, names=ls.names)
    out = run_named(rg.region, inputs, rg.names)
    g = out[rg.grads["table"]].to_numpy(order=("b", "v"))
    fd = numeric_grad(ls.region, "y", "table", inputs, ls.names)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


def test_over_scatter_gradient_matches_finite_differences():
    def body(vals, idx):
        s = scatter_add(vals, idx, dim="v", extent=(0, 3), over=("t",))
        sq = s * s
        y = reduce(red.sum, sq, ("b", "v"))
        return y

    vals = T(np.arange(6.0).reshape(2, 3), ("b", "t"))
    idx = T(np.array([[0, 1, 0], [2, 2, 0]]), ("b", "t"))
    ls = lift_step(body, vals=vals.layout, idx=idx.layout)
    inputs = {"vals": vals, "idx": idx}
    rg = grad(ls.region, "y", inputs, names=ls.names)
    out = run_named(rg.region, inputs, rg.names)
    g = out[rg.grads["vals"]].to_numpy(order=("b", "t"))
    fd = numeric_grad(ls.region, "y", "vals", inputs, ls.names)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


# ----------------------------------------------------------------------
# the index producers: argtopk / argsort (gradient-free)
# ----------------------------------------------------------------------


def test_argtopk_descends_and_ties_go_first():
    """Descending; ties FIRST-WINS (the partition law's stable choice): a
    duplicated maximum yields the lower lattice position first."""
    x = T(np.array([1.0, 5.0, 3.0, 5.0]), ("t",))
    top = argtopk(x, dim="t", k=3, k_name="r")
    assert top.names == ("r",)
    assert top.layout.dim("r").size == 3
    np.testing.assert_array_equal(top.to_numpy(), [1, 3, 2])  # first 5.0 wins


def test_argmax_is_argtopk_k1():
    x = T(np.array([[1.0, 9.0, 2.0], [7.0, 0.0, 3.0]]), ("b", "e"))
    am = argtopk(x, dim="e", k=1, k_name="m")
    assert am.names == ("b", "m")
    np.testing.assert_array_equal(am.to_numpy(), [[1], [0]])


def test_argsort_is_ascending_stable_and_composes_with_take():
    """Any differentiable reordering is take by sorted indices."""
    x = T(np.array([3.0, 1.0, 2.0, 1.0]), ("t",))
    order = argsort(x, dim="t")
    np.testing.assert_array_equal(order.to_numpy(), [1, 3, 2, 0])  # stable at the tie
    sorted_x = take(x, order, dim="t")
    np.testing.assert_array_equal(sorted_x.to_numpy(), [1.0, 1.0, 2.0, 3.0])


def test_producers_read_lattice_positions_not_zero_based_offsets():
    x = T(np.array([3.0, 1.0, 2.0]), ("t",)).shift(t=5)
    order = argsort(x, dim="t")
    np.testing.assert_array_equal(order.to_numpy(), [6, 7, 5])
    np.testing.assert_array_equal(take(x, order, dim="t").to_numpy(), [1.0, 2.0, 3.0])


def test_producer_refusals():
    x = T(np.arange(6.0).reshape(2, 3), ("b", "e"))
    with pytest.raises(ValueError, match=r"k=4 outside \[1, 3\]"):
        argtopk(x, dim="e", k=4, k_name="r")
    with pytest.raises(ValueError, match="collides with an existing dim"):
        argtopk(x, dim="e", k=2, k_name="b")


def test_topk_gradient_is_correct_by_composition():
    """The factoring pays: grad of sum(take(x, argtopk(x))) flows through
    TAKE alone — ones at the winning positions, zero elsewhere — matching
    finite differences with no adjoint rule for the producer."""

    def body(x):
        top = argtopk(x, dim="t", k=2, k_name="r")
        vals = take(x, top, dim="t")
        y = reduce(red.sum, vals, ("r",))
        return y

    x = T(np.array([4.0, 9.0, 1.0, 7.0]), ("t",))
    ls = lift_step(body, x=x.layout)
    rg = grad(ls.region, "y", {"x": x}, names=ls.names)
    out = run_named(rg.region, {"x": x}, rg.names)
    g = out[rg.grads["x"]].to_numpy(order=("t",))
    np.testing.assert_array_equal(g, [0.0, 1.0, 0.0, 1.0])
    fd = numeric_grad(ls.region, "y", "x", {"x": x}, ls.names)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)


def test_producers_run_as_region_ops_with_int_signature():
    from pdum.tl import infer_signatures

    def body(x):
        s = argsort(x, dim="t")
        return s

    x = T(np.array([2.0, 1.0]), ("t",))
    ls = lift_step(body, x=x.layout)
    out = run_named(ls.region, {"x": x}, ls.names)
    np.testing.assert_array_equal(out["s"].to_numpy(), [1, 0])
    assert infer_signatures(ls.region, {"x": x}, names=ls.names)["s"].carrier == "int"
    snode = ls.region.body[-1].args[0]
    assert snode.type.layout.dim("t").chart is None


def test_producer_cost_and_bound_dim_refusal():
    def body(x):
        top = argtopk(x, dim="e", k=2, k_name="r")
        return top

    x = T(np.arange(6.0).reshape(2, 3), ("b", "e"))
    ls = lift_step(body, x=x.layout)
    c = ops_count(ls.region, names=ls.names)
    assert c.per_var["top"]["argtopk"] == 6  # one bucket entry per element examined
    lsb = lift_step(body, x=x.bind(e="gpu").layout)
    with pytest.raises(NotImplementedError, match="distributed sort"):
        traffic(lsb.region, None, mesh(4), names=lsb.names)


def _st_body(logits, gum, tau, wte):
    sc = (logits + gum) / repeat_like(tau, logits)
    mx = reduce(red.max, sc, ("v",))
    ex = pointwise(exp, sc - repeat_like(mx, sc))
    soft = ex / repeat_like(reduce(red.sum, ex, ("v",)), ex)  # softmax((logits+gum)/tau)
    nxt = argtopk(sc, dim="v", k=1, k_name="c")  # the sample
    ehard = take(wte, nxt, dim="v")  # (c, d)
    esoft = reduce(red.sum, repeat_like(soft, wte) * wte, ("v",))  # (d,)
    esb = repeat_like(esoft, ehard)
    st = ehard + (esb - pointwise(stop_gradient, esb))  # hard + (soft - sg(soft))
    y = reduce(red.sum, st, ("c", "d"))
    yss = reduce(red.sum, esoft, ("d",))  # the soft path alone
    yhh = reduce(red.sum, ehard, ("c", "d"))  # the hard path alone
    return st, ehard, y, yss, yhh


def test_straight_through_topk_sampling_trains_tau():
    """The §1.8 sampling idiom over the §1.9 family: sample = argtopk of
    Gumbel-perturbed logits (gradient-free), hard embedding by take, soft
    embedding by contract, straight-through = hard + (soft −
    stop_gradient(soft)). Forward equals the HARD path exactly — which is
    WHY finite differences of the objective itself see zero in τ: ST's
    gradient is a DECLARED estimator, not the true derivative. The pin:
    each declared grad equals FD of the path it declares (τ and logits →
    the soft target; wte → hard target + soft target)."""
    rng = np.random.default_rng(3)
    logits = T(rng.standard_normal(4), ("v",))
    gum = T(-np.log(-np.log(rng.uniform(size=4))), ("v",))
    tau = T(np.asarray(0.7), ())
    wte = T(rng.standard_normal((4, 2)), ("v", "d"))
    inputs = {"logits": logits, "gum": gum, "tau": tau, "wte": wte}
    ls = lift_step(_st_body, logits=logits.layout, gum=gum.layout, tau=tau.layout, wte=wte.layout)
    fw = run_named(ls.region, inputs, ls.names)
    # forward IS the hard path, exactly
    np.testing.assert_array_equal(fw["st"].to_numpy(), fw["ehard"].to_numpy())
    np.testing.assert_array_equal(fw["y"].to_numpy(), fw["yhh"].to_numpy())
    rg = grad(ls.region, "y", inputs, names=ls.names)
    assert rg.grads["gum"] is not None  # noise participates in the soft path
    out = run_named(rg.region, inputs, rg.names)

    def g(wrt):
        return out[rg.grads[wrt]].to_numpy(order=inputs[wrt].names)

    # tau and logits train through the SOFT path alone (the declaration)
    np.testing.assert_allclose(g("tau"), numeric_grad(ls.region, "yss", "tau", inputs, ls.names), rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(
        g("logits"), numeric_grad(ls.region, "yss", "logits", inputs, ls.names), rtol=1e-4, atol=1e-8
    )
    # wte trains through BOTH: the take's true adjoint plus the soft path
    want = numeric_grad(ls.region, "yhh", "wte", inputs, ls.names) + numeric_grad(
        ls.region, "yss", "wte", inputs, ls.names
    )
    np.testing.assert_allclose(g("wte"), want, rtol=1e-4, atol=1e-8)
    assert float(np.abs(g("tau"))) > 0  # tau genuinely trains


def test_producers_lower_in_a_step_body():
    """The MoE router shape: argtopk in a step body, gates by take."""

    def router(logits):
        choice = argtopk(logits, dim="e", k=1, k_name="m")
        return choice

    logits = T(np.array([[0.1, 0.9], [0.8, 0.2]]), ("t", "e"))
    ls = lift_step(router, logits=logits.layout)
    assert any(n.op == "tl.argtopk" for n in walk_region(ls.region))
    out = run_named(ls.region, {"logits": logits}, ls.names)
    np.testing.assert_array_equal(out[ls.outputs[0]].to_numpy(), [[1], [0]])
