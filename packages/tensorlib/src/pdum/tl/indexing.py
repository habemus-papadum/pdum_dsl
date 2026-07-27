"""The indexing family (200 §1.9) — take/scatter_add, the adjoint pair.

`take` is a COMPUTATION, not a view. The layout algebra is affine and
data-independent — that is what keeps alignment decidable and adjoints
derivable — so gather never joins it: `take` materializes a fresh,
plainly-laid-out tensor, like every computation. Zero-cost data-dependent
views do not exist.

Indices are the VALUE door (250 §8): integer-carrier, unitless DATA whose
values are lattice positions in the taken dim's frame. They never become
Coordinates — CoordType stays an ambient credential — and the two doors
never bleed into each other. The coordinate law proves bounds at build
time BY CONSTRUCTION; data cannot be policed at build time, so the value
door's complement is a RUNTIME law: the reference refuses out-of-range
indices loudly. Structural facts (the dim exists, no name collisions,
domains agree) still refuse at build time, in `take_dims`/`scatter_dims`
— the single copy of the structural law that eager evaluation, Program
inference, and the dialect's type rules all share.

`take(table, idx, dim="v")`: idx dims NEW to the table replace the taken
dim IN PLACE; idx dims the table ALREADY carries align by name — the
tensor tier's one naming law (same name = same lattice, exactly as
pointwise and repeat_like read it), which is what makes the batched
forms fall out: `take(x, argsort(x, dim="t"), dim="t")` reorders each
line of a batch, and a per-token gate gather is the same spelling as
the embedding gather. out[**aligned**, spliced, rest] =
table[idx[aligned, spliced], **aligned**, rest]. Every surviving dim
keeps its frame verbatim (chart, labels, level ride); the consumed
frame disappears — its lattice went data. The adjoint is `scatter_add`
into the taken dim: duplicates SUM (a token appearing twice accumulates
both contributions — exactly the embedding gradient), which makes the
adjoint order-independent, hence deterministic. Indices are
gradient-free (`d_idx = None`).

`scatter_add(values, idx, dim="v", extent=..., over=...)` is user-facing
(routing needs it) and DECLARES its output frame — name and extent,
never inferred from the data (`max(idx)+1` would be a data-dependent
shape). Unlike take's, scatter's consumed/aligned split is genuinely
free — so it is DECLARED: `over=` names the consumed dims (reduce's
declared-dims precedent), defaulting to all idx dims (the total form);
idx dims not in `over` align and ride. Its adjoint is `take` — a
self-dual pair, like `repeat† = reduce`.

THE FACTORING: every other indexing operation decomposes into a
gradient-free INDEX PRODUCER plus `take`. `argtopk` (descending, ties
first-wins — the partition law's stable choice; `argmax` IS
`argtopk(k=1)`) and `argsort` (ascending, stable) produce integer
indices with NO adjoint rules — piecewise-constant in the data, zero
a.e.; top-k values are `take(x, argtopk(x, ...))`, their gradient
correct BY COMPOSITION through scatter_add; any differentiable
reordering is `take` by sorted indices. No multi-output instructions.
The produced rank lattice is plain — rank order is not the physical
axis, so the sorted dim's chart/labels do not survive onto it.
"""

from __future__ import annotations

import numpy as np

from .layout import Dim, Layout
from .tensor import Tensor

_OOB = (
    "{op}: indices out of range for dim {dim!r} [{start}, {stop}): saw "
    "min {lo}, max {hi} — the reference refuses out-of-range indices "
    "loudly (200 §1.9); device-tier behavior is a descent-license matter, "
    "never silent"
)


def _check_index_tensor(op: str, idx: Tensor) -> None:
    """The value-door demands on an index tensor: integer carrier, unitless.
    Data facts, checked where the data is — at evaluation."""
    if idx.carrier != "int":
        raise TypeError(
            f"{op}: indices are integer-carrier DATA (lattice positions in the "
            f"indexed dim's frame), got carrier {idx.carrier!r} — produce them "
            f"with an index producer (argtopk/argsort/iota) or an integer tensor; "
            f"a float never rounds itself into an address"
        )
    if idx.value_units is not None:
        raise TypeError(
            f"{op}: indices are unitless lattice positions, got value_units "
            f"{idx.value_units!r} — an index is an address, not a measurement"
        )


def _check_range(op: str, dim: str, iarr: np.ndarray, start: int, stop: int) -> None:
    if iarr.size == 0:
        return
    lo, hi = int(iarr.min()), int(iarr.max())
    if lo < start or hi >= stop:
        raise IndexError(_OOB.format(op=op, dim=dim, start=start, stop=stop, lo=lo, hi=hi))


def _as_domain(extent) -> tuple[int, int]:
    """The declared-extent convention shared with repeat: an int is a 0-based
    width, a pair is (start, stop)."""
    if isinstance(extent, tuple):
        return extent
    return (0, extent)


# ---- the structural law (build-time; layout-only) --------------------------


def _check_aligned(op: str, n: str, a: Dim, b: Dim) -> None:
    if (a.start, a.stop) != (b.start, b.stop):
        raise ValueError(
            f"{op}: aligned dim {n!r} disagrees in domain: [{a.start}, {a.stop}) "
            f"vs [{b.start}, {b.stop}) — same name IS the same lattice (the "
            f"naming law); slice/pad one side, or rename to splice a new dim"
        )
    if a.chart != b.chart or a.labels != b.labels:
        raise ValueError(
            f"{op}: aligned dim {n!r} disagrees in chart/labels — same name IS the same lattice; strip_charts or rename"
        )


def take_dims(table, idx, dim: str) -> tuple[tuple[Dim, ...], Dim]:
    """Output dims of a take. idx dims NEW to the table replace the taken
    dim IN PLACE (in idx order); idx dims the table already carries ALIGN
    by name (the naming law — the batched forms) and must agree with the
    table's. `table`/`idx` are Layouts (or anything with .dims/.dim).
    Structurally impossible takes refuse at build time."""
    d = table.dim(dim)  # KeyError names the dims it does have
    survivors = {x.name for x in table.dims if x.name != dim}
    for n in idx.names:
        if n in survivors:
            _check_aligned("take", n, table.dim(n), idx.dim(n))
    spliced = tuple(n for n in idx.names if n not in survivors)
    out: list[Dim] = []
    for x in table.dims:
        if x.name == dim:
            out.extend(idx.dim(n) for n in spliced)
        else:
            out.append(x)
    return tuple(out), d


def _over_names(idx, over) -> tuple[str, ...]:
    """The DECLARED consumed dims: default = all idx dims (the total form);
    idx dims not named ride aligned."""
    if over is None:
        return tuple(idx.names)
    names = (over,) if isinstance(over, str) else tuple(over)
    unknown = [n for n in names if n not in idx.names]
    if unknown:
        raise ValueError(f"scatter_add: over={unknown} are not idx dims {idx.names}")
    return names


def scatter_dims(values, idx, dim: str, extent, over=None) -> tuple[Dim, ...]:
    """Output dims of a scatter_add: the `over` dims are CONSUMED (every idx
    dim must be a values dim, agreeing in domain — the naming law); the
    declared dim takes the place of the first consumed dim; idx dims not in
    `over` align and ride; surviving dims verbatim."""
    start, stop = _as_domain(extent)
    consumed = _over_names(idx, over)
    missing = [n for n in idx.names if n not in values.names]
    if missing:
        raise ValueError(
            f"scatter_add: idx dim(s) {missing} are not values dims {values.names} "
            f"— values must carry every index dim (the consumed and aligned lattice)"
        )
    for n in idx.names:
        _check_aligned("scatter_add", n, values.dim(n), idx.dim(n))
    survivors = {x.name for x in values.dims if x.name not in consumed}
    if dim in survivors:
        raise ValueError(
            f"scatter_add: declared output dim {dim!r} collides with a surviving "
            f"values dim — pick a fresh name, or rename the values dim"
        )
    new = Dim(dim, 0, start, stop)
    out: list[Dim] = []
    placed = False
    for x in values.dims:
        if x.name in consumed:
            if not placed:
                out.append(new)
                placed = True
        else:
            out.append(x)
    if not placed:  # a rank-0/fully-aligned idx consumes nothing: the new dim leads
        out.insert(0, new)
    return tuple(out)


# ---- the computations (reference semantics) --------------------------------


def take(table: Tensor, idx: Tensor, *, dim: str) -> Tensor:
    """out[t..., rest...] = table[idx[t...], rest...] — gather (200 §1.9).

    Materializes a fresh, plainly-laid-out tensor; the taken dim is replaced
    in place by idx's dims and every surviving dim keeps its frame verbatim.
    idx values are lattice positions in the taken dim's frame [start, stop);
    out-of-range refuses loudly. Adjoint: scatter_add (duplicates sum);
    indices are gradient-free."""
    from .compute import _tensor_like

    out_dims, d = take_dims(table.layout, idx.layout, dim)
    _check_index_tensor("take", idx)
    survivors = {x.name for x in table.layout.dims if x.name != dim}
    aligned = tuple(n for n in idx.names if n in survivors)
    spliced = tuple(n for n in idx.names if n not in survivors)
    rest = tuple(x.name for x in table.layout.dims if x.name != dim and x.name not in aligned)
    iarr = idx.to_numpy(order=aligned + spliced) if idx.names else idx.to_numpy()
    _check_range("take", dim, iarr, d.start, d.stop)
    arr = table.to_numpy(order=aligned + (dim,) + rest)
    na = int(np.prod(arr.shape[: len(aligned)])) if aligned else 1
    ns = int(np.prod(iarr.shape[len(aligned) :])) if spliced else 1
    flat = arr.reshape((na, d.size) + arr.shape[len(aligned) + 1 :])
    picks = flat[np.arange(na)[:, None], (iarr - d.start).reshape(na, ns)]
    a_shape = arr.shape[: len(aligned)]
    s_shape = iarr.shape[len(aligned) :]
    got = picks.reshape(a_shape + s_shape + flat.shape[2:])
    src_order = aligned + spliced + rest
    perm = tuple(src_order.index(x.name) for x in out_dims)
    return _tensor_like(np.transpose(got, perm), out_dims, value_units=table.value_units)


def scatter_add(values: Tensor, idx: Tensor, *, dim: str, extent, over=None) -> Tensor:
    """out[v, aligned..., rest...] = Σ over consumed positions p with
    idx[aligned, p] == v of values[aligned, p, rest...]; zero where nothing
    lands — take's adjoint, user-facing (200 §1.9). The output frame is
    DECLARED: `dim` names it, `extent` (an int width or a (start, stop)
    pair, repeat's convention) is its domain; the new dim is plain —
    with_charts glues physics back on. `over` DECLARES the consumed dims
    (reduce's precedent), defaulting to all idx dims; idx dims not in
    `over` align by name and ride (the per-line scatter). Duplicates sum;
    addition makes the result order-independent, hence deterministic.
    Adjoint: take at the same indices; indices are gradient-free."""
    from .compute import _tensor_like

    out_dims = scatter_dims(values.layout, idx.layout, dim, extent, over)
    _check_index_tensor("scatter_add", idx)
    if values.carrier == "bool":
        raise TypeError(
            "scatter_add accumulates by ADDITION; bool values have no sum — "
            "cast to a numeric carrier first (a count is pw.mul with 1.0)"
        )
    start, stop = _as_domain(extent)
    consumed = _over_names(idx.layout, over)
    aligned = tuple(n for n in idx.names if n not in consumed)
    rest = tuple(x.name for x in values.layout.dims if x.name not in idx.names)
    iarr = idx.to_numpy(order=aligned + consumed) if idx.names else idx.to_numpy()
    _check_range("scatter_add", dim, iarr, start, stop)
    varr = values.to_numpy(order=aligned + consumed + rest)
    na = int(np.prod(varr.shape[: len(aligned)])) if aligned else 1
    nc = int(np.prod(iarr.shape[len(aligned) :])) if consumed else 1
    flat_v = varr.reshape((na, nc) + varr.shape[len(aligned) + len(consumed) :])
    acc = np.zeros((na, stop - start) + flat_v.shape[2:], dtype=varr.dtype)
    np.add.at(acc, (np.arange(na)[:, None], (iarr - start).reshape(na, nc)), flat_v)
    a_shape = varr.shape[: len(aligned)]
    got = acc.reshape(a_shape + (stop - start,) + flat_v.shape[2:])
    src_order = aligned + (dim,) + rest
    perm = tuple(src_order.index(x.name) for x in out_dims)
    return _tensor_like(np.transpose(got, perm), out_dims, value_units=values.value_units)


# ---- the index producers (gradient-free) -----------------------------------


def argtopk_dims(x, dim: str, k: int, k_name: str) -> tuple[Dim, ...]:
    d = x.dim(dim)
    if not 1 <= k <= d.size:
        raise ValueError(f"argtopk: k={k} outside [1, {d.size}] for dim {dim!r} [{d.start}, {d.stop})")
    clash = {y.name for y in x.dims if y.name != dim} & {k_name}
    if clash:
        raise ValueError(
            f"argtopk: rank dim {k_name!r} collides with an existing dim — the "
            f"produced lattice is new; pick a fresh name"
        )
    return tuple(Dim(k_name, 0, 0, k) if y.name == dim else y for y in x.dims)


def argsort_dims(x, dim: str) -> tuple[Dim, ...]:
    d = x.dim(dim)  # KeyError names the dims it does have
    plain = Dim(d.name, 0, d.start, d.stop)  # rank space: chart/labels do not survive
    return tuple(plain if y.name == dim else y for y in x.dims)


def argtopk(x: Tensor, *, dim: str, k: int, k_name: str) -> Tensor:
    """Indices of the k LARGEST values along `dim` — descending, ties
    first-wins (stable: the lowest lattice position of an equal value wins,
    the partition law's deterministic choice). The dim is replaced by the
    DECLARED rank dim `k_name` over [0, k); values are lattice positions in
    the sorted dim's frame. Integer output, NO adjoint rule (200 §1.9):
    top-k values are take(x, argtopk(x, ...)) — gradient correct by
    composition. argmax IS argtopk(k=1)."""
    from .compute import _tensor_like

    out_dims = argtopk_dims(x.layout, dim, k, k_name)
    d = x.layout.dim(dim)
    rest = tuple(n for n in x.names if n != dim)
    arr = x.to_numpy(order=rest + (dim,)) if rest else x.to_numpy(order=(dim,))
    order = np.argsort(-arr, axis=-1, kind="stable")[..., :k] + d.start
    src_order = rest + (k_name,)
    perm = tuple(src_order.index(y.name) for y in out_dims)
    return _tensor_like(np.transpose(order, perm), out_dims)


def argsort(x: Tensor, *, dim: str) -> Tensor:
    """Indices that sort `dim` ASCENDING, stable. The output ranges over the
    same domain re-read as rank space (plain: rank order is not the physical
    axis); values are lattice positions. Integer output, NO adjoint rule —
    any differentiable reordering is take by these indices."""
    from .compute import _tensor_like

    out_dims = argsort_dims(x.layout, dim)
    d = x.layout.dim(dim)
    rest = tuple(n for n in x.names if n != dim)
    arr = x.to_numpy(order=rest + (dim,)) if rest else x.to_numpy(order=(dim,))
    order = np.argsort(arr, axis=-1, kind="stable") + d.start
    src_order = rest + (dim,)
    perm = tuple(src_order.index(y.name) for y in out_dims)
    return _tensor_like(np.transpose(order, perm), out_dims)


def infer_take(shadows: dict, ins) -> Layout:
    """The Program-tier shadow rule — take_dims over operand shadows, wrapped
    dense (a take output is always freshly materialized)."""
    from .ir import _dense_like

    dims, _ = take_dims(shadows[ins.operands[0]], shadows[ins.operands[1]], ins.params["dim"])
    return _dense_like(dims)


def infer_scatter(shadows: dict, ins) -> Layout:
    from .ir import _dense_like

    return _dense_like(
        scatter_dims(
            shadows[ins.operands[0]],
            shadows[ins.operands[1]],
            ins.params["dim"],
            ins.params["extent"],
            ins.params.get("over"),
        )
    )


def infer_producer(shadows: dict, ins) -> Layout:
    from .ir import _dense_like

    src = shadows[ins.operands[0]]
    if ins.op == "argtopk":
        return _dense_like(argtopk_dims(src, ins.params["dim"], ins.params["k"], ins.params["k_name"]))
    return _dense_like(argsort_dims(src, ins.params["dim"]))
