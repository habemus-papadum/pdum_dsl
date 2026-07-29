"""The substrate-generic Region evaluator — ONE core, per-framework hooks (310).

283's emitter philosophy applied to interpreter columns: the dims machinery,
the take/scatter/fold algorithms, and the op dispatch exist ONCE here; a
framework column (torch_evaluator.py, jax_evaluator.py) contributes only a
``Substrate`` — ~16 array hooks plus a primitive marker table. The core
imports neither framework, so it loads under the default group set.

The evaluation law (unchanged from the torch column's landing): the type
rules already ran at region construction, so every node's type carries its
result dims — name, start, stop, in presentation order. Each node therefore
evaluates to a dense array whose axes are exactly its type dims, and this
layer computes DATA only. Layout ops that move coordinates but never values
(shift, rename, with_charts, ...) are data no-ops. Composite markers and
reducers are marker-DSL trees, evaluated over the substrate's primitive
table — one table is the whole porting surface per framework.

Where torch and jax genuinely agree — operators, basic indexing, advanced-
indexing gather — the core uses the shared spelling directly; hooks exist
only for the true divergences (mutation vs ``.at[]``, axis-op namespaces).

Untranslated ops raise ``Untranslatable`` naming the op (wgsl_executor's
law): tests skip loudly, nothing silently falls back.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from pdum.dsl.ir import Region
from pdum.tl.dialect import _thaw_params
from pdum.tl.markers import pw_marker, reducer
from pdum.tl.mdsl import CompositeMarker, CompositeReducer
from pdum.tl.nodes import Arg, Const, Prim


class Untranslatable(Exception):
    """Raised naming the op (or facet) a column does not evaluate yet."""


def _dims(t) -> tuple:
    dims = getattr(t, "dims", None)
    if dims is None:
        raise Untranslatable(f"non-tensor type {t!r}")
    return dims


def _names(dims) -> tuple[str, ...]:
    return tuple(d.name for d in dims)


def _sizes(dims) -> tuple[int, ...]:
    return tuple(d.stop - d.start for d in dims)


def _basic(x, axis, key):
    """Basic indexing at one axis — a spelling torch and jax share."""
    return x[(slice(None),) * axis + (key,)]


class Substrate:
    """The per-framework hook surface. A column subclasses and fills these;
    everything else the framework must do is reached through operators and
    basic/advanced indexing, which the frameworks already share."""

    name: str

    def __init__(self, device):
        self.device = device
        self.pw = self.pw_table()

    # arrays in / out
    def asarray(self, x):  # framework array on self.device; ints stay integral
        raise NotImplementedError

    def to_numpy(self, x) -> np.ndarray:
        raise NotImplementedError

    def is_array(self, x) -> bool:
        raise NotImplementedError

    # dtype policy (the column's float width is the substrate's choice; f64
    # for the conformance columns)
    def to_float(self, x):
        raise NotImplementedError

    def as_index(self, x):  # integer index carrier (i64)
        raise NotImplementedError

    # constructors
    def full(self, shape, value, kind: str):  # kind: "float" | "int"
        raise NotImplementedError

    def arange(self, start, stop):
        raise NotImplementedError

    # shape algebra
    def permute(self, x, perm):
        raise NotImplementedError

    def expand_dims(self, x, axis):
        raise NotImplementedError

    def broadcast_to(self, x, shape):
        raise NotImplementedError

    def moveaxis(self, x, src, dst):
        raise NotImplementedError

    def flip(self, x, axis):
        raise NotImplementedError

    def stack0(self, xs):
        raise NotImplementedError

    # the two genuinely divergent writes (mutation vs .at[])
    def paste(self, shape, fill, slices, data):  # canvas of fill, data pasted at slices
        raise NotImplementedError

    def scatter_rows_add(self, shape, rows, idx, vals):  # zeros(shape) += vals at [rows, idx]
        raise NotImplementedError

    # axis-op namespaces
    def argsort_stable(self, x):  # last axis, ascending, stable
        raise NotImplementedError

    def cum(self, name, x, axis):  # sum | prod | max | min
        raise NotImplementedError

    def red(self, name, x, axes):  # sum | prod | max | min | mean
        raise NotImplementedError

    def pw_table(self) -> dict:  # the primitive marker vocabulary
        raise NotImplementedError


@dataclass(frozen=True)
class Field:
    """A named-axis result: data axes are ``names``, in order."""

    data: object
    names: tuple[str, ...]
    substrate: Substrate

    def numpy(self, order: tuple[str, ...] | None = None) -> np.ndarray:
        arr = self.data
        if order is not None and tuple(order) != self.names:
            arr = self.substrate.permute(arr, tuple(self.names.index(n) for n in order))
        return self.substrate.to_numpy(arr)


def _eval_tree(node, args, table):
    """The marker-DSL tree over substrate values (compute._eval_tree's twin)."""
    if isinstance(node, Arg):
        return args[node.index]
    if isinstance(node, Const):
        v = node.value
        return float(v) if isinstance(v, Fraction) else v
    if isinstance(node, Prim):
        return table[node.op](*(_eval_tree(a, args, table) for a in node.args))
    raise TypeError(f"not a marker-DSL node: {node!r}")


class Evaluator:
    def __init__(self, sub: Substrate):
        self.sub = sub

    def _align(self, data, src, dst):
        """Permute axes by name: same dim set, possibly different order."""
        sn, dn = _names(src), _names(dst)
        if sn == dn:
            return data
        return self.sub.permute(data, tuple(sn.index(n) for n in dn))

    def _broadcast(self, data, src, dst):
        """repeat_like's data law: shared dims into dst order, a stride-0
        axis for every dim src lacks, expand to dst's sizes."""
        sn = _names(src)
        have = [n for n in _names(dst) if n in sn]
        data = self.sub.permute(data, tuple(sn.index(n) for n in have))
        for i, d in enumerate(dst):
            if d.name not in sn:
                data = self.sub.expand_dims(data, i)
        return self.sub.broadcast_to(data, _sizes(dst))

    def run(self, region: Region, values: list):
        memo: dict[int, object] = {}
        by_param = {id(p): v for p, v in zip(region.params, values)}

        def ev(n):
            if id(n) not in memo:
                memo[id(n)] = self._ev(n, ev, by_param)
            return memo[id(n)]

        return ev(region.body[-1])

    def _ev(self, n, ev, by_param):
        s = self.sub
        a = _thaw_params(dict(n.attrs))
        if id(n) in by_param:
            return by_param[id(n)]
        if n.op == "core.const":
            return a["value"]
        if n.op == "core.yield":
            return ev(n.args[0])
        if n.op == "core.tuple":
            return tuple(ev(x) for x in n.args)
        if n.op == "core.extract":
            return ev(n.args[0])[a["index"]]
        if n.op == "tl.const":
            kind = "int" if np.dtype(a.get("dtype", "float64")).kind in "iu" else "float"
            return s.full(_sizes(_dims(n.type)), a["value"], kind)
        if n.op == "tl.iota":
            d = next(x for x in _dims(n.type) if x.name == a["name"])
            return self._broadcast(s.arange(d.start, d.stop), (d,), _dims(n.type))
        if n.op == "tl.pointwise":
            return self._pointwise(n, a, ev)
        if n.op == "tl.reduce":
            return self._reduce(n, a, ev)
        if n.op == "tl.scan":
            return self._scan(n, a, ev)
        if n.op in ("tl.repeat_like", "tl.repeat"):
            return self._broadcast(ev(n.args[0]), _dims(n.args[0].type), _dims(n.type))
        if n.op in (
            "tl.rename",
            "tl.shift",
            "tl.with_charts",
            "tl.strip_charts",
            "tl.with_labels",
            "tl.bind",
            "tl.simplify",
            "tl.with_value_units",
        ):
            return ev(n.args[0])  # coordinates move, values do not
        if n.op == "tl.materialize":
            return self._align(ev(n.args[0]), _dims(n.args[0].type), _dims(n.type))
        if n.op == "tl.slice":
            out, src = _dims(n.type), _dims(n.args[0].type)
            data = ev(n.args[0])
            for i, (d, x) in enumerate(zip(out, src)):
                if (d.start, d.stop) != (x.start, x.stop):
                    data = _basic(data, i, slice(d.start - x.start, d.stop - x.start))
            return data
        if n.op == "tl.select":
            src = _dims(n.args[0].type)
            data = ev(n.args[0])
            for name, q in sorted(a["coords"].items(), key=lambda kv: -_names(src).index(kv[0])):
                i = _names(src).index(name)
                data = _basic(data, i, int(q) - src[i].start)
            survivors = tuple(d for d in src if d.name not in a["coords"])
            return self._align(data, survivors, _dims(n.type))
        if n.op == "tl.pad":
            out, src = _dims(n.type), _dims(n.args[0].type)
            slices = tuple(slice(x.start - d.start, x.stop - d.start) for d, x in zip(out, src))
            return s.paste(_sizes(out), float(a["fill"]), slices, ev(n.args[0]))
        if n.op == "tl.flip":
            src = _dims(n.args[0].type)
            return s.flip(ev(n.args[0]), next(i for i, d in enumerate(src) if d.name == a["name"]))
        if n.op == "tl.split":
            out, src = _dims(n.type), _dims(n.args[0].type)
            parts = set(a["parts"]) if not isinstance(a["parts"], dict) else set(a["parts"])
            # src presentation with the parts run collapsed back to the split dim
            expect = tuple(dict.fromkeys(a["name"] if d.name in parts else d.name for d in out))
            data = s.permute(ev(n.args[0]), tuple(_names(src).index(x) for x in expect))
            return data.reshape(_sizes(out))
        if n.op == "tl.merge":
            out, src = _dims(n.type), _dims(n.args[0].type)
            # out presentation with the merged dim expanded back into the parts run
            expect: list = []
            for d in out:
                expect.extend(tuple(a["parts"]) if d.name == a["name"] else (d.name,))
            data = s.permute(ev(n.args[0]), tuple(_names(src).index(x) for x in expect))
            return data.reshape(_sizes(out))
        if n.op == "tl.take":
            return self._take(n, a, ev)
        if n.op == "tl.scatter_add":
            return self._scatter_add(n, a, ev)
        if n.op in ("tl.argtopk", "tl.argsort"):
            return self._argranks(n, a, ev)
        if n.op == "tl.fold":
            return self._fold(n, a, ev)
        raise Untranslatable(n.op)

    def _pointwise(self, n, a, ev):
        s = self.sub
        out = _dims(n.type)
        vals = []
        for arg in n.args:
            v = ev(arg)
            if s.is_array(v):
                v = self._align(v, _dims(arg.type), out)
            else:  # scalars broadcast at real carrier (run_region's law)
                v = s.full(_sizes(out), float(v), "float")
            vals.append(v)
        f = pw_marker(a["f"])
        if isinstance(f, CompositeMarker):
            r = _eval_tree(f.body, vals, s.pw)
            if not s.is_array(r):
                return s.full(_sizes(out), float(r), "float")
            return s.broadcast_to(r, _sizes(out)) if tuple(r.shape) != _sizes(out) else r
        return s.pw[f.name](*vals)

    def _reduce(self, n, a, ev):
        s = self.sub
        f = reducer(a["f"])
        dims = (a["dims"],) if isinstance(a["dims"], str) else tuple(a["dims"])
        if isinstance(f, CompositeReducer):
            (dim,) = dims  # composites fold one dim at a time (compute.py's law)
            arrs, src = self._composite_elems(n, ev)
            state = self._composite_sweep(f, arrs, _names(src).index(dim), last_only=True)
            out = _eval_tree(f.project, state, s.pw)
            survivors = tuple(d for d in src if d.name != dim)
            return self._align(out, survivors, _dims(n.type))
        if a.get("zero") is not None:
            raise Untranslatable("tl.reduce zero= override")
        src = _dims(n.args[0].type)
        axes = tuple(i for i, d in enumerate(src) if d.name in dims)
        arr = ev(n.args[0])
        out = s.red(f.name, s.to_float(arr) if f.name == "mean" else arr, axes)
        survivors = tuple(d for d in src if d.name not in dims)
        return self._align(out, survivors, _dims(n.type))

    def _scan(self, n, a, ev):
        s = self.sub
        f = reducer(a["f"])
        dim = a["dim"]
        if isinstance(f, CompositeReducer):
            arrs, src = self._composite_elems(n, ev)
            out = self._composite_sweep(f, arrs, _names(src).index(dim), last_only=False)
            return self._align(out, src, _dims(n.type))
        if a.get("zero") is not None:
            raise Untranslatable("tl.scan zero= override")
        src = _dims(n.args[0].type)
        axis = _names(src).index(dim)
        arr = ev(n.args[0])
        if f.name == "mean":
            counts = s.to_float(s.arange(1, arr.shape[axis] + 1))
            shape = [1] * len(arr.shape)
            shape[axis] = -1
            out = s.cum("sum", s.to_float(arr), axis) / counts.reshape(shape)
        else:
            out = s.cum(f.name, arr, axis)
        return self._align(out, src, _dims(n.type))

    def _composite_elems(self, n, ev):
        """Element tensors of a composite reduce/scan, aligned to arg 0."""
        src = _dims(n.args[0].type)
        arrs = [self.sub.to_float(self._align(ev(arg), _dims(arg.type), src)) for arg in n.args]
        return arrs, src

    def _composite_sweep(self, f, arrs, axis, *, last_only):
        """compute._composite_sweep's twin: lift/combine sequentially."""
        s = self.sub
        moved = [s.moveaxis(x, axis, 0) for x in arrs]
        state, outs = None, []
        for t in range(moved[0].shape[0]):
            elem = [x[t] for x in moved]
            lifted = [_eval_tree(node, elem, s.pw) for node in f.lift]
            state = lifted if state is None else [_eval_tree(node, state + lifted, s.pw) for node in f.combine]
            if not last_only:
                outs.append(_eval_tree(f.project, state, s.pw))
        if last_only:
            if state is None:  # empty dim: the declared identity state
                shape = tuple(moved[0].shape[1:])
                state = [s.full(shape, float(v), "float") for v in f.init]
            return state
        return s.moveaxis(s.stack0(outs), 0, axis)

    def _take(self, n, a, ev):
        """indexing.take's semantics: idx dims new to the table SPLICE in
        place of the taken dim; idx dims the table carries ALIGN by name."""
        s = self.sub
        table, idx = n.args
        tdims, idims, out = _dims(table.type), _dims(idx.type), _dims(n.type)
        d = next(x for x in tdims if x.name == a["dim"])
        survivors = {x.name for x in tdims if x.name != a["dim"]}
        aligned = tuple(x for x in idims if x.name in survivors)
        spliced = tuple(x for x in idims if x.name not in survivors)
        rest = tuple(x for x in tdims if x.name != a["dim"] and x.name not in _names(aligned))
        iarr = s.as_index(self._align(ev(idx), idims, aligned + spliced)) - d.start
        arr = self._align(ev(table), tdims, aligned + (d,) + rest)
        na = int(np.prod(_sizes(aligned))) if aligned else 1
        ns = int(np.prod(_sizes(spliced))) if spliced else 1
        flat = arr.reshape((na, d.stop - d.start) + _sizes(rest))
        picks = flat[s.arange(0, na)[:, None], iarr.reshape(na, ns)]
        got = picks.reshape(_sizes(aligned) + _sizes(spliced) + _sizes(rest))
        return self._align(got, aligned + spliced + rest, out)

    def _scatter_add(self, n, a, ev):
        """indexing.scatter_add: the consumed (over=) dims collapse into the
        DECLARED output dim; duplicates sum."""
        s = self.sub
        values, idx = n.args
        vdims, idims, out = _dims(values.type), _dims(idx.type), _dims(n.type)
        start, stop = (a["extent"] if isinstance(a["extent"], tuple) else (0, a["extent"]))
        over = a.get("over")
        consumed = tuple(_names(idims)) if over is None else ((over,) if isinstance(over, str) else tuple(over))
        aligned = tuple(x for x in idims if x.name not in consumed)
        cons = tuple(x for x in idims if x.name in consumed)
        rest = tuple(x for x in vdims if x.name not in _names(idims))
        iarr = s.as_index(self._align(ev(idx), idims, aligned + cons)) - start
        varr = self._align(ev(values), vdims, aligned + cons + rest)
        na = int(np.prod(_sizes(aligned))) if aligned else 1
        nc = int(np.prod(_sizes(cons))) if cons else 1
        flat_v = varr.reshape((na, nc) + _sizes(rest))
        rows = s.broadcast_to(s.arange(0, na)[:, None], (na, nc))
        got = s.scatter_rows_add((na, stop - start) + _sizes(rest), rows, iarr.reshape(na, nc), flat_v)
        new = next(x for x in out if x.name == a["dim"])
        got = got.reshape(_sizes(aligned) + (stop - start,) + _sizes(rest))
        return self._align(got, aligned + (new,) + rest, out)

    def _argranks(self, n, a, ev):
        """argtopk (descending, ties first-wins) / argsort (ascending,
        stable): values are lattice positions — the dim's start rides out."""
        s = self.sub
        src, out = _dims(n.args[0].type), _dims(n.type)
        d = next(x for x in src if x.name == a["dim"])
        rest = tuple(x for x in src if x.name != a["dim"])
        arr = s.to_float(self._align(ev(n.args[0]), src, rest + (d,)))
        if n.op == "tl.argtopk":
            order = s.argsort_stable(-arr)[..., : a["k"]] + d.start
            tail = next(x for x in out if x.name == a["k_name"])
        else:
            order = s.argsort_stable(arr) + d.start
            tail = next(x for x in out if x.name == a["dim"])
        return self._align(order, rest + (tail,), out)

    def _fold(self, n, a, ev):
        """run_region's fold row: carry k states positionally, select element
        slices at ABSOLUTE coordinates, finish (final) or stack (emit)."""
        s = self.sub
        k = len(tuple(a["state"]))
        dim, out = a["dim"], tuple(a["out"])
        vals = [ev(x) for x in n.args]
        carried, srcs = list(vals[:k]), vals[k:]
        cdims = [_dims(x.type) for x in n.args[:k]]
        sdims = [_dims(x.type) for x in n.args[k:]]
        if srcs:
            lo, hi = next((d.start, d.stop) for d in sdims[0] if d.name == dim)
        else:
            lo, hi = a["extent"]
        step = n.regions[0]
        pdims = [_dims(p.type) for p in step.params]
        emitted = []
        for q in range(lo, hi):
            elems = []
            for src, sd in zip(srcs, sdims):
                i = _names(sd).index(dim)
                elems.append((_basic(src, i, q - sd[i].start), tuple(d for d in sd if d.name != dim)))
            pairs = [(c, cd) for c, cd in zip(carried, cdims)] + elems
            bound = [self._align(v, vd, pd) for (v, vd), pd in zip(pairs, pdims)]
            res = self.run(step, bound)
            res = res if isinstance(res, tuple) else (res,)
            ydims = self._yield_dims(step)
            carried = [self._align(r, yd, cd) for r, yd, cd in zip(res[:k], ydims, cdims)]
            if out[0] == "emit":
                emitted.append((res[k], ydims[k]))
        if out[0] == "final":
            return self._align(carried[out[1]], cdims[out[1]], _dims(n.type))
        stacked = s.stack0([self._align(e, ed, emitted[0][1]) for e, ed in emitted])
        lead = next(d for d in _dims(n.type) if d.name == dim)
        return self._align(stacked, (lead,) + tuple(emitted[0][1]), _dims(n.type))

    @staticmethod
    def _yield_dims(step: Region) -> list:
        yielded = step.body[-1].args[0]
        outs = tuple(yielded.args) if yielded.op == "core.tuple" else (yielded,)
        return [_dims(x.type) for x in outs]


def run_region_on(sub: Substrate, region: Region, values: list):
    """Evaluate a tensor-tier region on a substrate — ``values`` positional
    per param, each array in ITS PARAM's type-dim order."""
    return Evaluator(sub).run(region, [sub.asarray(v) for v in values])


def run_named_on(sub: Substrate, region: Region, inputs: dict, names) -> dict:
    """dialect.run_named's door on a substrate: params bind by their claimed
    names (arrays in the param's type-dim order), the yield's slots report
    under theirs — as ``Field``s carrying their axis names."""
    order = [names[id(p)] for p in region.params]
    missing = [k for k in order if k not in inputs]
    if missing:
        raise KeyError(f"missing input {missing[0]!r}")
    res = run_region_on(sub, region, [inputs[k] for k in order])
    yielded = region.body[-1].args[0]
    outs = tuple(yielded.args) if yielded.op == "core.tuple" else (yielded,)
    res = res if isinstance(res, tuple) else (res,)
    return {names[id(x)]: Field(v, _names(_dims(x.type)), sub) for x, v in zip(outs, res)}
