"""The Region -> PyTorch evaluator — the first reference-runtime CHECK column (310).

A graph-level interpreter over dialect regions: each node evaluates to a dense
``torch.Tensor`` whose axes are EXACTLY the node's type dims, in type order —
the type rules already ran at construction, so this layer never re-infers
layout. It computes data only. Layout ops that move coordinates but never
values (shift, rename, with_charts, ...) are data no-ops here; their whole
effect is already recorded in the result type's dims.

The column's role is the conformance CHECK: the same region a numpy reference
serves (``dialect.run_region``) evaluated on a second, independent substrate —
torch f64 on CPU or CUDA — against the zoo's own numpy denotation. It is NOT
the baseline artifact (idiomatic hand-written torch is, torch_zoo.py) and not
yet an rt Pair: it mounts behind pdum.rt's selection door when the graphics
team's skeleton lands (283; the interpreter-openness ask on PR #9).

Composite markers and reducers are marker-DSL TREES (mdsl.py), so this layer
evaluates the same declarations the reference does — zoo.gelu or zoo.flashsm
need no per-name porting, only the primitive table below.

Untranslated ops raise ``Untranslatable`` naming the op (wgsl_executor's law):
tests skip loudly, nothing silently falls back.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch

from pdum.dsl.ir import Region
from pdum.tl.dialect import _thaw_params
from pdum.tl.markers import pw_marker, reducer
from pdum.tl.mdsl import CompositeMarker, CompositeReducer
from pdum.tl.nodes import Arg, Const, Prim

# the executor-fingerprint seam (kernel.py's content door / 283's Pair seat):
# a version change here is a new artifact world
TORCH_FP = ("region_torch", "torch", torch.__version__.split("+")[0])


class Untranslatable(Exception):
    """Raised naming the op (or facet) this column does not evaluate yet."""


def _dims(t) -> tuple:
    dims = getattr(t, "dims", None)
    if dims is None:
        raise Untranslatable(f"non-tensor type {t!r}")
    return dims


def _names(dims) -> tuple[str, ...]:
    return tuple(d.name for d in dims)


def _sizes(dims) -> tuple[int, ...]:
    return tuple(d.stop - d.start for d in dims)


def _align(data: torch.Tensor, src, dst) -> torch.Tensor:
    """Permute axes by name: same dim set, possibly different presentation."""
    sn, dn = _names(src), _names(dst)
    if sn == dn:
        return data
    return data.permute(tuple(sn.index(n) for n in dn))


def _broadcast(data: torch.Tensor, src, dst) -> torch.Tensor:
    """repeat_like's data law: permute the shared dims into dst order, add a
    stride-0 axis for every dim src lacks, expand to dst's sizes."""
    sn = _names(src)
    have = [n for n in _names(dst) if n in sn]
    data = data.permute(tuple(sn.index(n) for n in have))
    for i, d in enumerate(dst):
        if d.name not in sn:
            data = data.unsqueeze(i)
    return data.expand(_sizes(dst))


def _pw_table(dtype):
    """The primitive marker vocabulary over torch (pdum.dsl.markers names).

    numpy upcasts integer inputs through float-only ufuncs and true-divides
    ints to f64; torch refuses or drops to the default dtype — so float-only
    rows cast integral inputs to the column's dtype first.
    """

    def f(x):
        return x.to(dtype) if isinstance(x, torch.Tensor) and not x.is_floating_point() else x

    return {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: f(a) / f(b),
        "neg": lambda a: -a,
        "exp": lambda a: torch.exp(f(a)),
        "log": lambda a: torch.log(f(a)),
        "maximum": torch.maximum,
        "minimum": torch.minimum,
        "tanh": lambda a: torch.tanh(f(a)),
        "sqrt": lambda a: torch.sqrt(f(a)),
        "sin": lambda a: torch.sin(f(a)),
        "cos": lambda a: torch.cos(f(a)),
        "abs": torch.abs,
        "floor": lambda a: torch.floor(f(a)),
        "stop_gradient": lambda a: a,
        "where": lambda c, a, b: torch.where(c.bool() if isinstance(c, torch.Tensor) else c, a, b),
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "le": lambda a, b: a <= b,
        "lt": lambda a, b: a < b,
        "ge": lambda a, b: a >= b,
        "gt": lambda a, b: a > b,
    }


def _eval_tree(node, args, table):
    """The marker-DSL tree over torch values (compute._eval_tree's twin)."""
    if isinstance(node, Arg):
        return args[node.index]
    if isinstance(node, Const):
        v = node.value
        return float(v) if isinstance(v, Fraction) else v
    if isinstance(node, Prim):
        return table[node.op](*(_eval_tree(a, args, table) for a in node.args))
    raise TypeError(f"not a marker-DSL node: {node!r}")


_PLAIN_RED = {
    "sum": torch.sum,
    "prod": torch.prod,
    "max": torch.amax,
    "min": torch.amin,
}


@dataclass(frozen=True)
class TorchField:
    """A named-axis result: data axes are ``names``, in order."""

    data: torch.Tensor
    names: tuple[str, ...]

    def numpy(self, order: tuple[str, ...] | None = None) -> np.ndarray:
        arr = self.data
        if order is not None and tuple(order) != self.names:
            arr = arr.permute(tuple(self.names.index(n) for n in order))
        return arr.cpu().numpy()


class _Eval:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.pw = _pw_table(dtype)

    def as_input(self, arr) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(arr) if not isinstance(arr, torch.Tensor) else arr)
        t = t.to(self.device)
        return t.to(self.dtype) if t.is_floating_point() else t

    def run(self, region: Region, values: list):
        memo: dict[int, object] = {}
        by_param = {}
        for p, v in zip(region.params, values):
            by_param[id(p)] = v

        def ev(n):
            if id(n) not in memo:
                memo[id(n)] = self._ev(n, ev, by_param)
            return memo[id(n)]

        return ev(region.body[-1])

    def _ev(self, n, ev, by_param):
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
            dt = torch.int64 if np.dtype(a.get("dtype", "float64")).kind in "iu" else self.dtype
            return torch.full(_sizes(_dims(n.type)), a["value"], dtype=dt, device=self.device)
        if n.op == "tl.iota":
            d = next(x for x in _dims(n.type) if x.name == a["name"])
            ramp = torch.arange(d.start, d.stop, dtype=torch.int64, device=self.device)
            return _broadcast(ramp, (d,), _dims(n.type))
        if n.op == "tl.pointwise":
            return self._pointwise(n, a, ev)
        if n.op == "tl.reduce":
            return self._reduce(n, a, ev)
        if n.op == "tl.scan":
            return self._scan(n, a, ev)
        if n.op in ("tl.repeat_like", "tl.repeat"):
            return _broadcast(ev(n.args[0]), _dims(n.args[0].type), _dims(n.type))
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
            return _align(ev(n.args[0]), _dims(n.args[0].type), _dims(n.type)).contiguous()
        if n.op == "tl.slice":
            return self._slice(n, ev)
        if n.op == "tl.select":
            return self._select(n, a, ev)
        if n.op == "tl.pad":
            return self._pad(n, a, ev)
        if n.op == "tl.flip":
            src = _dims(n.args[0].type)
            return torch.flip(ev(n.args[0]), (next(i for i, d in enumerate(src) if d.name == a["name"]),))
        if n.op == "tl.split":
            return self._split(n, a, ev)
        if n.op == "tl.merge":
            return self._merge(n, a, ev)
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
        out = _dims(n.type)
        vals = []
        for arg in n.args:
            v = ev(arg)
            if isinstance(v, torch.Tensor):
                v = _align(v, _dims(arg.type), out)
            else:  # scalars broadcast at real carrier (run_region's law)
                v = torch.full(_sizes(out), float(v), dtype=self.dtype, device=self.device)
            vals.append(v)
        f = pw_marker(a["f"])
        if isinstance(f, CompositeMarker):
            r = _eval_tree(f.body, vals, self.pw)
            r = r if isinstance(r, torch.Tensor) else torch.full(_sizes(out), float(r), dtype=self.dtype)
            return r.expand(_sizes(out)) if r.shape != tuple(_sizes(out)) else r
        return self.pw[f.name](*vals)

    def _reduce(self, n, a, ev):
        f = reducer(a["f"])
        dims = (a["dims"],) if isinstance(a["dims"], str) else tuple(a["dims"])
        if isinstance(f, CompositeReducer):
            (dim,) = dims  # composites fold one dim at a time (compute.py's law)
            arrs, src = self._composite_elems(n, ev)
            axis = _names(src).index(dim)
            state = self._composite_sweep(f, arrs, axis, last_only=True)
            out = _eval_tree(f.project, state, self.pw)
            survivors = tuple(d for d in src if d.name != dim)
            return _align(out, survivors, _dims(n.type))
        if a.get("zero") is not None:
            raise Untranslatable("tl.reduce zero= override")
        src = _dims(n.args[0].type)
        axes = tuple(i for i, d in enumerate(src) if d.name in dims)
        arr = ev(n.args[0])
        if f.name == "mean":
            arr = arr.to(self.dtype) if not arr.is_floating_point() else arr
            out = arr.mean(dim=axes)
        elif f.name == "prod":
            out = arr
            for ax in sorted(axes, reverse=True):
                out = torch.prod(out, dim=ax)
        else:
            out = _PLAIN_RED[f.name](arr, dim=axes)
        survivors = tuple(d for d in src if d.name not in dims)
        return _align(out, survivors, _dims(n.type))

    def _scan(self, n, a, ev):
        f = reducer(a["f"])
        dim = a["dim"]
        if isinstance(f, CompositeReducer):
            arrs, src = self._composite_elems(n, ev)
            axis = _names(src).index(dim)
            out = self._composite_sweep(f, arrs, axis, last_only=False)
            return _align(out, src, _dims(n.type))
        if a.get("zero") is not None:
            raise Untranslatable("tl.scan zero= override")
        src = _dims(n.args[0].type)
        axis = _names(src).index(dim)
        arr = ev(n.args[0])
        if f.name == "sum":
            out = torch.cumsum(arr, dim=axis)
        elif f.name == "prod":
            out = torch.cumprod(arr, dim=axis)
        elif f.name in ("max", "min"):
            out = (torch.cummax if f.name == "max" else torch.cummin)(arr, dim=axis).values
        elif f.name == "mean":
            counts = torch.arange(1, arr.shape[axis] + 1, dtype=self.dtype, device=self.device)
            shape = [1] * arr.ndim
            shape[axis] = -1
            out = torch.cumsum(arr.to(self.dtype), dim=axis) / counts.reshape(shape)
        else:
            raise Untranslatable(f"tl.scan f={f.name}")
        return _align(out, src, _dims(n.type))

    def _composite_elems(self, n, ev):
        """Element tensors of a composite reduce/scan, aligned to arg 0."""
        src = _dims(n.args[0].type)
        arrs = [_align(ev(arg), _dims(arg.type), src).to(self.dtype) for arg in n.args]
        return arrs, src

    def _composite_sweep(self, f, arrs, axis, *, last_only):
        """compute._composite_sweep over torch: lift/combine sequentially."""
        moved = [torch.movedim(x, axis, 0) for x in arrs]
        steps = moved[0].shape[0]
        state, outs = None, []
        for t in range(steps):
            elem = [x[t] for x in moved]
            lifted = [_eval_tree(node, elem, self.pw) for node in f.lift]
            state = lifted if state is None else [_eval_tree(node, state + lifted, self.pw) for node in f.combine]
            if not last_only:
                outs.append(_eval_tree(f.project, state, self.pw))
        if last_only:
            if state is None:  # empty dim: the declared identity state
                shape = tuple(moved[0].shape[1:])
                state = [torch.full(shape, float(v), dtype=self.dtype, device=self.device) for v in f.init]
            return state
        return torch.movedim(torch.stack(outs, dim=0), 0, axis)

    def _slice(self, n, ev):
        out, src = _dims(n.type), _dims(n.args[0].type)
        data = ev(n.args[0])
        for i, (d, s) in enumerate(zip(out, src)):
            if (d.start, d.stop) != (s.start, s.stop):
                data = data.narrow(i, d.start - s.start, d.stop - d.start)
        return data

    def _select(self, n, a, ev):
        src = _dims(n.args[0].type)
        data = ev(n.args[0])
        for name, q in sorted(a["coords"].items(), key=lambda kv: -_names(src).index(kv[0])):
            i = _names(src).index(name)
            data = data.select(i, int(q) - src[i].start)
        survivors = tuple(d for d in src if d.name not in a["coords"])
        return _align(data, survivors, _dims(n.type))

    def _pad(self, n, a, ev):
        out, src = _dims(n.type), _dims(n.args[0].type)
        data = ev(n.args[0])
        canvas = torch.full(_sizes(out), float(a["fill"]), dtype=data.dtype, device=self.device)
        idx = tuple(slice(s.start - d.start, s.stop - d.start) for d, s in zip(out, src))
        canvas[idx] = data
        return canvas

    def _split(self, n, a, ev):
        out, src = _dims(n.type), _dims(n.args[0].type)
        parts = set(a["parts"]) if not isinstance(a["parts"], dict) else set(a["parts"])
        # src presentation with the parts run collapsed back to the split dim
        expect = tuple(dict.fromkeys(a["name"] if d.name in parts else d.name for d in out))
        data = ev(n.args[0]).permute(tuple(_names(src).index(x) for x in expect))
        return data.reshape(_sizes(out))

    def _merge(self, n, a, ev):
        out, src = _dims(n.type), _dims(n.args[0].type)
        parts = tuple(a["parts"])
        # out presentation with the merged dim expanded back into the parts run
        expect = []
        for d in out:
            if d.name == a["name"]:
                expect.extend(parts)
            else:
                expect.append(d.name)
        data = ev(n.args[0]).permute(tuple(_names(src).index(x) for x in expect))
        return data.reshape(_sizes(out))

    def _take(self, n, a, ev):
        """indexing.take's semantics: idx dims new to the table SPLICE in place
        of the taken dim; idx dims the table carries ALIGN by name."""
        table, idx = n.args
        tdims, idims, out = _dims(table.type), _dims(idx.type), _dims(n.type)
        d = next(x for x in tdims if x.name == a["dim"])
        survivors = {x.name for x in tdims if x.name != a["dim"]}
        aligned = tuple(x for x in idims if x.name in survivors)
        spliced = tuple(x for x in idims if x.name not in survivors)
        rest = tuple(x for x in tdims if x.name != a["dim"] and x.name not in _names(aligned))
        iarr = _align(ev(idx), idims, aligned + spliced).to(torch.long) - d.start
        arr = _align(ev(table), tdims, aligned + (d,) + rest)
        na = int(np.prod(_sizes(aligned))) if aligned else 1
        ns = int(np.prod(_sizes(spliced))) if spliced else 1
        flat = arr.reshape((na, d.stop - d.start) + tuple(_sizes(rest)))
        picks = flat[torch.arange(na, device=self.device)[:, None], iarr.reshape(na, ns)]
        got = picks.reshape(_sizes(aligned) + _sizes(spliced) + tuple(_sizes(rest)))
        return _align(got, aligned + spliced + rest, out)

    def _scatter_add(self, n, a, ev):
        """indexing.scatter_add: consumed (over=) dims collapse into the
        DECLARED output dim; duplicates sum (index_put_ accumulate)."""
        values, idx = n.args
        vdims, idims, out = _dims(values.type), _dims(idx.type), _dims(n.type)
        start, stop = (a["extent"] if isinstance(a["extent"], tuple) else (0, a["extent"]))
        over = a.get("over")
        consumed = tuple(_names(idims)) if over is None else ((over,) if isinstance(over, str) else tuple(over))
        aligned = tuple(x for x in idims if x.name not in consumed)
        cons = tuple(x for x in idims if x.name in consumed)
        rest = tuple(x for x in vdims if x.name not in _names(idims))
        iarr = _align(ev(idx), idims, aligned + cons).to(torch.long) - start
        varr = _align(ev(values), vdims, aligned + cons + rest)
        na = int(np.prod(_sizes(aligned))) if aligned else 1
        nc = int(np.prod(_sizes(cons))) if cons else 1
        flat_v = varr.reshape((na, nc) + tuple(_sizes(rest)))
        acc = torch.zeros((na, stop - start) + tuple(_sizes(rest)), dtype=flat_v.dtype, device=self.device)
        rows = torch.arange(na, device=self.device)[:, None].expand(na, nc)
        acc.index_put_((rows, iarr.reshape(na, nc)), flat_v, accumulate=True)
        new = next(x for x in out if x.name == a["dim"])
        got = acc.reshape(_sizes(aligned) + (stop - start,) + tuple(_sizes(rest)))
        return _align(got, aligned + (new,) + rest, out)

    def _argranks(self, n, a, ev):
        """argtopk (descending, ties first-wins) / argsort (ascending, stable):
        values are lattice positions — the source dim's start rides out."""
        src, out = _dims(n.args[0].type), _dims(n.type)
        d = next(x for x in src if x.name == a["dim"])
        rest = tuple(x for x in src if x.name != a["dim"])
        arr = _align(ev(n.args[0]), src, rest + (d,))
        arr = arr.to(self.dtype) if not arr.is_floating_point() else arr
        if n.op == "tl.argtopk":
            order = torch.argsort(-arr, dim=-1, stable=True)[..., : a["k"]] + d.start
            tail = next(x for x in out if x.name == a["k_name"])
        else:
            order = torch.argsort(arr, dim=-1, stable=True) + d.start
            tail = next(x for x in out if x.name == a["dim"])
        return _align(order, rest + (tail,), out)

    def _fold(self, n, a, ev):
        """run_region's fold row: carry k states positionally, select element
        slices at ABSOLUTE coordinates, finish (final) or stack (emit)."""
        k, m = len(tuple(a["state"])), len(tuple(a["element"]))
        dim, out = a["dim"], tuple(a["out"])
        vals = [ev(x) for x in n.args]
        carried, srcs = list(vals[:k]), vals[k:]
        cdims = [_dims(x.type) for x in n.args[:k]]
        sdims = [_dims(x.type) for x in n.args[k:]]
        if m:
            lo, hi = next((d.start, d.stop) for d in sdims[0] if d.name == dim)
        else:
            lo, hi = a["extent"]
        step = n.regions[0]
        pdims = [_dims(p.type) for p in step.params]
        emitted = []
        for q in range(lo, hi):
            elems = []
            for s, sd in zip(srcs, sdims):
                i = _names(sd).index(dim)
                elems.append((s.select(i, q - sd[i].start), tuple(d for d in sd if d.name != dim)))
            bound = [
                _align(v, vd, pd)
                for (v, vd), pd in zip([(c, cd) for c, cd in zip(carried, cdims)] + elems, pdims)
            ]
            res = self.run(step, bound)
            res = res if isinstance(res, tuple) else (res,)
            ydims = self._yield_dims(step)
            carried = [_align(r, yd, cd) for r, yd, cd in zip(res[:k], ydims, cdims)]
            if out[0] == "emit":
                emitted.append((res[k], ydims[k]))
        if out[0] == "final":
            return _align(carried[out[1]], cdims[out[1]], _dims(n.type))
        stacked = torch.stack([_align(e, ed, emitted[0][1]) for e, ed in emitted], dim=0)
        lead = next(d for d in _dims(n.type) if d.name == dim)
        return _align(stacked, (lead,) + tuple(emitted[0][1]), _dims(n.type))

    @staticmethod
    def _yield_dims(step: Region) -> list:
        yielded = step.body[-1].args[0]
        outs = tuple(yielded.args) if yielded.op == "core.tuple" else (yielded,)
        return [_dims(x.type) for x in outs]


def run_region_torch(region: Region, values: list, *, device="cpu", dtype=torch.float64):
    """Evaluate a tensor-tier region over torch — ``values`` positional per
    param, each array in ITS PARAM's type-dim order (numpy or torch)."""
    e = _Eval(torch.device(device), dtype)
    return e.run(region, [e.as_input(v) for v in values])


def run_named_torch(region: Region, inputs: dict, names, *, device="cpu", dtype=torch.float64) -> dict:
    """dialect.run_named's door on the torch substrate: params bind by their
    claimed names (arrays in the param's type-dim order), the yield's slots
    report under theirs — as ``TorchField``s carrying their axis names."""
    order = [names[id(p)] for p in region.params]
    missing = [kk for kk in order if kk not in inputs]
    if missing:
        raise KeyError(f"missing input {missing[0]!r}")
    res = run_region_torch(region, [inputs[kk] for kk in order], device=device, dtype=dtype)
    yielded = region.body[-1].args[0]
    outs = tuple(yielded.args) if yielded.op == "core.tuple" else (yielded,)
    res = res if isinstance(res, tuple) else (res,)
    return {
        names[id(x)]: TorchField(v, _names(_dims(x.type)))
        for x, v in zip(outs, res)
    }
