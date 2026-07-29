"""The Region -> Triton translator — the tile tier's CHECK column (320 §8).

Translates a TILE-TIER region (one per-tile body, the binding law's frame)
into one Triton program: params become pointers, every value is a block
whose axes are its type dims, and the whole kernel launches on a (1,)
grid — the K-A/launch machinery owns real grids later.

The mechanism is REQUEST-DRIVEN index emission, the layout algebra doing
the addressing it always claimed to do:

- pure layout chains (slice/shift/split/stage over a param) never
  materialize — they compose COORDINATE EXPRESSIONS, and the load happens
  once, at the leaf, with the composed pointer math and a mask;
- `tl.iota` IS its coordinate expression (closed forms stay free), so
  masks built from iota comparisons translate to inline integer
  arithmetic — including inside fold steps, where the scan coordinate is
  the loop variable;
- the tile-fold becomes an in-kernel `for` loop with carried block
  variables; element sources re-emit per iteration with the scan dim
  bound to the loop index (the absolute-coordinate law, verbatim);
- non-power-of-two extents pad to the next power of two: loads carry
  masks with `other=0.0`, every reduction identity-fills its padded
  lanes first, and the final store masks — padding is never observable.

We take Triton's block semantics and refuse its pointer STYLE: no
program computes a base offset; params arrive as tiles through the
binding law. Untranslated ops raise Untranslatable naming the op
(wgsl_executor's law). Generated source goes through a real file —
triton's JIT reads source via inspect, so exec() strings cannot carry it
(the NoSourceError lesson, recorded).
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

from pdum.dsl.ir import Region
from pdum.tl.dialect import _thaw_params
from region_evaluator import Untranslatable

# the executor-fingerprint seam (283's Pair seat)
TRITON_FP = ("tile_triton", "triton")

_IDENT = {"sum": "0.0", "max": "float('-inf')", "min": "float('inf')", "prod": "1.0"}
_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/", "le": "<=", "lt": "<", "ge": ">=", "gt": ">"}
_CALLS = {"exp": "tl.exp", "sqrt": "tl.sqrt", "log": "tl.log", "sin": "tl.sin", "cos": "tl.cos"}


def _pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _dims(t):
    d = getattr(t, "dims", None)
    if d is None:
        raise Untranslatable(f"non-tensor type {t!r}")
    return d


class _Coord:
    """One dim's coordinate in the current emission space: a full-rank
    integer block expression (or a scalar, e.g. a fold's loop variable),
    its validity mask, and whether it is the dim's canonical grid."""

    def __init__(self, expr: str, valid: str | None, canonical: bool):
        self.expr, self.valid, self.canonical = expr, valid, canonical


def _suffix(rank: int, axis: int) -> str:
    return "[" + ", ".join(":" if i == axis else "None" for i in range(rank)) + "]"


def _canonical_space(axes) -> dict:
    """Coordinate/validity grids for an ordered dim tuple: absolute coords
    (start + arange), broadcast to full rank by None-indexing."""
    rank = len(axes)
    space = {}
    for i, d in enumerate(axes):
        size, p = d.stop - d.start, _pow2(d.stop - d.start)
        sfx = _suffix(rank, i) if rank > 1 else ""
        expr = f"({d.start} + tl.arange(0, {p})){sfx}"
        valid = f"(tl.arange(0, {p}) < {size}){sfx}" if p != size else None
        space[d.name] = _Coord(expr, valid, True)
    return space


class _Gen:
    def __init__(self, region: Region):
        self.region = region
        self.lines: list[str] = []
        self.indent = 1
        self.n = 0
        self.bind: dict[int, tuple[str, tuple]] = {}  # id(step param) -> (var, dims)
        self.blockmemo: dict[int, tuple[str, tuple]] = {}
        self.pstrides: dict[int, tuple] = {}  # id(region param) -> element strides
        for i, p in enumerate(region.params):
            sizes = tuple(d.stop - d.start for d in _dims(p.type))
            strides = []
            acc = 1
            for s in reversed(sizes):
                strides.append(acc)
                acc *= s
            self.pstrides[id(p)] = tuple(reversed(strides))

    def v(self) -> str:
        self.n += 1
        return f"v{self.n}"

    def ln(self, s: str) -> None:
        self.lines.append("    " * self.indent + s)

    # --- the two emission modes ------------------------------------------------

    def emit_at(self, n, axes, space) -> str:
        """The value of ``n`` at the space's coordinate grids — an expression
        over full-rank blocks. Layout ops transform coordinates; leaves load
        or generate; materialized sub-blocks align by None-expansion."""
        a = _thaw_params(dict(n.attrs))
        if id(n) in self.bind:
            var, vdims = self.bind[id(n)]
            return self._use(var, vdims, axes, space)
        if n.op == "core.const":
            return repr(float(a["value"]))
        if n.op == "tl.const":
            return repr(float(a["value"]))
        if n.op == "tl.iota":
            return space[a["name"]].expr
        if n.op in ("tl.stage", "tl.materialize", "tl.slice", "tl.simplify", "tl.with_charts", "tl.strip_charts"):
            return self.emit_at(n.args[0], axes, space)
        if n.op in ("tl.repeat_like", "tl.repeat"):
            return self.emit_at(n.args[0], axes, space)  # full-rank grids broadcast for free
        if n.op == "tl.rename":
            sub = dict(space)
            for old, new in a["mapping"].items():
                sub[old] = space[new]
            return self.emit_at(n.args[0], axes, sub)
        if n.op == "tl.shift":
            sub = dict(space)
            for name, delta in a["deltas"].items():
                c = space[name]
                sub[name] = _Coord(f"({c.expr} - {delta})", c.valid, False)
            return self.emit_at(n.args[0], axes, sub)
        if n.op == "tl.split":
            parts = tuple(a["parts"].items()) if isinstance(a["parts"], dict) else tuple(a["parts"])
            (po, _eo), (pi, ei) = parts  # two parts: outer, inner (the flagships' shape)
            co, ci = space[po], space[pi]
            valid = " & ".join(x for x in (co.valid, ci.valid) if x) or None
            sub = dict(space)
            sub[a["name"]] = _Coord(f"(({co.expr}) * {ei} + ({ci.expr}))", valid, False)
            return self.emit_at(n.args[0], axes, sub)
        if n.op == "tl.pointwise":
            args = [self.emit_at(x, axes, space) for x in n.args]
            f = a["f"]
            if f in _INFIX:
                return f"({args[0]} {_INFIX[f]} {args[1]})"
            if f in _CALLS:
                return f"{_CALLS[f]}({args[0]})"
            if f == "maximum":
                return f"tl.maximum({args[0]}, {args[1]})"
            if f == "minimum":
                return f"tl.minimum({args[0]}, {args[1]})"
            if f == "where":
                return f"tl.where({args[0]}, {args[1]}, {args[2]})"
            if f == "neg":
                return f"(-{args[0]})"
            raise Untranslatable(f"tl.pointwise f={f}")
        if n.op == "core.param":
            return self._load(n, space)
        if n.op in ("tl.reduce", "tl.fold"):
            var, vdims = self.emit_block(n)
            return self._use(var, vdims, axes, space)
        raise Untranslatable(n.op)

    def _use(self, var: str, vdims, axes, space) -> str:
        """A materialized block consumed inside a space: its dims must sit at
        canonical grids, in the same relative order — alignment is pure
        None-expansion (the emitter keeps orders consistent by type law)."""
        names = [d.name for d in vdims]
        axnames = [d.name for d in axes]
        pos = []
        for nm in names:
            if nm not in axnames or not space[nm].canonical:
                raise Untranslatable(f"block {var} consumed at non-canonical coordinate {nm!r}")
            pos.append(axnames.index(nm))
        if pos != sorted(pos):  # permute the block's axes into the space's order
            perm = tuple(sorted(range(len(pos)), key=lambda i: pos[i]))
            tv = self.v()
            self.ln(f"{tv} = tl.trans({var}, {perm})")
            var, pos = tv, sorted(pos)
        if len(axes) <= 1 or len(names) == len(axes):
            return var
        sfx = "[" + ", ".join(":" if i in pos else "None" for i in range(len(axes))) + "]"
        return f"{var}{sfx}"

    def _load(self, p, space) -> str:
        idx = self.region.params.index(p)
        strides = self.pstrides[id(p)]
        dims = _dims(p.type)
        # buffers are origin-at-domain-start: offsets are RELATIVE coords;
        # guards (below) stay absolute — the domain law splits exactly here
        off = " + ".join(f"(({space[d.name].expr}) - {d.start}) * {s}" for d, s in zip(dims, strides))
        guards = []
        for d in dims:
            c = space[d.name]
            if c.valid:
                guards.append(c.valid)
            if not c.canonical:  # composed coordinates get the domain guard
                guards.append(f"(({c.expr}) >= {d.start}) & (({c.expr}) < {d.stop})")
        mask = " & ".join(dict.fromkeys(guards)) if guards else None
        var = self.v()
        if mask:
            self.ln(f"{var} = tl.load(p{idx} + {off}, mask={mask}, other=0.0)")
        else:
            self.ln(f"{var} = tl.load(p{idx} + {off})")
        return var

    def emit_block(self, n) -> tuple[str, tuple]:
        """Materialize ``n`` as a block variable with axes = its type dims."""
        if id(n) in self.blockmemo:
            return self.blockmemo[id(n)]
        a = _thaw_params(dict(n.attrs))
        if id(n) in self.bind:
            out = self.bind[id(n)]
        elif n.op == "tl.const":
            dims = _dims(n.type)
            shape = ", ".join(str(_pow2(d.stop - d.start)) for d in dims)
            var = self.v()
            self.ln(f"{var} = tl.full(({shape},), {float(a['value'])!r}, tl.float32)")
            out = (var, dims)
        elif n.op == "tl.reduce":
            out = self._block_reduce(n, a)
        elif n.op == "tl.fold":
            out = self._block_fold(n, a)
        else:
            dims = _dims(n.type)
            space = _canonical_space(dims)
            var = self.v()
            self.ln(f"{var} = {self.emit_at(n, dims, space)}")
            out = (var, dims)
        self.blockmemo[id(n)] = out
        return out

    def _block_reduce(self, n, a):
        f = a["f"]
        if f not in _IDENT or a.get("zero") is not None:
            raise Untranslatable(f"tl.reduce f={f}")
        src = _dims(n.args[0].type)
        rdims = (a["dims"],) if isinstance(a["dims"], str) else tuple(a["dims"])
        space = _canonical_space(src)
        var = self.v()
        self.ln(f"{var} = {self.emit_at(n.args[0], src, space)}")
        for d in src:  # identity-fill the padded lanes before folding them in
            if d.name in rdims and space[d.name].valid:
                self.ln(f"{var} = tl.where({space[d.name].valid}, {var}, {_IDENT[f]})")
        call = {"sum": "tl.sum", "max": "tl.max", "min": "tl.min", "prod": "tl.prod"}[f]
        for i in sorted((i for i, d in enumerate(src) if d.name in rdims), reverse=True):
            self.ln(f"{var} = {call}({var}, axis={i})")
        return var, _dims(n.type)

    def _block_fold(self, n, a):
        k = len(tuple(a["state"]))
        dim, out = a["dim"], tuple(a["out"])
        if out[0] != "final":
            raise Untranslatable("tl.fold out=('emit',)")
        step = n.regions[0]
        inits, srcs = n.args[:k], n.args[k:]
        lo, hi = next((d.start, d.stop) for d in _dims(srcs[0].type) if d.name == dim)
        carried = [self.emit_block(x) for x in inits]
        cvars = []
        for var, dims in carried:  # loop-carried variables need their own names
            cv = self.v()
            self.ln(f"{cv} = {var}")
            cvars.append((cv, dims))
        q = self.v()
        self.ln(f"for {q} in range({lo}, {hi}):")
        self.indent += 1
        outer_memo, outer_bind = self.blockmemo, dict(self.bind)
        self.blockmemo = {}
        elems = []
        for src in srcs:
            edims = tuple(d for d in _dims(src.type) if d.name != dim)
            space = _canonical_space(edims)
            space[dim] = _Coord(q, None, False)
            ev = self.v()
            self.ln(f"{ev} = {self.emit_at(src, edims, space)}")
            elems.append((ev, edims))
        for p, bound in zip(step.params, cvars + elems):
            self.bind[id(p)] = bound
        yld = step.body[-1].args[0]
        outs = tuple(yld.args) if yld.op == "core.tuple" else (yld,)
        news = [self.emit_block(x) for x in outs[:k]]
        for (cv, cdims), (nv, ndims) in zip(cvars, news):
            if tuple(d.name for d in cdims) != tuple(d.name for d in ndims):
                raise Untranslatable("fold carry reorders its dims — not spelled yet")
            self.ln(f"{cv} = {nv}")
        self.indent -= 1
        self.blockmemo, self.bind = outer_memo, outer_bind
        return cvars[out[1]]

    # --- the kernel ------------------------------------------------------------

    def render(self) -> tuple[str, tuple]:
        yld = self.region.body[-1].args[0]
        if yld.op == "core.tuple":
            raise Untranslatable("multi-output tile kernels")
        var, dims = self.emit_block(yld)
        sizes = tuple(d.stop - d.start for d in dims)
        strides = []
        acc = 1
        for s in reversed(sizes):
            strides.append(acc)
            acc *= s
        strides = tuple(reversed(strides))
        space = _canonical_space(dims)
        off = " + ".join(f"(({space[d.name].expr}) - {d.start}) * {s}" for d, s in zip(dims, strides))
        mask = " & ".join(c.valid for c in (space[d.name] for d in dims) if c.valid)
        store = f"tl.store(out + {off}, {var}" + (f", mask={mask})" if mask else ")")
        self.ln(store)
        args = ", ".join([f"p{i}" for i in range(len(self.region.params))] + ["out"])
        head = ["import triton", "import triton.language as tl", "", "@triton.jit", f"def tile_kernel({args}):"]
        return "\n".join(head + self.lines) + "\n", sizes


def compile_tile(region: Region):
    """Region -> a runner: run(values) -> np.ndarray (f32, CUDA). ``values``
    are numpy arrays positional per param, in the param's type-dim order."""
    src, out_sizes = _Gen(region).render(), None
    src, out_sizes = src[0], src[1]
    tmp = Path(tempfile.mkdtemp(prefix="pdum_triton_")) / "tile_kernel.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location(f"pdum_triton_{tmp.parent.name}", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    def run(values) -> np.ndarray:
        import torch

        ins = [torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32).cuda() for v in values]
        out = torch.empty(out_sizes, dtype=torch.float32, device="cuda")
        mod.tile_kernel[(1,)](*ins, out)
        torch.cuda.synchronize()
        return out.cpu().numpy()

    run.source = src
    return run
