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
  lanes first, and the final store masks — padding is never observable;
- a sum over one shared dim of a two-operand product emits
  `tl.dot(..., input_precision="ieee")` when every block dim reaches 16:
  triton's TTIR combine pass otherwise rewrites the raw mul+sum into
  TF32 tensor-core MMA — a silent 2^-11 demotion (measured 5.7e-3 on
  flash scores). Below 16 neither fires, so both paths stay ieee, and
  the runner refuses outright if `.tf32` ever reaches the PTX.

We take Triton's block semantics and refuse its pointer STYLE: no
program computes a base offset; params arrive as tiles through the
binding law. Untranslated ops raise Untranslatable naming the op
(wgsl_executor's law). Generated source goes through a real file —
triton's JIT reads source via inspect, so exec() strings cannot carry it
(the NoSourceError lesson, recorded).
"""

from __future__ import annotations

import importlib.util
import re
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


def _all_nodes(region):
    seen = set()
    stack = list(region.body)
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        yield n
        stack.extend(n.args)
        for r in n.regions:
            stack.extend(r.body)


class _Gen:
    def __init__(self, region: Region, launch: tuple = (), prune: tuple = ()):
        self.region = region
        self.launch = dict(launch)  # output dim name -> tile extent (340 §2)
        self.prune = {e[0]: (e[1], e[2]) for e in prune}  # scan dim -> affine [lo, hi) in pid
        self.pidv: dict[str, str] = {}  # launched dim name -> pid coordinate var
        self.lines: list[str] = []
        self.indent = 1
        self.n = 0
        self.bind: dict[int, tuple[str, tuple]] = {}  # id(step param) -> (var, dims)
        self.blockmemo: dict[int, tuple[str, tuple]] = {}
        self.foldmemo: dict[tuple, list] = {}  # (step, args, dim) -> carried vars
        self.pstrides: dict[int, tuple] = {}  # id(region param) -> element strides
        for i, p in enumerate(region.params):
            sizes = tuple(d.stop - d.start for d in _dims(p.type))
            strides = []
            acc = 1
            for s in reversed(sizes):
                strides.append(acc)
                acc *= s
            self.pstrides[id(p)] = tuple(reversed(strides))

    def psize(self, d) -> int:
        """A dim's block extent: its launch tile when gridded, else its size."""
        return _pow2(self.launch.get(d.name, d.stop - d.start))

    def space(self, axes) -> dict:
        """Coordinate/validity grids for an ordered dim tuple: absolute coords
        (start + arange), broadcast to full rank by None-indexing. Launched
        dims shift their start by the program's tile coordinate — the one
        place tile coordinates become addresses (340 §2) — and guard the
        ragged tail absolutely."""
        rank = len(axes)
        space = {}
        for i, d in enumerate(axes):
            size = d.stop - d.start
            sfx = _suffix(rank, i) if rank > 1 else ""
            tile = self.launch.get(d.name)
            if tile is None:
                p = _pow2(size)
                expr = f"({d.start} + tl.arange(0, {p})){sfx}"
                valid = f"(tl.arange(0, {p}) < {size}){sfx}" if p != size else None
            else:
                p = _pow2(tile)
                base = f"{self.pidv[d.name]} * {tile}"
                expr = f"({d.start} + {base} + tl.arange(0, {p})){sfx}"
                covered = p == tile and size % tile == 0
                valid = None if covered else f"(({base} + tl.arange(0, {p})) < {size}){sfx}"
            space[d.name] = _Coord(expr, valid, True)
        return space

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
            f = a["f"]
            if self._apply_f(f, ["", "", ""]) is not None:
                args = [self.emit_at(x, axes, space) for x in n.args]
                return self._apply_f(f, args)
            try:  # autodiff's f.dN slopes are composite markers: expression
                from pdum.tl.markers import MARKERS  # trees over Arg leaves (§7.8);

                cm = MARKERS[f]  # LAZY arg emission — a slope that drops an
            except KeyError:  # operand must not emit its (dead) loads
                cm = None
            if cm is not None and hasattr(cm, "body"):
                lazy = {}

                def arg(i):
                    if i not in lazy:
                        lazy[i] = self.emit_at(n.args[i], axes, space)
                    return lazy[i]

                return self._marker_expr(cm.body, arg)
            raise Untranslatable(f"tl.pointwise f={f}")
        if n.op == "core.param":
            return self._load(n, space)
        if n.op == "tl.reduce" and any(
            d.name in space and not space[d.name].canonical for d in _dims(n.type)
        ):
            return self._reduce_at(n, axes, space)
        if n.op in ("tl.reduce", "tl.fold"):
            var, vdims = self.emit_block(n)
            return self._use(var, vdims, axes, space)
        raise Untranslatable(n.op)

    def _dot_at(self, n, axes, space, rdims, src, sub, axes2, guards) -> str | None:
        """The dot fast path AT ambient coordinates (§7.8's board): a
        two-operand product-sum whose cores are 2D re-emits per tile as
        an ieee tl.dot — no rank-3 intermediate, the tensor-core form a
        hand FA kernel writes. Kept tile widths are read off the ambient
        coordinates' own arange; anything narrower than the MMA floor
        (or shaped otherwise) falls back to mul+sum."""
        kept = tuple(_dims(n.type))  # the reduce's OWN kept dims — the ambient
        if len(rdims) != 1 or len(axes) != 2 or len(kept) != 2:  # space maps them
            return None  # through splits (t rides as a composed ki coordinate)
        r = rdims[0]
        mul = n.args[0]
        while mul.op in ("tl.with_charts", "tl.strip_charts", "tl.simplify"):
            mul = mul.args[0]
        if mul.op != "tl.pointwise" or len(mul.args) != 2:
            return None
        if _thaw_params(dict(mul.attrs)).get("f") != "mul":
            return None
        cores = []
        for x in mul.args:
            while x.op in ("tl.with_charts", "tl.strip_charts", "tl.simplify", "tl.repeat_like", "tl.repeat"):
                x = x.args[0]
            cores.append(x)
        named = {}
        for x, c in zip(mul.args, cores):
            d = _dims(c.type)
            keep = tuple(k for k in d if k.name != r)
            if len(d) != 2 or len(keep) != 1 or keep[0].name in named:
                return None
            named[keep[0].name] = x  # emit the ORIGINAL arg: views ride for free
        if set(named) != {kept[0].name, kept[1].name}:
            return None
        rdim = next(x for x in src if x.name == r)
        rp = _pow2(rdim.stop - rdim.start)
        if rp < 16:
            return None
        widths = []
        for ax in kept:  # the tile width is literal in the ambient coordinate
            c = space.get(ax.name)
            ws = re.findall(r"tl\.arange\(0, (\d+)\)", c.expr) if c is not None else []
            if len(ws) != 1 or int(ws[0]) < 16:
                return None
            widths.append(int(ws[0]))
        halves = []
        for pos, ax in enumerate(kept):
            v = self.v()
            self.ln(f"{v} = {self.emit_at(named[ax.name], tuple(axes2), sub)}")
            for g in guards:  # zero the padded contraction lanes: 0*0 contributes 0
                self.ln(f"{v} = tl.where({g}, {v}, 0.0)")
            v2 = self.v()  # squeeze the broadcast axis: (W, 1, R) / (1, W, R) -> (W, R)
            self.ln(f"{v2} = tl.reshape({v}, ({widths[pos]}, {rp}))")
            halves.append(v2)
        va, vb = halves
        tb = self.v()
        self.ln(f"{tb} = tl.trans({vb})")
        out = self.v()
        self.ln(f'{out} = tl.dot({va}, {tb}, input_precision="ieee")')
        return out

    def _apply_f(self, f: str, args) -> str | None:
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
        return None

    def _marker_expr(self, x, arg) -> str:
        """A composite marker's slope tree (autodiff's f.dN) as an
        expression — Arg leaves emit the pointwise's operands on demand."""
        if hasattr(x, "index"):  # Arg
            return arg(x.index)
        if hasattr(x, "op"):  # Prim
            sub = [self._marker_expr(a, arg) for a in x.args]
            e = self._apply_f(x.op, sub)
            if e is None:
                raise Untranslatable(f"marker prim {x.op!r}")
            return e
        if hasattr(x, "value"):  # Const
            return repr(float(x.value))
        raise Untranslatable(f"marker leaf {type(x).__name__}")

    def _reduce_at(self, n, axes, space) -> str:
        """A reduce evaluated AT ambient coordinates (330 §7.8): tile-local
        reductions travel into consumers, so a prologue's contraction is
        re-emitted inside the sweep — kept dims ride the ambient coords
        (rank-extended one axis per reduced dim), reduced dims get fresh
        trailing grids, identity-filled and folded flat."""
        a = _thaw_params(dict(n.attrs))
        f = a["f"]
        mean = f == "mean"
        if mean:
            f = "sum"
        if f not in _IDENT or a.get("zero") is not None:
            raise Untranslatable(f"tl.reduce f={a['f']} at composed coordinates")
        rdims = (a["dims"],) if isinstance(a["dims"], str) else tuple(a["dims"])
        src = _dims(n.args[0].type)
        R, E = len(axes), len(rdims)
        ext = "[" + ", ".join([":"] * R + ["None"] * E) + "]"

        def extend(c):
            if c is None:
                return c
            e = f"({c.expr}){ext}" if "arange" in c.expr else c.expr  # scalars broadcast
            v = f"({c.valid}){ext}" if c.valid and "arange" in c.valid else c.valid
            return _Coord(e, v, c.canonical)

        sub = {nm: extend(c) for nm, c in space.items()}
        axes2, guards = list(axes), []
        for j, nm in enumerate(rdims):
            d = next(x for x in src if x.name == nm)
            size = d.stop - d.start
            p = _pow2(size)
            sfx = _suffix(R + E, R + j)
            valid = f"(tl.arange(0, {p}) < {size}){sfx}" if p != size else None
            sub[nm] = _Coord(f"({d.start} + tl.arange(0, {p})){sfx}", valid, True)
            axes2.append(d)
            if valid:
                guards.append(valid)
        if not mean:
            dot = self._dot_at(n, axes, space, rdims, src, sub, axes2, guards)
            if dot is not None:
                return dot
        var = self.v()
        self.ln(f"{var} = {self.emit_at(n.args[0], tuple(axes2), sub)}")
        for g in guards:  # identity-fill the padded reduced lanes
            self.ln(f"{var} = tl.where({g}, {var}, {_IDENT[f]})")
        call = {"sum": "tl.sum", "max": "tl.max", "min": "tl.min", "prod": "tl.prod"}[f]
        for j in range(E - 1, -1, -1):
            self.ln(f"{var} = {call}({var}, axis={R + j})")
        if mean:
            count = 1
            for nm in rdims:
                d = next(x for x in src if x.name == nm)
                count *= d.stop - d.start
            self.ln(f"{var} = {var} / {count}")
        return var

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
            shape = ", ".join(str(self.psize(d)) for d in dims)
            var = self.v()
            self.ln(f"{var} = tl.full(({shape},), {float(a['value'])!r}, tl.float32)")
            out = (var, dims)
        elif n.op == "tl.reduce":
            out = self._block_reduce(n, a)
        elif n.op == "tl.fold":
            out = self._block_fold(n, a)
        else:
            dims = _dims(n.type)
            space = self.space(dims)
            var = self.v()
            self.ln(f"{var} = {self.emit_at(n, dims, space)}")
            out = (var, dims)
        self.blockmemo[id(n)] = out
        return out

    def _block_reduce(self, n, a):
        f = a["f"]
        mean = f == "mean"
        if mean:
            f = "sum"  # mean lowers as sum with a divide-by-N finalize (330 §7.6):
        if f not in _IDENT or a.get("zero") is not None:  # N static, one scalar op
            raise Untranslatable(f"tl.reduce f={f}")
        src = _dims(n.args[0].type)
        rdims = (a["dims"],) if isinstance(a["dims"], str) else tuple(a["dims"])
        dot = None if mean else self._contraction_dot(n, f, rdims)
        if dot is not None:
            return dot
        space = self.space(src)
        var = self.v()
        self.ln(f"{var} = {self.emit_at(n.args[0], src, space)}")
        for d in src:  # identity-fill the padded lanes before folding them in
            if d.name in rdims and space[d.name].valid:
                self.ln(f"{var} = tl.where({space[d.name].valid}, {var}, {_IDENT[f]})")
        call = {"sum": "tl.sum", "max": "tl.max", "min": "tl.min", "prod": "tl.prod"}[f]
        for i in sorted((i for i, d in enumerate(src) if d.name in rdims), reverse=True):
            self.ln(f"{var} = {call}({var}, axis={i})")
        if mean:
            count = 1
            for d in src:
                if d.name in rdims:
                    count *= d.stop - d.start
            self.ln(f"{var} = {var} / {count}")
        return var, _dims(n.type)

    def _contraction_dot(self, n, f, rdims):
        """Sum over one shared dim of a two-operand product, as an ieee dot
        (the docstring's precision law); anything stricter falls back."""
        if f != "sum" or len(rdims) != 1:
            return None
        r = rdims[0]
        mul = n.args[0]
        if mul.op != "tl.pointwise" or len(mul.args) != 2:
            return None
        if _thaw_params(dict(mul.attrs)).get("f") != "mul":
            return None
        out = _dims(n.type)
        if len(out) != 2 or len(_dims(mul.type)) != 3:
            return None
        if {d.name for d in _dims(mul.type)} != {out[0].name, out[1].name, r}:
            return None
        named = {}
        for c in (x.args[0] if x.op == "tl.repeat_like" else x for x in mul.args):
            d = _dims(c.type)
            keep = tuple(x for x in d if x.name != r)
            if len(d) != 2 or len(keep) != 1:
                return None
            named[keep[0].name] = (c, d)
        if set(named) != {out[0].name, out[1].name}:
            return None
        (ca, da), (cb, db) = named[out[0].name], named[out[1].name]
        ra = next(x for x in da if x.name == r)
        rb = next(x for x in db if x.name == r)
        if (ra.start, ra.stop) != (rb.start, rb.stop):
            return None
        if min(self.psize(d) for d in (out[0], out[1], ra)) < 16:
            return None
        va, da2 = self.emit_block(ca)
        vb, db2 = self.emit_block(cb)
        if da2[0].name == r:  # dot wants A as (kept, r), B as (r, kept)
            ta = self.v()
            self.ln(f"{ta} = tl.trans({va})")
            va = ta
        if db2[1].name == r:
            tb = self.v()
            self.ln(f"{tb} = tl.trans({vb})")
            vb = tb
        size, p = ra.stop - ra.start, _pow2(ra.stop - ra.start)
        if p != size:  # zero BOTH operands' padded lanes: 0*0 contributes 0, nan-safe
            ok = f"(tl.arange(0, {p}) < {size})"
            fa, fb = self.v(), self.v()
            self.ln(f"{fa} = tl.where({ok}[None, :], {va}, 0.0)")
            self.ln(f"{fb} = tl.where({ok}[:, None], {vb}, 0.0)")
            va, vb = fa, fb
        var = self.v()
        self.ln(f'{var} = tl.dot({va}, {vb}, input_precision="ieee")')
        return var, out

    def _fold_bounds(self, dim, lo, hi) -> tuple[str, str]:
        """Mask-derived bounds (340 §4b): the plan's per-program affine
        [lo, hi) clamps the sweep — a scalar tl.maximum/minimum on the pid,
        skipping tiles the template proved inert. No prune, no change."""
        pr = self.prune.get(dim)
        if pr is None:
            return str(lo), str(hi)
        (l0, dl, cl), (h0, dh, ch) = pr
        if len(self.pidv) == 1:
            pv = next(iter(self.pidv.values()))

            def form(a, d, c):
                e = f"{a} + {d} * {pv}"
                return f"({e}) // {c}" if c != 1 else f"{e}"

            lo_e = f"tl.maximum({lo}, {form(l0, dl, cl)})" if dl else str(max(lo, l0 // cl))
            hi_e = f"tl.minimum({hi}, {form(h0, dh, ch)})" if dh else str(min(hi, h0 // ch))
            return lo_e, hi_e
        if not self.pidv and dl == 0 and dh == 0:  # ungridded: constants only
            return str(max(lo, l0 // cl)), str(min(hi, h0 // ch))
        return str(lo), str(hi)

    def _block_fold(self, n, a):
        k = len(tuple(a["state"]))
        dim, out = a["dim"], tuple(a["out"])
        if out[0] != "final":
            raise Untranslatable("tl.fold out=('emit',)")
        step = n.regions[0]
        inits, srcs = n.args[:k], n.args[k:]
        # sibling folds sharing (step, args, dim) differ only in WHICH final
        # they surface — one loop carries all states, the sibling reads its
        # carry from the same sweep (flash's o and den finals: 340 §7.4
        # measured the second sweep as ~2x on the s-axis)
        key = (id(step), tuple(map(id, n.args)), dim)
        hit = self.foldmemo.get(key)
        if hit is not None:
            return hit[out[1]]
        lo, hi = next((d.start, d.stop) for d in _dims(srcs[0].type) if d.name == dim)
        lo_e, hi_e = self._fold_bounds(dim, lo, hi)
        carried = [self.emit_block(x) for x in inits]
        cvars = []
        for var, dims in carried:  # loop-carried variables need their own names
            cv = self.v()
            self.ln(f"{cv} = {var}")
            cvars.append((cv, dims))
        q = self.v()
        self.ln(f"for {q} in range({lo_e}, {hi_e}):")
        self.indent += 1
        outer_memo, outer_bind = self.blockmemo, dict(self.bind)
        self.blockmemo = {}
        elems = []
        for src in srcs:
            edims = tuple(d for d in _dims(src.type) if d.name != dim)
            space = self.space(edims)
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
        self.foldmemo[key] = cvars
        return cvars[out[1]]

    # --- the kernel ------------------------------------------------------------

    def _prelude(self, out_dims) -> int:
        """The pid decomposition (340 §2): launch dims in output order,
        row-major, program count DERIVED as a product of cdivs. Legality is
        checked here — a launched dim carried by a fold or consumed by a
        reduce is refused loudly (the partition law's precondition)."""
        names = {d.name for d in out_dims}
        carried = set()

        def scan(r):
            for n in _all_nodes(r):
                a = _thaw_params(dict(n.attrs))
                if n.op == "tl.fold":
                    carried.add(a["dim"])
                if n.op == "tl.reduce":
                    rd = a["dims"]
                    carried.update((rd,) if isinstance(rd, str) else rd)

        scan(self.region)
        gdims = [d for d in out_dims if d.name in self.launch]
        for name in self.launch:
            if name not in names:
                raise Untranslatable(f"launch dim {name!r} is not an output dim")
            if name in carried:
                raise Untranslatable(f"launch dim {name!r} is carried/reduced — not griddable")
        counts = [-((d.start - d.stop) // self.launch[d.name]) for d in gdims]
        self.ln("pid = tl.program_id(0)")
        rest = "pid"
        for d, cnt in zip(reversed(gdims), reversed(counts)):
            self.pidv[d.name] = f"pid_{d.name}"
            self.ln(f"pid_{d.name} = {rest} % {cnt}" if d is not gdims[0] else f"pid_{d.name} = {rest}")
            rest = f"({rest} // {cnt})"
        grid = 1
        for cnt in counts:
            grid *= cnt
        return grid

    def render(self) -> tuple[str, list, int]:
        yld = self.region.body[-1].args[0]
        outs = tuple(yld.args) if yld.op == "core.tuple" else (yld,)
        # multi-output = the output plus surfaced artifacts (§7.8): extra
        # stores of finals the sweep already computed, nothing more
        grid = self._prelude(_dims(outs[0].type)) if self.launch else 1
        onames = ["out"] if len(outs) == 1 else [f"out{i}" for i in range(len(outs))]
        all_sizes = []
        for o, oname in zip(outs, onames):
            var, dims = self.emit_block(o)
            sizes = tuple(d.stop - d.start for d in dims)
            strides = []
            acc = 1
            for s in reversed(sizes):
                strides.append(acc)
                acc *= s
            strides = tuple(reversed(strides))
            space = self.space(dims)
            off = " + ".join(f"(({space[d.name].expr}) - {d.start}) * {s}" for d, s in zip(dims, strides))
            mask = " & ".join(c.valid for c in (space[d.name] for d in dims) if c.valid)
            store = f"tl.store({oname} + {off}, {var}" + (f", mask={mask})" if mask else ")")
            self.ln(store)
            all_sizes.append(sizes)
        args = ", ".join([f"p{i}" for i in range(len(self.region.params))] + onames)
        head = ["import triton", "import triton.language as tl", "", "@triton.jit", f"def tile_kernel({args}):"]
        return "\n".join(head + self.lines) + "\n", all_sizes, grid


def compile_tile(region: Region, launch: tuple = (), prune: tuple = ()):
    """Region -> a runner: run(values) -> np.ndarray (f32, CUDA). ``values``
    are numpy arrays positional per param, in the param's type-dim order.
    ``launch`` is the plan artifact (340 §2): ordered (output dim, tile)
    pairs; the program count is derived, never chosen. ``prune`` carries
    mask-derived fold bounds (340 §4b), affine in the program id."""
    src, out_sizes, grid = _Gen(region, launch, prune).render()
    tmp = Path(tempfile.mkdtemp(prefix="pdum_triton_")) / "tile_kernel.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location(f"pdum_triton_{tmp.parent.name}", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    def run(values) -> np.ndarray:
        import torch

        ins = [torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32).cuda() for v in values]
        outs = [torch.empty(s, dtype=torch.float32, device="cuda") for s in out_sizes]
        mod.tile_kernel[(grid,)](*ins, *outs)
        torch.cuda.synchronize()
        out = outs[0]
        if not run.checked:  # the docstring's precision law: refuse demoted PTX
            run.checked = True
            for entry in mod.tile_kernel.device_caches.values():
                cache = entry[0] if isinstance(entry, tuple) else entry
                for ck in cache.values():
                    if ".tf32" in getattr(ck, "asm", {}).get("ptx", ""):
                        raise RuntimeError("tf32 reached the PTX — the translated column is ieee f32")
        if len(outs) > 1:  # the output plus its surfaced artifacts (§7.8)
            return tuple(o.cpu().numpy() for o in outs)
        return out.cpu().numpy()

    run.checked = False
    run.source = src
    run.grid = grid
    return run
