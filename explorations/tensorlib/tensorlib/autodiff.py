"""Reverse-mode AD as a program transformation on the linear SSA IR.

`grad(prog, target, input_layouts, seed=None)` returns (joint_program,
grads): the forward program truncated at `target`, extended with backward
instructions computing the cotangent of every forward variable, and a map
var -> gradient-var (None where no gradient flows). One backward pass yields
d(target)/d(v) for ALL v — reverse mode's native output. The generated
program is ordinary IR: run it with `ir.run`, inspect it, differentiate it
again.

Contracts and conventions:
- Seed: scalar targets seed with 1 automatically; non-scalar targets REQUIRE
  `seed` (the name of an extra runtime input aligned with the target) —
  silent ones-seeding is a footgun we refuse.
- The tape is the program: adjoints reference forward variables directly
  (e.g. d(exp) reuses the forward output).
- Cotangents live on the LATTICE: the transform strips charts/labels from
  every forward value it touches and from the seed. Whether gradients should
  inherit charts is a real design question deferred (CONCERNS); stripping
  makes alignment of accumulated contributions a pure lattice matter.
- Differentiability is structural: markers declare their partials
  (comparisons and iota declare none — the carrier discipline, enforced by
  the rule table rather than a type system for now). Cotangents reaching a
  gradient-free op are dropped.
- Fan-out accumulates: a var consumed n times receives n contributions,
  summed pointwise.
- Adjoints of layout ops are layout(+compute) programs, per the COMPUTE.md
  table: repeat†=reduce-sum, slice†=pad-0, pad†=slice (fill cotangent
  discarded), shift/flip/rename/split/merge are relabelings,
  select†=repeat-at-point+pad, window/stencil†=per-tap overlap-add,
  decimate†=zero-stuffing (materialize+merge; aligned domains only),
  diagonal†=masked embedding (binary only), scan(sum)†=reverse scan.
"""

from __future__ import annotations

from .ir import PW, Instr, Program, infer

_MISSING_SENTINEL = object()


class _Builder:
    def __init__(self, taken: set[str]):
        self.taken = set(taken)
        self.instrs: list[Instr] = []
        self.n = 0

    def fresh(self, hint: str) -> str:
        while True:
            name = f"%{hint}{self.n}"
            self.n += 1
            if name not in self.taken:
                self.taken.add(name)
                return name

    def emit(self, op: str, operands: tuple[str, ...] = (), params: dict | None = None, hint: str = "g") -> str:
        var = self.fresh(hint)
        self.instrs.append(Instr(var, op, operands, params or {}))
        return var

    def fresh_dim(self, base: str, used: set[str]) -> str:
        i = 0
        while f"_{base}{i}" in used:
            i += 1
        used.add(f"_{base}{i}")
        return f"_{base}{i}"


def grad(
    prog: Program,
    target: str,
    input_layouts: dict,
    seed: str | None = None,
    wrt: tuple[str, ...] | None = None,
) -> tuple[Program, dict[str, str | None]]:
    idx = prog.vars.index(target)
    fwd = prog.instrs[: idx + 1]
    shadows = infer(Program(fwd), input_layouts)
    b = _Builder(set(prog.vars))

    stripped: dict[str, str] = {}

    def strip(v: str) -> str:
        if v not in stripped:
            stripped[v] = b.emit("strip_charts", (v,), {}, hint="s")
        return stripped[v]

    def extents_of(layout) -> tuple:
        return tuple((d.name, (d.start, d.stop)) for d in layout.dims)

    def zeros_like(layout) -> str:
        return b.emit("const", (), {"value": 0.0, "dims": extents_of(layout)}, hint="z")

    # ---- seed ----------------------------------------------------------
    tshape = shadows[target]
    cot: dict[str, list[str]] = {}
    if seed is None:
        if len(tshape.dims) != 0:
            raise ValueError(
                f"target {target!r} is not a scalar; pass seed= (the name of a "
                f"runtime input aligned with the target) — reverse mode "
                f"computes vector-Jacobian products"
            )
        cot[target] = [b.emit("const", (), {"value": 1.0, "dims": ()}, hint="seed")]
    else:
        b.instrs.append(Instr(seed, "input", (), {}))
        b.taken.add(seed)
        cot[target] = [strip(seed)]

    final: dict[str, str] = {}

    def finalize(v: str) -> str | None:
        if v in final:
            return final[v]
        parts = cot.get(v)
        if not parts:
            return None
        acc = parts[0]
        for p in parts[1:]:
            acc = b.emit("pointwise", (acc, p), {"f": "add"}, hint="acc")
        final[v] = acc
        return acc

    def contribute(v: str, gv: str) -> None:
        cot.setdefault(v, []).append(gv)

    # ---- per-instruction adjoint rules --------------------------------
    def pw_rule(ins: Instr, c: str) -> None:
        f = ins.params["f"]
        A = ins.operands
        if f == "add":
            contribute(A[0], c)
            contribute(A[1], c)
        elif f == "sub":
            contribute(A[0], c)
            contribute(A[1], b.emit("pointwise", (c,), {"f": "neg"}))
        elif f == "neg":
            contribute(A[0], b.emit("pointwise", (c,), {"f": "neg"}))
        elif f == "mul":
            contribute(A[0], b.emit("pointwise", (c, strip(A[1])), {"f": "mul"}))
            contribute(A[1], b.emit("pointwise", (c, strip(A[0])), {"f": "mul"}))
        elif f == "div":
            contribute(A[0], b.emit("pointwise", (c, strip(A[1])), {"f": "div"}))
            bb = b.emit("pointwise", (strip(A[1]), strip(A[1])), {"f": "mul"})
            num = b.emit("pointwise", (c, strip(A[0])), {"f": "mul"})
            frac = b.emit("pointwise", (num, bb), {"f": "div"})
            contribute(A[1], b.emit("pointwise", (frac,), {"f": "neg"}))
        elif f == "exp":
            contribute(A[0], b.emit("pointwise", (c, strip(ins.var)), {"f": "mul"}))
        elif f == "log":
            contribute(A[0], b.emit("pointwise", (c, strip(A[0])), {"f": "div"}))
        elif f == "maximum":
            m0 = b.emit("pointwise", (strip(A[0]), strip(A[1])), {"f": "ge"})
            contribute(A[0], b.emit("pointwise", (c, m0), {"f": "mul"}))
            m1 = b.emit("pointwise", (strip(A[1]), strip(A[0])), {"f": "gt"})
            contribute(A[1], b.emit("pointwise", (c, m1), {"f": "mul"}))
        elif f == "minimum":
            m0 = b.emit("pointwise", (strip(A[0]), strip(A[1])), {"f": "le"})
            contribute(A[0], b.emit("pointwise", (c, m0), {"f": "mul"}))
            m1 = b.emit("pointwise", (strip(A[1]), strip(A[0])), {"f": "lt"})
            contribute(A[1], b.emit("pointwise", (c, m1), {"f": "mul"}))
        elif f == "where":
            z = zeros_like(shadows[ins.var])
            contribute(A[1], b.emit("pointwise", (strip(A[0]), c, z), {"f": "where"}))
            contribute(A[2], b.emit("pointwise", (strip(A[0]), z, c), {"f": "where"}))
        elif f in PW:
            pass  # comparisons etc.: declared gradient-free; cotangent drops
        else:
            raise KeyError(f"unknown marker {f!r}")

    def repeats_over(v: str, names, src_layout) -> str:
        cur = v
        for name in names:
            d = src_layout.dim(name)
            cur = b.emit("repeat", (cur,), {"name": name, "extent": (d.start, d.stop)})
        return cur

    def reduce_rule(ins: Instr, c: str) -> None:
        f = ins.params["f"]
        dims = ins.params["dims"]
        names = (dims,) if isinstance(dims, str) else tuple(dims)
        A = ins.operands[0]
        a_shape = shadows[A]
        if f == "sum":
            contribute(A, repeats_over(c, names, a_shape))
        elif f == "mean":
            r = repeats_over(c, names, a_shape)
            n = 1
            for name in names:
                n *= a_shape.dim(name).size
            nb = b.emit("const", (), {"value": float(n), "dims": extents_of(a_shape)})
            contribute(A, b.emit("pointwise", (r, nb), {"f": "div"}))
        elif f in ("max", "min"):
            rc = repeats_over(c, names, a_shape)
            rm = repeats_over(strip(ins.var), names, a_shape)
            m = b.emit("pointwise", (strip(A), rm), {"f": "eq"})
            contribute(A, b.emit("pointwise", (rc, m), {"f": "mul"}))
        else:
            raise NotImplementedError(f"reduce({f}) has no adjoint rule yet")

    def scan_rule(ins: Instr, c: str) -> None:
        if ins.params["f"] != "sum":
            raise NotImplementedError("only scan(sum) is differentiable so far")
        dim = ins.params["dim"]
        f1 = b.emit("flip", (c,), {"name": dim})
        s1 = b.emit("scan", (f1,), {"f": "sum", "dim": dim})
        contribute(ins.operands[0], b.emit("flip", (s1,), {"name": dim}))

    def tap_rule(ins: Instr, c: str) -> None:
        """Shared adjoint for window and stencil: per-tap overlap-add."""
        p = ins.params
        name, k_name = p["name"], p.get("k_name") or f"{p['name']}_k"
        dilation = p.get("dilation", 1)
        A = ins.operands[0]
        src = shadows[A].dim(name)
        kdim = shadows[ins.var].dim(k_name)
        for kappa in range(kdim.start, kdim.stop):
            t1 = b.emit("select", (c,), {"coords": {k_name: kappa}})
            t2 = b.emit("shift", (t1,), {"deltas": {name: kappa * dilation}})
            anchor = shadows[ins.var].dim(name)
            lo = max(src.start, anchor.start + kappa * dilation)
            hi = min(src.stop, anchor.stop + kappa * dilation)
            hi = max(lo, hi)
            if hi < lo:
                lo = hi = src.start  # fully out-of-range tap: empty contribution
            t3 = b.emit("slice", (t2,), {"ranges": {name: (lo, hi)}})
            t4 = b.emit("pad", (t3,), {"fill": 0.0, "extents": {name: (src.start, src.stop)}})
            contribute(A, t4)

    def decimate_rule(ins: Instr, c: str) -> None:
        p = ins.params
        name, f = p["name"], p["factor"]
        A = ins.operands[0]
        src = shadows[A].dim(name)
        phase = src.delta_to_lattice(p.get("phase", 0)) % f
        s, e = src.start, src.stop
        out = shadows[ins.var].dim(name)
        if (e - s) % f or (e - s) // f != out.size:
            raise NotImplementedError(
                f"decimate adjoint needs a factor-divisible domain: [{s}, {e}) with factor {f} (pad the source first)"
            )
        slot = (phase - s) % f  # interleave slot of the kept residue class
        used = {d.name for d in shadows[ins.var].dims}
        cname = b.fresh_dim("c", used)
        ph = b.fresh_dim("ph", used)
        r0 = b.emit("rename", (c,), {"mapping": {name: cname}})
        r1 = b.emit("repeat", (r0,), {"name": ph, "extent": (0, f)})
        i1 = b.emit("iota", (r1,), {"name": ph})
        r1_dims = tuple((cname if d.name == name else d.name, (d.start, d.stop)) for d in shadows[ins.var].dims) + (
            (ph, (0, f)),
        )
        cp = b.emit("const", (), {"value": slot, "dims": r1_dims, "dtype": "int64"})
        m = b.emit("pointwise", (i1, cp), {"f": "eq"})
        z0 = b.emit("const", (), {"value": 0.0, "dims": r1_dims})
        w = b.emit("pointwise", (m, r1, z0), {"f": "where"})
        others = tuple(n for n, _ in r1_dims if n not in (cname, ph))
        mo = b.emit("materialize", (w,), {"order": others + (cname, ph)})
        mg = b.emit("merge", (mo,), {"parts": (cname, ph), "name": name, "start": s})
        contribute(A, mg)

    def diagonal_rule(ins: Instr, c: str) -> None:
        parts = tuple(ins.params["parts"])
        if len(parts) != 2:
            raise NotImplementedError("n-ary diagonal adjoint not written yet")
        x, y = parts
        z = ins.params["name"]
        A = ins.operands[0]
        xdom, ydom = shadows[A].dim(x), shadows[A].dim(y)
        r1 = c if z == x else b.emit("rename", (c,), {"mapping": {z: x}})
        r2 = b.emit("repeat", (r1,), {"name": y, "extent": (ydom.start, ydom.stop)})
        ix = b.emit("iota", (r2,), {"name": x})
        iy = b.emit("iota", (r2,), {"name": y})
        m = b.emit("pointwise", (ix, iy), {"f": "eq"})
        r2_dims = tuple((x if d.name == z else d.name, (d.start, d.stop)) for d in shadows[ins.var].dims) + (
            (y, (ydom.start, ydom.stop)),
        )
        z0 = b.emit("const", (), {"value": 0.0, "dims": r2_dims})
        w = b.emit("pointwise", (m, r2, z0), {"f": "where"})
        zdim = shadows[ins.var].dim(z)
        if (zdim.start, zdim.stop) != (xdom.start, xdom.stop):
            w = b.emit("pad", (w,), {"fill": 0.0, "extents": {x: (xdom.start, xdom.stop)}})
        contribute(A, w)

    def layout_rule(ins: Instr, c: str) -> None:
        A = ins.operands[0]
        a_shape = shadows[A]
        p = ins.params
        if ins.op == "slice":
            extents = {n: (a_shape.dim(n).start, a_shape.dim(n).stop) for n in p["ranges"]}
            contribute(A, b.emit("pad", (c,), {"fill": 0.0, "extents": extents}))
        elif ins.op == "pad":
            ranges = {n: (a_shape.dim(n).start, a_shape.dim(n).stop) for n in p["extents"]}
            contribute(A, b.emit("slice", (c,), {"ranges": ranges}))
        elif ins.op == "shift":
            deltas = {n: -a_shape.dim(n).delta_to_lattice(v) for n, v in p["deltas"].items()}
            contribute(A, b.emit("shift", (c,), {"deltas": deltas}))
        elif ins.op == "flip":
            contribute(A, b.emit("flip", (c,), {"name": p["name"]}))
        elif ins.op == "rename":
            inv = {new: old for old, new in p["mapping"].items()}
            contribute(A, b.emit("rename", (c,), {"mapping": inv}))
        elif ins.op == "repeat":
            contribute(A, b.emit("reduce", (c,), {"f": "sum", "dims": (p["name"],)}))
        elif ins.op == "select":
            cur = c
            names = list(p["coords"])
            for n, coord in p["coords"].items():
                i = a_shape.dim(n).to_lattice(coord)
                cur = b.emit("repeat", (cur,), {"name": n, "extent": (i, i + 1)})
            extents = {n: (a_shape.dim(n).start, a_shape.dim(n).stop) for n in names}
            contribute(A, b.emit("pad", (cur,), {"fill": 0.0, "extents": extents}))
        elif ins.op == "split":
            parts = tuple(p["parts"])
            others = tuple(d.name for d in shadows[ins.var].dims if d.name not in parts)
            mo = b.emit("materialize", (c,), {"order": others + parts})
            contribute(
                A,
                b.emit(
                    "merge",
                    (mo,),
                    {"parts": parts, "name": p["name"], "start": a_shape.dim(p["name"]).start},
                ),
            )
        elif ins.op == "merge":
            parts = {n: (a_shape.dim(n).start, a_shape.dim(n).stop) for n in p["parts"]}
            contribute(A, b.emit("split", (c,), {"name": p["name"], "parts": parts}))
        elif ins.op in ("window", "stencil"):
            tap_rule(ins, c)
        elif ins.op == "decimate":
            decimate_rule(ins, c)
        elif ins.op == "diagonal":
            diagonal_rule(ins, c)
        elif ins.op in ("strip_charts", "simplify", "materialize"):
            contribute(A, c)  # value-preserving (materialize: identity copy)
        else:
            raise NotImplementedError(f"no adjoint rule for {ins.op!r}")

    # ---- reverse walk --------------------------------------------------
    for ins in reversed(fwd):
        c = finalize(ins.var)
        if c is None:
            continue
        if ins.op in ("input", "const", "iota"):
            continue  # leaves: gradient stops (iota/const are gradient-free)
        if ins.op == "pointwise":
            pw_rule(ins, c)
        elif ins.op == "reduce":
            reduce_rule(ins, c)
        elif ins.op == "scan":
            scan_rule(ins, c)
        else:
            layout_rule(ins, c)

    grads: dict[str, str | None] = {}
    names = wrt if wrt is not None else tuple(i.var for i in fwd)
    for v in names:
        grads[v] = final.get(v)
    return Program(tuple(fwd) + tuple(b.instrs)), grads


# ----------------------------------------------------------------------
# validation harness
# ----------------------------------------------------------------------


def numeric_grad(prog: Program, target: str, wrt_var: str, inputs: dict, eps: float = 1e-6):
    """Central finite differences of a SCALAR target w.r.t. one input tensor.
    Rebuilds perturbed inputs via from_numpy (0-based, uncharted) — use
    simple inputs in FD tests."""
    import numpy as np

    from .ir import run
    from .tensor import Tensor

    base = inputs[wrt_var]
    arr = base.to_numpy().astype(np.float64)
    names = base.names
    g = np.zeros_like(arr)
    for idx in np.ndindex(*arr.shape):
        out = []
        for sign in (+1, -1):
            pert = arr.copy()
            pert[idx] += sign * eps
            env = run(prog, {**inputs, wrt_var: Tensor.from_numpy(pert, names)})
            out.append(float(env[target].item()))
        g[idx] = (out[0] - out[1]) / (2 * eps)
    return g
