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
- Gradients carry their primal's COORDINATE charts and labels: AD
  differentiates value-with-respect-to-value, never with respect to
  coordinates, so the gradient entry for the sample at x = 0.75 um is
  labeled x = 0.75 um. This holds BY CONSTRUCTION: every contribution is
  restamped with its primal's charts/labels at the moment it is recorded
  (which also absorbs select's axis-compensation and promotion). Composite
  rules (decimate, diagonal) work on the bare lattice internally and are
  restamped on the way out.
- VALUE units transform as unit(dL/dv) = unit(L)/unit(v). Pass
  `target_unit` (the unit of the effective scalar) and gradients of inputs
  with declared value_units are annotated accordingly; a runtime seed
  should itself carry unit(L)/unit(target).
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
  decimate†=zero-stuffing (materialize+merge; factor-divisible source
  domain only),
  diagonal†=masked embedding (binary only), scan(sum)†=reverse scan.
- Composite reducers differentiate via BPTT-as-IR: the state cotangent of a
  structured-state scan is itself a linear recurrence in reversed time, run
  as a generated matrix-linrec scan over derived Jacobian trees
  (composite_scan_adjoint); reduce† = embed-at-last, then scan†.
- Tie caveat asymmetry: reduce(max/min) gives every tied element the full
  cotangent (sums to c x #ties); pointwise maximum/minimum split ties
  cleanly via the ge/gt asymmetry. Both are standard; only the reduce form
  over-counts.
"""

from __future__ import annotations

from functools import lru_cache

from .ir import PW, RED, Instr, Program, _fold_extent, _fold_parts, _fold_step_layouts, infer
from .mdsl import COMPOSITE_MARKERS, COMPOSITE_REDUCERS, Const
from .signatures import infer_signatures
from .units import ONE

_GRADIENT_FREE = frozenset({"eq", "ne", "le", "lt", "ge", "gt"})


@lru_cache(maxsize=None)
def _revolve_cost(s: int, length: int) -> float:
    """Minimum recomputed forward-steps to reverse `length` steps with `s`
    checkpoint slots, under the revolve recursion (split at c: advance the
    head, reverse the tail with s-1 slots, reverse the head with s slots).
    A range is a store-all leaf — zero re-advance — once it fits in the
    available slots (length <= 1, or s >= length)."""
    if length <= 1 or s >= length:
        return 0.0
    if s < 1:
        return float("inf")  # cannot subdivide a multi-step range with no slot
    return min(m + _revolve_cost(s - 1, length - m) + _revolve_cost(s, m) for m in range(1, length))


@lru_cache(maxsize=None)
def _revolve_split(s: int, length: int) -> int:
    """Head length (steps to advance before the next checkpoint) for the
    optimal revolve split of `length` steps with `s` slots. This is exactly
    the Griewank & Walther binomial rule — with s slots and r recomputations
    one reverses up to C(s+r, s) steps — computed here by a memoized DP over
    the recompute-cost recurrence rather than the closed form, since fold
    lengths are modest at trace time. Ties are broken toward the LONGER head
    (advance as far as possible), matching revolve's convention. Only called
    for a genuine split: 1 <= s and 2 <= length and s < length."""
    best = _revolve_cost(s, length)
    return max(m for m in range(1, length) if m + _revolve_cost(s - 1, length - m) + _revolve_cost(s, m) == best)


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
    target_unit=None,
    fold_segments: int | None = None,
    fold_slots: int | None = None,
) -> tuple[Program, dict[str, str | None]]:
    if target not in prog.vars:
        raise KeyError(f"target {target!r} is not defined by the program")
    if seed is not None and seed in prog.vars:
        raise ValueError(f"seed name {seed!r} collides with a program variable; pick a fresh name")
    if fold_segments is not None and fold_slots is not None:
        raise ValueError(
            "pass fold_segments (uniform K-way checkpointing) OR fold_slots "
            "(binomial revolve), not both — they name different schedules"
        )
    if fold_slots is not None and int(fold_slots) < 1:
        raise ValueError("fold_slots must be >= 1 (the number of checkpoint slots)")
    idx = prog.vars.index(target)
    fwd = prog.instrs[: idx + 1]
    shadows = infer(Program(fwd), input_layouts)
    # the signature pass makes target_unit INFERABLE (and checks declared
    # units while it's at it — conflicting declarations refuse loudly here)
    sigs = infer_signatures(Program(fwd), input_layouts)
    if target_unit is None:
        target_unit = sigs[target].unit
    b = _Builder(set(prog.vars))

    def extents_of(layout) -> tuple:
        return tuple((d.name, (d.start, d.stop)) for d in layout.dims)

    def restamp(gv: str, layout) -> str:
        """Stamp the primal's charts/labels onto a contribution — the
        gradients-carry-their-primal's-labeling invariant, by construction.
        Also normalizes away select's axis-compensation and any stray
        labeling a composite rule left behind."""
        charts, labels = {}, {}
        for d in layout.dims:
            if d.labels is not None:
                labels[d.name] = d.labels
            else:
                charts[d.name] = d.chart  # a Chart, or None to clear
        if charts:
            gv = b.emit("with_charts", (gv,), {"charts": charts}, hint="st")
        if labels:
            gv = b.emit("with_labels", (gv,), {"labels": labels}, hint="st")
        return gv

    def const_like(layout, value, dtype=None) -> str:
        params = {"value": value, "dims": extents_of(layout)}
        if dtype is not None:
            params["dtype"] = dtype
        return restamp(b.emit("const", (), params, hint="z"), layout)

    def zeros_like(layout) -> str:
        return const_like(layout, 0.0)

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
        cot[target] = [restamp(seed, tshape)]

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
        cot.setdefault(v, []).append(restamp(gv, shadows[v]))

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
            contribute(A[0], b.emit("pointwise", (c, A[1]), {"f": "mul"}))
            contribute(A[1], b.emit("pointwise", (c, A[0]), {"f": "mul"}))
        elif f == "div":
            contribute(A[0], b.emit("pointwise", (c, A[1]), {"f": "div"}))
            bb = b.emit("pointwise", (A[1], A[1]), {"f": "mul"})
            num = b.emit("pointwise", (c, A[0]), {"f": "mul"})
            frac = b.emit("pointwise", (num, bb), {"f": "div"})
            contribute(A[1], b.emit("pointwise", (frac,), {"f": "neg"}))
        elif f == "exp":
            contribute(A[0], b.emit("pointwise", (c, ins.var), {"f": "mul"}))
        elif f == "log":
            contribute(A[0], b.emit("pointwise", (c, A[0]), {"f": "div"}))
        elif f == "maximum":
            m0 = b.emit("pointwise", (A[0], A[1]), {"f": "ge"})
            contribute(A[0], b.emit("pointwise", (c, m0), {"f": "mul"}))
            m1 = b.emit("pointwise", (A[1], A[0]), {"f": "gt"})
            contribute(A[1], b.emit("pointwise", (c, m1), {"f": "mul"}))
        elif f == "minimum":
            m0 = b.emit("pointwise", (A[0], A[1]), {"f": "le"})
            contribute(A[0], b.emit("pointwise", (c, m0), {"f": "mul"}))
            m1 = b.emit("pointwise", (A[1], A[0]), {"f": "lt"})
            contribute(A[1], b.emit("pointwise", (c, m1), {"f": "mul"}))
        elif f == "where":
            z = zeros_like(shadows[ins.var])
            contribute(A[1], b.emit("pointwise", (A[0], c, z), {"f": "where"}))
            contribute(A[2], b.emit("pointwise", (A[0], z, c), {"f": "where"}))
        elif f == "tanh":
            sq = b.emit("pointwise", (ins.var, ins.var), {"f": "mul"})
            one = const_like(shadows[ins.var], 1.0)
            om = b.emit("pointwise", (one, sq), {"f": "sub"})
            contribute(A[0], b.emit("pointwise", (c, om), {"f": "mul"}))
        elif f == "sqrt":
            two = const_like(shadows[ins.var], 2.0)
            den = b.emit("pointwise", (two, ins.var), {"f": "mul"})
            contribute(A[0], b.emit("pointwise", (c, den), {"f": "div"}))
        elif f == "sin":
            cv = b.emit("pointwise", (A[0],), {"f": "cos"})
            contribute(A[0], b.emit("pointwise", (c, cv), {"f": "mul"}))
        elif f == "cos":
            sv = b.emit("pointwise", (A[0],), {"f": "sin"})
            neg = b.emit("pointwise", (sv,), {"f": "neg"})
            contribute(A[0], b.emit("pointwise", (c, neg), {"f": "mul"}))
        elif f in _GRADIENT_FREE:
            pass  # declared gradient-free (bool-carrier outputs); cotangent drops
        elif f in COMPOSITE_MARKERS:
            # the marker DSL pays off: partials are DERIVED by tree
            # rewriting, so composite markers differentiate automatically
            cm = COMPOSITE_MARKERS[f]
            for i, operand in enumerate(A):
                p = cm.partial(i)
                if isinstance(p.body, Const) and p.body.value == 0:
                    continue
                pv = b.emit("pointwise", A, {"f": p.name})
                contribute(operand, b.emit("pointwise", (c, pv), {"f": "mul"}))
        elif f in PW:
            raise NotImplementedError(
                f"marker {f!r} has no gradient rule — add one to pw_rule or "
                f"declare it in _GRADIENT_FREE (silent zero gradients are how "
                f"models rot)"
            )
        else:
            raise KeyError(f"unknown marker {f!r}")

    def repeats_over(v: str, names, src_layout) -> str:
        cur = v
        for name in names:
            d = src_layout.dim(name)
            cur = b.emit(
                "repeat",
                (cur,),
                {
                    "name": name,
                    "extent": (d.start, d.stop),
                    "chart": d.chart,
                    "labels": d.labels,
                },
            )
        return cur

    def composite_scan_adjoint(fname: str, dim: str, elems: tuple, sc: str) -> None:
        """BPTT for a structured-state scan, emitted as IR. With state
        s_t = C(s_{t-1}, lift(e_t)), y_t = P(s_t), the state cotangent obeys
        ŝ_t = Pᵀ(s_t)·ȳ_t + C_leftᵀ(s_t, l_{t+1})·ŝ_{t+1} — itself a LINEAR
        recurrence in reversed time, run as a generated matrix-linrec
        composite scan (adjoint_scanner). All Jacobian entries are derived
        partials of the combine/lift/project trees, evaluated pointwise at
        the forward trajectory (re-scanned per state component — reference
        inefficiency, deliberate). Because `init` is the monoid identity,
        C(init, r) = r makes ∂C/∂right the identity at t=start, so the
        boundary needs no special case; the first reversed element's M slot
        is garbage but provably never projected."""
        f = COMPOSITE_REDUCERS[fname]
        k = f.state
        cs, ls, p_marker = f.component_markers()
        ddim = shadows[elems[0]].dim(dim)
        if ddim.size == 0:
            for e in elems:
                contribute(e, zeros_like(shadows[e]))
            return
        s0, s1 = ddim.start, ddim.stop

        def lat(v: str) -> str:
            return b.emit("strip_charts", (v,), {}, hint="lat")

        def acc_sum(terms: list) -> str:
            total = terms[0]
            for t in terms[1:]:
                total = b.emit("pointwise", (total, t), {"f": "add"})
            return total

        sc = lat(sc)
        se = tuple(lat(e) for e in elems)
        # forward trajectories: state components s_j and lifted elements l_j
        sjs = tuple(b.emit("scan", se, {"f": f.state_scanner(j).name, "dim": dim}) for j in range(k))
        ljs = tuple(b.emit("pointwise", se, {"f": ls[j].name}) for j in range(k))
        sprev, lnext = [], []
        for j in range(k):  # s_{t-1} (init-filled at start) and l_{t+1}
            sh = b.emit("shift", (sjs[j],), {"deltas": {dim: 1}})
            sl = b.emit("slice", (sh,), {"ranges": {dim: (s0 + 1, s1)}})
            sprev.append(b.emit("pad", (sl,), {"fill": float(f.init[j]), "extents": {dim: (s0, s1)}}))
            sh = b.emit("shift", (ljs[j],), {"deltas": {dim: -1}})
            sl = b.emit("slice", (sh,), {"ranges": {dim: (s0, s1 - 1)}})
            lnext.append(b.emit("pad", (sl,), {"fill": 0.0, "extents": {dim: (s0, s1)}}))
        # reversed-time backward elements: M[i][x] = ∂C_x/∂left_i at
        # (s_t, l_{t+1}), then the injection g_i = ∂P/∂s_i (s_t) · ȳ_t
        felems = []
        for i in range(k):
            for x in range(k):
                mv = b.emit("pointwise", sjs + tuple(lnext), {"f": cs[x].partial(i).name})
                felems.append(b.emit("flip", (mv,), {"name": dim}))
        gs = []
        for i in range(k):
            pv = b.emit("pointwise", sjs, {"f": p_marker.partial(i).name})
            gs.append(b.emit("pointwise", (pv, sc), {"f": "mul"}))
            felems.append(b.emit("flip", (gs[i],), {"name": dim}))
        # k backward scans (one projection each), flipped back to t-order
        shat = tuple(
            b.emit(
                "flip",
                (b.emit("scan", tuple(felems), {"f": f.adjoint_scanner(i).name, "dim": dim}),),
                {"name": dim},
            )
            for i in range(k)
        )
        # element cotangents: l̄_j = Σ_i ∂C_i/∂right_j (s_{t-1}, l_t) · ŝ_i,
        # then ē = Jᵀ(lift) · l̄, contributed per operand slot
        lbar: list[str | None] = []
        for j in range(k):
            terms = []
            for i in range(k):
                pd = cs[i].partial(k + j)
                if isinstance(pd.body, Const) and pd.body.value == 0:
                    continue
                dv = b.emit("pointwise", tuple(sprev) + tuple(ljs), {"f": pd.name})
                terms.append(b.emit("pointwise", (dv, shat[i]), {"f": "mul"}))
            lbar.append(acc_sum(terms) if terms else None)
        for i in range(f.element):
            terms = []
            for j in range(k):
                pd = ls[j].partial(i)
                if lbar[j] is None or (isinstance(pd.body, Const) and pd.body.value == 0):
                    continue
                dv = b.emit("pointwise", se, {"f": pd.name})
                terms.append(b.emit("pointwise", (dv, lbar[j]), {"f": "mul"}))
            if terms:
                contribute(elems[i], acc_sum(terms))

    def fold_rule(ins: Instr, c: str) -> None:
        """The adjoint of a fold is a REVERSE fold over the step's VJP
        program — derived by self-application: wrap the step with cotangent
        inputs and a scalarized target (sum of cot·out inner products), run
        `grad` on the wrapper, and the resulting joint program IS the
        backward step. It carries the state cotangent in reversed time and
        consumes (s_{t-1}, e_t, ȳ_t) as elements — standard BPTT with
        per-step recompute, generated rather than hand-written.

        Memory: with `fold_segments=K` the time axis is cut into K equal
        segments; only segment-BOUNDARY states are computed up front, and
        each segment's trajectory is recomputed just-in-time during its own
        backward sweep (Chen-style uniform checkpointing: ~T/K + K states
        live instead of T; K≈√T minimizes). `fold_slots=S` instead runs the
        binomial revolve schedule over the same pieces (below): ~O(S) live
        states, recompute up the binomial curve, and no divisibility
        constraint on T. K=1 (the default) is the store-everything adjoint."""
        step, dim, state_names, elem_names, carry, (out_kind, out_var) = _fold_parts(ins.params)
        k = len(state_names)
        inits, elems = ins.operands[:k], ins.operands[k:]
        start, stop = _fold_extent(ins, shadows)
        if stop - start == 0:
            for sn, iv in zip(state_names, inits):
                if out_kind == "final" and carry[sn] == out_var:
                    contribute(iv, c)
                else:
                    contribute(iv, zeros_like(shadows[iv]))
            for ev in elems:
                contribute(ev, zeros_like(shadows[ev]))
            return

        def flp(v: str) -> str:
            return b.emit("flip", (v,), {"name": dim})

        # the step program may be chart-aware (staggered grids re-stamp
        # charts internally), so the adjoint keeps the primal's charts
        # throughout: cotangents already arrive restamped to the fold's
        # output shadow, and only generated consts need stamping
        slayouts = _fold_step_layouts(ins, shadows)
        ss = infer(step, slayouts)
        # normalize the cotangent to emit-form along dim (final = emit-at-last)
        if out_kind == "final":
            r = b.emit("repeat", (c,), {"name": dim, "extent": (stop - 1, stop)})
            yb = b.emit("pad", (r,), {"fill": 0.0, "extents": {dim: (start, stop)}})
        else:
            yb = c
        # the VJP wrapper: step + cotangent inputs + scalarized target
        wb = _Builder(set(step.vars))
        winstrs = list(step.instrs)
        wlayouts = dict(slayouts)
        terms = []

        def scalarize(v: str, layout) -> str:
            cin = wb.fresh("ct")
            winstrs.append(Instr(cin, "input", (), {}))
            wlayouts[cin] = layout
            pr = wb.fresh("pr")
            winstrs.append(Instr(pr, "pointwise", (v, cin), {"f": "mul"}))
            names = tuple(d.name for d in layout.dims)
            if not names:
                return cin, pr
            rs = wb.fresh("rs")
            winstrs.append(Instr(rs, "reduce", (pr,), {"f": "sum", "dims": names}))
            return cin, rs

        cot_state = {}
        for sn in state_names:
            cot_state[sn], t0 = scalarize(carry[sn], ss[carry[sn]])
            terms.append(t0)
        cot_out, t0 = scalarize(out_var, ss[out_var])
        terms.append(t0)
        target = terms[0]
        for tv in terms[1:]:
            nv = wb.fresh("L")
            winstrs.append(Instr(nv, "pointwise", (target, tv), {"f": "add"}))
            target = nv
        jp, g = grad(Program(tuple(winstrs)), target, wlayouts)
        # missing gradients become explicit zeros so the reverse fold always
        # has a var to carry/emit
        taken = set(jp.vars)
        extra, zn = [], 0

        def ensure(gv, layout):
            nonlocal zn
            if gv is not None:
                return gv
            while f"%fz{zn}" in taken:
                zn += 1
            name = f"%fz{zn}"
            taken.add(name)
            extra.append(
                Instr(
                    name, "const", (), {"value": 0.0, "dims": tuple((d.name, (d.start, d.stop)) for d in layout.dims)}
                )
            )
            charts = {d.name: d.chart for d in layout.dims if d.chart is not None}
            labels = {d.name: d.labels for d in layout.dims if d.labels is not None}
            for op, key, data in (("with_charts", "charts", charts), ("with_labels", "labels", labels)):
                if data:
                    prev, name = name, f"{name}s"
                    taken.add(name)
                    extra.append(Instr(name, op, (prev,), {key: data}))
            return name

        carry_back = {sn: ensure(g[sn], slayouts[sn]) for sn in state_names}
        ejp = Program(tuple(jp.instrs) + tuple(extra))
        adj_params = {
            "step": ejp,
            "dim": dim,
            "state": tuple(cot_state[sn] for sn in state_names),
            "element": tuple(state_names) + tuple(elem_names) + (cot_out,),
            "carry": {cot_state[sn]: carry_back[sn] for sn in state_names},
        }
        base = dict(ins.params)
        T = stop - start

        # ---- certified pieces, shared by every schedule -----------------
        # Both the uniform (fold_segments) and binomial (fold_slots) paths
        # are SCHEDULES over the same two operations: `advance` runs the
        # forward step to move a boundary state across a range (out=final),
        # and `leaf_backward` recomputes one range's trajectory just-in-time
        # and reverse-folds it (contributing element cotangents and returning
        # the range's start-state cotangent). Nothing about the certified
        # backward step depends on how the ranges are chosen.
        def seg_ops(state_vars, lo, hi):
            """(operands, param overrides, element slices) for [lo, hi)."""
            if elems:
                se = tuple(b.emit("slice", (ev,), {"ranges": {dim: (lo, hi)}}) for ev in elems)
                return tuple(state_vars) + se, {}, se
            return tuple(state_vars), {"extent": (lo, hi)}, ()

        def zero_state():
            return {
                sn: restamp(
                    b.emit(
                        "const",
                        (),
                        {"value": 0.0, "dims": tuple((d.name, (d.start, d.stop)) for d in slayouts[sn].dims)},
                    ),
                    slayouts[sn],
                )
                for sn in state_names
            }

        def advance(s0, lo, hi):
            """State at `hi` from state `s0` at `lo` (out=final boundary fold)."""
            ops_, extra_p, _ = seg_ops(tuple(s0[sn] for sn in state_names), lo, hi)
            return {sn: b.emit("fold", ops_, {**base, **extra_p, "out": ("final", carry[sn])}) for sn in state_names}

        def leaf_backward(lo, hi, s0, cur, full):
            """Reverse [lo, hi) from boundary state `s0` and incoming state
            cotangent `cur` (the cotangent at `hi`); contribute this range's
            element cotangents and return the cotangent at `lo`. `full` is
            True only when the range is the whole fold (no slice/pad needed)."""
            ops_, extra_p, se = seg_ops(tuple(s0[sn] for sn in state_names), lo, hi)
            # trajectory (value AFTER each step), then s_{t-1} via shift +
            # where(t == lo, boundary, ...) — the boundary is an iota mask,
            # because the "fill" here is a TENSOR, not a scalar
            sprev = {}
            for sn in state_names:
                traj = b.emit("fold", ops_, {**base, **extra_p, "out": ("emit", carry[sn])})
                sh = b.emit("shift", (traj,), {"deltas": {dim: 1}})
                sl = b.emit("slice", (sh,), {"ranges": {dim: (lo + 1, hi)}})
                pd = b.emit("pad", (sl,), {"fill": 0.0, "extents": {dim: (lo, hi)}})
                ri = b.emit("repeat", (s0[sn],), {"name": dim, "extent": (lo, hi)})
                it = b.emit("iota", (pd,), {"name": dim})
                sdims = tuple((d.name, (d.start, d.stop)) for d in slayouts[sn].dims) + ((dim, (lo, hi)),)
                cs = b.emit("const", (), {"value": lo, "dims": sdims, "dtype": "int64"})
                cs = restamp(cs, slayouts[sn])  # partial stamp: scan dim stays bare
                mask = b.emit("pointwise", (it, cs), {"f": "eq"})
                sprev[sn] = b.emit("pointwise", (mask, ri, pd), {"f": "where"})
            ybj = yb if full else b.emit("slice", (yb,), {"ranges": {dim: (lo, hi)}})
            adj_ops = (
                tuple(cur[sn] for sn in state_names)
                + tuple(flp(sprev[sn]) for sn in state_names)
                + tuple(flp(x) for x in se)
                + (flp(ybj),)
            )
            for en, ev in zip(elem_names, elems):
                if g[en] is None:
                    continue
                fv = flp(b.emit("fold", adj_ops, {**adj_params, "out": ("emit", g[en])}))
                if not full:
                    fv = b.emit("pad", (fv,), {"fill": 0.0, "extents": {dim: (start, stop)}})
                contribute(ev, fv)  # ranges accumulate via cotangent fan-in
            return {sn: b.emit("fold", adj_ops, {**adj_params, "out": ("final", carry_back[sn])}) for sn in state_names}

        if fold_slots is not None:
            # ---- binomial revolve (Griewank & Walther) ------------------
            # A RECURSIVE schedule over the same pieces. With S checkpoint
            # slots, reverse [lo, hi): store the state at a split point c,
            # recurse on the tail [c, hi) with S-1 slots (the checkpoint holds
            # one), then — that slot now free — recurse on the head [lo, c)
            # with S slots. Leaves (a single step, or any range that already
            # fits in the available slots) get the store-all `leaf_backward`.
            # The split is chosen by _revolve_split: the optimal offline
            # schedule (memoized DP over the recompute-cost recurrence — the
            # same optimum revolve reaches in closed form; T is modest at
            # trace time). No divisibility constraint: arbitrary T works.
            # Memory ~O(S·state) live checkpoints + one leaf trajectory;
            # compute grows by the binomial recompute factor. ŝ chains across
            # every seam exactly as the uniform path does, because leaves are
            # visited in strictly decreasing time order.
            S = int(fold_slots)
            cur = zero_state()  # cotangent at time `stop`

            def revolve(lo, hi, s, boundary):
                # `boundary` is the state at `lo`; on return, `cur` has been
                # carried from the cotangent at `hi` to the cotangent at `lo`
                nonlocal cur
                span = hi - lo
                if span <= 1 or s >= span:
                    cur = leaf_backward(lo, hi, boundary, cur, full=(lo == start and hi == stop))
                    return
                c = lo + _revolve_split(s, span)
                state_c = advance(boundary, lo, c)  # the checkpoint (one slot)
                revolve(c, hi, s - 1, state_c)  # tail first (later times)
                revolve(lo, c, s, boundary)  # then head, checkpoint freed

            revolve(start, stop, S, dict(zip(state_names, inits)))
            for sn, iv in zip(state_names, inits):
                if g[sn] is None:
                    continue
                contribute(iv, cur[sn])
            return

        # ---- uniform (Chen-style) segmentation, or store-all when K=1 ---
        K = 1 if fold_segments is None else min(int(fold_segments), T)
        if K < 1:
            raise ValueError("fold_segments must be >= 1")
        if T % K:
            raise ValueError(f"fold_segments={K} must divide the fold extent {T} (pad the dim or pick a divisor)")
        L = T // K

        # forward pass over segments: keep only segment-START states
        seg_start = [dict(zip(state_names, inits))]
        for j in range(K - 1):
            lo, hi = start + j * L, start + (j + 1) * L
            seg_start.append(advance(seg_start[-1], lo, hi))

        # backward, segment by segment (reversed): recompute the segment's
        # trajectory from its boundary state, then run the reverse fold —
        # ŝ chains across the seam (segment j's final reverse carry is the
        # cotangent of segment j-1's end state)
        cur = zero_state()
        for j in reversed(range(K)):
            lo, hi = start + j * L, start + (j + 1) * L
            cur = leaf_backward(lo, hi, seg_start[j], cur, full=(K == 1))
        for sn, iv in zip(state_names, inits):
            if g[sn] is None:
                continue
            contribute(iv, cur[sn])

    def reduce_rule(ins: Instr, c: str) -> None:
        f = ins.params["f"]
        dims = ins.params["dims"]
        names = (dims,) if isinstance(dims, str) else tuple(dims)
        if f not in RED and f in COMPOSITE_REDUCERS:
            # reduce = select the last slot of the scan, so its adjoint is
            # embed-at-last (zeros elsewhere) then the scan adjoint
            (dim,) = names
            ddim = shadows[ins.operands[0]].dim(dim)
            if ddim.size == 0:
                for e in ins.operands:
                    contribute(e, zeros_like(shadows[e]))
                return
            lc = b.emit("strip_charts", (c,), {}, hint="lat")
            r = b.emit("repeat", (lc,), {"name": dim, "extent": (ddim.stop - 1, ddim.stop)})
            yb = b.emit("pad", (r,), {"fill": 0.0, "extents": {dim: (ddim.start, ddim.stop)}})
            composite_scan_adjoint(f, dim, ins.operands, yb)
            return
        A = ins.operands[0]
        a_shape = shadows[A]
        if f == "sum":
            contribute(A, repeats_over(c, names, a_shape))
        elif f == "mean":
            r = repeats_over(c, names, a_shape)
            n = 1
            for name in names:
                n *= a_shape.dim(name).size
            nb = const_like(a_shape, float(n))
            contribute(A, b.emit("pointwise", (r, nb), {"f": "div"}))
        elif f in ("max", "min"):
            rc = repeats_over(c, names, a_shape)
            rm = repeats_over(ins.var, names, a_shape)
            m = b.emit("pointwise", (A, rm), {"f": "eq"})
            contribute(A, b.emit("pointwise", (rc, m), {"f": "mul"}))
        else:
            raise NotImplementedError(f"reduce({f}) has no adjoint rule yet")

    def scan_rule(ins: Instr, c: str) -> None:
        if ins.params["f"] in COMPOSITE_REDUCERS and ins.params["f"] not in RED:
            composite_scan_adjoint(ins.params["f"], ins.params["dim"], ins.operands, c)
            return
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
            if hi <= lo:
                # empty overlap, including taps entirely outside the source:
                # anchor the empty slice inside src so the pad-back is legal
                lo = hi = src.start
            t3 = b.emit("slice", (t2,), {"ranges": {name: (lo, hi)}})
            t4 = b.emit("pad", (t3,), {"fill": 0.0, "extents": {name: (src.start, src.stop)}})
            contribute(A, t4)

    def decimate_rule(ins: Instr, c: str) -> None:
        p = ins.params
        name, f = p["name"], p["factor"]
        A = ins.operands[0]
        c = b.emit("strip_charts", (c,), {}, hint="lat")  # lattice-mode internals
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
        if shadows[ins.var].dim(z).size == 0:
            # disjoint parts: the diagonal read nothing, the gradient is zero
            contribute(A, zeros_like(shadows[A]))
            return
        c = b.emit("strip_charts", (c,), {}, hint="lat")  # lattice-mode internals
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
        elif ins.op in (
            "strip_charts",
            "with_charts",
            "with_labels",
            "with_value_units",
            "bind",
            "simplify",
            "materialize",
        ):
            contribute(A, c)  # value-preserving metadata / identity copy
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
        elif ins.op == "fold":
            fold_rule(ins, c)
        else:
            layout_rule(ins, c)

    grads: dict[str, str | None] = {}
    names = wrt if wrt is not None else tuple(i.var for i in fwd)
    for v in names:
        gv = final.get(v)
        if gv is not None and target_unit is not None:
            uv = sigs[v].unit  # inferred, so INTERMEDIATE grads annotate too
            gu = target_unit if uv is None or uv == ONE else target_unit / uv
            gv = b.emit("with_value_units", (gv,), {"value_units": gu}, hint="vu")
        grads[v] = gv
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
    charts = {d.name: d.chart for d in base.layout.dims if d.chart is not None}
    labels = {d.name: d.labels for d in base.layout.dims if d.labels is not None}

    def rebuild(pert):
        t = Tensor.from_numpy(pert, names)
        if charts:
            t = t.with_charts(**charts)
        if labels:
            t = t.with_labels(**labels)
        if base.value_units is not None:
            t = t.with_value_units(base.value_units)
        return t

    g = np.zeros_like(arr)
    for idx in np.ndindex(*arr.shape):
        out = []
        for sign in (+1, -1):
            pert = arr.copy()
            pert[idx] += sign * eps
            env = run(prog, {**inputs, wrt_var: rebuild(pert)})
            out.append(float(env[target].item()))
        g[idx] = (out[0] - out[1]) / (2 * eps)
    return g
