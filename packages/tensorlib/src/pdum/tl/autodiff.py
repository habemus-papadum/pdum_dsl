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
- Adjoints of layout ops are layout(+compute) programs: repeat†=reduce-sum, slice†=pad-0, pad†=slice (fill cotangent
  discarded), shift/flip/rename/split/merge are relabelings,
  select†=repeat-at-point+pad, window/stencil†=per-tap overlap-add,
  decimate†=zero-stuffing (materialize+merge; factor-divisible source
  domain only),
  diagonal†=masked embedding (binary only), scan(sum)†=reverse scan.
- Composite reducers differentiate via BPTT-as-IR: the state cotangent of a
  structured-state scan is itself a linear recurrence in reversed time, run
  as a generated matrix-linrec scan over derived Jacobian trees
  (composite_scan_adjoint); reduce† = embed-at-last, then scan†.
- The at-kink law (S.2, re-pinned at P4): the derivative table is one-sided
  and PARTITIONS — at a tie exactly one operand receives the cotangent,
  first-wins. Pointwise maximum/minimum select via the ge/gt asymmetry;
  reduce(max/min) derives as a chain of single-dim reduces with a
  first-occurrence mask, inheriting the partition. Gradient mass is
  conserved regardless of tie structure (test_at_kink pins the points).
- Primitive linearizations come from THE one derivative table
  (derivative.TABLE): a primitive's slope is registered once as a derived
  slope marker (f.d{i}) and applied over the operands — the same interface
  composite partials use. None = gradient-free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from pdum.dsl import events

from .derivative import TABLE
from .layout import _dense_like
from .markers import PW, RED
from .mdsl import CompositeMarker
from .nodes import Arg, Const
from .registry import MARKERS, REDUCERS
from .signatures import infer_signatures
from .units import ONE


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


def grad(
    region,
    target: str,
    inputs: dict | None = None,
    seed: str | None = None,
    wrt: tuple[str, ...] | None = None,
    target_unit=None,
    fold_segments: int | None = None,
    fold_slots: int | None = None,
    *,
    names=None,
) -> "RegionGrad":
    """Reverse-mode AD over a dialect region: returns a ``RegionGrad`` whose
    joint region yields (target, *grads). ``names`` is the naming-law
    assignment (LiftedStep/Assemblage/ZooModel carry it). The module
    docstring's contracts — seeds, restamping, units, the at-kink law, the
    fold schedules — all hold unchanged."""
    # the second compile-ish seam (200 §1.10): adjoint derivation announces
    # itself as a span, nesting the build events it causes
    with events.span("adjoint.derive", (target, wrt)):
        return _region_grad(region, target, inputs, seed, wrt, target_unit, fold_segments, fold_slots, names)


# ----------------------------------------------------------------------
# validation harness
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# the Region face (the excavation, LEVELS) — grad over dialect regions
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RegionGrad:
    """The joint region and its naming. ``region`` yields (target, *grads);
    ``outputs`` names the yield slots in order; ``grads`` maps each
    requested name to its gradient's name (None = no gradient flows);
    ``names`` is the naming-law assignment over the joint (id -> name);
    ``nodes`` maps yielded names to nodes. With ``seed=`` the joint gains
    one extra param, passed LAST at run time."""

    region: object
    grads: dict
    names: dict
    outputs: tuple
    nodes: dict


def _region_grad(region, target, inputs, seed, wrt, target_unit, fold_segments, fold_slots, names):
    from pdum.dsl.ir import Builder, Region
    from pdum.dsl.naming import Namer
    from pdum.dsl.ops import CORE_OPS

    from .dialect import (
        TL_OPS,
        _freeze_params,
        _thaw_params,
        region_names,
        tensor_type_of_layout,
        walk_region,
    )

    if names is None:
        raise ValueError(
            "region grad joins on names — pass names=region_names(region, param_names) "
            "(LiftedStep.names / Assemblage.names / ZooModel.names carry it)"
        )
    by_name: dict = {}
    for p_ in region.params:  # declared params are addressable even when no
        by_name.setdefault(names[id(p_)], p_)  # output reaches them (grad -> None)
    for nd in walk_region(region):
        nm = names.get(id(nd))
        if nm is not None:
            by_name.setdefault(nm, nd)
    if target not in by_name:
        raise KeyError(f"target {target!r} is not defined by the program")
    if seed is not None and seed in by_name:
        raise ValueError(f"seed name {seed!r} collides with a program variable; pick a fresh name")
    if fold_segments is not None and fold_slots is not None:
        raise ValueError(
            "pass fold_segments (uniform K-way checkpointing) OR fold_slots "
            "(binomial revolve), not both — they name different schedules"
        )
    if fold_slots is not None and int(fold_slots) < 1:
        raise ValueError("fold_slots must be >= 1 (the number of checkpoint slots)")
    tnode = by_name[target]

    fwd: list = []  # the target's ancestor DAG, topological (= export order)
    seenw: set = set()

    def anc(nd):
        if id(nd) in seenw:
            return
        seenw.add(id(nd))
        for a in nd.args:
            anc(a)
        fwd.append(nd)

    anc(tnode)

    sigs = infer_signatures(region, inputs or {}, names=names)
    if target_unit is None:
        target_unit = sigs[target].unit

    b = Builder({**CORE_OPS, **TL_OPS})
    namer = Namer(taken=set(names.values()))
    jnames: dict = dict(names)

    def emit(op, operands=(), params=None, hint="g", regions=()):
        node = b.emit("tl." + op, *operands, regions=regions, **_freeze_params(dict(params or {})))
        jnames[id(node)] = namer.derive(f"%{hint}")
        return node

    def lay(nd):
        return nd.type.layout

    def extents_of(layout) -> tuple:
        return tuple((d.name, (d.start, d.stop)) for d in layout.dims)

    def restamp(gv, layout):
        charts, labels = {}, {}
        for d in layout.dims:
            if d.labels is not None:
                labels[d.name] = d.labels
            else:
                charts[d.name] = d.chart  # a Chart, or None to clear
        if charts:
            gv = emit("with_charts", (gv,), {"charts": charts}, hint="st")
        if labels:
            gv = emit("with_labels", (gv,), {"labels": labels}, hint="st")
        levels = {d.name: d.level for d in layout.dims}
        if any(lv is not None for lv in levels.values()):
            gv = emit("bind", (gv,), {"levels": levels}, hint="st")
        return gv

    def const_like(layout, value, dtype=None):
        params = {"value": value, "dims": extents_of(layout)}
        if dtype is not None:
            params["dtype"] = dtype
        return restamp(emit("const", (), params, hint="z"), layout)

    def zeros_like(layout):
        return const_like(layout, 0.0)

    # ---- seed ----------------------------------------------------------
    tlay = lay(tnode)
    cot: dict = {}
    seed_param = None
    if seed is None:
        if len(tlay.dims) != 0:
            raise ValueError(
                f"target {target!r} is not a scalar; pass seed= (the name of a "
                f"runtime input aligned with the target) — reverse mode "
                f"computes vector-Jacobian products"
            )
        cot[id(tnode)] = [emit("const", (), {"value": 1.0, "dims": ()}, hint="seed")]
    else:
        seed_param = b.param(("grad.seed",), tensor_type_of_layout(tlay))
        jnames[id(seed_param)] = namer.claim(seed)
        cot[id(tnode)] = [restamp(seed_param, tlay)]

    final: dict = {}

    def finalize(nd):
        if id(nd) in final:
            return final[id(nd)]
        parts = cot.get(id(nd))
        if not parts:
            return None
        acc = parts[0]
        for p_ in parts[1:]:
            acc = emit("pointwise", (acc, p_), {"f": "add"}, hint="acc")
        final[id(nd)] = acc
        return acc

    def contribute(nd, gv):
        if getattr(nd.type, "layout", None) is None:
            return  # a deferred scalar const: gradient discarded at the leaf
        cot.setdefault(id(nd), []).append(restamp(gv, lay(nd)))

    # ---- per-node adjoint rules (ports of the incumbent's, 1:1) --------
    def pw_rule(nd, c):
        f = dict(nd.attrs)["f"]
        A = nd.args
        if f in TABLE:
            for i, operand in enumerate(A):
                rule = TABLE[f][i]
                if rule is None:
                    continue
                slope = rule(*(Arg(j) for j in range(len(A))))
                if isinstance(slope, Const) and slope.value == 0:
                    continue
                if isinstance(slope, Const) and slope.value == 1:
                    contribute(operand, c)
                    continue
                dn = f"{f}.d{i}"
                pm = MARKERS.derive(dn, lambda dn=dn, slope=slope: CompositeMarker(dn, len(A), slope))
                pv = emit("pointwise", A, {"f": pm.name})
                contribute(operand, emit("pointwise", (c, pv), {"f": "mul"}))
        elif f in MARKERS:
            cm = MARKERS[f]
            for i, operand in enumerate(A):
                p_ = cm.partial(i)
                if isinstance(p_.body, Const) and p_.body.value == 0:
                    continue
                pv = emit("pointwise", A, {"f": p_.name})
                contribute(operand, emit("pointwise", (c, pv), {"f": "mul"}))
        elif f in PW:
            raise NotImplementedError(
                f"marker {f!r} has no entry in the derivative table — a "
                f"primitive joins the core WITH its linearization (silent "
                f"zero gradients are how models rot)"
            )
        else:
            raise KeyError(f"unknown marker {f!r}")

    def repeats_over(v, dim_names, src_layout):
        cur = v
        rebind = {}
        for name in dim_names:
            d = src_layout.dim(name)
            cur = emit(
                "repeat",
                (cur,),
                {"name": name, "extent": (d.start, d.stop), "chart": d.chart, "labels": d.labels},
            )
            if d.level is not None:
                rebind[name] = d.level
        if rebind:
            cur = emit("bind", (cur,), {"levels": rebind})
        return cur

    def composite_scan_adjoint(fname, dim, elems, sc):
        f = REDUCERS[fname]
        k = f.state
        cs, ls, p_marker = f.component_markers()
        ddim = lay(elems[0]).dim(dim)
        if ddim.size == 0:
            for e in elems:
                contribute(e, zeros_like(lay(e)))
            return
        s0, s1 = ddim.start, ddim.stop

        def lat(v):
            return emit("strip_charts", (v,), {}, hint="lat")

        def acc_sum(terms):
            total = terms[0]
            for t in terms[1:]:
                total = emit("pointwise", (total, t), {"f": "add"})
            return total

        sc = lat(sc)
        se = tuple(lat(e) for e in elems)
        sjs = tuple(emit("scan", se, {"f": f.state_scanner(j).name, "dim": dim}) for j in range(k))
        ljs = tuple(emit("pointwise", se, {"f": ls[j].name}) for j in range(k))
        sprev, lnext = [], []
        for j in range(k):
            sh = emit("shift", (sjs[j],), {"deltas": {dim: 1}})
            sl = emit("slice", (sh,), {"ranges": {dim: (s0 + 1, s1)}})
            sprev.append(emit("pad", (sl,), {"fill": float(f.init[j]), "extents": {dim: (s0, s1)}}))
            sh = emit("shift", (ljs[j],), {"deltas": {dim: -1}})
            sl = emit("slice", (sh,), {"ranges": {dim: (s0, s1 - 1)}})
            lnext.append(emit("pad", (sl,), {"fill": 0.0, "extents": {dim: (s0, s1)}}))
        felems = []
        for i in range(k):
            for x in range(k):
                mv = emit("pointwise", sjs + tuple(lnext), {"f": cs[x].partial(i).name})
                felems.append(emit("flip", (mv,), {"name": dim}))
        gs = []
        for i in range(k):
            pv = emit("pointwise", sjs, {"f": p_marker.partial(i).name})
            gs.append(emit("pointwise", (pv, sc), {"f": "mul"}))
            felems.append(emit("flip", (gs[i],), {"name": dim}))
        shat = tuple(
            emit(
                "flip",
                (emit("scan", tuple(felems), {"f": f.adjoint_scanner(i).name, "dim": dim}),),
                {"name": dim},
            )
            for i in range(k)
        )
        lbar = []
        for j in range(k):
            terms = []
            for i in range(k):
                pd = cs[i].partial(k + j)
                if isinstance(pd.body, Const) and pd.body.value == 0:
                    continue
                dv = emit("pointwise", tuple(sprev) + tuple(ljs), {"f": pd.name})
                terms.append(emit("pointwise", (dv, shat[i]), {"f": "mul"}))
            lbar.append(acc_sum(terms) if terms else None)
        for i in range(f.element):
            terms = []
            for j in range(k):
                pd = ls[j].partial(i)
                if lbar[j] is None or (isinstance(pd.body, Const) and pd.body.value == 0):
                    continue
                dv = emit("pointwise", se, {"f": pd.name})
                terms.append(emit("pointwise", (dv, lbar[j]), {"f": "mul"}))
            if terms:
                contribute(elems[i], acc_sum(terms))

    def _with_yield(step, ynew):
        wb2 = Builder({**CORE_OPS, **TL_OPS})
        y = ynew[0] if len(ynew) == 1 else wb2.emit("core.tuple", *ynew)
        return Region(params=step.params, body=(wb2.emit("core.yield", y),))

    def fold_rule(nd, c):
        """The port of the incumbent fold adjoint onto the positional
        contract: the adjoint step is the recursive region-grad of the
        scalarized wrapper; every re-out of a step is a variant region
        sharing all nodes with a different yield."""
        at = _thaw_params(dict(nd.attrs))
        state_names, elem_names = tuple(at["state"]), tuple(at["element"])
        k = len(state_names)
        step = nd.regions[0]
        inits, elems = nd.args[:k], nd.args[k:]
        out = tuple(at["out"])
        out_kind = out[0]
        dim = at["dim"]
        if elems:
            d0 = lay(elems[0]).dim(dim)
            start, stop = d0.start, d0.stop
        else:
            start, stop = tuple(at["extent"])
        yielded = step.body[-1].args[0]
        ynodes = tuple(yielded.args) if yielded.op == "core.tuple" else (yielded,)
        if stop - start == 0:
            for i, iv in enumerate(inits):
                if out_kind == "final" and out[1] == i:
                    contribute(iv, c)
                else:
                    contribute(iv, zeros_like(lay(iv)))
            for ev_ in elems:
                contribute(ev_, zeros_like(lay(ev_)))
            return

        def flp(v):
            return emit("flip", (v,), {"name": dim})

        slayouts = {nm: lay(p_) for nm, p_ in zip(state_names + elem_names, step.params)}
        if out_kind == "final":
            r = emit("repeat", (c,), {"name": dim, "extent": (stop - 1, stop)})
            yb = emit("pad", (r,), {"fill": 0.0, "extents": {dim: (start, stop)}})
        else:
            yb = c

        # the VJP wrapper: the step's own params + cotangent params, target =
        # sum of cot·carry inner products + cot·out — then grad, recursively
        wb = Builder({**CORE_OPS, **TL_OPS})
        cot_params = {}
        terms = []

        def scalarize(vnode):
            cin = wb.param(("ct", len(cot_params)), tensor_type_of_layout(lay(vnode)))
            pr = wb.emit("tl.pointwise", vnode, cin, f="mul")
            dnames = tuple(d.name for d in lay(vnode).dims)
            if not dnames:
                return cin, pr
            return cin, wb.emit("tl.reduce", pr, f="sum", dims=dnames)

        for i, sn in enumerate(state_names):
            cot_params[sn], t0 = scalarize(ynodes[i])
            terms.append(t0)
        out_node = ynodes[out[1]] if out_kind == "final" else ynodes[k]
        cot_out, t0 = scalarize(out_node)
        terms.append(t0)
        tgt = terms[0]
        for tv in terms[1:]:
            tgt = wb.emit("tl.pointwise", tgt, tv, f="add")
        ct_names = tuple(f"%ct_{sn}" for sn in state_names)
        wparams = tuple(step.params) + tuple(cot_params[sn] for sn in state_names) + (cot_out,)
        wrapper = Region(params=wparams, body=(wb.emit("core.yield", tgt),))
        wnames = region_names(wrapper, state_names + elem_names + ct_names + ("%ct_out",))
        wg = _region_grad(
            wrapper, wnames[id(tgt)], {}, None, state_names + elem_names, None, None, None, wnames
        )

        def gnode(v):
            return None if wg.grads[v] is None else wg.nodes[wg.grads[v]]

        def ensure(gn, layout):
            if gn is not None:
                return gn
            v = wb.emit("tl.const", value=0.0, dims=extents_of(layout))
            charts = {d.name: d.chart for d in layout.dims if d.chart is not None}
            labels = {d.name: d.labels for d in layout.dims if d.labels is not None}
            if charts:
                v = wb.emit("tl.with_charts", v, charts=tuple(sorted(charts.items())))
            if labels:
                v = wb.emit("tl.with_labels", v, labels=tuple(sorted(labels.items())))
            levels = {d.name: d.level for d in layout.dims if d.level is not None}
            if levels:
                v = wb.emit("tl.bind", v, levels=tuple(sorted(levels.items())))
            return v

        cb_nodes = tuple(ensure(gnode(sn), slayouts[sn]) for sn in state_names)
        adj_param_order = tuple(cot_params[sn] for sn in state_names) + tuple(step.params) + (cot_out,)
        adj_at = {"dim": dim, "state": ct_names, "element": state_names + elem_names + ("%ct_out",)}

        def adj_step(emit_node=None):
            y = cb_nodes + ((emit_node,) if emit_node is not None else ())
            wb2 = Builder({**CORE_OPS, **TL_OPS})
            yt = y[0] if len(y) == 1 else wb2.emit("core.tuple", *y)
            return Region(params=adj_param_order, body=(wb2.emit("core.yield", yt),))

        adj_base = adj_step()
        traj_steps = {i: _with_yield(step, ynodes[:k] + (ynodes[i],)) for i in range(k)}
        base_at = {"dim": dim, "state": state_names, "element": elem_names}

        def seg_ops(state_nodes, lo, hi):
            if elems:
                se = tuple(emit("slice", (ev_,), {"ranges": {dim: (lo, hi)}}) for ev_ in elems)
                return tuple(state_nodes) + se, {}, se
            return tuple(state_nodes), {"extent": (lo, hi)}, ()

        def zero_state():
            return {sn: zeros_like(slayouts[sn]) for sn in state_names}

        def advance(s0, lo, hi):
            ops_, extra_p, _ = seg_ops(tuple(s0[sn] for sn in state_names), lo, hi)
            return {
                sn: emit("fold", ops_, {**base_at, **extra_p, "out": ("final", i)}, regions=(step,))
                for i, sn in enumerate(state_names)
            }

        def leaf_backward(lo, hi, s0, cur, full):
            ops_, extra_p, se = seg_ops(tuple(s0[sn] for sn in state_names), lo, hi)
            sprev = {}
            for i, sn in enumerate(state_names):
                traj = emit("fold", ops_, {**base_at, **extra_p, "out": ("emit",)}, regions=(traj_steps[i],))
                sh = emit("shift", (traj,), {"deltas": {dim: 1}})
                sl = emit("slice", (sh,), {"ranges": {dim: (lo + 1, hi)}})
                pd = emit("pad", (sl,), {"fill": 0.0, "extents": {dim: (lo, hi)}})
                ri = emit("repeat", (s0[sn],), {"name": dim, "extent": (lo, hi)})
                it = emit("iota", (pd,), {"name": dim})
                sdims = extents_of(slayouts[sn]) + ((dim, (lo, hi)),)
                cs = emit("const", (), {"value": lo, "dims": sdims, "dtype": "int64"})
                cs = restamp(cs, slayouts[sn])  # partial stamp: scan dim stays bare
                mask = emit("pointwise", (it, cs), {"f": "eq"})
                sprev[sn] = emit("pointwise", (mask, ri, pd), {"f": "where"})
            ybj = yb if full else emit("slice", (yb,), {"ranges": {dim: (lo, hi)}})
            adj_ops = (
                tuple(cur[sn] for sn in state_names)
                + tuple(flp(sprev[sn]) for sn in state_names)
                + tuple(flp(x) for x in se)
                + (flp(ybj),)
            )
            for en, ev_ in zip(elem_names, elems):
                gn = gnode(en)
                if gn is None:
                    continue
                fv = flp(emit("fold", adj_ops, {**adj_at, "out": ("emit",)}, regions=(adj_step(gn),)))
                if not full:
                    fv = emit("pad", (fv,), {"fill": 0.0, "extents": {dim: (start, stop)}})
                contribute(ev_, fv)
            return {
                sn: emit("fold", adj_ops, {**adj_at, "out": ("final", i)}, regions=(adj_base,))
                for i, sn in enumerate(state_names)
            }

        T = stop - start
        if fold_slots is not None:
            S = int(fold_slots)
            cur = zero_state()

            def revolve(lo, hi, s, boundary):
                nonlocal cur
                span = hi - lo
                if span <= 1 or s >= span:
                    cur = leaf_backward(lo, hi, boundary, cur, full=(lo == start and hi == stop))
                    return
                c_ = lo + _revolve_split(s, span)
                state_c = advance(boundary, lo, c_)
                revolve(c_, hi, s - 1, state_c)
                revolve(lo, c_, s, boundary)

            revolve(start, stop, S, dict(zip(state_names, inits)))
            for sn, iv in zip(state_names, inits):
                if gnode(sn) is None:
                    continue
                contribute(iv, cur[sn])
            return

        K = 1 if fold_segments is None else min(int(fold_segments), T)
        if K < 1:
            raise ValueError("fold_segments must be >= 1")
        if T % K:
            raise ValueError(f"fold_segments={K} must divide the fold extent {T} (pad the dim or pick a divisor)")
        L = T // K
        seg_start = [dict(zip(state_names, inits))]
        for j in range(K - 1):
            lo, hi = start + j * L, start + (j + 1) * L
            seg_start.append(advance(seg_start[-1], lo, hi))
        cur = zero_state()
        for j in reversed(range(K)):
            lo, hi = start + j * L, start + (j + 1) * L
            cur = leaf_backward(lo, hi, seg_start[j], cur, full=(K == 1))
        for sn, iv in zip(state_names, inits):
            if gnode(sn) is None:
                continue
            contribute(iv, cur[sn])

    def reduce_rule(nd, c):
        p = _thaw_params(dict(nd.attrs))
        f = p["f"]
        dims = p["dims"]
        dim_names = (dims,) if isinstance(dims, str) else tuple(dims)
        if f not in RED and f in REDUCERS:
            (dim,) = dim_names
            ddim = lay(nd.args[0]).dim(dim)
            if ddim.size == 0:
                for e in nd.args:
                    contribute(e, zeros_like(lay(e)))
                return
            lc = emit("strip_charts", (c,), {}, hint="lat")
            r = emit("repeat", (lc,), {"name": dim, "extent": (ddim.stop - 1, ddim.stop)})
            yb = emit("pad", (r,), {"fill": 0.0, "extents": {dim: (ddim.start, ddim.stop)}})
            composite_scan_adjoint(f, dim, nd.args, yb)
            return
        A = nd.args[0]
        a_shape = lay(A)
        if f == "sum":
            contribute(A, repeats_over(c, dim_names, a_shape))
        elif f == "mean":
            r = repeats_over(c, dim_names, a_shape)
            count = 1
            for name in dim_names:
                count *= a_shape.dim(name).size
            nb = const_like(a_shape, float(count))
            contribute(A, emit("pointwise", (r, nb), {"f": "div"}))
        elif f in ("max", "min"):
            plain = tuple(replace(d, chart=None, labels=None) for d in a_shape.dims)

            def rep(v, d0):
                r = emit("repeat", (v,), {"name": d0.name, "extent": (d0.start, d0.stop)}, hint="km")
                if d0.level is not None:
                    r = emit("bind", (r,), {"levels": {d0.name: d0.level}}, hint="km")
                return r

            links = []
            srcp, sdims = emit("strip_charts", (A,), {}, hint="kk"), plain
            for i, name in enumerate(dim_names):
                last = i == len(dim_names) - 1
                dst = nd if last else emit("reduce", (srcp,), {"f": f, "dims": (name,)}, hint="kr")
                dstp = emit("strip_charts", (dst,), {}, hint="kk") if last else dst
                links.append((srcp, sdims, name))
                sdims = tuple(d for d in sdims if d.name != name)
                srcp = dstp
            prev, cotan = srcp, emit("strip_charts", (c,), {}, hint="kc")
            for srcp, sdims, name in reversed(links):
                d0 = next(d for d in sdims if d.name == name)
                lay_ = _dense_like(sdims)
                one, zero = const_like(lay_, 1.0), const_like(lay_, 0.0)
                m = emit("pointwise", (srcp, rep(prev, d0)), {"f": "eq"}, hint="kt")
                mnum = emit("pointwise", (m, one, zero), {"f": "where"}, hint="kn")
                s = emit("scan", (mnum,), {"f": "sum", "dim": name}, hint="ks")
                fw = emit("pointwise", (s, one), {"f": "eq"}, hint="kf")
                first = emit("pointwise", (m, fw), {"f": "mul"}, hint="kw")
                cotan = emit("pointwise", (rep(cotan, d0), first), {"f": "mul"}, hint="kg")
                prev = srcp
            contribute(A, cotan)
        else:
            raise NotImplementedError(f"reduce({f}) has no adjoint rule yet")

    def scan_rule(nd, c):
        p = dict(nd.attrs)
        if p["f"] in REDUCERS and p["f"] not in RED:
            composite_scan_adjoint(p["f"], p["dim"], nd.args, c)
            return
        if p["f"] != "sum":
            raise NotImplementedError("only scan(sum) is differentiable so far")
        dim = p["dim"]
        f1 = emit("flip", (c,), {"name": dim})
        s1 = emit("scan", (f1,), {"f": "sum", "dim": dim})
        contribute(nd.args[0], emit("flip", (s1,), {"name": dim}))

    def tap_rule(nd, c, p):
        name, k_name = p["name"], p.get("k_name") or f"{p['name']}_k"
        dilation = p.get("dilation", 1)
        A = nd.args[0]
        src = lay(A).dim(name)
        kdim = lay(nd).dim(k_name)
        for kappa in range(kdim.start, kdim.stop):
            t1 = emit("select", (c,), {"coords": {k_name: kappa}})
            t2 = emit("shift", (t1,), {"deltas": {name: kappa * dilation}})
            anchor = lay(nd).dim(name)
            lo = max(src.start, anchor.start + kappa * dilation)
            hi = min(src.stop, anchor.stop + kappa * dilation)
            if hi <= lo:
                lo = hi = src.start
            t3 = emit("slice", (t2,), {"ranges": {name: (lo, hi)}})
            t4 = emit("pad", (t3,), {"fill": 0.0, "extents": {name: (src.start, src.stop)}})
            contribute(A, t4)

    def decimate_rule(nd, c, p):
        name, f = p["name"], p["factor"]
        A = nd.args[0]
        c = emit("strip_charts", (c,), {}, hint="lat")
        src = lay(A).dim(name)
        phase = src.delta_to_lattice(p.get("phase", 0)) % f
        s, e = src.start, src.stop
        outd = lay(nd).dim(name)
        if (e - s) % f or (e - s) // f != outd.size:
            raise NotImplementedError(
                f"decimate adjoint needs a factor-divisible domain: [{s}, {e}) with factor {f} (pad the source first)"
            )
        slot = (phase - s) % f
        used = {d.name for d in lay(nd).dims}

        def fresh_dim(base):
            i = 0
            while f"_{base}{i}" in used:
                i += 1
            used.add(f"_{base}{i}")
            return f"_{base}{i}"

        cname = fresh_dim("c")
        ph = fresh_dim("ph")
        r0 = emit("rename", (c,), {"mapping": {name: cname}})
        r1 = emit("repeat", (r0,), {"name": ph, "extent": (0, f)})
        i1 = emit("iota", (r1,), {"name": ph})
        r1_dims = tuple((cname if d.name == name else d.name, (d.start, d.stop)) for d in lay(nd).dims) + (
            (ph, (0, f)),
        )
        cp = emit("const", (), {"value": slot, "dims": r1_dims, "dtype": "int64"})
        m = emit("pointwise", (i1, cp), {"f": "eq"})
        z0 = emit("const", (), {"value": 0.0, "dims": r1_dims})
        w = emit("pointwise", (m, r1, z0), {"f": "where"})
        others = tuple(nm for nm, _ in r1_dims if nm not in (cname, ph))
        mo = emit("materialize", (w,), {"order": others + (cname, ph)})
        mg = emit("merge", (mo,), {"parts": (cname, ph), "name": name, "start": s})
        contribute(A, mg)

    def diagonal_rule(nd, c, p):
        parts = tuple(p["parts"])
        if len(parts) != 2:
            raise NotImplementedError("n-ary diagonal adjoint not written yet")
        x, y = parts
        z = p["name"]
        A = nd.args[0]
        xdom, ydom = lay(A).dim(x), lay(A).dim(y)
        if lay(nd).dim(z).size == 0:
            contribute(A, zeros_like(lay(A)))
            return
        c = emit("strip_charts", (c,), {}, hint="lat")
        r1 = c if z == x else emit("rename", (c,), {"mapping": {z: x}})
        r2 = emit("repeat", (r1,), {"name": y, "extent": (ydom.start, ydom.stop)})
        ix = emit("iota", (r2,), {"name": x})
        iy = emit("iota", (r2,), {"name": y})
        m = emit("pointwise", (ix, iy), {"f": "eq"})
        r2_dims = tuple((x if d.name == z else d.name, (d.start, d.stop)) for d in lay(nd).dims) + (
            (y, (ydom.start, ydom.stop)),
        )
        z0 = emit("const", (), {"value": 0.0, "dims": r2_dims})
        w = emit("pointwise", (m, r2, z0), {"f": "where"})
        zdim = lay(nd).dim(z)
        if (zdim.start, zdim.stop) != (xdom.start, xdom.stop):
            w = emit("pad", (w,), {"fill": 0.0, "extents": {x: (xdom.start, xdom.stop)}})
        contribute(A, w)

    def layout_rule(nd, c):
        base = nd.op[3:]
        A = nd.args[0]
        a_shape = lay(A)
        p = _thaw_params(dict(nd.attrs))
        if base == "slice":
            extents = {nm: (a_shape.dim(nm).start, a_shape.dim(nm).stop) for nm in p["ranges"]}
            contribute(A, emit("pad", (c,), {"fill": 0.0, "extents": extents}))
        elif base == "pad":
            ranges = {nm: (a_shape.dim(nm).start, a_shape.dim(nm).stop) for nm in p["extents"]}
            contribute(A, emit("slice", (c,), {"ranges": ranges}))
        elif base == "shift":
            deltas = {nm: -a_shape.dim(nm).delta_to_lattice(v) for nm, v in p["deltas"].items()}
            contribute(A, emit("shift", (c,), {"deltas": deltas}))
        elif base == "flip":
            contribute(A, emit("flip", (c,), {"name": p["name"]}))
        elif base == "rename":
            inv = {new: old for old, new in p["mapping"].items()}
            contribute(A, emit("rename", (c,), {"mapping": inv}))
        elif base == "repeat":
            contribute(A, emit("reduce", (c,), {"f": "sum", "dims": (p["name"],)}))
        elif base == "select":
            cur = c
            coord_names = list(p["coords"])
            for nm, coord in p["coords"].items():
                i = a_shape.dim(nm).to_lattice(coord)
                cur = emit("repeat", (cur,), {"name": nm, "extent": (i, i + 1)})
            extents = {nm: (a_shape.dim(nm).start, a_shape.dim(nm).stop) for nm in coord_names}
            contribute(A, emit("pad", (cur,), {"fill": 0.0, "extents": extents}))
        elif base == "split":
            parts = tuple(p["parts"])
            others = tuple(d.name for d in lay(nd).dims if d.name not in parts)
            mo = emit("materialize", (c,), {"order": others + parts})
            contribute(
                A,
                emit("merge", (mo,), {"parts": parts, "name": p["name"], "start": a_shape.dim(p["name"]).start}),
            )
        elif base == "merge":
            # ORDERED parts: pre-tupled so the canonical attr sort cannot
            # reorder the mixed-radix nesting (the lifting packer precedent)
            parts = tuple((nm, (a_shape.dim(nm).start, a_shape.dim(nm).stop)) for nm in p["parts"])
            contribute(A, emit("split", (c,), {"name": p["name"], "parts": parts}))
        elif base in ("window", "stencil"):
            tap_rule(nd, c, p)
        elif base == "decimate":
            decimate_rule(nd, c, p)
        elif base == "diagonal":
            diagonal_rule(nd, c, p)
        elif base in (
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
            raise NotImplementedError(f"no adjoint rule for {base!r}")

    # ---- reverse walk --------------------------------------------------
    for nd in reversed(fwd):
        c = finalize(nd)
        if c is None:
            continue
        if nd.op in ("core.param", "core.const") or not nd.op.startswith("tl."):
            continue
        base = nd.op[3:]
        if base in ("const", "iota", "random"):
            continue
        if base in ("argtopk", "argsort"):
            continue
        if base == "repeat_like":
            have = {d.name for d in lay(nd.args[0]).dims}
            added = tuple(d.name for d in lay(nd).dims if d.name not in have)
            gv = emit("reduce", (c,), {"f": "sum", "dims": added}) if added else c
            contribute(nd.args[0], gv)
        elif base == "round_to":
            if dict(nd.attrs).get("grad", "straight_through") != "zero":
                contribute(nd.args[0], c)
        elif base == "take":
            d = lay(nd.args[0]).dim(dict(nd.attrs)["dim"])
            survivors = {x.name for x in lay(nd.args[0]).dims if x.name != d.name}
            spliced = tuple(nm for nm in lay(nd.args[1]).names if nm not in survivors)
            gv = emit(
                "scatter_add",
                (c, nd.args[1]),
                {"dim": d.name, "extent": (d.start, d.stop), "over": spliced},
            )
            contribute(nd.args[0], gv)
        elif base == "scatter_add":
            gv = emit("take", (c, nd.args[1]), {"dim": dict(nd.attrs)["dim"]})
            contribute(nd.args[0], gv)
        elif base == "pointwise":
            pw_rule(nd, c)
        elif base == "reduce":
            reduce_rule(nd, c)
        elif base == "scan":
            scan_rule(nd, c)
        elif base == "fold":
            fold_rule(nd, c)
        else:
            layout_rule(nd, c)

    # ---- outputs -------------------------------------------------------
    grads: dict = {}
    req = tuple(wrt) if wrt is not None else tuple(names[id(p_)] for p_ in region.params)
    ynodes_out = [tnode]
    outputs = [target]
    for v in req:
        node_v = by_name.get(v)
        if node_v is None:
            raise KeyError(f"wrt name {v!r} is not defined by the program")
        gn = finalize(node_v)
        if gn is not None and target_unit is not None:
            uv = sigs[v].unit if v in sigs else None
            gu = target_unit if uv is None or uv == ONE else target_unit / uv
            gn = emit("with_value_units", (gn,), {"value_units": gu}, hint="vu")
        if gn is None:
            grads[v] = None
            continue
        grads[v] = jnames[id(gn)]
        ynodes_out.append(gn)
        outputs.append(jnames[id(gn)])
    y = ynodes_out[0] if len(ynodes_out) == 1 else b.emit("core.tuple", *ynodes_out)
    joint_params = tuple(region.params) + ((seed_param,) if seed_param is not None else ())
    joint = Region(params=joint_params, body=(b.emit("core.yield", y),))
    return RegionGrad(joint, grads, jnames, tuple(outputs), {jnames[id(x)]: x for x in ynodes_out})


def numeric_grad(region, target: str, wrt_var: str, inputs: dict, names, eps: float = 1e-6):
    """Central finite differences of a SCALAR target w.r.t. one input tensor,
    on the region face. Rebuilds perturbed inputs via from_numpy with the
    base tensor's stamps — use simple inputs in FD tests."""
    import numpy as np

    from .dialect import run_named
    from .tensor import Tensor

    base = inputs[wrt_var]
    arr = base.to_numpy().astype(np.float64)
    tnames = base.names
    charts = {d.name: d.chart for d in base.layout.dims if d.chart is not None}
    labels = {d.name: d.labels for d in base.layout.dims if d.labels is not None}
    levels = {d.name: d.level for d in base.layout.dims if d.level is not None}

    def rebuild(pert):
        t = Tensor.from_numpy(pert, tnames)
        if charts:
            t = t.with_charts(**charts)
        if labels:
            t = t.with_labels(**labels)
        if levels:
            t = t.bind(**levels)
        if base.value_units is not None:
            t = t.with_value_units(base.value_units)
        return t

    g = np.zeros_like(arr)
    for idx in np.ndindex(*arr.shape):
        out = []
        for sign in (+1, -1):
            pert = arr.copy()
            pert[idx] += sign * eps
            vals = run_named(region, {**inputs, wrt_var: rebuild(pert)}, names)
            out.append(float(vals[target].to_numpy()))
        g[idx] = (out[0] - out[1]) / (2 * eps)
    return g
