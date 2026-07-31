"""The fusion registry (330 §2, §7.3–4): four template rows + the refusal law.

Template 1 — **map chain**: a region of pointwise/layout/iota work with no
reduction and no effects. Reuse-free, so no staging: the group IS a tile
kernel already, and its certificate is proved-exact by construction (the
generated kernel and the subgraph share a content key).

Template 4 — **stencil**: the map chain WITH neighborhood reuse — a param
read at two or more distinct shifted offsets. The generator stages the
halo once to "shared" and reads every neighbor from the staged copy; the
certificate stays proved-exact (erasure strips the stage).

Template 3 — **row normalization / flash**: the reduce-max/sub/exp/
reduce-sum/div chain. Bare, its inputs stage once (proved-exact). BETWEEN
two contractions over the same dim — with a CLOSED mask forest (pointwise
over iota leaves; an iota's backing is a lattice credential, never
values) — it is the flash composition: the generator emits the
online-softmax fold (three carried states, the mask riding as a split
element source, q as a stride-0 repeat, two fold nodes sharing the step
for the two finals) and certification runs licensed-differential under
``flash.online-softmax`` with the template's OWN adversarial families —
the template knows its failure modes.

Template 2 — **contraction + epilogue**: the ``rl/mul/reduce`` contract
idiom, optionally followed by a pointwise epilogue (bias, activation —
tensor traffic riding broadcast ``repeat_like``). The generator emits the
GEMM tile shape (split k, fold over ko with staged operand slices, the
carry spelled by absence) and REBUILDS the epilogue over the accumulator
— the fused values never touch memory, the same structure a hand author
writes. Certification consumes the declared reassociation license, and
the normalization chain reaches the unfused subgraph's content key: the
fusion is PROVED, not trusted.

Everything else REFUSES — a red, uncompiled group with the reason in the
rulebook voice, served by the reference or a framework column. Never
silent mediocrity (330's law). Plans are DATA: partition, template
assignment, parameters (the k-tile width is a plan parameter), and the
confidence color — yellow (certified, analytic only) until a measurement
attaches (green; the measurement analysis lives with the machine that
evaluates it, keyed through the analysis cache).

V1 recognizes WHOLE regions — one group per plan. The anchor-and-absorb
partitioner over multi-group regions is §7.5's step, deliberately after
the templates it would assign.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pdum.dsl.ir import Builder, Region
from pdum.dsl.ops import CORE_OPS

from .analysis import defanalysis
from .certify import Certificate, certify
from .dialect import TL_OPS, _minus_dim, check_tier, walk_region
from .licenses import FLASH_ONLINE_SOFTMAX, GEMM_F16_TILES
from .tensor import Tensor

OPS = {**CORE_OPS, **TL_OPS}

_MAP_OPS = frozenset({"core.param", "core.const", "core.yield", "tl.pointwise", "tl.iota", "tl.const"}) | frozenset(
    f"tl.{n}"
    for n in (
        "slice",
        "select",
        "shift",
        "rename",
        "repeat",
        "flip",
        "split",
        "merge",
        "pad",
        "window",
        "stencil",
        "strip_charts",
        "with_charts",
        "with_labels",
        "simplify",
        "repeat_like",
    )
)
_CHART_OPS = ("tl.with_charts", "tl.strip_charts", "tl.simplify")
_EXACT_OPS = frozenset(
    {"tl.pointwise", "tl.repeat_like", "tl.rename", "tl.repeat", "tl.iota", "core.const", "tl.const"}
) | frozenset(_CHART_OPS)  # recompute-exact vocabulary: prologues and epilogues both draw from it (§7.6)
_REASSOC = tuple(lic for lic in GEMM_F16_TILES if lic.kind == "reassociation")


def _unchart(n):
    """Charts are metadata views — matchers look through them (§7.6)."""
    while n.op in _CHART_OPS:
        n = n.args[0]
    return n


@dataclass(frozen=True)
class Group:
    """One fusion group: the template that claimed it, the generated tile
    kernel, its certificate, and the confidence color. Red groups carry
    the refusal reason and no kernel — the reference serves them."""

    template: str  # "map-chain" | "contraction-epilogue" | "uncompiled"
    kernel: Region | None = field(repr=False)  # IR reprs recurse the DAG: keep failures printable
    certificate: Certificate | None
    confidence: str  # "green" | "yellow" | "red"
    params: tuple = ()  # plan parameters, e.g. (("ki", 4),)
    reason: str = ""
    launch: tuple = ()  # (output dim, tile) pairs — the 340 §2 plan artifact
    prune: tuple = ()  # (scan dim, (lo0, dlo), (hi0, dhi)) — mask-derived bounds (340 §4b)
    artifacts: tuple = ()  # surfaced fold-carried finals, in tuple-yield order after the output (§7.8)


@dataclass(frozen=True)
class Plan:
    """A fusion plan is DATA — inspectable, diffable, comparable."""

    groups: tuple[Group, ...]


def _is_map_chain(region: Region) -> str | None:
    """None when the region is a pure map chain; else the first offender."""
    for n in walk_region(region):
        if n.op not in _MAP_OPS:
            return n.op
    return None


def _match_contraction_epilogue(region: Region):
    """The contraction row, §7.6 shape: exactly one reduce — sum over ONE
    dim of a two-operand product whose operands both CARRY the contracted
    dim — and every other node drawn from the recompute-exact vocabulary
    (prologues absorbed upstream, epilogues downstream, charts looked
    through). Operands may be computed chains — params are the trivial
    case. A broadcast pair with disjoint kept dims is the GEMM shape;
    same-space operands are the ROWSUM shape (the softmax adjoint's
    rowsum(dP*P)). Anything else declines: a matcher never guesses."""
    yld = region.body[-1].args[0]
    if yld.op == "core.tuple":
        return None
    reduces = [n for n in walk_region(region) if n.op == "tl.reduce"]
    red = None
    if len(reduces) == 1:
        red = reduces[0]
    else:  # §7.8: tile-local reduces may live in operand prologues — the
        for r in reduces:  # OUTER reduce is the one whose cone holds the rest
            cone, stack = set(), list(r.args)
            while stack:
                x = stack.pop()
                if id(x) in cone:
                    continue
                cone.add(id(x))
                stack.extend(x.args)
            if all(o is r or id(o) in cone for o in reduces):
                red = r
                break
    if red is None:
        return None
    a = dict(red.attrs)
    dims = (a["dims"],) if isinstance(a["dims"], str) else tuple(a["dims"])
    if a["f"] not in ("sum", "mean") or len(dims) not in (1, 2) or a.get("zero") is not None:
        return None  # mean contracts too: the fold sums, a scale finalizes (§7.6)
    for r in reduces:  # a prologue reduce must be TILE-LOCAL: disjoint from the
        if r is red:  # swept dims — a swept reduction cannot travel (§7.8)
            continue
        rd = dict(r.attrs)["dims"]
        rdt = {rd} if isinstance(rd, str) else set(rd)
        if rdt & set(dims) or dict(r.attrs).get("zero") is not None:
            return None
    prod = _unchart(red.args[0])
    if prod.op == "tl.pointwise" and dict(prod.attrs).get("f") == "mul" and len(prod.args) == 2:
        cores = []
        for x in prod.args:
            x = _unchart(x)
            if x.op == "tl.repeat_like":  # a broadcast rides; its LIKE is a dims-only credential
                x = _unchart(x.args[0])
            cores.append(x)
    else:  # PLAIN shape: a sum with no product at all (bias gradients) —
        cores = [prod]  # the degenerate rowsum, one operand riding the fold
    if any(k not in {d.name for d in _dims(c.type)} for c in cores for k in dims):
        return None
    keeps = [tuple(d.name for d in _dims(c.type) if d.name not in dims) for c in cores]
    out = tuple(d.name for d in _dims(red.type))
    if set(out) != set().union(*map(set, keeps)):
        return None
    ca = cores[0]
    cb = cores[1] if len(cores) == 2 else None
    if cb is None:
        shape = "plain"
    else:
        full_a = tuple(d.name for d in _dims(ca.type))
        full_b = tuple(d.name for d in _dims(cb.type))
        # same-space operands are the ROWSUM shape (plain mul); different spaces
        # take the broadcast pair, which joins riders and disjoint kepts alike —
        # operand order stays the original spelling's (the step mirrors it)
        shape = "rowsum" if full_a == full_b else "gemm"
    core = {id(red), id(prod)} | {id(r) for r in reduces}  # prologue reduces validated above
    for n in walk_region(region):  # the whole region is exact vocabulary, or we decline
        if id(n) in core or n.op in ("core.yield", "core.param"):
            continue
        if n.op not in _EXACT_OPS:
            return None
    total = 1
    for k in dims:
        kdim = next(d for d in _dims(ca.type) if d.name == k)
        kb = kdim if cb is None else next(d for d in _dims(cb.type) if d.name == k)
        if (kdim.start, kdim.stop) != (kb.start, kb.stop) or kdim.start != 0:
            return None
        total *= kdim.stop
    if any(d.start != 0 for d in _dims(red.type)):
        return None
    # the fold runs over the WIDEST contracted dim; the rest reduce inside
    # the step whole (330 §7.6 C — what a hand author writes)
    k = max(dims, key=lambda nm: next(d.stop for d in _dims(ca.type) if d.name == nm))
    extent = next(d.stop for d in _dims(ca.type) if d.name == k)
    rest = tuple(nm for nm in dims if nm != k)
    return {
        "reduce": red,
        "k": k,
        "extent": extent,
        "rest": rest,
        "total": total,
        "a": ca,
        "b": cb,
        "shape": shape,
        "f": a["f"],
    }


def _dims(t):
    return t.dims


def _pick_ki(extent: int) -> int:
    """The v1 tile width: the largest divisor <= 32 that leaves >= 2 tiles;
    a plan PARAMETER, not a truth — the measured loop revises it."""
    for ki in range(min(32, extent // 2), 0, -1):
        if extent % ki == 0:
            return ki
    return extent


def _generate_contraction(region: Region, m: dict, ki: int) -> Region:
    """Emit the fused tile kernel: split k, fold over ko with staged operand
    slices (gemm_tile's shape), then the epilogue REBUILT over the
    accumulator — fused values never touch memory. Operands may be computed
    prologue chains (§7.6): they rebuild ahead of the split and ride as
    fold element sources, computed per k-tile by request-driven emission —
    the flash mask's pattern, recompute-exact by per-element identity."""
    k, extent, red = m["k"], m["extent"], m["reduce"]
    ko = extent // ki
    b = Builder(OPS)
    newp = {id(p): b.param(i, p.type) for i, p in enumerate(region.params)}
    memo: dict[int, object] = dict(newp)  # shared by prologue and epilogue rebuilds

    shape = m.get("shape", "gemm")
    at = b.emit("tl.split", _rebuild_into(b, memo, m["a"]), name=k, parts=(("ko", ko), ("ki", ki)))
    bt = None
    if shape != "plain":
        bt = b.emit("tl.split", _rebuild_into(b, memo, m["b"]), name=k, parts=(("ko", ko), ("ki", ki)))
    acc0 = b.emit("tl.const", value=0.0, dims=tuple((d.name, d.stop) for d in _dims(red.type)))

    sb = Builder(OPS)
    p_acc = sb.param(0, acc0.type)
    p_a = sb.param(1, _minus_dim(at.type, "ko"))
    a_s = sb.emit("tl.stage", p_a, level="shared")
    if shape == "plain":  # one operand, no product: the degenerate rowsum
        prod, params = a_s, (p_acc, p_a)
    else:
        p_b = sb.param(2, _minus_dim(bt.type, "ko"))
        b_s = sb.emit("tl.stage", p_b, level="shared")
        params = (p_acc, p_a, p_b)
        if shape == "rowsum":  # same-space operands: no broadcast pair
            prod = sb.emit("tl.pointwise", a_s, b_s, f="mul")
        else:
            prod = sb.emit(
                "tl.pointwise", sb.emit("tl.repeat_like", a_s, b_s), sb.emit("tl.repeat_like", b_s, a_s), f="mul"
            )
    part = sb.emit("tl.reduce", prod, dims=("ki", *m.get("rest", ())), f="sum")
    nxt = sb.emit("tl.pointwise", p_acc, part, f="add")
    step = Region(params=params, body=(sb.emit("core.yield", nxt),))

    srcs = (at,) if shape == "plain" else (at, bt)
    names = ("a",) if shape == "plain" else ("a", "b")
    fold = b.emit(
        "tl.fold", acc0, *srcs, regions=(step,), dim="ko", state=("acc",), element=names, out=("final", 0)
    )

    # the epilogue, rebuilt over the accumulator: reduce -> fold, params -> new params
    memo[id(red)] = fold
    if m.get("f") == "mean":  # the divide-by-N finalize: N static, one scalar op
        from pdum.dsl.types import f64

        memo[id(red)] = b.emit(
            "tl.pointwise", fold, b.emit("core.const", type=f64, value=1.0 / m.get("total", extent)), f="mul"
        )
    yld = _rebuild_into(b, memo, region.body[-1].args[0])
    fused = Region(params=tuple(newp[id(p)] for p in region.params), body=(b.emit("core.yield", yld),))
    return check_tier(fused, "tile")


def _rebuild_into(b: Builder, memo: dict, n):
    """Generic re-emission with a seeded memo (params, holes) — the one
    rebuild shape every generator shares."""
    if id(n) in memo:
        return memo[id(n)]
    args = tuple(_rebuild_into(b, memo, x) for x in n.args)
    explicit = {} if OPS[n.op].type_rule is not None else {"type": n.type}
    out = b.emit(n.op, *args, loc=n.loc, **explicit, **dict(n.attrs))
    memo[id(n)] = out
    return out


def _stage_params(region: Region, staged_ids: frozenset) -> Region:
    """Template 3a/4's generator: the same chain with the named params
    STAGED once to "shared" — one load, all reads over the staged copy.
    Erasure strips the stages, so the certificate is exact by key."""
    b = Builder(OPS)
    newp = {id(p): b.param(i, p.type) for i, p in enumerate(region.params)}
    memo: dict[int, object] = dict(newp)
    for pid in staged_ids:
        memo[pid] = b.emit("tl.stage", newp[pid], level="shared")
    yld = _rebuild_into(b, memo, region.body[-1].args[0])
    staged = Region(params=tuple(newp[id(p)] for p in region.params), body=(b.emit("core.yield", yld),))
    return check_tier(staged, "tile")


def _layout_root(n):
    while n.op in ("tl.shift", "tl.slice", "tl.stage", "tl.rename", "tl.pad", "tl.simplify"):
        n = n.args[0]
    return n


def _neighborhood_params(region: Region) -> frozenset:
    """The stencil signal (330 §1): a param read at >= 2 DISTINCT shifted
    offsets has overlap — staging its halo once pays."""
    shifts: dict[int, set] = {}
    for n in walk_region(region):
        if n.op == "tl.shift":
            root = _layout_root(n.args[0])
            if root.op == "core.param":
                shifts.setdefault(id(root), set()).add(n.attrs)
    return frozenset(pid for pid, deltas in shifts.items() if len(deltas) >= 2)


def _match_softmax(pr):
    """The row-normalization chain, as the zoo spells it:
    ``div(exp(sm - rl(max_s(sm))), rl(sum_s(exp(...))))`` — both reduces
    over the SAME single dim. Returns {sm, s} or None."""
    if pr.op != "tl.pointwise" or dict(pr.attrs).get("f") != "div" or len(pr.args) != 2:
        return None
    e, rld = pr.args
    if e.op != "tl.pointwise" or dict(e.attrs).get("f") != "exp" or rld.op != "tl.repeat_like":
        return None
    sume = rld.args[0]
    if sume.op != "tl.reduce" or dict(sume.attrs)["f"] != "sum" or sume.args[0] is not e:
        return None
    sub = e.args[0]
    if sub.op != "tl.pointwise" or dict(sub.attrs).get("f") != "sub" or len(sub.args) != 2:
        return None
    sm, rlm = sub.args
    if rlm.op != "tl.repeat_like":
        return None
    maxm = rlm.args[0]
    if maxm.op != "tl.reduce" or dict(maxm.attrs)["f"] != "max" or maxm.args[0] is not sm:
        return None
    if rlm.args[1] is not sm or rld.args[1] not in (sm, e):
        return None
    sdims = dict(sume.attrs)["dims"]
    mdims = dict(maxm.attrs)["dims"]
    sd = (sdims,) if isinstance(sdims, str) else tuple(sdims)
    md = (mdims,) if isinstance(mdims, str) else tuple(mdims)
    if sd != md or len(sd) != 1:
        return None
    return {"sm": sm, "s": sd[0], "max": maxm, "den": sume}


def _match_rowstat(pr):
    """The row-statistics core (330 §7.6), as the zoo spells layernorm —
    TWO-PASS, the stable form: ``div(xc, rl(sqrt(mean(xc*xc) + eps)))``
    with ``xc = x - rl(mean(x))``, both means over the SAME single dim.
    Structurally softmax's two-sweep shape; scale-shift is epilogue.
    Returns {x, feat} or None."""
    if pr.op != "tl.pointwise" or dict(pr.attrs).get("f") != "div" or len(pr.args) != 2:
        return None
    xc, rld = pr.args
    if rld.op != "tl.repeat_like":
        return None
    sd = rld.args[0]
    if sd.op != "tl.pointwise" or dict(sd.attrs).get("f") != "sqrt":
        return None
    add = sd.args[0]
    if add.op != "tl.pointwise" or dict(add.attrs).get("f") != "add" or len(add.args) != 2:
        return None
    var, eps = add.args
    if eps.op not in ("core.const", "tl.const"):
        return None
    if var.op != "tl.reduce" or dict(var.attrs)["f"] != "mean":
        return None
    sq = var.args[0]
    if sq.op != "tl.pointwise" or dict(sq.attrs).get("f") != "mul" or sq.args[0] is not sq.args[1]:
        return None
    if sq.args[0] is not xc:
        return None
    if xc.op != "tl.pointwise" or dict(xc.attrs).get("f") != "sub" or len(xc.args) != 2:
        return None
    x, rlm = xc.args
    if rlm.op != "tl.repeat_like":
        return None
    mu = rlm.args[0]
    if mu.op != "tl.reduce" or dict(mu.attrs)["f"] != "mean" or mu.args[0] is not x:
        return None
    vd, md = (dict(r.attrs)["dims"] for r in (var, mu))
    vt = (vd,) if isinstance(vd, str) else tuple(vd)
    mt = (md,) if isinstance(md, str) else tuple(md)
    if vt != mt or len(vt) != 1:
        return None
    return {"x": x, "feat": vt[0]}


def _contract_core(red, kname=None):
    """reduce(sum, mul(rl(x, y), rl(y, x)), (dim,)) -> (x, y, dim) or None."""
    if red.op != "tl.reduce":
        return None
    a = dict(red.attrs)
    dims = (a["dims"],) if isinstance(a["dims"], str) else tuple(a["dims"])
    if a["f"] != "sum" or len(dims) != 1 or a.get("zero") is not None:
        return None
    if kname is not None and dims[0] != kname:
        return None
    prod = red.args[0]
    if prod.op != "tl.pointwise" or dict(prod.attrs).get("f") != "mul" or len(prod.args) != 2:
        return None
    ra, rb = prod.args
    if ra.op != "tl.repeat_like" or rb.op != "tl.repeat_like":
        return None
    x, y = ra.args[0], rb.args[0]
    if (x, y) != (rb.args[1], ra.args[1]):
        return None
    return x, y, dims[0]


def _forest_is_closed(n, stop) -> bool:
    """A mask forest is CLOSED when it reads no data: pointwise/consts over
    iota leaves (an iota's backing is a lattice credential, never values)."""
    if n.op == "tl.iota":
        return True  # leaf: the backing's VALUES are never read
    if n.op in ("core.const", "tl.const"):
        return True
    if n.op in ("tl.pointwise", "tl.repeat_like") or n.op in _MAP_OPS:
        return all(_forest_is_closed(x, stop) for x in n.args)
    return False


def _match_flash(region: Region):
    """The flash composition (330 §2): contraction -> closed-form mask ->
    row normalization -> contraction over the same dim. Strict v1 shapes:
    q:(t,e), k:(s,e), v:(s,o) params; riders arrive with the partitioner.
    The ARTIFACT variant (§7.8): a tuple yield (out, m, den) surfacing
    the composition's own row statistics — the fold already carries
    both, so surfacing costs stores, never FLOPs."""
    yld = region.body[-1].args[0]
    artifacts = False
    if yld.op == "core.tuple":
        if len(yld.args) != 3:
            return None
        yld, m_out, den_out = yld.args
        artifacts = True
    out = _contract_core(yld)
    if out is None:
        return None
    pr, v, s = out
    if artifacts:  # the surfaced statistics must BE this composition's own
        sm0 = _match_softmax(pr)
        if sm0 is None or m_out is not sm0["max"] or den_out is not sm0["den"]:
            return None
    if v.op != "core.param":
        return None
    sm_m = _match_softmax(pr)
    if sm_m is None or sm_m["s"] != s:
        return None
    smn = sm_m["sm"]
    if smn.op != "tl.pointwise" or dict(smn.attrs).get("f") != "where" or len(smn.args) != 3:
        return None
    mask, sc, fill = smn.args
    if fill.op not in ("core.const", "tl.const") or not _forest_is_closed(mask, None):
        return None
    core = _contract_core(sc)
    if core is None:
        return None
    q, k, e = core
    if q.op != "core.param" or k.op != "core.param":
        return None
    if {id(q), id(k), id(v)} != {id(p) for p in region.params}:
        return None
    qd, kd, vd = [tuple(d.name for d in _dims(p.type)) for p in (q, k, v)]
    if len(qd) != 2 or kd != (s, e) or len(vd) != 2 or vd[0] != s:
        return None
    sdim = next(d for d in _dims(k.type) if d.name == s)
    if sdim.start != 0:
        return None
    return {
        "q": q,
        "k": k,
        "v": v,
        "mask": mask,
        "fill": float(dict(fill.attrs)["value"]),
        "t": next(n for n in qd if n != e),
        "s": s,
        "e": e,
        "o": vd[1],
        "extent": sdim.stop,
        "artifacts": artifacts,
    }


def _generate_flash(region: Region, m: dict, si: int) -> Region:
    """The generalized flash_tile shape: the mask forest rebuilds as a
    closed form over a const lattice, splits along s, and rides as an
    element source; q rides as a stride-0 repeat; two fold nodes share
    the step for the two finals (o, den)."""
    s, t, fill = m["s"], m["t"], m["fill"]
    so = m["extent"] // si
    tdim = next(d for d in _dims(m["q"].type) if d.name == t)
    odim = next(d for d in _dims(m["v"].type) if d.name == m["o"])
    b = Builder(OPS)
    newp = {id(p): b.param(i, p.type) for i, p in enumerate(region.params)}
    nq, nk, nv = newp[id(m["q"])], newp[id(m["k"])], newp[id(m["v"])]

    lattice = b.emit("tl.const", value=0.0, dims=((t, tdim.stop), (s, m["extent"])))
    memo: dict[int, object] = dict(newp)
    for n in walk_region(region):  # iota backings re-root on the lattice: values were never read
        if n.op == "tl.iota":
            memo[id(n)] = b.emit("tl.iota", lattice, name=dict(n.attrs)["name"])
    mask = _rebuild_into(b, memo, m["mask"])

    kt = b.emit("tl.split", nk, name=s, parts=(("so", so), ("si", si)))
    vt = b.emit("tl.split", nv, name=s, parts=(("so", so), ("si", si)))
    mt = b.emit("tl.split", mask, name=s, parts=(("so", so), ("si", si)))
    qr = b.emit("tl.repeat", nq, name="so", extent=so)
    m0 = b.emit("tl.const", value=-1e30, dims=((t, tdim.stop),))
    den0 = b.emit("tl.const", value=0.0, dims=((t, tdim.stop),))
    o0 = b.emit("tl.const", value=0.0, dims=((t, tdim.stop), (m["o"], odim.stop)))

    sb = Builder(OPS)
    p_m = sb.param(0, m0.type)
    p_den = sb.param(1, den0.type)
    p_o = sb.param(2, o0.type)
    p_q = sb.param(3, _minus_dim(qr.type, "so"))
    p_k = sb.param(4, _minus_dim(kt.type, "so"))
    p_v = sb.param(5, _minus_dim(vt.type, "so"))
    p_mk = sb.param(6, _minus_dim(mt.type, "so"))
    ks = sb.emit("tl.stage", p_k, level="shared")
    vs = sb.emit("tl.stage", p_v, level="shared")
    prod = sb.emit("tl.pointwise", sb.emit("tl.repeat_like", p_q, ks), sb.emit("tl.repeat_like", ks, p_q), f="mul")
    sc = sb.emit("tl.reduce", prod, dims=(m["e"],), f="sum")
    from pdum.dsl.types import f64

    sm = sb.emit("tl.pointwise", p_mk, sc, sb.emit("core.const", type=f64, value=fill), f="where")
    m_new = sb.emit("tl.pointwise", p_m, sb.emit("tl.reduce", sm, dims=("si",), f="max"), f="maximum")
    alpha = sb.emit("tl.pointwise", sb.emit("tl.pointwise", p_m, m_new, f="sub"), f="exp")
    p_w = sb.emit("tl.pointwise", sb.emit("tl.pointwise", sm, sb.emit("tl.repeat_like", m_new, sm), f="sub"), f="exp")
    den_new = sb.emit(
        "tl.pointwise",
        sb.emit("tl.pointwise", p_den, alpha, f="mul"),
        sb.emit("tl.reduce", p_w, dims=("si",), f="sum"),
        f="add",
    )
    pv = sb.emit("tl.pointwise", sb.emit("tl.repeat_like", p_w, vs), sb.emit("tl.repeat_like", vs, p_w), f="mul")
    o_new = sb.emit(
        "tl.pointwise",
        sb.emit("tl.pointwise", p_o, sb.emit("tl.repeat_like", alpha, p_o), f="mul"),
        sb.emit("tl.reduce", pv, dims=("si",), f="sum"),
        f="add",
    )
    step = Region(
        params=(p_m, p_den, p_o, p_q, p_k, p_v, p_mk),
        body=(sb.emit("core.yield", sb.emit("core.tuple", m_new, den_new, o_new)),),
    )

    fold_args = (m0, den0, o0, qr, kt, vt, mt)
    fold_kw = dict(dim="so", state=("m", "den", "o"), element=("q", "k", "v", "mask"))
    f_o = b.emit("tl.fold", *fold_args, regions=(step,), out=("final", 2), **fold_kw)
    f_den = b.emit("tl.fold", *fold_args, regions=(step,), out=("final", 1), **fold_kw)
    out = b.emit("tl.pointwise", f_o, b.emit("tl.repeat_like", f_den, f_o), f="div")
    if m.get("artifacts"):  # §7.8: the carried finals surface — stores, never FLOPs
        f_m = b.emit("tl.fold", *fold_args, regions=(step,), out=("final", 0), **fold_kw)
        out = b.emit("core.tuple", out, f_m, f_den)
    fused = Region(params=tuple(newp[id(p)] for p in region.params), body=(b.emit("core.yield", out),))
    return check_tier(fused, "tile")


def _score_families(region: Region, seed: int = 11):
    """Adversarial families for the reducing rows (the template knows its
    failure modes): unit gaussians, wide scores (exp near saturation,
    max-shifted; for contractions, cancellation at magnitude), and a
    dominant row (flash's rescale at its extreme; magnitude disparity
    for a contraction's reassociation)."""

    masks = set()  # params consumed (through views) as where-conditions draw as MASKS
    for n in walk_region(region):  # derivative markers too: where.d1 keeps the carrier discipline
        if n.op == "tl.pointwise" and dict(n.attrs).get("f", "").startswith("where"):
            r = n.args[0]
            while r.op in ("tl.rename", "tl.repeat_like", "tl.repeat", "tl.slice", "tl.shift") + _CHART_OPS:
                r = r.args[0]
            if r.op == "core.param":
                masks.add(id(r))

    def draw(scale, subseed, spike=False):
        def factory():
            rng = np.random.default_rng(subseed)
            out = {}
            for i, p in enumerate(region.params):
                dims = _dims(p.type)
                shape = tuple(d.stop - d.start for d in dims)
                if id(p) in masks:
                    out[f"p{i}"] = Tensor.from_numpy(rng.random(shape) < 0.5, tuple(d.name for d in dims))
                    continue
                arr = scale * rng.standard_normal(shape)
                if spike and i == 1:
                    arr[0] *= 40.0
                out[f"p{i}"] = Tensor.from_numpy(arr, tuple(d.name for d in dims))
            return out

        return factory

    return (
        ("gaussian", draw(1.0, seed)),
        ("wide-scores", draw(8.0, seed + 1)),
        ("dominant-key", draw(1.0, seed + 2, spike=True)),
    )


def _prune_flash(region: Region, *, tile: int, si: int):
    """Mask-derived fold bounds (340 §4b): the flash template DECLARES its
    step inert on mask-false tiles (fill enters max as a no-op, exp
    underflows to exact zero), so the general emptiness engine turns the
    closed mask into per-program [lo, hi) over s-tiles — affine in the
    program id, or no prune at all. Legality edge: a row with NO live
    tile has uniform-softmax semantics and refuses pruning outright."""
    from .launch import fit_affine, tri_eval

    fm = _match_flash(region)
    if fm is None:
        return None
    t, s, mask, S = fm["t"], fm["s"], fm["mask"], fm["extent"]
    so = S // si
    td = next(d for d in _dims(fm["q"].type) if d.name == t)
    sbox = [(q * si, (q + 1) * si - 1) for q in range(so)]
    los, his = [], []
    for g in range(-((td.start - td.stop) // tile)):
        tb = (td.start + g * tile, min(td.start + (g + 1) * tile, td.stop) - 1)
        act = [q for q in range(so) if tri_eval(mask, {t: tb, s: sbox[q]})[1] != "F"]
        if not act:
            return None  # a fully-masked program: uniform-softmax semantics
        los.append(act[0])
        his.append(act[-1] + 1)
    for r in range(td.start, td.stop):  # every ROW keeps a live tile (the m-chain law)
        if all(tri_eval(mask, {t: (r, r), s: sb})[1] == "F" for sb in sbox):
            return None
    flo, fhi = fit_affine(los, 0, so), fit_affine(his, 0, so)
    return (flo, fhi) if flo is not None and fhi is not None else None


_PRUNE_FLASH = defanalysis("fusion.prune-flash", 1)(_prune_flash)


_CERT_CAP = 128  # §7.7: a plan-level constant, not a machine fact


@defanalysis("certify.certificate", 1)
def _CERT_FACT(tile: Region, naive: Region, *, kind: str):
    """Certificates are FACTS (330 §4): keyed by the twin pair's content,
    never re-derived warm — the differential run (the expensive half of
    the §7.6 rows) rides the cache like every other analysis."""
    if kind == "flash":
        c = certify(tile, naive, licenses=FLASH_ONLINE_SOFTMAX, families=_score_families(naive))
    else:
        c = certify(tile, naive, licenses=_REASSOC, families=_score_families(naive))
    return {
        "verdict": c.verdict,
        "licenses": list(c.licenses),
        "families": list(c.families),
        "key_reached": c.key_reached,
    }


def _cert_fact(tile: Region, naive: Region, kind: str) -> Certificate:
    d = _CERT_FACT(tile, naive, kind=kind).value
    return Certificate(d["verdict"], tuple(d["licenses"]), tuple(d["families"]), d["key_reached"])


def _shrink(region: Region, cap: int) -> Region | None:
    """The §7.7 twin: the SAME program over clamped extents. Rebuilds the
    region with every dim stop at most ``cap`` — param types re-derived,
    extent-carrying attrs (tl.const dims, tl.repeat) clamped, everything
    else re-typed by rule. None when a dim's start is nonzero (nothing
    to clamp against; the caller certifies full size, honestly)."""
    from .dialect import tensor_type_of_layout

    if any(d.start != 0 for p in region.params for d in _dims(p.type)):
        return None
    b = Builder(OPS)
    memo: dict[int, object] = {}
    for i, p in enumerate(region.params):
        dims = _dims(p.type)
        shape = tuple(min(d.stop, cap) for d in dims)
        t = tensor_type_of_layout(Tensor.from_numpy(np.zeros(shape), tuple(d.name for d in dims)).layout)
        memo[id(p)] = b.param(i, t)

    def rebuild(n):
        if id(n) in memo:
            return memo[id(n)]
        args = tuple(rebuild(x) for x in n.args)
        attrs = dict(n.attrs)
        if n.op == "tl.const" and "dims" in attrs:
            attrs["dims"] = tuple(  # widths, or (start, stop) pairs — both forms clamp
                (nm, (e[0], min(int(e[1]), cap)) if isinstance(e, tuple) else min(int(e), cap))
                for nm, e in tuple(attrs["dims"])
            )
        if n.op == "tl.repeat" and "extent" in attrs:
            e = attrs["extent"]  # a bare width, or the adjoint's (start, stop) pair
            attrs["extent"] = (e[0], min(int(e[1]), cap)) if isinstance(e, tuple) else min(int(e), cap)
        explicit = {} if OPS[n.op].type_rule is not None else {"type": n.type}  # scalars: size-free
        out = b.emit(n.op, *args, loc=n.loc, **explicit, **attrs)
        memo[id(n)] = out
        return out

    yld = rebuild(region.body[-1].args[0])
    return Region(params=tuple(memo[id(p)] for p in region.params), body=(b.emit("core.yield", yld),))


def _oversized(region: Region, cap: int) -> bool:
    return any(d.stop - d.start > cap for p in region.params for d in _dims(p.type))


def plan_region(region: Region, machine=None, floor: int = 1024) -> Plan:
    """The v1 pass: recognize the whole region against the registry, most
    specific template first; certify what matched; refuse the rest LOUDLY.
    With a ``machine`` (340): each compiled group carries the analytic
    default launch — smallest feasible tile above the floor; no feasible
    launch keeps grid (1,) and the translator's tripwire owns the rest."""
    plan = _recognize(region)
    if machine is None:
        return plan
    from dataclasses import replace

    from .launch import propose

    groups = []
    for g in plan.groups:
        cands = propose(g.kernel, machine, floor) if g.kernel is not None else ()
        if cands:
            g = replace(g, launch=cands[0])
        if g.template == "flash" and len(g.launch) == 1:
            pr = _PRUNE_FLASH(region, tile=g.launch[0][1], si=dict(g.params)["si"]).value
            if pr is not None:
                g = replace(g, prune=(("so", *pr),))
        groups.append(g)
    return Plan(tuple(groups))


def _recognize(region: Region) -> Plan:
    fm = _match_flash(region)
    if fm is not None:
        si = _pick_ki(fm["extent"])
        kernel = _generate_flash(region, fm, si)
        csrc, ckern = region, kernel
        if _oversized(region, _CERT_CAP):  # §7.7: certify the shrunk twin —
            shr = _shrink(region, _CERT_CAP)  # the program, never the size
            fm2 = _match_flash(shr) if shr is not None else None
            if fm2 is not None:
                csrc = shr
                ckern = _generate_flash(shr, fm2, _pick_ki(fm2["extent"]))
        cert = _cert_fact(ckern, csrc, "flash")
        arts = ("m", "den") if fm.get("artifacts") else ()
        return Plan((Group("flash", kernel, cert, "yellow", params=(("si", si),), artifacts=arts),))
    m = _match_contraction_epilogue(region)
    if m is not None:
        ki = _pick_ki(m["extent"])
        kernel = _generate_contraction(region, m, ki)
        csrc, ckern = region, kernel
        if _oversized(region, _CERT_CAP):  # §7.7 again: the differential sites
            shr = _shrink(region, _CERT_CAP)  # are the OOM class, and only they
            m2 = _match_contraction_epilogue(shr) if shr is not None else None
            if m2 is not None:
                csrc = shr
                ckern = _generate_contraction(shr, m2, _pick_ki(m2["extent"]))
        # families are the differential fallback: computed-operand kernels the
        # normalizer cannot walk back to the twin's key still certify (§7.6)
        cert = _cert_fact(ckern, csrc, "contraction")
        return Plan((Group("contraction-epilogue", kernel, cert, "yellow", params=(("ki", ki),)),))
    sm = _match_softmax(region.body[-1].args[0])
    if sm is not None:
        rows = frozenset(
            id(p) for p in region.params if any(d.name == sm["s"] for d in _dims(p.type))
        )
        kernel = _stage_params(region, rows)
        cert = certify(kernel, region)
        return Plan((Group("row-normalization", kernel, cert, "yellow"),))
    rs = next((m for n in walk_region(region) if (m := _match_rowstat(n)) is not None), None)
    if rs is not None:
        reduces = [n for n in walk_region(region) if n.op == "tl.reduce"]
        feat_ok = all(
            dict(r.attrs)["f"] == "mean"
            and (lambda d: ((d,) if isinstance(d, str) else tuple(d)) == (rs["feat"],))(dict(r.attrs)["dims"])
            for r in reduces
        )
        vocab_ok = all(
            n.op in _EXACT_OPS or n.op in ("core.yield", "core.param", "tl.reduce") for n in walk_region(region)
        )
        if len(reduces) == 2 and feat_ok and vocab_ok:
            rows = frozenset(
                id(p) for p in region.params if any(d.name == rs["feat"] for d in _dims(p.type))
            )
            kernel = _stage_params(region, rows)
            cert = certify(kernel, region)
            return Plan((Group("row-statistics", kernel, cert, "yellow"),))
    offender = _is_map_chain(region)
    if offender is None:
        halos = _neighborhood_params(region)
        if halos:
            kernel = _stage_params(region, halos)
            cert = certify(kernel, region)
            idxs = tuple(i for i, p in enumerate(region.params) if id(p) in halos)
            return Plan((Group("stencil", kernel, cert, "yellow", params=(("staged", idxs),)),))
        kernel = check_tier(region, "tile")
        cert = certify(kernel, region)
        return Plan((Group("map-chain", kernel, cert, "yellow"),))
    return Plan(
        (
            Group(
                "uncompiled",
                None,
                None,
                "red",
                reason=(
                    f"no recognized schedule: {offender} has no template row — "
                    f"the reference serves this group (330 §2)"
                ),
            ),
        )
    )
