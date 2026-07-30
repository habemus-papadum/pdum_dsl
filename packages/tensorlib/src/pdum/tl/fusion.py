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

from dataclasses import dataclass

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
_EPILOGUE_OPS = frozenset({"tl.pointwise", "tl.repeat_like", "core.const", "tl.const", "core.param"})
_REASSOC = tuple(lic for lic in GEMM_F16_TILES if lic.kind == "reassociation")


@dataclass(frozen=True)
class Group:
    """One fusion group: the template that claimed it, the generated tile
    kernel, its certificate, and the confidence color. Red groups carry
    the refusal reason and no kernel — the reference serves them."""

    template: str  # "map-chain" | "contraction-epilogue" | "uncompiled"
    kernel: Region | None
    certificate: Certificate | None
    confidence: str  # "green" | "yellow" | "red"
    params: tuple = ()  # plan parameters, e.g. (("ki", 4),)
    reason: str = ""
    launch: tuple = ()  # (output dim, tile) pairs — the 340 §2 plan artifact
    prune: tuple = ()  # (scan dim, (lo0, dlo), (hi0, dhi)) — mask-derived bounds (340 §4b)


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
    """The strict v1 shape: exactly one reduce — sum over ONE dim, over
    ``mul(rl(a, b), rl(b, a))`` with a and b params — and everything
    between it and the yield drawn from the epilogue vocabulary
    (pointwise/repeat_like/consts, broadcasts riding). Anything else
    declines: a matcher never guesses (330 §2)."""
    yld = region.body[-1].args[0]
    if yld.op == "core.tuple":
        return None
    reduces = [n for n in walk_region(region) if n.op == "tl.reduce"]
    if len(reduces) != 1:
        return None
    red = reduces[0]
    a = dict(red.attrs)
    dims = (a["dims"],) if isinstance(a["dims"], str) else tuple(a["dims"])
    if a["f"] != "sum" or len(dims) != 1 or a.get("zero") is not None:
        return None
    (k,) = dims
    prod = red.args[0]
    if prod.op != "tl.pointwise" or dict(prod.attrs).get("f") != "mul" or len(prod.args) != 2:
        return None
    ra, rb = prod.args
    if ra.op != "tl.repeat_like" or rb.op != "tl.repeat_like":
        return None
    pa, pb = ra.args[0], rb.args[0]
    core = {id(red), id(prod), id(ra), id(rb), id(pa), id(pb), id(ra.args[1]), id(rb.args[1])}
    for v in (pa, pb):  # free views ride the operand (renamed activations);
        while v.op == "tl.rename":  # the LIKES are dims-only credentials — any
            core.add(id(v))  # broadcast pair contracts, adjoints included
            v = v.args[0]
            core.add(id(v))
        if v.op != "core.param":
            return None
    for n in walk_region(region):  # the whole region is core + epilogue, or we decline
        if id(n) in core or n.op == "core.yield":
            continue
        if n.op not in _EPILOGUE_OPS:
            return None
    kdim = next((d for d in _dims(pa.type) if d.name == k), None)
    kb = next((d for d in _dims(pb.type) if d.name == k), None)
    if kdim is None or kb is None or kdim.start != 0 or any(d.start != 0 for d in _dims(red.type)):
        return None
    return {"reduce": red, "k": k, "extent": kdim.stop, "a": pa, "b": pb}


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
    accumulator — fused values never touch memory."""
    k, extent, red = m["k"], m["extent"], m["reduce"]
    ko = extent // ki
    b = Builder(OPS)
    newp = {id(p): b.param(i, p.type) for i, p in enumerate(region.params)}

    def operand(x):  # free-view chains ride into the kernel
        if x.op == "core.param":
            return newp[id(x)]
        return b.emit(x.op, operand(x.args[0]), **dict(x.attrs))

    at = b.emit("tl.split", operand(m["a"]), name=k, parts=(("ko", ko), ("ki", ki)))
    bt = b.emit("tl.split", operand(m["b"]), name=k, parts=(("ko", ko), ("ki", ki)))
    acc0 = b.emit("tl.const", value=0.0, dims=tuple((d.name, d.stop) for d in _dims(red.type)))

    sb = Builder(OPS)
    p_acc = sb.param(0, acc0.type)
    p_a = sb.param(1, _minus_dim(at.type, "ko"))
    p_b = sb.param(2, _minus_dim(bt.type, "ko"))
    a_s = sb.emit("tl.stage", p_a, level="shared")
    b_s = sb.emit("tl.stage", p_b, level="shared")
    prod = sb.emit("tl.pointwise", sb.emit("tl.repeat_like", a_s, b_s), sb.emit("tl.repeat_like", b_s, a_s), f="mul")
    part = sb.emit("tl.reduce", prod, dims=("ki",), f="sum")
    nxt = sb.emit("tl.pointwise", p_acc, part, f="add")
    step = Region(params=(p_acc, p_a, p_b), body=(sb.emit("core.yield", nxt),))

    fold = b.emit(
        "tl.fold", acc0, at, bt, regions=(step,), dim="ko", state=("acc",), element=("a", "b"), out=("final", 0)
    )

    # the epilogue, rebuilt over the accumulator: reduce -> fold, params -> new params
    memo: dict[int, object] = {id(red): fold, **newp}

    def rebuild(n):
        if id(n) in memo:
            return memo[id(n)]
        args = tuple(rebuild(x) for x in n.args)
        explicit = {} if OPS[n.op].type_rule is not None else {"type": n.type}
        out = b.emit(n.op, *args, loc=n.loc, **explicit, **dict(n.attrs))
        memo[id(n)] = out
        return out

    yld = rebuild(region.body[-1].args[0])
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
    return {"sm": sm, "s": sd[0]}


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
    q:(t,e), k:(s,e), v:(s,o) params; riders arrive with the partitioner."""
    yld = region.body[-1].args[0]
    out = _contract_core(yld)
    if out is None:
        return None
    pr, v, s = out
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
    fused = Region(params=tuple(newp[id(p)] for p in region.params), body=(b.emit("core.yield", out),))
    return check_tier(fused, "tile")


def _score_families(region: Region, seed: int = 11):
    """Template 3's own adversarial families (the template knows its
    failure modes): unit gaussians, wide scores (exp near saturation,
    max-shifted), and a dominant key (the rescale at its extreme)."""

    def draw(scale, subseed, spike=False):
        def factory():
            rng = np.random.default_rng(subseed)
            out = {}
            for i, p in enumerate(region.params):
                dims = _dims(p.type)
                arr = scale * rng.standard_normal(tuple(d.stop - d.start for d in dims))
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
        cert = certify(kernel, region, licenses=FLASH_ONLINE_SOFTMAX, families=_score_families(region))
        return Plan((Group("flash", kernel, cert, "yellow", params=(("si", si),)),))
    m = _match_contraction_epilogue(region)
    if m is not None:
        ki = _pick_ki(m["extent"])
        kernel = _generate_contraction(region, m, ki)
        cert = certify(kernel, region, licenses=_REASSOC)
        return Plan((Group("contraction-epilogue", kernel, cert, "yellow", params=(("ki", ki),)),))
    sm = _match_softmax(region.body[-1].args[0])
    if sm is not None:
        rows = frozenset(
            id(p) for p in region.params if any(d.name == sm["s"] for d in _dims(p.type))
        )
        kernel = _stage_params(region, rows)
        cert = certify(kernel, region)
        return Plan((Group("row-normalization", kernel, cert, "yellow"),))
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
