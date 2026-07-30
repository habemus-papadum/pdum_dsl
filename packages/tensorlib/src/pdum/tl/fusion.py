"""The fusion registry, first two rows (330 §2, §7.3).

Template 1 — **map chain**: a region of pointwise/layout/iota work with no
reduction and no effects. Reuse-free, so no staging: the group IS a tile
kernel already, and its certificate is proved-exact by construction (the
generated kernel and the subgraph share a content key).

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

from pdum.dsl.ir import Builder, Region
from pdum.dsl.ops import CORE_OPS

from .certify import Certificate, certify
from .dialect import TL_OPS, _minus_dim, check_tier, walk_region
from .licenses import GEMM_F16_TILES

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
    if (pa, pb) != (rb.args[1], ra.args[1]) or pa.op != "core.param" or pb.op != "core.param":
        return None
    core = {id(red), id(prod), id(ra), id(rb), id(pa), id(pb)}
    for n in walk_region(region):  # the whole region is core + epilogue, or we decline
        if id(n) in core or n.op == "core.yield":
            continue
        if n.op not in _EPILOGUE_OPS:
            return None
    kdim = next((d for d in _dims(pa.type) if d.name == k), None)
    if kdim is None or kdim.start != 0 or any(d.start != 0 for d in _dims(red.type)):
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
    at = b.emit("tl.split", newp[id(m["a"])], name=k, parts=(("ko", ko), ("ki", ki)))
    bt = b.emit("tl.split", newp[id(m["b"])], name=k, parts=(("ko", ko), ("ki", ki)))
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


def plan_region(region: Region) -> Plan:
    """The v1 pass: recognize the whole region against the registry, in
    template order; certify what matched; refuse the rest LOUDLY."""
    m = _match_contraction_epilogue(region)
    if m is not None:
        ki = _pick_ki(m["extent"])
        kernel = _generate_contraction(region, m, ki)
        cert = certify(kernel, region, licenses=_REASSOC)
        group = Group("contraction-epilogue", kernel, cert, "yellow", params=(("ki", ki),))
        return Plan((group,))
    offender = _is_map_chain(region)
    if offender is None:
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
