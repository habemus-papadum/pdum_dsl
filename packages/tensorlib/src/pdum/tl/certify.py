"""The equivalence harness (320 §6): certified rewrite chains over regions.

A tile program and its naive twin are the same denotation spelled at two
disciplines; the harness's job is to say WHY, with the strongest available
certificate:

1. **proved-exact** — normalization by EXACT rewrites (stage erasure,
   split/merge cancellation, split-commutation, nested-reduce fusion)
   reaches the twin's content key. Zero numerics: the region hash IS the
   proof.
2. **proved-licensed** — the chain also consumed LICENSED rewrites (each
   tagged with a license kind and admitted only when the caller declares a
   license of that kind — the closed taxonomy, licenses.py). The key is
   still reached; the certificate names what was consumed. The k-sum
   re-bracketing is the worked case: fold-of-accumulation inlines to a
   reduce, and a reduce over split parts collapses to the source dim —
   both `reassociation`.
3. **licensed-differential** — normalization stops short (flash's
   online-softmax lemma is algebra, not syntax), so the declared licenses
   gate an ADVERSARIAL differential (260's law: never random draws alone;
   families come from the caller), tolerances quoted FROM the licenses.

A deviation with no declared license REFUSES loudly — a tile program that
neither normalizes to its twin nor names its deviation is not certified.

Rewrites are bottom-up region rebuilds (erase_stages' shape); the driver
runs the rule set to a key fixpoint. Matching is deliberately narrow —
a rule that does not recognize a shape declines, and the harness falls
through to the next certificate tier; nothing is ever rewritten on a
guess. Known gap, recorded: the non-divisible-tail family arrives with a
pad-then-split flagship variant (the pad fill is exact for sums).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pdum.dsl.ir import Builder, Region
from pdum.dsl.ops import CORE_OPS

from .dialect import TL_OPS, run_region, walk_region
from .transforms import erase_stages

OPS = {**CORE_OPS, **TL_OPS}
_ASSOC_COMM = frozenset({"sum", "prod", "max", "min"})  # declared monoids (markers.red)


def _emit_like(b, n, args, regs):
    explicit = {} if OPS[n.op].type_rule is not None else {"type": n.type}
    return b.emit(n.op, *args, regions=regs, loc=n.loc, **explicit, **dict(n.attrs))


def _rewrite(region: Region, rule) -> Region:
    """One bottom-up pass: rule(b, n, args, regs) returns a replacement node
    or None; params keep identity; sub-regions rewrite under the same rule."""
    b = Builder(OPS)
    memo: dict[int, object] = {}

    def go(n):
        if id(n) in memo:
            return memo[id(n)]
        if n.op == "core.param":
            out = n
        else:
            args = tuple(go(a) for a in n.args)
            regs = tuple(_rewrite(r, rule) for r in n.regions)
            out = rule(b, n, args, regs)
            if out is None:
                out = _emit_like(b, n, args, regs)
        memo[id(n)] = out
        return out

    return Region(params=region.params, body=tuple(go(x) for x in region.body))


def _dims_of(attrs) -> tuple:
    d = attrs["dims"]
    return (d,) if isinstance(d, str) else tuple(d)


# --- the rules, narrowest-possible matches ------------------------------------


def _r_cancel_split_merge(b, n, args, regs):
    """merge(split(x, k -> parts), parts -> k) == x — exact, a pure layout
    round trip."""
    if n.op != "tl.merge" or args[0].op != "tl.split":
        return None
    ma, sa = dict(n.attrs), dict(args[0].attrs)
    if (
        ma["name"] == sa["name"]
        and tuple(ma["parts"]) == tuple(p for p, _ in sa["parts"])
        and ma.get("start", 0) == 0
    ):
        return args[0].args[0]
    return None


def _r_fuse_reduce(b, n, args, regs):
    """reduce_f(reduce_f(x, inner), outer) == reduce_f(x, outer + inner) for
    a declared associative-commutative monoid — exact (one reduction, one
    bracketing; the SPLIT of a reduction is where the license lives)."""
    if n.op != "tl.reduce" or args[0].op != "tl.reduce":
        return None
    oa, ia = dict(n.attrs), dict(args[0].attrs)
    if oa["f"] != ia["f"] or oa["f"] not in _ASSOC_COMM or oa.get("zero") or ia.get("zero"):
        return None
    return b.emit("tl.reduce", args[0].args[0], f=oa["f"], dims=_dims_of(oa) + _dims_of(ia))


def _r_commute_split(b, n, args, regs):
    """F(split(x), split(y), ...) == split(F(x, y, ...)) for pointwise and
    repeat_like over operands split identically — exact (layout ops move
    coordinates, never values)."""
    if n.op not in ("tl.pointwise", "tl.repeat_like"):
        return None
    tens = [x for x in args if x.op != "core.const"]
    if not tens or any(x.op != "tl.split" for x in tens):
        return None
    specs = {(dict(x.attrs)["name"], tuple(dict(x.attrs)["parts"])) for x in tens}
    if len(specs) != 1:
        return None
    ((name, parts),) = specs
    inner = tuple(x.args[0] if x.op == "tl.split" else x for x in args)
    return b.emit("tl.split", b.emit(n.op, *inner, **dict(n.attrs)), name=name, parts=parts)


def _r_inline_fold(b, n, args, regs):
    """LICENSED (reassociation): a one-state fold from a zero init whose
    step yields ``acc + R(elements)`` IS the sum over the scan dim of R
    over the sources — the accumulation loop re-bracketed as one reduce.
    R evaluated per-slice equals the slice of R over the full sources
    (pointwise/reduce-over-other-dims commute with the fold's absolute
    select), so the substitution is exact; the SUM's bracketing is the
    licensed part."""
    if n.op != "tl.fold":
        return None
    a = dict(n.attrs)
    if len(tuple(a["state"])) != 1 or tuple(a["out"]) != ("final", 0):
        return None
    init, srcs = args[0], args[1:]
    if init.op != "tl.const" or dict(init.attrs).get("value") != 0.0 or not srcs:
        return None
    step = regs[0]
    if any(x.regions for x in walk_region(step)):
        return None  # nested control flow: decline, never guess
    p_acc = step.params[0]
    yld = step.body[-1].args[0]
    if yld.op != "tl.pointwise" or dict(yld.attrs).get("f") != "add" or len(yld.args) != 2:
        return None
    if yld.args[0] is p_acc:
        r = yld.args[1]
    elif yld.args[1] is p_acc:
        r = yld.args[0]
    else:
        return None
    if sum(1 for x in walk_region(step) for arg in x.args if arg is p_acc) != 1:
        return None  # the carry must be pure accumulation
    sub = {id(p): s for p, s in zip(step.params[1:], srcs)}
    memo: dict[int, object] = {}

    def subst(x):
        if id(x) in sub:
            return sub[id(x)]
        if id(x) in memo:
            return memo[id(x)]
        out = _emit_like(b, x, tuple(subst(g) for g in x.args), ())
        memo[id(x)] = out
        return out

    return b.emit("tl.reduce", subst(r), f="sum", dims=(a["dim"],))


def _r_cancel_reduce_split(b, n, args, regs):
    """LICENSED (reassociation): reduce over BOTH parts of a split is the
    reduce over the source dim — the tiling re-bracketing itself."""
    if n.op != "tl.reduce" or args[0].op != "tl.split":
        return None
    oa, sa = dict(n.attrs), dict(args[0].attrs)
    if oa["f"] not in _ASSOC_COMM or oa.get("zero"):
        return None
    pnames = tuple(p for p, _ in sa["parts"])
    dims = _dims_of(oa)
    if not set(pnames) <= set(dims):
        return None
    newdims = tuple(
        sa["name"] if d == pnames[0] else d for d in dims if d == pnames[0] or d not in pnames
    )
    return b.emit("tl.reduce", args[0].args[0], f=oa["f"], dims=newdims)


_RULES = (
    ("exact", _r_cancel_split_merge),
    ("exact", _r_fuse_reduce),
    ("exact", _r_commute_split),
    ("reassociation", _r_inline_fold),
    ("reassociation", _r_cancel_reduce_split),
)


def normalize(region: Region, kinds: frozenset = frozenset()) -> tuple[Region, frozenset]:
    """Run the rule set to a content-key fixpoint. ``kinds`` names the
    license kinds the caller has DECLARED; licensed rules outside it never
    run. Returns (normal form, the licensed kinds actually consumed)."""
    used: set[str] = set()
    cur = region
    for _ in range(16):
        before = cur.key
        for kind, rule in _RULES:
            if kind != "exact" and kind not in kinds:
                continue
            nxt = _rewrite(cur, rule)
            if nxt.key != cur.key:
                cur = nxt
                if kind != "exact":
                    used.add(kind)
        if cur.key == before:
            break
    return cur, frozenset(used)


@dataclass(frozen=True)
class Certificate:
    verdict: str  # "proved-exact" | "proved-licensed" | "licensed-differential"
    licenses: tuple[str, ...]  # declared-license names backing the verdict
    families: tuple[str, ...]  # adversarial families run (differential tier only)
    key_reached: bool


def certify(tile: Region, naive: Region, *, licenses=(), families=()) -> Certificate:
    """Certify that ``tile`` and ``naive`` are one denotation (320 §6).
    ``families`` is a tuple of (name, factory) where factory() returns the
    flagship's inputs — dict name -> Tensor, in param order — for one
    adversarial draw. Raises on an unlicensed deviation or a failed
    differential; the error quotes what was missing."""
    kinds = frozenset(lic.kind for lic in licenses)
    norm_t, used = normalize(erase_stages(tile), kinds)
    norm_n, _ = normalize(naive, kinds)
    if norm_t.key == norm_n.key:
        if used:
            names = tuple(sorted(lic.name for lic in licenses if lic.kind in used))
            return Certificate("proved-licensed", names, (), True)
        return Certificate("proved-exact", (), (), True)
    if not licenses:
        raise ValueError(
            "unlicensed deviation: normalization did not reach the twin's content key and "
            "no license is declared — a tile program either normalizes to its twin or names "
            "its deviation (320 §6)"
        )
    rtol = max(lic.rtol for lic in licenses)
    atol = max(lic.atol for lic in licenses)
    names = tuple(sorted(lic.name for lic in licenses))
    ran = []
    for fname, factory in families:
        vals = list(factory().values())
        got_n = run_region(naive, vals)
        order = got_n.names
        got_t = run_region(tile, list(vals))
        np.testing.assert_allclose(
            got_t.to_numpy(order=order),
            got_n.to_numpy(order=order),
            rtol=rtol,
            atol=atol,
            err_msg=f"family {fname!r} exceeded the declared license bound ({', '.join(names)})",
        )
        ran.append(fname)
    if not ran:
        raise ValueError("licensed-differential certification needs adversarial families (260's law)")
    return Certificate("licensed-differential", names, tuple(ran), False)
