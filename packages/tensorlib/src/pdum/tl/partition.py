"""The anchor-and-absorb partitioner (330 §7.5): whole-model regions
carve into template-shaped groups, and everything unclaimed refuses
loudly with its offender named.

Claims run most-specific-first over the anchors — every reduce must
live somewhere: flash compositions (3 reduces), then bare row
normalizations (2), then single-dim contractions (1) — the specificity
ladder, so upstream absorption never steals a taller chain's root.
Contraction claims absorb BOTH ways (§7.6): epilogues walk downstream
while the root has exactly one consumer; operand prologues walk
upstream through the recompute-exact vocabulary while every consumer
lies inside the claim. A fork ends the walk in either direction, and
the boundary it leaves is a materialized tensor. Interior members may
not leak (a member consumed outside the claim invalidates it — v1
groups have one output, the translator's law). Unclaimed residue
becomes map-chain groups where it is pure map work and red groups
where it is not. Constants and charts are FREE — content-addressing
shares them across the graph, so they never block a claim and every
rebuild duplicates them.

Every carved group is REBUILT canonically — params renumbered in
first-use order — so structurally identical groups from different
layers share one content key, and certification, launch, pruning, and
measurement are paid once per distinct kernel, not once per layer
(the 330 §4 law at model scale). plan_model dedups by that key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pdum.dsl.ir import Builder, Region

from .dialect import walk_region
from .fusion import (
    _CHART_OPS,
    OPS,
    Group,
    _contract_core,
    _dims,
    _forest_is_closed,
    _match_rowstat,
    _match_softmax,
    _unchart,
    plan_region,
)

_CONSTS = ("core.const", "tl.const")
# zero-cost views: shared by claims, duplicated at rebuild; charts are
# metadata and never block a claim (§7.6's vocabulary ruling)
_FREE = _CONSTS + ("tl.rename", "tl.repeat_like") + _CHART_OPS
_EPILOGUE = ("tl.pointwise", "tl.repeat_like") + _CHART_OPS
_COMPUTE = ("tl.pointwise", "tl.iota", "tl.repeat")  # upstream-absorbable compute (§7.6)


@dataclass(frozen=True)
class Carve:
    """One carved group: the canonical kernel (the caching unit), the
    plan the recognizer produced for it, and the claim's footprint in
    the model graph."""

    kernel: Region = field(repr=False)  # IR reprs recurse the DAG: keep failures printable
    group: Group
    nodes: int  # interior (non-const) model nodes this carve claims
    root_op: str
    root: object = field(default=None, repr=False)  # the model node this carve computes
    bounds: tuple = field(default=(), repr=False)  # the model nodes its params bind to, in param order


@dataclass(frozen=True)
class ModelPlan:
    """A whole-model plan is DATA: the carves in dependency order,
    the interior node count, and coverage derived, never asserted."""

    carves: tuple
    interior: int

    def coverage(self) -> float:
        return sum(c.nodes for c in self.carves if c.group.confidence != "red") / self.interior

    def distinct_kernels(self) -> int:
        return len({c.kernel.key for c in self.carves})


def _graph(region: Region):
    """Interior nodes in dependency order + consumer map (consts excluded
    from the interior — content-addressing shares them; they are free)."""
    yld = region.body[-1]
    params = {id(p) for p in region.params}
    order, consumers = [], {}
    for n in walk_region(region):
        if id(n) in params:
            continue
        for a in n.args:
            consumers.setdefault(id(a), []).append(n)
        if n is yld or n.op == "core.tuple":
            continue  # sinks: joint grad regions yield a tuple of outputs
        order.append(n)
    interior = [n for n in order if n.op not in _FREE]
    return interior, consumers


def _cone(root, stops: set, allow: set | None = None):
    """The producer cone of ``root``, exclusive of ``stops``, optionally
    restricted to ``allow`` (a residue component); params never join."""
    members, stack = {}, [root]
    while stack:
        n = stack.pop()
        if id(n) in members or id(n) in stops or n.op == "core.param":
            continue
        if allow is not None and id(n) not in allow and n is not root:
            continue
        members[id(n)] = n
        stack.extend(n.args)
    return members


def _attach_free_fringe(members):
    """Free views referenced by members join them (duplicated at rebuild)
    — boundaries are never free: they punch through to real values."""
    grew = True
    while grew:
        grew = False
        for m in list(members.values()):
            for a in m.args:
                if id(a) not in members and a.op in _FREE and a.op not in _CONSTS:
                    members[id(a)] = a
                    grew = True
    return members


def _leaks(root, members, consumers) -> bool:
    """An interior member consumed outside the claim would need a second
    output — v1 refuses the claim instead (the translator's law)."""
    return any(
        id(c) not in members
        for m in members.values()
        if m is not root and m.op not in _FREE
        for c in consumers.get(id(m), ())
    )


def _absorb_up(members, consumers, claimed):
    """Upstream absorption (330 §7.6), the mirror of the epilogue walk: a
    compute node joins the claim while EVERY consumer already lies inside
    it — a fork is a boundary, because someone else needs that value
    materialized anyway. Free views attach each round so the walk sees
    the compute above them; recompute never crosses a boundary."""
    while True:
        _attach_free_fringe(members)
        added = False
        for m in list(members.values()):
            for a in m.args:
                if id(a) in members or id(a) in claimed or a.op not in _COMPUTE:
                    continue
                if any(id(c) not in members for c in consumers.get(id(a), ())):
                    continue
                members[id(a)] = a
                added = True
        if not added:
            return members


def _absorb(root, members, consumers, claimed):
    """Walk downstream through the epilogue vocabulary while the root has
    exactly one consumer; side operands ride (a repeat_like joins, its
    value becomes a boundary; consts are free; bare values bound)."""
    while True:
        cs = consumers.get(id(root), ())
        if len(cs) != 1:
            return root
        c = cs[0]
        if c.op not in _EPILOGUE or id(c) in claimed or c.op == "core.yield":
            return root
        if c.op == "tl.repeat_like" and c.args[0] is root:
            return root  # broadcast-INTO-elsewhere is another anchor's product, not epilogue
        for side in c.args:
            if side is root or id(side) in members or side.op in _CONSTS:
                continue
            if side.op == "tl.repeat_like" and id(side.args[1]) in members:
                members[id(side)] = side  # its value stays a boundary
        members[id(c)] = c
        root = c


def _claim_flash(r):
    """The strict flash shape on bare nodes (q/k/v are values, not params
    — the carve makes them params). Riders decline here, by design: the
    matcher never guesses."""
    out = _contract_core(r)
    if out is None:
        return None
    pr, v, s = out
    sm = _match_softmax(pr)
    if sm is None or sm["s"] != s:
        return None
    smn = sm["sm"]
    if smn.op != "tl.pointwise" or dict(smn.attrs).get("f") != "where" or len(smn.args) != 3:
        return None
    mask, sc, fill = smn.args
    if fill.op not in _CONSTS or not _forest_is_closed(mask, None):
        return None
    core = _contract_core(sc)
    if core is None:
        return None
    q, k, e = core
    qd, kd, vd = [tuple(d.name for d in _dims(x.type)) for x in (q, k, v)]
    if len(qd) != 2 or kd != (s, e) or len(vd) != 2 or vd[0] != s:
        return None
    return q, k, v


def _rebuild_dechart(b: Builder, memo: dict, n):
    """_rebuild_into, with chart views ELIDED: charts are type-preserving
    autodiff bookkeeping, and layer-specific chart attrs would split the
    content keys the canonical carve exists to share (§7.6)."""
    if id(n) in memo:
        return memo[id(n)]
    if n.op in _CHART_OPS:
        out = _rebuild_dechart(b, memo, n.args[0])
        memo[id(n)] = out
        return out
    args = tuple(_rebuild_dechart(b, memo, x) for x in n.args)
    explicit = {} if OPS[n.op].type_rule is not None else {"type": n.type}
    out = b.emit(n.op, *args, loc=n.loc, **explicit, **dict(n.attrs))
    memo[id(n)] = out
    return out


def _extract(root, members) -> Region:
    """Rebuild a claim as a standalone canonical region: boundaries become
    params in first-use order — the content key layers share."""
    bounds, seen, stack = [], set(), [root]
    while stack:  # deterministic: DFS mirrors the rebuild's arg order
        n = stack.pop()
        for a in reversed(n.args):
            if id(a) in members:
                stack.append(a)
            elif id(a) not in seen and a.op not in _CONSTS:
                seen.add(id(a))
                bounds.append(a)
    b = Builder(OPS)
    memo = {id(x): b.param(i, x.type) for i, x in enumerate(bounds)}
    params = tuple(memo[id(x)] for x in bounds)
    region = Region(params=params, body=(b.emit("core.yield", _rebuild_dechart(b, memo, root)),))
    return region, tuple(bounds)


def _components(nodes, consumers):
    """Weakly-connected components of the unclaimed residue, each with
    its externally-consumed roots."""
    pool = {id(n): n for n in nodes}
    seen, comps = set(), []
    edges = {}
    for n in nodes:
        edges.setdefault(id(n), set()).update(id(a) for a in n.args if id(a) in pool)
        for a in n.args:
            if id(a) in pool:
                edges.setdefault(id(a), set()).add(id(n))
    for n in nodes:
        if id(n) in seen:
            continue
        comp, stack = [], [id(n)]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            comp.append(pool[i])
            stack.extend(edges.get(i, ()))
        roots = [
            m for m in comp if any(id(c) not in {id(x) for x in comp} for c in consumers.get(id(m), ()))
        ]
        comps.append((comp, roots))
    return comps


def carve_model(region: Region):
    """The partition alone — claims + residue, no per-group planning.
    Returns (claims, interior_count): claims are (root, members) with
    members keyed by id."""
    interior, consumers = _graph(region)
    claimed: set[int] = set()
    claims = []

    def take(root, members):
        _attach_free_fringe(members)
        if _leaks(root, members, consumers):
            return False
        claims.append((root, members))
        claimed.update(i for i, m in members.items() if m.op not in _FREE)
        return True

    for r in interior:  # flash first: most specific
        if r.op != "tl.reduce" or id(r) in claimed:
            continue
        f = _claim_flash(r)
        if f is not None:
            members = _cone(r, {id(x) for x in f})
            if not any(i in claimed for i in members):
                take(r, members)
    for n in interior:  # bare row normalizations BEFORE single-reduce claims: the
        if id(n) in claimed:  # specificity ladder (3 reduces > 2 > 1) keeps upstream
            continue  # absorption from stealing a two-reduce chain's root (§7.6)
        sm = _match_softmax(n)
        if sm is None:
            continue
        members = _cone(n, {id(sm["sm"])})
        if not any(i in claimed for i in members):
            take(n, members)
    for n in interior:  # row-statistics chains (layernorm): two means, one dim,
        if id(n) in claimed:  # scale-shift absorbed as epilogue (§7.6)
            continue
        rs = _match_rowstat(n)
        if rs is None:
            continue
        members = _cone(n, {id(rs["x"])})
        if not any(i in claimed for i in members):
            root = _absorb(n, members, consumers, claimed)
            take(root, members)
    for r in interior:  # single-dim contractions: prologues absorbed upstream (§7.6),
        if r.op != "tl.reduce" or id(r) in claimed:  # epilogues downstream
            continue
        a = dict(r.attrs)
        dims = (a["dims"],) if isinstance(a["dims"], str) else tuple(a["dims"])
        if a.get("f") not in ("sum", "mean") or len(dims) not in (1, 2) or a.get("zero") is not None:
            continue
        mul = _unchart(r.args[0])
        if mul.op == "tl.pointwise" and dict(mul.attrs).get("f") == "mul" and len(mul.args) == 2:
            cores = []
            for x in mul.args:
                x = _unchart(x)
                if x.op == "tl.repeat_like":
                    x = _unchart(x.args[0])
                cores.append(x)
        else:  # PLAIN: a sum with no product (bias gradients) — one operand
            cores = [mul]
        if any(k not in {d.name for d in _dims(c.type)} for c in cores for k in dims):
            continue  # both operands must CARRY every contracted dim (the dims credential)
        keeps = [tuple(d.name for d in _dims(c.type) if d.name not in dims) for c in cores]
        out = tuple(d.name for d in _dims(r.type))
        if set(out) != set().union(*map(set, keeps)):
            continue
        members = {id(r): r}
        if len(cores) == 2:  # plain sums leave their operand to absorption:
            members[id(mul)] = mul  # it may be another anchor's root
        if any(i in claimed for i in members):
            continue
        _absorb_up(members, consumers, claimed)
        root = _absorb(r, members, consumers, claimed)
        take(root, members)
    residue = [n for n in interior if id(n) not in claimed]
    for comp, roots in _components(residue, consumers):
        comp_ids = {id(m) for m in comp}
        pool = {id(m): m for m in comp}
        rootset = [id(x) for x in roots]
        while True:  # split per root; promote internally-bound nodes until the
            parts, used = [], set()  # split tiles the component (fixpoint)
            for ri in rootset:
                stops = (set(rootset) - {ri}) | used
                mem = _attach_free_fringe(_cone(pool[ri], stops, allow=comp_ids))
                parts.append((pool[ri], mem))
                used |= {i for i in mem if i in comp_ids}
            promote = []
            for rt, mem in parts:
                for m in mem.values():
                    for a in m.args:
                        if id(a) in comp_ids and id(a) not in mem and id(a) not in rootset:
                            if id(a) not in promote:
                                promote.append(id(a))
            if not promote:
                claims.extend(parts)
                break
            rootset.extend(promote)
    return claims, len(interior)


def plan_model(region: Region, machine=None, floor: int = 1024) -> ModelPlan:
    """Carve, then map each carve through the ONE recognizer — deduped by
    canonical kernel key, so repeated layers plan once."""
    claims, interior = carve_model(region)
    seen: dict[str, Group] = {}
    carves = []
    for root, members in claims:
        n_interior = sum(1 for m in members.values() if m.op not in _FREE)
        if root is None:
            ops = sorted({m.op for m in members.values() if m.op not in _FREE})
            g = Group("uncompiled", None, None, "red", reason=f"multi-output residue over {ops}")
            carves.append(Carve(region, g, n_interior, "residue"))
            continue
        kernel, bounds = _extract(root, members)
        g = seen.get(kernel.key)
        if g is None:
            g = plan_region(kernel, machine=machine, floor=floor).groups[0]
            seen[kernel.key] = g
        carves.append(Carve(kernel, g, n_interior, root.op, root, bounds))
    return ModelPlan(tuple(carves), interior)
