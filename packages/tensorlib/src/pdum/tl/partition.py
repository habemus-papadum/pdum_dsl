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
    _uncore,
    plan_region,
)

_CONSTS = ("core.const", "tl.const")
# zero-cost views: shared by claims, duplicated at rebuild; charts are
# metadata and never block a claim (§7.6's ruling); tl.repeat is
# repeat_like's literal-extent twin — a stride-0 broadcast, never work
_FREE = _CONSTS + ("tl.rename", "tl.repeat_like", "tl.repeat") + _CHART_OPS
_EPILOGUE = ("tl.pointwise", "tl.repeat_like") + _CHART_OPS
_COMPUTE = ("tl.pointwise", "tl.iota")  # upstream-absorbable compute (§7.6)


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
    artifacts: tuple = field(default=(), repr=False)  # surfaced statistics nodes, tuple-yield order (§7.8)


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


def _outside(a, members, consumers, root=None) -> bool:
    """Is a's VALUE consumed outside ``members``? Free views forward the
    value, they do not consume it — the check punches THROUGH free
    members to their consumers (charts and broadcasts are SHARED across
    claims by design, so membership of the view proves nothing). A path
    that reaches the claim's ROOT is the export itself, never a leak."""
    for c in consumers.get(id(a), ()):
        if c is root:
            continue
        if id(c) not in members:
            return True
        if c.op in _FREE and _outside(c, members, consumers, root):
            return True
    return False


def _leaks(root, members, consumers, travel=()) -> bool:
    """An interior member consumed outside the claim would need a second
    output — v1 refuses the claim instead (the translator's law). Travel
    members are exempt (§7.8): they are COPIES, duplicable by law."""
    return any(
        _outside(m, members, consumers, root)
        for m in members.values()
        if m is not root and m.op not in _FREE and id(m) not in travel
    )


def _absorb_up(members, consumers, claimed, travel=()):
    """Upstream absorption (330 §7.6), the mirror of the epilogue walk: a
    compute node joins the claim while EVERY consumer already lies inside
    it — a fork is a boundary, because someone else needs that value
    materialized anyway. Free views attach each round so the walk sees
    the compute above them; recompute never crosses a boundary — EXCEPT
    along the §7.8 travel set, whose cones copy in unconditionally (they
    are duplicable by classification; forks do not matter for copies)."""
    while True:
        _attach_free_fringe(members)
        added = False
        for m in list(members.values()):
            for a in m.args:
                if id(a) in members:
                    continue
                if id(a) in travel:
                    members[id(a)] = a
                    added = True
                    continue
                if id(a) in claimed or a.op not in _COMPUTE:
                    continue
                if _outside(a, members, consumers):
                    continue  # the fork law, punched through shared free views
                members[id(a)] = a
                added = True
        if not added:
            return members


def _absorb_travel(members, travel):
    """Travel-only absorption (§7.8): free views attach, travel copies
    join unconditionally, to fixpoint — the §7.6 consumer law untouched
    (row and residue claims never absorb ordinary compute upstream)."""
    while True:
        _attach_free_fringe(members)
        added = False
        for m in list(members.values()):
            for a in m.args:
                if id(a) not in members and id(a) in travel:
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
    t = next(n for n in qd if n != e)
    return {"q": q, "k": k, "v": v, "m": sm["max"], "den": sm["den"], "t": t, "s": s}


def _duplicable_cone(x, members, stats, sweep):
    """The §7.8 classification: x's cone within the claim, stopped at
    params and artifacts — map work and tile-local reductions travel; a
    SWEPT-dim reduction refuses (artifacts exist for it). Returns the
    cone (id -> node) or None."""
    cone, stack = {}, [x]
    while stack:
        n = stack.pop()
        if id(n) in cone or id(n) in stats or id(n) not in members:
            continue  # artifact or boundary: already materialized
        if n.op == "tl.reduce":
            d = dict(n.attrs)["dims"]
            dt = {d} if isinstance(d, str) else set(d)
            if dt & sweep:
                return None
        elif n.op not in _FREE and n.op not in _COMPUTE:
            return None
        cone[id(n)] = n
        stack.extend(n.args)
    return cone


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


def _extract(root, members, artifacts=()) -> Region:
    """Rebuild a claim as a standalone canonical region: boundaries become
    params in first-use order — the content key layers share. With
    artifacts (§7.8) the region yields a TUPLE (output, *artifacts):
    the surfaced statistics, in declared order."""
    roots = (root, *artifacts)
    bounds, seen, stack = [], set(), []
    for r in roots:
        stack = [r]
        while stack:  # deterministic: DFS mirrors the rebuild's arg order
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            for a in reversed(n.args):
                if id(a) in members:
                    stack.append(a)
                elif id(a) not in seen and a.op not in _CONSTS:
                    seen.add(id(a))
                    bounds.append(a)
    b = Builder(OPS)
    memo = {id(x): b.param(i, x.type) for i, x in enumerate(bounds)}
    params = tuple(memo[id(x)] for x in bounds)
    built = tuple(_rebuild_dechart(b, memo, r) for r in roots)
    yld = built[0] if len(built) == 1 else b.emit("core.tuple", *built)
    region = Region(params=params, body=(b.emit("core.yield", yld),))
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


def _dissolve(claims, consumers, travel, region):
    """Root-travel (§7.8's completing amendment): a carve whose root
    serves ONLY other claims, and whose every reduction is tile-local
    for each consumer's sweep, need not materialize — its members join
    the travel set and consumers re-derive per tile. Runs to fixpoint:
    dissolving dP frees the dS pieces, which reshapes dQ/dK. A root the
    region itself yields stays materialized; a reduction over a
    consumer's swept dim holds its carve back automatically — D's
    rowsum over s cannot travel into dQ's s-sweep, exactly FA's
    precomputed-per-row D."""
    yld = region.body[-1]
    sinks = {id(yld)} | ({id(yld.args[0])} if yld.args[0].op == "core.tuple" else set())

    def value_consumers(x, out, seen):
        for c in consumers.get(id(x), ()):
            if id(c) in seen:
                continue
            seen.add(id(c))
            if c.op in _FREE:
                out.append(c)
                value_consumers(c, out, seen)
            else:
                out.append(c)
        return out

    while True:
        owner: dict[int, set] = {}
        for ci, (root, members, _arts) in enumerate(claims):
            for i in members:
                owner.setdefault(i, set()).add(ci)
        sweeps = []
        for _root, members, _arts in claims:
            sw: set = set()
            for m in members.values():
                if m.op == "tl.reduce" and id(m) not in travel:
                    d = dict(m.attrs)["dims"]
                    sw |= {d} if isinstance(d, str) else set(d)
            sweeps.append(sw)
        dissolved = None
        order = sorted(  # pure-map pieces first: a rowsum must not ride into
            range(len(claims)),  # a piece that could otherwise dissolve whole
            key=lambda ci: sum(1 for m in claims[ci][1].values() if m.op == "tl.reduce"),
        )
        for ci in order:
            root, members, arts = claims[ci]
            if root is None or arts:
                continue  # rootless residue and artifact carves stay put
            if any(
                m.op not in _FREE and m.op not in _COMPUTE and m.op != "tl.reduce" and id(m) not in travel
                for m in members.values()
            ):
                continue  # something unspelled (a scan): not duplicable
            mine: set = set()
            for m in members.values():
                if m.op == "tl.reduce":  # travel copies included: the copy must
                    d = dict(m.attrs)["dims"]  # be legal where it finally RESTS
                    mine |= {d} if isinstance(d, str) else set(d)
            cons = value_consumers(root, [], set())
            ok = bool(cons)
            eaters: set = set()
            for c in cons:
                if id(c) in sinks:
                    ok = False  # the region yields it: an external output
                    break
                if c.op in _FREE:
                    continue  # a view forwards; its consumers were followed
                who = owner.get(id(c), set()) - {ci}
                if not who:
                    ok = False
                    break
                eaters |= who
            if not ok:
                continue
            if any(mine & sweeps[cj] for cj in eaters):
                continue  # a swept-dim reduction cannot travel (§7.8)
            # the traffic gate (§7.8 ruling 3's analytic default): dissolving
            # trades the root's round trip (write + read-back) for each eater
            # re-reading the cone's boundaries — dissolve only when that wins;
            # at toy sizes materialization IS cheaper and the carve stays
            root_elems = 1
            for d in _dims(root.type):
                root_elems *= d.stop - d.start
            binputs, seenb = 0, set()
            for m in members.values():
                for a in m.args:
                    if id(a) not in members and id(a) not in seenb and a.op not in _CONSTS:
                        seenb.add(id(a))
                        e = 1
                        for d in getattr(a.type, "dims", ()):
                            e *= d.stop - d.start
                        binputs += e
            if 2 * root_elems <= len(eaters) * binputs:
                continue
            dissolved = ci
            break
        if dissolved is None:
            return claims
        root, members, _arts = claims.pop(dissolved)
        travel.update(members)
        travel[id(root)] = root
        for _r, mem, _a in claims:
            _absorb_travel(mem, travel)


def carve_model(region: Region):
    """The partition alone — claims + residue, no per-group planning.
    Returns (claims, interior_count, travel): claims are (root, members,
    artifacts) with members keyed by id; travel is the §7.8 set — nodes
    whose re-derivation cones COPY into consuming claims."""
    interior, consumers = _graph(region)
    claimed: set[int] = set()
    travel: dict[int, object] = {}
    claims = []

    def take(root, members, arts=()):
        _attach_free_fringe(members)
        if _leaks(root, members, consumers, travel):
            return False
        claims.append((root, members, arts))
        claimed.update(i for i, m in members.items() if m.op not in _FREE)
        return True

    def take_flash_artifacts(r, f, members):
        """The §7.8 path: a leaking flash claim survives by surfacing its
        row statistics as ARTIFACTS and marking every other leaked
        interior's cone as TRAVEL — consumers re-derive, never reload."""
        _attach_free_fringe(members)
        stats = {id(f["m"]), id(f["den"])}
        sweep = {f["t"], f["s"]}
        new_travel: dict[int, object] = {}
        for m in members.values():
            if m is r or id(m) in stats:
                continue
            if not _outside(m, members, consumers, r):
                continue
            while m.op in _FREE and m.args:  # a leaked free view punches through
                m = m.args[0]  # to the VALUE it reads — classify that
            if m is r or id(m) in stats or id(m) not in members:
                continue
            cone = _duplicable_cone(m, members, stats, sweep)
            if cone is None:
                return False
            new_travel.update(cone)
        claims.append((r, members, (f["m"], f["den"])))
        travel.update(new_travel)
        claimed.update(i for i, m in members.items() if m.op not in _FREE)
        return True

    for r in interior:  # flash first: most specific
        if r.op != "tl.reduce" or id(r) in claimed:
            continue
        f = _claim_flash(r)
        if f is not None:
            members = _cone(r, {id(f["q"]), id(f["k"]), id(f["v"])})
            if not any(i in claimed and i not in travel for i in members):
                if not take(r, dict(members)):  # leaking? the §7.8 artifact path
                    take_flash_artifacts(r, f, members)
    for n in interior:  # bare row normalizations BEFORE single-reduce claims: the
        if id(n) in claimed:  # specificity ladder (3 reduces > 2 > 1) keeps upstream
            continue  # absorption from stealing a two-reduce chain's root (§7.6)
        sm = _match_softmax(n)
        if sm is None:
            continue
        members = _cone(n, {id(sm["sm"])})
        _absorb_travel(members, travel)
        if not any(i in claimed and i not in travel for i in members):
            take(n, members)
    for n in interior:  # row-statistics chains (layernorm): two means, one dim,
        if id(n) in claimed:  # scale-shift absorbed as epilogue (§7.6)
            continue
        rs = _match_rowstat(n)
        if rs is None:
            continue
        members = _cone(n, {id(rs["x"])})
        _absorb_travel(members, travel)
        if not any(i in claimed and i not in travel for i in members):
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
            cores = [_uncore(x) for x in mul.args]
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
        _absorb_up(members, consumers, claimed, travel)
        root = _absorb(r, members, consumers, claimed)
        take(root, members)
    residue = [n for n in interior if id(n) not in claimed]
    for comp, roots in _components(residue, consumers):
        comp_ids = {id(m) for m in comp}
        allow = comp_ids | set(travel)  # travel copies rejoin cones (§7.8) but
        pool = {id(m): m for m in comp}  # never promote: nobody materializes them
        rootset = [id(x) for x in roots]
        while True:  # split per root; promote internally-bound nodes until the
            parts, used = [], set()  # split tiles the component (fixpoint)
            for ri in rootset:
                stops = (set(rootset) - {ri}) | used
                mem = _absorb_travel(_cone(pool[ri], stops, allow=allow), travel)
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
                claims.extend((rt, mem, ()) for rt, mem in parts)
                break
            rootset.extend(promote)
    _dissolve(claims, consumers, travel, region)
    return claims, len(interior), travel


def plan_model(region: Region, machine=None, floor: int = 1024) -> ModelPlan:
    """Carve, then map each carve through the ONE recognizer — deduped by
    canonical kernel key, so repeated layers plan once."""
    claims, interior, _travel = carve_model(region)
    seen: dict[str, Group] = {}
    carves = []
    counted: set[int] = set()  # travel copies count ONCE, in their first claim (§7.8)
    for root, members, arts in claims:
        n_interior = sum(1 for i, m in members.items() if m.op not in _FREE and i not in counted)
        counted.update(i for i, m in members.items() if m.op not in _FREE)
        if root is None:
            ops = sorted({m.op for m in members.values() if m.op not in _FREE})
            g = Group("uncompiled", None, None, "red", reason=f"multi-output residue over {ops}")
            carves.append(Carve(region, g, n_interior, "residue"))
            continue
        kernel, bounds = _extract(root, members, arts)
        g = seen.get(kernel.key)
        if g is None:
            g = plan_region(kernel, machine=machine, floor=floor).groups[0]
            seen[kernel.key] = g
        carves.append(Carve(kernel, g, n_interior, root.op, root, bounds, arts))
    return ModelPlan(tuple(carves), interior)
