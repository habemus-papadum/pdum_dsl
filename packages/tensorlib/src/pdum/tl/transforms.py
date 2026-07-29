"""L1 program transformations: requested-gradients DCE and min-cut
rematerialization (activation checkpointing).

`dce(prog, keep)` — backward reachability from the requested outputs.
"Which gradients do I care about" needs no special cases: prune backward
instructions that don't reach a requested gradient and the saved-activation
set shrinks automatically (frozen early layers stop saving what only their
weight-gradients needed). Run it before any memory planning.

`checkpoint(prog, target, input_layouts)` — the min-cut saved-set selection
(the AOTAutograd partitioner formulation, sharpened by exact sizes):

- The joint program splits at `target` into forward and backward. The
  backward reads a set R of forward values; naively all of R stays live
  across the boundary.
- Instead, choose a SAVED set S and recompute everything else in R from
  S ∪ inputs during the backward — placed lazily, just before first use,
  so recomputed values don't all coexist.
- Choosing S is a min cut between sources (primal inputs + outputs of
  recompute-BANNED ops) and a sink fed by R, with node capacities = exact
  bytes from layouts. The representation sharpens the classic setup:
  iota/const cost 0 (closed forms are free to "save"), views cost ∞ at the
  view (never save an alias — cut at its root or recompute the view, which
  is free), and banned ops (default: reduce/scan/fold — the contractions
  and recurrences whose recompute would double real FLOPs) source fresh
  demand so the cut must save at-or-after them. Pointwise chains and
  layout ops recompute freely — the fusion-cheap region.

Recomputation in pure SSA is duplication under fresh names (`v.rc`): two
variables, one semantic value — name≠value is the price REPRESENTATIONS.md
predicted. Gradient var names are untouched, so `grad`'s returned mapping
stays valid on the transformed program.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .dialect import LAYOUT_OP_NAMES

_VIEW = LAYOUT_OP_NAMES | {"with_value_units"}
_CLOSED = frozenset({"iota", "const"})
_DEFAULT_BAN = frozenset({"reduce", "scan", "fold"})
_ITEM = 8


def erase_stages(region):
    """The erasure oracle's stage half (320 §6): a stage moves residence and
    presentation order, never values-by-name — denotationally IDENTITY, so
    the erased region's denotation is bit-exactly the staged one's (the
    megatron ``level=None`` precedent, one level down). Rebuild without the
    ``tl.stage`` nodes, recursing into sub-regions; types re-infer, and
    every consumer aligns by name, so the order a stage chose disappears
    with it. Params are kept BY IDENTITY (callers' name maps stay valid)."""
    from pdum.dsl.ir import Builder, Region
    from pdum.dsl.ops import CORE_OPS

    from .dialect import TL_OPS

    ops = {**CORE_OPS, **TL_OPS}
    b = Builder(ops)
    memo: dict[int, object] = {}

    def rebuild(n):
        if id(n) in memo:
            return memo[id(n)]
        if n.op == "tl.stage":
            out = rebuild(n.args[0])
        elif n.op == "core.param":
            out = n
        else:
            args = tuple(rebuild(a) for a in n.args)
            regs = tuple(erase_stages(r) for r in n.regions)
            explicit = {} if ops[n.op].type_rule is not None else {"type": n.type}
            out = b.emit(n.op, *args, regions=regs, loc=n.loc, **explicit, **dict(n.attrs))
        memo[id(n)] = out
        return out

    return Region(params=region.params, body=tuple(rebuild(x) for x in region.body))


def dce(region, keep, *, names=None):
    """Drop values that don't (transitively) feed a kept name: a region IS
    its yield-reachable graph, so DCE re-yields exactly ``keep`` (a name
    sequence over the naming law — pass ``names``) and reachability does
    the pruning; params that stop feeding an output drop from the
    signature."""
    return _region_dce(region, tuple(keep), names)


def _region_dce(region, keep, names):
    """The Region face: rebuild the yield over the kept nodes (keep order =
    output order); everything else prunes by unreachability."""
    from pdum.dsl.ir import Builder, Region
    from pdum.dsl.ops import CORE_OPS

    from .dialect import TL_OPS, walk_region  # lazy: dialect's consumers import this module

    if names is None:
        raise ValueError(
            "region dce joins on names — pass names=region_names(region, param_names) "
            "(LiftedStep.names / Assemblage.names carry it)"
        )
    by_name: dict = {}
    for n in walk_region(region):
        nm = names.get(id(n))
        if nm is not None:
            by_name.setdefault(nm, n)
    missing = [k for k in keep if k not in by_name]
    if missing:
        raise KeyError(f"kept vars not defined by the program: {missing}")
    kept = [by_name[k] for k in keep]
    live: set[int] = set()
    stack = list(kept)
    while stack:
        n = stack.pop()
        if id(n) in live:
            continue
        live.add(id(n))
        stack.extend(n.args)
    b = Builder({**CORE_OPS, **TL_OPS})
    yielded = kept[0] if len(kept) == 1 else b.emit("core.tuple", *kept)
    return Region(
        params=tuple(p for p in region.params if id(p) in live),
        body=(b.emit("core.yield", yielded),),
    )


class _MinCut:
    """Max-flow / min-cut (Edmonds-Karp) over hashable node keys."""

    def __init__(self):
        self.adj: dict = {}

    def add(self, u, v, cap: int) -> None:
        fw = self.adj.setdefault(u, [])
        bw = self.adj.setdefault(v, [])
        fw.append([v, cap, len(bw)])
        bw.append([u, 0, len(fw) - 1])

    def solve(self, s, t) -> set:
        """Run to saturation; return the source side of the residual graph."""
        while True:
            parent = {s: None}
            q = deque([s])
            while q and t not in parent:
                u = q.popleft()
                for i, (v, cap, _) in enumerate(self.adj.get(u, ())):
                    if cap > 0 and v not in parent:
                        parent[v] = (u, i)
                        q.append(v)
            if t not in parent:
                break
            path, v = [], t
            while parent[v] is not None:
                u, i = parent[v]
                path.append((u, i))
                v = u
            aug = min(self.adj[u][i][1] for u, i in path)
            for u, i in path:
                edge = self.adj[u][i]
                edge[1] -= aug
                self.adj[edge[0]][edge[2]][1] += aug
        seen = {s}
        q = deque([s])
        while q:
            u = q.popleft()
            for v, cap, _ in self.adj.get(u, ()):
                if cap > 0 and v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen


@dataclass(frozen=True)
class RegionCheckpoint:
    """The Region face's report: ``region`` is the transformed joint;
    ``names`` extends the input assignment over the recompute clones
    (``<name>.rc`` — two node objects, one semantic value: the
    name-vs-value price made explicit as object identity)."""

    region: object
    saved: tuple
    recomputed: tuple
    bytes_before: int
    bytes_after: int
    names: dict


def checkpoint(region, target: str, input_layouts: dict | None = None, ban=None, keep=(), *, names=None):
    """Min-cut saved-set selection (module docstring) over the JOINT region
    (a RegionGrad's); pass ``names``. Recomputation is node CLONING — the
    DAG needs no lazy placement, because the default schedule (walk order)
    visits a clone immediately before its first consumer."""
    return _region_checkpoint(region, target, ban, keep, names)


def _region_checkpoint(region, target, ban, keep, names):
    from pdum.dsl.ir import Builder, Region
    from pdum.dsl.naming import Namer
    from pdum.dsl.ops import CORE_OPS

    from .dialect import TL_OPS, walk_region  # lazy: dialect's consumers import this module

    if names is None:
        raise ValueError(
            "region checkpoint joins on names — pass names=region_names(region, param_names) "
            "(RegionGrad.names carries it)"
        )
    ban = _DEFAULT_BAN if ban is None else frozenset(ban)
    by_name: dict = {}
    node_of: dict = {}
    for nd in walk_region(region):
        node_of[id(nd)] = nd
        nm = names.get(id(nd))
        if nm is not None:
            by_name.setdefault(nm, nd)
    if target not in by_name:
        raise KeyError(f"target {target!r} is not defined by the program")
    tnode = by_name[target]

    fwd: list = []  # target's ancestors, topological
    fids: set = set()

    def anc(nd):
        if id(nd) in fids:
            return
        fids.add(id(nd))
        for a in nd.args:
            anc(a)
        fwd.append(nd)

    anc(tnode)
    order = {id(nd): i for i, nd in enumerate(fwd)}
    bwd = [nd for nd in walk_region(region) if id(nd) not in fids]

    def base_of(nd) -> str:
        return nd.op[3:] if nd.op.startswith("tl.") else nd.op

    def is_view(nd) -> bool:
        return base_of(nd) in _VIEW

    def nbytes(nd) -> int:
        if nd.op == "core.param" or nd.op == "core.const" or is_view(nd) or base_of(nd) in _CLOSED:
            return 0
        n = 1
        for d in nd.type.layout.dims:
            n *= d.size
        return n * _ITEM

    def root(nd):
        while is_view(nd):
            nd = nd.args[0]
        return nd

    # outputs are not consumption: yield/tuple plumbing never puts a forward
    # value in the backward's read set (the same law the analyses apply)
    R = sorted(
        {id(a) for nd in bwd if nd.op not in ("core.yield", "core.tuple") for a in nd.args if id(a) in fids},
        key=order.__getitem__,
    )
    forced = {id(by_name[target])} | {id(by_name[k]) for k in keep if k in by_name}
    forced &= fids
    if not R:
        return RegionCheckpoint(region, (), (), 0, 0, dict(names))

    rel: set = set()
    stack = list(R)
    while stack:
        v = stack.pop()
        if v in rel:
            continue
        rel.add(v)
        stack.extend(id(a) for a in node_of[v].args if id(a) in fids and id(a) not in rel)

    inf = sum(nbytes(node_of[v]) for v in rel) + 1
    net = _MinCut()
    for v in sorted(rel, key=order.__getitem__):
        nd = node_of[v]
        if v in forced or nd.op == "core.param" or nd.op == "core.const" or base_of(nd) in _CLOSED:
            cap = 0
        elif is_view(nd):
            cap = inf
        else:
            cap = nbytes(nd)
        net.add(("i", v), ("o", v), cap)
        for a in nd.args:
            if id(a) in rel:
                net.add(("o", id(a)), ("i", v), inf)
        if nd.op == "core.param" or base_of(nd) in ban:
            net.add("src", ("i", v), inf)
    for v in R:
        net.add(("o", v), "snk", inf)
    reach = net.solve("src", "snk")
    cut = {v for v in rel if ("i", v) in reach and ("o", v) not in reach}
    available = cut | forced | {v for v in rel if node_of[v].op == "core.param"}

    need: set = set()
    stack = [v for v in R if v not in available]
    while stack:
        v = stack.pop()
        if v in need:
            continue
        need.add(v)
        stack.extend(id(a) for a in node_of[v].args if id(a) in fids and id(a) not in available and id(a) not in need)
    blocked = [names.get(v, "?") for v in need if base_of(node_of[v]) in ban]
    if blocked:
        raise AssertionError(f"min-cut violated the recompute ban: {blocked}")  # structural bug guard

    b = Builder({**CORE_OPS, **TL_OPS})
    namer = Namer(taken=set(names.values()))
    names2 = dict(names)
    clones: dict = {}

    def clone(nd):
        if id(nd) in clones:
            return clones[id(nd)]
        args2 = tuple(clone(a) if id(a) in need else a for a in nd.args)
        c = b.emit(nd.op, *args2, regions=nd.regions, type=nd.type, **dict(nd.attrs))
        base_name = names.get(id(nd))
        names2[id(c)] = namer.derive(f"{base_name}.rc" if base_name else "%rc")
        clones[id(nd)] = c
        return c

    rebuilt: dict = {}

    def rb(nd):
        if id(nd) in rebuilt:
            return rebuilt[id(nd)]
        args2 = tuple(clone(a) if id(a) in need else (rb(a) if id(a) not in fids else a) for a in nd.args)
        if args2 == nd.args:
            rebuilt[id(nd)] = nd
            return nd
        c = b.emit(nd.op, *args2, regions=nd.regions, type=nd.type, **dict(nd.attrs))
        nm = names.get(id(nd))
        if nm is not None:
            names2[id(c)] = nm  # same name: same semantic value, rebuilt args
        rebuilt[id(nd)] = c
        return c

    body2 = tuple(rb(nd) for nd in region.body)
    out_region = Region(params=region.params, body=body2)

    saved = tuple(
        (names.get(v, "?"), nbytes(node_of[v])) for v in sorted(cut, key=order.__getitem__) if nbytes(node_of[v]) > 0
    )
    naive_roots = {id(root(node_of[v])) for v in R} - forced
    bytes_before = sum(nbytes(node_of[r]) for r in naive_roots)
    return RegionCheckpoint(
        region=out_region,
        saved=saved,
        recomputed=tuple(names.get(v, "?") for v in sorted(need, key=order.__getitem__)),
        bytes_before=bytes_before,
        bytes_after=sum(nb for _, nb in saved),
        names=names2,
    )
