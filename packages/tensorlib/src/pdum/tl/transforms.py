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

from .ir import _LAYOUT_OPS, Instr, Program, infer

_VIEW = frozenset(_LAYOUT_OPS) | {"with_value_units"}
_CLOSED = frozenset({"iota", "const"})
_DEFAULT_BAN = frozenset({"reduce", "scan", "fold"})
_ITEM = 8


def dce(prog: Program, keep) -> Program:
    """Drop instructions that don't (transitively) feed a kept var."""
    keep = tuple(keep)
    defined = set(prog.vars)
    missing = [k for k in keep if k not in defined]
    if missing:
        raise KeyError(f"kept vars not defined by the program: {missing}")
    live = set(keep)
    for ins in reversed(prog.instrs):
        if ins.var in live:
            live.update(ins.operands)
    return Program(tuple(i for i in prog.instrs if i.var in live))


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
class CheckpointReport:
    program: Program
    saved: tuple  # ((var, bytes), ...) — allocations chosen by the cut
    recomputed: tuple  # forward vars duplicated into the backward
    bytes_before: int  # naive boundary: root allocations of every backward read
    bytes_after: int  # sum of saved bytes


def checkpoint(prog: Program, target: str, input_layouts: dict, ban=None, keep=()) -> CheckpointReport:
    ban = _DEFAULT_BAN if ban is None else frozenset(ban)
    idx = prog.vars.index(target)
    fwd, bwd = prog.instrs[: idx + 1], prog.instrs[idx + 1 :]
    fvars = {i.var for i in fwd}
    defs = {i.var: i for i in prog.instrs}
    shadows = infer(prog, input_layouts)

    def nbytes(v: str) -> int:
        if defs[v].op in _VIEW or defs[v].op in _CLOSED or defs[v].op == "input":
            return 0
        n = 1
        for d in shadows[v].dims:
            n *= d.size
        return n * _ITEM

    def root(v: str) -> str:
        while defs[v].op in _VIEW:
            v = defs[v].operands[0]
        return v

    R = sorted({o for ins in bwd for o in ins.operands if o in fvars})
    forced = ({target} | set(keep)) & fvars
    if not R:
        return CheckpointReport(prog, (), (), 0, 0)

    # relevant subgraph: forward ancestors of R
    rel: set[str] = set()
    stack = list(R)
    while stack:
        v = stack.pop()
        if v in rel:
            continue
        rel.add(v)
        stack.extend(o for o in defs[v].operands if o in fvars and o not in rel)

    inf = sum(nbytes(v) for v in rel) + 1
    net = _MinCut()
    for v in rel:
        op = defs[v].op
        if v in forced or op == "input" or op in _CLOSED:
            cap = 0  # free to keep: outputs-anyway, resident, closed forms
        elif op in _VIEW:
            cap = inf  # never save an alias; save its root or recompute
        else:
            cap = nbytes(v)
        net.add(("i", v), ("o", v), cap)
        for o in defs[v].operands:
            if o in rel:
                net.add(("o", o), ("i", v), inf)
        if op == "input" or op in ban:
            net.add("src", ("i", v), inf)  # fresh demand: not re-derivable
    for v in R:
        net.add(("o", v), "snk", inf)
    reach = net.solve("src", "snk")
    cut = {v for v in rel if ("i", v) in reach and ("o", v) not in reach}
    available = cut | forced | {v for v in rel if defs[v].op == "input"}

    # recompute set: unavailable ancestors of R, stopping at available vars
    need: set[str] = set()
    stack = [v for v in R if v not in available]
    while stack:
        v = stack.pop()
        if v in need:
            continue
        need.add(v)
        stack.extend(o for o in defs[v].operands if o in fvars and o not in available and o not in need)
    blocked = [v for v in need if defs[v].op in ban]
    if blocked:
        raise AssertionError(f"min-cut violated the recompute ban: {blocked}")  # structural bug guard

    taken = set(prog.vars)

    def fresh(v: str) -> str:
        nm = f"{v}.rc"
        while nm in taken:
            nm += "_"
        taken.add(nm)
        return nm

    ren = {v: fresh(v) for v in need}
    order = {v: i for i, v in enumerate(prog.vars)}
    emitted: set[str] = set()
    out: list[Instr] = list(fwd)

    def emit_rc(operands) -> None:
        # lazily materialize the unemitted recompute ancestors, fwd order —
        # just-in-time recompute is what actually moves the peak
        todo, seen = [], set()
        stack = [o for o in operands if o in need and o not in emitted]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            todo.append(v)
            stack.extend(o for o in defs[v].operands if o in need and o not in emitted and o not in seen)
        for v in sorted(todo, key=order.__getitem__):
            ins = defs[v]
            out.append(Instr(ren[v], ins.op, tuple(ren.get(o, o) for o in ins.operands), dict(ins.params)))
            emitted.add(v)

    for ins in bwd:
        emit_rc(ins.operands)
        out.append(Instr(ins.var, ins.op, tuple(ren.get(o, o) for o in ins.operands), dict(ins.params)))

    saved = tuple((v, nbytes(v)) for v in sorted(cut, key=order.__getitem__) if nbytes(v) > 0)
    naive_roots = {root(v) for v in R} - forced
    bytes_before = sum(nbytes(r) for r in naive_roots)
    return CheckpointReport(
        program=Program(tuple(out)),
        saved=saved,
        recomputed=tuple(sorted(need, key=order.__getitem__)),
        bytes_before=bytes_before,
        bytes_after=sum(nb for _, nb in saved),
    )
