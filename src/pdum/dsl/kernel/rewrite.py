"""The one pass mechanism: declarative rules, one greedy driver, stage legality.

Every compiler activity downstream — simplification, backend decompositions,
param legalization, transform columns, even rendering — is ``(pattern, fn)``
data run by this driver. Any proposed new mechanism must first prove it
cannot be a rule.

The driver is **greedy, directional, deterministic**: bottom-up, first
matching rule wins (rule order is priority), per-node fixpoint with a loud
non-termination guard, DAG sharing preserved via memoized rebuild. Upward
cascades (a child's rewrite enabling the parent's) resolve in one pass;
a stage needing another sweep runs itself again — determinism and
termination are worth more than automatic global fixpoints here.

Why greedy and not equality saturation as the core (measured, ch06): egglog
solves phase-ordering beautifully (``x*2 + x*3 → x*5`` in ~1 ms) but costs
~20 ms to saturate kernel-sized programs (our whole miss budget), needs
bounded iteration (heuristic output vs. golden tests and content-addressed
determinism), imports in ~1.5 s (the kernel is zero-dep), and cannot express
our non-equational passes (slot numbering, AD's constructive transforms,
rendering). Equality saturation remains what §12 always said: an optional
optimizer *pass* — a ``Region -> Region`` satellite — where its power is
pure upside.

Stage legality is always on when declared: after a stage's rules reach
fixpoint, every surviving op must belong to a legal namespace — MLIR's
conversion-target idea reduced to an O(ops) prefix check.

Book: ``docs/book/ch06-everything-is-a-rule.ipynb``. Architecture: §2.7.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .ir import Builder, Node, Region, VerifyError

Match = dict  # capture name -> Node, plus "root"


@dataclass(frozen=True)
class Pat:
    """Matches on op name(s), positional sub-patterns (a str is a capture
    name; repeated names demand structurally equal nodes), and a guard."""

    op: str | tuple[str, ...] | None = None
    args: tuple | None = None
    guard: Callable[[Match], bool] | None = None


Rule = tuple  # (Pat, fn(builder, match) -> Node | None)
RuleSet = list


@dataclass(frozen=True)
class Stage:
    name: str
    rules: RuleSet
    legal: frozenset = frozenset()  # legal op namespaces at output; empty = unchecked


@dataclass
class MatchLog:
    entries: list = field(default_factory=list)  # (stage_name, before, after)


def _match(pat: Pat, node: Node, binds: Match) -> bool:
    if pat.op is not None and (node.op != pat.op if isinstance(pat.op, str) else node.op not in pat.op):
        return False
    if pat.args is not None:
        if len(node.args) != len(pat.args):
            return False
        for sub, arg in zip(pat.args, node.args):
            if isinstance(sub, str):
                if sub in binds and binds[sub] != arg:
                    return False  # nonlinear pattern: same name, same structure
                binds[sub] = arg
            elif not _match(sub, arg, binds):
                return False
    return pat.guard is None or bool(pat.guard(binds))


_MAX_STEPS = 64


def _count(region: Region) -> int:
    seen: set[int] = set()

    def visit(n: Node) -> None:
        if id(n) not in seen:
            seen.add(id(n))
            for a in n.args:
                visit(a)
            for r in n.regions:
                for m in r.body:
                    visit(m)

    for n in region.body:
        visit(n)
    return len(seen)


def rewrite(region: Region, rules: RuleSet, ops: dict, *, name: str = "", log: MatchLog | None = None) -> Region:
    builder = Builder(ops)
    index: dict = {}
    for pat, fn in rules:
        for key in (pat.op,) if (pat.op is None or isinstance(pat.op, str)) else pat.op:
            index.setdefault(key, []).append((pat, fn))
    wildcards = index.get(None, ())
    memo: dict[int, Node] = {}
    # Non-termination guard: a global budget scaled to program size. A per-node
    # counter is not enough — a depth-growing rule (x -> neg(neg(x))) evades it
    # and blows the Python stack instead of failing loudly.
    budget = [4 * (_count(region) + 16)]  # legit stages fire O(n); depth-growers trip this before the stack

    def apply_rules(node: Node) -> Node:
        for step in range(_MAX_STEPS):
            for pat, fn in (*index.get(node.op, ()), *wildcards):
                binds: Match = {"root": node}
                if _match(pat, node, binds):
                    builder.default_loc = node.loc  # fresh nodes inherit; survivors keep their own
                    try:
                        out = fn(builder, binds)
                    finally:
                        builder.default_loc = None
                    if out is not None and out != node:
                        budget[0] -= 1
                        if budget[0] < 0:
                            raise VerifyError(f"[{name}] rewrite did not stabilize (budget exhausted at {node.op!r})")
                        if log is not None:
                            log.entries.append((name, node, out))
                        node = walk(out)  # fresh subtree: rebuild + rules, bottom-up
                        break
            else:
                return node
        raise VerifyError(f"[{name}] rewrite did not stabilize at {node.op!r} after {_MAX_STEPS} steps")

    def walk(node: Node) -> Node:
        if id(node) in memo:
            return memo[id(node)]
        if node.op == "core.param":  # binders are never rewritten
            memo[id(node)] = node
            return node
        args = tuple(walk(a) for a in node.args)
        regions = tuple(walk_region(r) for r in node.regions)
        rebuilt = node if args == node.args and regions == node.regions else Node(
            node.op, node.type, args, regions, node.attrs, node.loc
        )
        out = apply_rules(rebuilt)
        memo[id(node)] = out
        return out

    def walk_region(r: Region) -> Region:
        body = tuple(walk(n) for n in r.body)
        return r if body == r.body else Region(r.params, body)

    return walk_region(region)


def check_legal(region: Region, namespaces: frozenset, stage_name: str) -> None:
    seen: set[int] = set()

    def visit(node: Node) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if node.op.split(".", 1)[0] not in namespaces:
            from .ir import format_loc

            where = f" at {format_loc(node.loc)}" if node.loc else ""
            raise VerifyError(f"[{stage_name}] illegal op {node.op!r}{where}; legal namespaces: {sorted(namespaces)}")
        for a in node.args:
            visit(a)
        for r in node.regions:
            for m in r.body:
                visit(m)

    for n in region.body:
        visit(n)


def run_stage(region: Region, stage: Stage, ops: dict, log: MatchLog | None = None) -> Region:
    """Rules to fixpoint, then the conversion-target check: which dialects may
    exist after this stage is machine-checked, not folklore."""
    out = rewrite(region, stage.rules, ops, name=stage.name, log=log)
    if stage.legal:
        check_legal(out, stage.legal, stage.name)
    return out
