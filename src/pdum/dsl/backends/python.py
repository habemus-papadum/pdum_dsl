"""The Python backend: the reference target, and the proof the seam is real.

Renders a legalized Region to *readable* Python source (ch09 prints it) and
executes it with the exact ABI every real target uses: ``kernel(staging,
leaves)``, where every input — captures AND scalar arguments — arrives as
bytes in the staging buffer, unpacked at the offsets the plan chose. That is
deliberately NOT the fastest way to call Python from Python; it is the same
calling convention as a uniform buffer, so the marshaling story is proven
end-to-end on CPU before a GPU ever sees it.

Honest deviations of the reference target, stated once:

- ``core.if`` renders as a real ``if``/``else`` statement — branches are
  LAZY, so the guard-then-divide idiom (``1.0/x if x > 0.0 else 0.0``) is
  safe (the step-8 review caught the eager first draft crashing on exactly
  the guarded input). Each node is emitted in the deepest region that
  dominates all its uses: branch-exclusive work stays inside its branch;
  anything shared hoists to the join point, where some live path needs it
  anyway.
- Python has one float; ``f32`` computes as f64 here. The WGSL backend is
  where narrowing gets real (§2.10 ``type_map``).
- Results return natively (a Python value) — the ``ResultPlan`` mirror
  becomes physical with DPS targets (ch10); ch08 already proved it round-trips.

Composite (tuple-typed) *arguments* are refused loudly for now: their leaf
slots exist in the plan, but the IR keeps one logical ``core.param`` — the
arg-side normalize lands with the arrays step (see the step-7 review note).
"""

from __future__ import annotations

import math

from ..kernel.ir import Node, Region, VerifyError
from ..kernel.pack import PackPlan
from ..kernel.registry import DEFAULT, Backend
from ..kernel.types import Scalar

_BIN = {"core.add": "+", "core.sub": "-", "core.mul": "*", "core.div": "/", "core.mod": "%", "core.pow": "**"}
_PREDS = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
_CASTS = {"f64": "float", "f32": "float", "i64": "int", "i32": "int", "u64": "int", "u32": "int", "bool": "bool"}


def render(region: Region, plan: PackPlan, name: str = "kernel") -> str:
    """Legalized Region -> Python source. One assignment per node, DAG-shared
    nodes emitted once in their OWNER region (the deepest region dominating
    every use — the lazy-branch placement rule), names dense in topo order."""
    args_by_index = {s.source.index: s for s in plan.slots if s.source.root == "arg" and not s.source.sub}

    # Pass 1: topo order, direct users, and region anchors ((if_id, branch) uses).
    topo: list[Node] = []
    seen: set[int] = set()
    users: dict[int, list[Node]] = {}
    anchors: dict[int, list[tuple]] = {}
    region_result: dict[tuple, Node] = {}

    def walk(node: Node) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        for a in node.args:
            users.setdefault(id(a), []).append(node)
            walk(a)
        for i, r in enumerate(node.regions):
            for n in r.body:
                inner = n.args[0] if n.op == "core.yield" else n
                anchors.setdefault(id(inner), []).append((id(node), i))
                if n.op == "core.yield":
                    region_result[(id(node), i)] = inner
                walk(inner)
        topo.append(node)

    result = None
    for n in region.body:
        inner = n.args[0] if n.op == "core.yield" else n
        result = inner if n.op == "core.yield" else result
        anchors.setdefault(id(inner), []).append(None)  # anchored at the root
        walk(inner)

    # Pass 2: owner region per node = longest common prefix of all use paths.
    def lca(a: tuple, b: tuple) -> tuple:
        out = []
        for x, y in zip(a, b):
            if x != y:
                break
            out.append(x)
        return tuple(out)

    owner: dict[int, tuple] = {}
    for node in reversed(topo):  # users precede their args here, so owner(user) is final
        paths = [owner[id(u)] for u in users.get(id(node), ())]
        paths += [() if a is None else (*owner[a[0]], a) for a in anchors.get(id(node), ())]
        o = paths[0]
        for q in paths[1:]:
            o = lca(o, q)
        owner[id(node)] = o

    names = {id(n): f"v{k}" for k, n in enumerate(topo)}
    blocks: dict[tuple, list[Node]] = {}
    for n in topo:
        blocks.setdefault(owner[id(n)], []).append(n)

    def expr_of(node: Node) -> str:
        attrs = dict(node.attrs)
        arg = [names[id(a)] for a in node.args]
        if node.op == "core.param":
            spec = args_by_index.get(attrs["index"])
            if spec is None:
                raise VerifyError(f"composite argument {attrs['index']} has no scalar slot (arrays step)")
            return f"_u({spec.dest.fmt!r}, staging, {spec.dest.offset})[0]"
        if node.op == "abi.slot":
            return f"_u({attrs['fmt']!r}, staging, {attrs['offset']})[0]"
        if node.op == "core.const":
            v = attrs["value"]
            if isinstance(v, float) and not math.isfinite(v):
                return f"float({str(v)!r})"  # repr(inf) is not a Python literal
            return repr(v)
        if node.op in _BIN:
            return f"{arg[0]} {_BIN[node.op]} {arg[1]}"
        if node.op == "core.neg":
            return f"-{arg[0]}"
        if node.op == "core.cmp":
            return f"{arg[0]} {_PREDS[attrs['pred']]} {arg[1]}"
        if node.op == "core.cast":
            to = attrs["to"]
            if not isinstance(to, Scalar):
                raise VerifyError(f"python backend cannot cast to {to!r}")
            return f"{_CASTS[to.kind]}({arg[0]})"
        if node.op == "core.select":
            return f"({arg[1]} if {arg[0]} else {arg[2]})"
        if node.op == "core.vec":
            return "(" + ", ".join(arg) + ("," if len(arg) == 1 else "") + ")"
        if node.op == "core.extract":
            return f"{arg[0]}[{attrs['index']}]"
        raise VerifyError(f"python backend has no rendering for {node.op!r}")

    def emit_block(path: tuple, indent: str) -> list[str]:
        out: list[str] = []
        for node in blocks.get(path, ()):
            if node.op == "core.if":
                res = names[id(node)]
                out.append(f"{indent}if {names[id(node.args[0])]}:")
                for i, kw in ((0, "if"), (1, "else")):
                    sub = (*path, (id(node), i))
                    if i:
                        out.append(f"{indent}else:")
                    out += emit_block(sub, indent + "    ")
                    out.append(f"{indent}    {res} = {names[id(region_result[(id(node), i)])]}")
            else:
                out.append(f"{indent}{names[id(node)]} = {expr_of(node)}")
        return out

    body = "\n".join(emit_block((), "    ")) or "    pass"
    head = f"from struct import unpack_from as _u\n\ndef {name}(staging, leaves):\n"
    return f"{head}{body}\n    return {names[id(result)]}\n"


def compile_source(source: str, name: str = "kernel"):
    ns: dict = {}
    exec(compile(source, f"<pdum-python:{name}>", "exec"), ns)  # noqa: S102 — this IS the backend
    artifact = ns[name]
    artifact.__pdum_source__ = source  # the artifact carries its own listing (ch09 autopsy)
    return artifact


PYTHON = Backend(name="python", render=render, compile=compile_source, fp=("python", 1))


def install(registry) -> None:
    """The explicit seam (same shape as ``stdlib.install``): batteries call it
    with DEFAULT; a hand-built Registry can call it directly."""
    registry.register_backend(PYTHON, default=True)


install(DEFAULT)
