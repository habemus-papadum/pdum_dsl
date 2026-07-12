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

from ...backends._emit import emit_dominated
from ...kernel.ir import Node, Region, VerifyError
from ...kernel.pack import PackPlan
from ...kernel.registry import DEFAULT, Backend
from ...kernel.types import Scalar

_BIN = {"core.add": "+", "core.sub": "-", "core.mul": "*", "core.div": "/", "core.mod": "%", "core.pow": "**"}
_PREDS = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
_CASTS = {"f64": "float", "f32": "float", "i64": "int", "i32": "int", "u64": "int", "u32": "int", "bool": "bool"}


def render(region: Region, plan: PackPlan, name: str = "kernel") -> str:
    """Legalized Region -> Python source. One assignment per node, DAG-shared
    nodes emitted once in their OWNER region (the shared dominator-placed
    walker in ``_emit`` — the lazy-branch rule), names dense in topo order."""
    args_by_index = {s.source.index: s for s in plan.slots if s.source.root == "arg" and not s.source.sub}

    def expr_of(node: Node, names: dict) -> str:
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

    def statement(node, nm):
        return f"{nm[id(node)]} = {expr_of(node, nm)}"

    def branch_join(node, nm, result_of, emit_block, path, ind):
        res = nm[id(node)]
        out = [f"{ind}if {nm[id(node.args[0])]}:"]
        out += emit_block((*path, (id(node), 0)), ind + "    ")
        out.append(f"{ind}    {res} = {result_of(0)}")
        out.append(f"{ind}else:")
        out += emit_block((*path, (id(node), 1)), ind + "    ")
        out.append(f"{ind}    {res} = {result_of(1)}")
        return out

    lines, names, result = emit_dominated(region, statement, branch_join, indent="    ")
    body = "\n".join(lines) or "    pass"
    head = f"from struct import unpack_from as _u\n\ndef {name}(staging, leaves):\n"
    guard = '    if leaves: raise TypeError("the python backend takes no launcher data (out= is for device targets)")\n'
    return f"{head}{guard}{body}\n    return {names[id(result)]}\n"


def compile_source(source: str, name: str = "kernel"):
    ns: dict = {}
    exec(compile(source, f"<pdum-python:{name}>", "exec"), ns)  # noqa: S102 — this IS the backend
    artifact = ns[name]
    artifact.__pdum_source__ = source  # the artifact carries its own listing (ch09 autopsy)
    return artifact


PYTHON = Backend(
    name="demo.simple_shader.python", render=render, compile=compile_source, fp=("demo.simple_shader.python", 1)
)


def install(registry) -> None:
    """The explicit seam (same shape as ``stdlib.install``): batteries call it
    with DEFAULT; a hand-built Registry can call it directly."""
    registry.register_backend(PYTHON, default=True)


install(DEFAULT)
