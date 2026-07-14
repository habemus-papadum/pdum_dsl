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


def render(region: Region, plan: PackPlan, backend=None, name: str = "kernel") -> str:
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
        if node.op == "array.buffer":  # leaves-channel position, resolved from THE plan (like abi offsets)
            src = attrs["src"]
            chan = [s for s in plan.slots if s.dest is None]
            for k, s in enumerate(chan):
                if s.source.root == src[0] and (s.source.index, *s.source.sub) == (*src[1:], 0):
                    return f"leaves[{k}]"
            raise VerifyError(f"no buffer leaf for {src!r}")
        if node.op == "array.dim":  # an argument array's shape/stride staging slot, from the plan
            src, sub = attrs["src"], attrs["sub"]
            for s in plan.slots:
                if s.source.root == src[0] and (s.source.index, *s.source.sub) == (*src[1:], sub):
                    return f"_u({s.dest.fmt!r}, staging, {s.dest.offset})[0]"
            raise VerifyError(f"no dim slot for {src!r}[{sub}]")
        if node.op == "core.const":
            v = attrs["value"]
            if isinstance(v, float) and not math.isfinite(v):
                return f"float({str(v)!r})"  # repr(inf) is not a Python literal
            return repr(v)
        if node.op in _BIN:
            # Numeric policy (070): TRUNC div/mod, like C — never Python's floored
            # // %. Integers go through EXACT helpers (float rounding would lose
            # precision past 2^53); float mod is math.fmod (sign of the dividend).
            if node.op in ("core.div", "core.mod") and isinstance(node.type, Scalar) and node.type.kind[0] in "iu":
                return f"{'_tdiv' if node.op == 'core.div' else '_tmod'}({arg[0]}, {arg[1]})"
            if node.op == "core.mod":
                return f"math.fmod({arg[0]}, {arg[1]})"
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
        if node.op in ("core.vec", "core.tuple"):
            return "(" + ", ".join(arg) + ("," if len(arg) == 1 else "") + ")"
        if node.op == "core.extract":
            return f"{arg[0]}[{attrs['index']}]"
        table = backend.code_for_op if backend is not None else CODE_FOR_OP
        template = table.get(node.op)  # surface D: THIS record's spellings (extend()-local)
        if template:
            return template.format(*arg)
        if template is None and node.op in table:  # spell(None) claims native support this renderer lacks
            raise VerifyError(f"{node.op!r} was spelled None ('native') but this renderer has no native handling")
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

    def loop_join(node, nm, result_of, emit_block, path, ind):
        res, (lo, hi, init) = nm[id(node)], (nm[id(a)] for a in node.args)
        iv, carry = node.regions[0].params
        out = [f"{ind}{res} = {init}"]  # res IS the carry between iterations
        out.append(f"{ind}for {nm[id(iv)]} in range({lo}, {hi}):")
        out.append(f"{ind}    {nm[id(carry)]} = {res}")
        out += emit_block((*path, (id(node), 0)), ind + "    ")
        out.append(f"{ind}    {res} = {result_of(0)}")
        return out

    lines, names, result = emit_dominated(region, statement, branch_join, indent="    ", loop=loop_join)
    body = "\n".join(lines) or "    pass"
    head = (
        "import math\nfrom struct import unpack_from as _u\n"
        "from pdum.dsl.kernel.registry import Out as _Out\n\n"
        "def _tdiv(a, b):  # exact trunc division (numeric policy: C semantics)\n"
        "    q = a // b\n"
        "    return q + 1 if q < 0 and q * b != a else q\n\n"
        "def _tmod(a, b):\n"
        "    return a - _tdiv(a, b) * b\n\n"
        f"def {name}(staging, leaves):\n"
    )
    guard = (  # buffer leaves are welcome (arrays, ch12); launcher data is peeled BY TYPE (§ Out contract)
        "    if any(isinstance(x, _Out) for x in leaves):\n"
        '        raise TypeError("the python backend takes no launcher data (out= is for device targets)")\n'
    )
    return f"{head}{guard}{body}\n    return {names[id(result)]}\n"


def compile_source(source: str, name: str = "kernel"):
    ns: dict = {}
    exec(compile(source, f"<pdum-python:{name}>", "exec"), ns)  # noqa: S102 — this IS the backend
    artifact = ns[name]
    artifact.__pdum_source__ = source  # the artifact carries its own listing (ch09 autopsy)
    return artifact


CODE_FOR_OP: dict = {
    "core.tuple": None,  # native tuple spelling; batteries add math.* templates
    "array.load": "{0}[{1}]",  # static, like the C target's — no install-order dependence
}

PYTHON = Backend(
    name="demo.simple_shader.python",
    render=render,
    compile=compile_source,
    fp=("demo.simple_shader.python", 1),
    code_for_op=CODE_FOR_OP,
)


def install(registry) -> None:
    """The explicit seam (same shape as ``stdlib.install``). Each registry
    gets its OWN record copy with a fresh spelling table — module singletons
    shared across registries would let one registry's spell() rewrite
    another's world (review-caught)."""
    from dataclasses import replace

    registry.register_backend(replace(PYTHON, code_for_op=dict(PYTHON.code_for_op)), default=True)


install(DEFAULT)
