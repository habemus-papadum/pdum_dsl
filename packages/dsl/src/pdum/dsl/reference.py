"""The reference evaluator: the value tier's oracle (design 200 §1.1).

Renders a legalized Region to *readable* Python source and executes it with
the exact ABI every real target will use: ``kernel(staging, leaves)``, every
input — captures AND scalar arguments — arriving as bytes in the staging
buffer at the offsets the plan chose. Deliberately small, deliberately slow:
this is the twin every future device backend is differential-tested
against, never a production path.

**Oracle execution is always spelled**: ``reference(f)(...)``. A plain call
on a kind with no routed backend refuses (registry.backend_for); it does not
degrade to interpretation.

Reference semantics, stated once:

- ``core.if`` renders as a real ``if``/``else`` — branches are LAZY, so the
  guard-then-divide idiom is safe; each node is emitted in the deepest
  region that dominates all its uses.
- Python has one float; ``f32`` computes as f64 here. Narrowing gets real in
  device backends.
- Numeric policy (distilled): TRUNC integer div/mod via exact helpers
  (float rounding would lose precision past 2^53); float mod is
  ``math.fmod`` (sign of the dividend).
- Composite (record-typed) *arguments* are refused loudly: their leaf slots
  exist in the plan, but the IR keeps one logical ``core.param``.
"""

from __future__ import annotations

import math

from .ir import Node, Region, VerifyError
from .pack import PackPlan
from .registry import Backend
from .render import emit_dominated
from .types import Scalar

_BIN = {"core.add": "+", "core.sub": "-", "core.mul": "*", "core.div": "/", "core.mod": "%", "core.pow": "**"}
_PREDS = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
_CASTS = {"f64": "float", "f32": "float", "i64": "int", "i32": "int", "u64": "int", "u32": "int", "bool": "bool"}


def render(region: Region, plan: PackPlan, backend=None, name: str = "kernel") -> str:
    """Legalized Region -> Python source. One assignment per node, DAG-shared
    nodes emitted once in their OWNER region (the dominator-placed walker in
    ``render`` — the lazy-branch rule), names dense in topo order."""
    args_by_index = {s.source.index: s for s in plan.slots if s.source.root == "arg" and not s.source.sub}

    def expr_of(node: Node, names: dict) -> str:
        attrs = dict(node.attrs)
        arg = [names[id(a)] for a in node.args]
        if node.op == "core.param":
            spec = args_by_index.get(attrs["index"])
            if spec is None:
                raise VerifyError(
                    f"argument {attrs['index']} has no scalar slot — composite (record-typed) "
                    "arguments are not marshalable yet; pass fields as separate arguments"
                )
            return f"_u({spec.dest.fmt!r}, staging, {spec.dest.offset})[0]"
        if node.op == "abi.slot":
            return f"_u({attrs['fmt']!r}, staging, {attrs['offset']})[0]"
        if node.op == "core.const":
            v = attrs["value"]
            if isinstance(v, float) and not math.isfinite(v):
                return f"float({str(v)!r})"  # repr(inf) is not a Python literal
            return repr(v)
        if node.op in _BIN:
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
                raise VerifyError(f"the reference evaluator cannot cast to {to!r}")
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
            raise VerifyError(f"{node.op!r} was spelled None ('native') but the reference has no native handling")
        raise VerifyError(f"the reference evaluator has no rendering for {node.op!r}")

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
        "import math\nfrom struct import unpack_from as _u\n\n"
        "def _tdiv(a, b):  # exact trunc division (numeric policy: C semantics)\n"
        "    q = a // b\n"
        "    return q + 1 if q < 0 and q * b != a else q\n\n"
        "def _tmod(a, b):\n"
        "    return a - _tdiv(a, b) * b\n\n"
        f"def {name}(staging, leaves):\n"
    )
    return f"{head}{body}\n    return {names[id(result)]}\n"


def compile_source(source: str, name: str = "kernel"):
    ns: dict = {}
    exec(compile(source, f"<pdum-reference:{name}>", "exec"), ns)  # noqa: S102 — this IS the oracle
    artifact = ns[name]
    artifact.__pdum_source__ = source  # the artifact carries its own listing
    return artifact


CODE_FOR_OP: dict = {
    "core.tuple": None,  # native tuple spelling; intrinsics add math.* templates
}

REFERENCE = Backend(
    name="reference",
    render=render,
    compile=compile_source,
    fp=("reference", 1),
    code_for_op=CODE_FOR_OP,
)


def install(registry) -> None:
    """Register the oracle. NEVER default, NEVER routed: reference execution
    is reachable only through the spelled door below. Each registry gets its
    OWN record copy with a fresh spelling table."""
    from dataclasses import replace

    registry.register_backend(replace(REFERENCE, code_for_op=dict(REFERENCE.code_for_op)))


def reference(target, *, registry=None):
    """The spelled oracle door: ``reference(f)(args...)`` runs ``f`` on the
    reference evaluator through the same two-tier dispatch as any backend
    (specialization cache, guards, marshaling — the thesis machinery is
    exercised, only the artifact is the readable Python twin)."""
    if registry is None:
        from .registry import DEFAULT as registry  # noqa: N811

    def run(*args):
        return registry.dispatch(target, args, backend="reference")

    return run
