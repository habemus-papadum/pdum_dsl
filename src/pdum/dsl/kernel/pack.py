"""Marshaling: one logical value -> N physical parameters, in both directions.

M0's disease was per-frame flattening — every call re-walked the env looking
for uniforms. The cure is structural (architecture §2.11, §4.3): a
``PackPlan`` is built ONCE per cache entry from **types alone** — values
never shape a plan — and per call one generic loop writes bytes into a
reused staging buffer. Marshaling is **bidirectional from the start**
(design 040 §3b): kernels are destination-passing at the ABI, so the result
side mirrors the input side as a ``ResultPlan`` (allocate destinations from
types; unflatten result bytes back into the logical value).

Three layers, all kernel-owned:

- **Leaves** — the closed vocabulary backends must be total over:
  ``ScalarLeaf`` packs by format; ``BufferLeaf``/``ShapeLeaf``/``StrideLeaf``
  arrive with the ndarray kind (stdlib) — buffers travel the *leaves
  channel* to the launcher, never staging. The architecture's ``EnvLeaf``
  recursion is the ``FnType`` walker: a captured kernel's leaves ARE its
  env's leaves with prefixed paths — the same paths lowering stamps on
  ``core.env`` (root capture *i* = ``(i,)``, callee *j* under it = ``(i, j)``).
- **Plans** — ``LeafPath -> SlotSpec(source, convert, dest) -> PackPlan``.
  The ``dest`` vocabulary is backend-owned; the kernel assumes only that
  byte destinations expose ``offset``/``fmt``. ``PackedDest`` is the
  reference dense layout (tests, the book); real backends bring std140 etc.
  at step 9. ``SlotSpec.convert`` is the units seat (step 15): a unit tweak
  swaps a converter — bytes change, plans and artifacts never do.
- **Stages** — the marshaling decision is IR, printable and golden-testable
  (P2 graft): ``NORMALIZE_ENV`` folds ``core.extract``/``core.field``-of-env
  into the env *path* (captures become leaf-typed), then
  ``legalize_params(plan)`` rewrites every ``core.env`` into a physical
  ``abi.slot`` op, and the stage's legality set {core, abi} machine-checks
  that no logical capture survived — per-frame flatten is now structurally
  impossible. ``core.param`` binders are positional runtime arguments; their
  physical spelling belongs to the backend renderer (steps 8/9).

Book: ``docs/book/ch08-one-value-n-parameters.ipynb``.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass

from .ir import VerifyError
from .ops import OpDef
from .rewrite import Pat, Stage
from .types import SCALAR_KINDS, FnType, Record, Scalar, Tuple, Type, Vec
from .valuekind import BUILTINS, KindTable


@dataclass(frozen=True, slots=True)
class ScalarLeaf:
    kind: str  # "f64", "i32", "bool", ...


@dataclass(frozen=True, slots=True)
class BufferLeaf:
    pass  # array payload: pointer-like, launcher-bound, never byte-packed


@dataclass(frozen=True, slots=True)
class ShapeLeaf:
    axis: int


@dataclass(frozen=True, slots=True)
class StrideLeaf:
    axis: int


def _composite_leaves(children) -> Callable:
    """One prefix-and-recurse walk, shared by every composite Type: child *i*
    contributes its own entries under path ``(i, *sub)``. Lowering stamps the
    same paths on ``core.env`` — one path language, defined once."""

    def walk(t: Type, table: KindTable) -> tuple:
        return tuple(((i, *sub), leaf) for i, e in enumerate(children(t)) for sub, leaf in table.leaf_entries(e))

    return walk


# The three Type-keyed marshaling aspects. `leaves` plans, `child` descends one
# step (the compiled extractor's only per-frame work), `rebuild` reassembles a
# result. Registered on the table, so a layered table can override any of them.
_CHILDREN = {
    Tuple: lambda t: t.elems,
    FnType: lambda t: t.env_types,  # the EnvLeaf recursion
    Record: lambda t: tuple(ty for _, ty in t.fields),
}
for _cls, _children in _CHILDREN.items():
    BUILTINS.register_aspect("leaves", _cls, _composite_leaves(_children))
BUILTINS.register_aspect("leaves", Scalar, lambda t, table: (((), ScalarLeaf(t.kind)),))
# Vec is IR-only (never a capture), but it IS a result type — a fragment
# shader yields vec4. Its leaves are its lanes; it rebuilds to a plain tuple.
BUILTINS.register_aspect("leaves", Vec, lambda t, table: tuple(((i,), ScalarLeaf(t.elem.kind)) for i in range(t.n)))
BUILTINS.register_aspect("child", Vec, lambda t, i: (t.elem, lambda v: v[i]))

# `child(t, i) -> (subtype, getter)`: FnType-typed values (Handle, Pipeline)
# expose `.captures`; tuples index; records read the field by name.
BUILTINS.register_aspect("child", Tuple, lambda t, i: (t.elems[i], lambda v: v[i]))
BUILTINS.register_aspect("child", FnType, lambda t, i: (t.env_types[i], lambda v: v.captures[i]))
BUILTINS.register_aspect(
    "child", Record, lambda t, i: (t.fields[i][1], lambda v, n=t.fields[i][0]: getattr(v, n))
)

BUILTINS.register_aspect("rebuild", Scalar, lambda t, it, rec: next(it))
BUILTINS.register_aspect("rebuild", Tuple, lambda t, it, rec: tuple(rec(e, it) for e in t.elems))
BUILTINS.register_aspect("rebuild", Vec, lambda t, it, rec: tuple(rec(t.elem, it) for _ in range(t.n)))


@dataclass(frozen=True, slots=True)
class LeafPath:
    root: str  # "env" | "arg" | "out"
    index: int  # capture slot / argument position
    sub: tuple = ()  # steps inside the composite (= lowering's env-path tail)


@dataclass(frozen=True, slots=True)
class SlotSpec:
    source: LeafPath
    convert: Callable | None  # the units seat: applied to the value, never the plan
    dest: object  # backend-owned; None = leaves channel; byte dests expose offset/fmt


@dataclass(frozen=True, slots=True)
class PackPlan:
    slots: tuple
    staging_size: int


@dataclass(frozen=True, slots=True)
class PackedDest:
    """Reference dense little-endian layout — the book's and tests' toy.
    Real backends own their PhysicalDest vocabulary (UniformSlot, KernelArg…)."""

    offset: int
    fmt: str


_FMTS = {"f64": "<d", "f32": "<f", "i64": "<q", "i32": "<i", "u64": "<Q", "u32": "<I", "bool": "<?"}
if set(_FMTS) != SCALAR_KINDS:  # an `assert` here would vanish under -O, taking the guard with it
    raise ImportError(f"scalar lattice and pack formats disagree: {SCALAR_KINDS ^ set(_FMTS)}")


def packed_dest(path: LeafPath, leaf: object, offset: int) -> tuple:
    if not isinstance(leaf, ScalarLeaf):
        return None, offset  # buffer-class leaf: travels the leaves channel
    fmt = _FMTS[leaf.kind]
    return PackedDest(offset, fmt), offset + struct.calcsize(fmt)


def _build_slots(roots: tuple, table: KindTable, dest_for) -> tuple:
    """The one byte-layout loop — `(slots, size)` from `(root, index, type)`
    triples. Inputs and results share it, so a backend's alignment policy can
    never apply to one direction and not the other."""
    slots, offset = [], 0
    for root, i, t in roots:
        for sub, leaf in table.leaf_entries(t):
            path = LeafPath(root, i, sub)
            dest, offset = dest_for(path, leaf, offset)
            slots.append(SlotSpec(path, None, dest))
    return tuple(slots), offset


def plan_from_types(env_types: tuple, arg_types: tuple, table: KindTable, dest_for=packed_dest) -> PackPlan:
    """Built once per cache entry, from types alone — values never shape a plan."""
    roots = tuple(
        (root, i, t) for root, types in (("env", env_types), ("arg", arg_types)) for i, t in enumerate(types)
    )
    return PackPlan(*_build_slots(roots, table, dest_for))


def build_extractor(env_types: tuple, arg_types: tuple, plan: PackPlan, table: KindTable) -> Callable:
    """``(env_values, args) -> leaf values`` aligned with ``plan.slots``.

    The plan's leaf paths are **compiled once** into one getter per slot — a
    chain of pure index/attribute reads, resolved against the TYPES at build
    time (architecture §4.3.10). The hot path therefore does no kind dispatch,
    no recursion, and no intermediate tuples: M0's per-frame flatten is gone
    from the extractor too, not just from the IR."""
    getters: list = []
    for spec in plan.slots:
        types = env_types if spec.source.root == "env" else arg_types
        t, i = types[spec.source.index], spec.source.index
        chain: list = [(lambda vs, args, i=i: vs[i]) if spec.source.root == "env" else (lambda vs, args, i=i: args[i])]
        for step in spec.source.sub:  # descend the composite by TYPE — statically known
            t, get = table.aspect("child", t)(t, step)
            prev = chain[-1]
            chain.append(lambda vs, args, prev=prev, get=get: get(prev(vs, args)))
        getters.append(chain[-1])
    return lambda env_values, args: tuple(get(env_values, args) for get in getters)


def pack_into(plan: PackPlan, staging, values) -> tuple:
    """The generic packer: byte slots into staging, the rest out the leaves channel."""
    leaves = []
    for spec, v in zip(plan.slots, values, strict=True):
        if spec.convert is not None:
            v = spec.convert(v)
        if spec.dest is None:
            leaves.append(v)
        else:
            struct.pack_into(spec.dest.fmt, staging, spec.dest.offset, v)
    return tuple(leaves)


@dataclass(frozen=True, slots=True)
class ResultPlan:
    """The output mirror (DPS): destinations allocated from the result type,
    result bytes unflattened back into the logical value."""

    result_type: Type
    slots: tuple
    size: int


def result_plan(t: Type, table: KindTable, dest_for=packed_dest) -> ResultPlan:
    """The output mirror. Rebuildability is checked HERE, from the type alone —
    a result the packer could allocate but never reassemble must fail at plan
    build, not after the kernel has run."""
    plan = ResultPlan(t, *_build_slots((("out", 0, t),), table, dest_for))
    unflatten(t, (None,) * len(plan.slots), table)  # dry run: resolves every rebuild rule
    return plan


def unflatten(t: Type, leaf_values, table: KindTable) -> object:
    """Leaves -> the logical value, by the ``rebuild`` aspect (MRO-dispatched
    and layerable, exactly like ``leaves``)."""
    it = iter(leaf_values)

    def rec(t: Type, it) -> object:
        return table.aspect("rebuild", t)(t, it, rec)

    return rec(t, it)


def unpack_result(plan: ResultPlan, buf, table: KindTable, leaves: tuple = ()) -> object:
    """The mirror of ``pack_into``: byte slots read from the result buffer,
    buffer-class slots taken (in order) from the device's ``leaves`` channel."""
    channel = iter(leaves)
    values = tuple(
        next(channel) if s.dest is None else struct.unpack_from(s.dest.fmt, buf, s.dest.offset)[0]
        for s in plan.slots
    )
    return unflatten(plan.result_type, values, table)


ABI_OPS = {"abi.slot": OpDef("abi.slot")}  # no type rule: type carried over; attrs: src, offset, fmt


def _fold_extract(b, m):
    env, root = m["e"], m["root"]
    return b.emit("core.env", type=root.type, slot=dict(env.attrs)["slot"] + (dict(root.attrs)["index"],))


def _fold_field(b, m):
    env, root = m["e"], m["root"]
    names = [name for name, _ in env.type.fields]
    return b.emit("core.env", type=root.type, slot=dict(env.attrs)["slot"] + (names.index(dict(root.attrs)["name"]),))


NORMALIZE_ENV = Stage(
    "normalize-env",
    [
        (Pat("core.extract", args=("e",), guard=lambda m: m["e"].op == "core.env"), _fold_extract),
        (Pat("core.field", args=("e",), guard=lambda m: m["e"].op == "core.env"), _fold_field),
    ],
)


def legalize_params(plan: PackPlan) -> Stage:
    """Every logical ``core.env`` becomes a physical ``abi.slot``; the
    {core, abi} legality set proves none survived. A whole-composite capture
    (no leaf slot for its path) is refused loudly — run NORMALIZE_ENV first."""
    by_path = {(s.source.index, *s.source.sub): s for s in plan.slots if s.source.root == "env"}

    def fire(b, m):
        node = m["root"]
        path = dict(node.attrs)["slot"]
        spec = by_path.get(path)
        if spec is None:
            if isinstance(node.type, FnType):  # a captured KERNEL used as a value
                raise VerifyError(
                    f"env path {path} is a captured kernel used as a value, not called. Normalizing "
                    f"cannot help (there is no extract-of-a-kernel): a kernel capture is a callee — "
                    f"call it, or wait for first-class kernel values"
                )
            raise VerifyError(
                f"no slot for env path {path}: a composite capture used whole "
                f"(run NORMALIZE_ENV first), or the plan was built from different types"
            )
        if spec.dest is None:  # a leaves-channel capture (arrays): its ABI op is the backend's
            raise VerifyError(
                f"env path {path} is a leaves-channel leaf; the byte-slot stage cannot legalize it "
                f"— the buffer-binding op ships with the kind that produces the leaf (ndarray, ch12)"
            )
        return b.emit("abi.slot", type=node.type, src=("env", *path), offset=spec.dest.offset, fmt=spec.dest.fmt)

    # `forbid` is what actually proves the point: a namespace target cannot say
    # "core.env is gone" (core.env IS core), so per-frame flatten would creep
    # back the day this rule became partial. Now it is machine-checked.
    return Stage(
        "legalize-params",
        [(Pat("core.env"), fire)],
        legal=frozenset({"core", "abi"}),
        forbid=frozenset({"core.env"}),
    )
