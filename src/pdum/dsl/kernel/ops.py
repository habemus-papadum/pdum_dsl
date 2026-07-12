"""``OpDef`` and the core dialect table: ops as data, never Node subclasses.

A dialect is a ``dict[str, OpDef]``; installing one is a dict merge. Type
rules run in the **honest** world (f64/i64 — backend ``type_map``s narrow at
render, never here). Ops whose type cannot be computed from operands
(``core.env``, ``core.const``, ``core.load``) have ``type_rule=None`` and
demand an explicit ``type=`` at emit.

THE STRICT-CORE RULE (settled 2026-07-12, superseding the earlier
one-promotion-semantics note): **core arithmetic and comparison are strict**
— operands must share a type; every conversion is an explicit ``core.cast``
in the IR. There is NO promotion in the kernel: promotion, where a language
wants it, is a *dialect's lowering policy* (auto-insert casts — the friendly
option) or absent (force the user to write ``float(i)`` — the strict
option). This is Julia's architecture (promotion is stdlib methods, not
compiler magic) and MLIR/LLVM/WGSL's (strict operands everywhere). Payoffs:
emitters never invent conversions at render; AD rules see matching types by
construction; the content hash reflects the exact computation. The residual,
documented surprise: the same surface ``x + y`` may lower with different
cast insertions under different dialects — visible in printed IR, never
ambient.

The unit type is ``Tuple(())`` — no lattice addition needed; ``core.yield``
types itself as the tuple of what it yields (a region's result vocabulary).

Book: ``docs/book/ch05-programs-are-values.ipynb``. Architecture: §2.6.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .types import Record, Scalar, Tuple, Type, Vec, boolean

UNIT = Tuple(())


@dataclass(frozen=True)
class OpDef:
    name: str
    type_rule: Callable | None = None  # (arg_types, attrs, regions) -> Type
    traits: frozenset = frozenset()
    nregions: int = 0


PURE = frozenset({"Pure"})
PURE_COMM = frozenset({"Pure", "Commutative"})


def _arith(args, attrs, regions) -> Type:
    if args[0] != args[1]:
        raise TypeError(f"core arithmetic is strict: {args[0]!r} vs {args[1]!r} — insert an explicit core.cast")
    return args[0]


def _same(args, attrs, regions) -> Type:
    return args[0]


def _cmp(args, attrs, regions) -> Type:
    if args[0] != args[1]:
        raise TypeError(f"core.cmp is strict: {args[0]!r} vs {args[1]!r} — insert an explicit core.cast")
    return boolean


def _select(args, attrs, regions) -> Type:
    if args[0] != boolean:
        raise TypeError(f"select condition must be bool, got {args[0]!r}")
    if args[1] != args[2]:
        raise TypeError(f"select branches disagree: {args[1]!r} vs {args[2]!r}")
    return args[1]


def _vec(args, attrs, regions) -> Type:
    if not (2 <= len(args) <= 4 and isinstance(args[0], Scalar) and all(t == args[0] for t in args)):
        raise TypeError(f"core.vec wants 2-4 elements of one scalar type, got {args!r}")
    return Vec(args[0], len(args))


def _extract(args, attrs, regions) -> Type:
    (t,) = args
    if not isinstance(t, Vec) or not 0 <= attrs["index"] < t.n:
        raise TypeError(f"cannot extract index {attrs.get('index')!r} from {t!r}")
    return t.elem


def _field(args, attrs, regions) -> Type:
    (t,) = args
    if not isinstance(t, Record) or attrs["name"] not in dict(t.fields):
        raise TypeError(f"no field {attrs.get('name')!r} on {t!r}")
    return dict(t.fields)[attrs["name"]]


def _cast(args, attrs, regions) -> Type:
    return attrs["to"]


def _yielded(region) -> Type:
    types = tuple(a.type for a in region.body[-1].args)
    return types[0] if len(types) == 1 else Tuple(types)


def _yield_rule(args, attrs, regions) -> Type:
    return args[0] if len(args) == 1 else Tuple(tuple(args))


def _if(args, attrs, regions) -> Type:
    then, other = _yielded(regions[0]), _yielded(regions[1])
    if then != other:
        raise TypeError(f"core.if branches yield {then!r} vs {other!r}")
    return then


def _for(args, attrs, regions) -> Type:
    carries = args[2:]  # (lo, hi, *carries) -> the carries flow through
    return carries[0] if len(carries) == 1 else Tuple(tuple(carries))


def _call(args, attrs, regions) -> Type:
    return _yielded(regions[0])


CORE_OPS: dict[str, OpDef] = {
    "core.add": OpDef("core.add", _arith, PURE_COMM),
    "core.sub": OpDef("core.sub", _arith, PURE),
    "core.mul": OpDef("core.mul", _arith, PURE_COMM),
    "core.div": OpDef("core.div", _arith, PURE),
    "core.mod": OpDef("core.mod", _arith, PURE),
    "core.pow": OpDef("core.pow", _arith, PURE),
    "core.neg": OpDef("core.neg", _same, PURE),
    "core.cmp": OpDef("core.cmp", _cmp, PURE),  # attrs: pred = lt|gt|le|ge|eq|ne
    "core.select": OpDef("core.select", _select, PURE),
    "core.vec": OpDef("core.vec", _vec, PURE),
    "core.extract": OpDef("core.extract", _extract, PURE),  # attrs: index
    "core.field": OpDef("core.field", _field, PURE),  # attrs: name
    "core.cast": OpDef("core.cast", _cast, PURE),  # attrs: to = Type
    "core.env": OpDef("core.env"),  # attrs: slot; type supplied by lowering
    "core.const": OpDef("core.const"),  # attrs: value; the Literal carve-out
    "core.param": OpDef("core.param"),  # attrs: index; a region binder
    "core.load": OpDef("core.load"),  # boundary-only effects (stores sink)
    "core.store": OpDef("core.store", lambda a, at, r: UNIT),
    "core.yield": OpDef("core.yield", _yield_rule),
    "core.if": OpDef("core.if", _if, nregions=2),
    "core.for": OpDef("core.for", _for, nregions=1),
    "core.call": OpDef("core.call", _call, nregions=1),
}
