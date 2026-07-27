"""The entire IR: one frozen node type, structured control flow as regions.

The load-bearing negative invariant: **no field reachable from ``Node`` can
hold a runtime value.** ``attrs`` is the single value-shaped slot, and it *is*
the compile-time-constant carve-out (``core.const``, ``Literal`` lifts) —
inside structural identity, visible in printed IR. A runtime capture is
``core.env(slot=k)``: a slot number, never a value. numba's
``ir.FreeVar(idx, name, value)`` is unrepresentable here by construction, and
the anti-pattern gate (``tests/test_ir.py``) checks the field annotations
mechanically.

Identity is two-layered, like everything in this system: Python ``==``/hash
are structural (deep, for tests and small work), and ``Node.key`` is a
memoized sha256 **content key** — the artifact-tier cache key. ``loc`` is
excluded from both: where code came from is not what it is. The content key
is in-process (it folds in ``hash()`` of types to disambiguate reprs); the
future disk cache re-keys structurally per the architecture (§4.4).

Exactly three region-carrying ops exist (``core.if``, ``core.for``,
``core.call``); a proposed fourth is priced at ~180 lines × live transform
columns before acceptance (architecture §2.5).

Book: ``docs/book/ch05-programs-are-values.ipynb``. Architecture: §2.5.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Hashable

from .types import Type

Attr = tuple[str, Hashable]


class VerifyError(Exception):
    """Malformed IR, refused at construction/verification — never at render."""


@dataclass(frozen=True, slots=True)
class Loc:
    file: str
    line: int
    col: int = 0


@dataclass(frozen=True, slots=True)
class CallLoc:
    """Provenance through inlining: the callee's loc within the caller's site."""

    callee: Loc | CallLoc | FusedLoc
    caller: Loc | CallLoc | FusedLoc


@dataclass(frozen=True, slots=True)
class FusedLoc:
    """A rewrite merged several nodes; every contributor, kept."""

    locs: tuple


Provenance = Loc | CallLoc | FusedLoc


def format_loc(p) -> str:
    """Render provenance compactly: 'a.py:5 (inlined from b.py:40)', '{a, b}'."""
    if isinstance(p, Loc):
        return f"{p.file}:{p.line}" + (f":{p.col}" if p.col else "")
    if isinstance(p, CallLoc):
        return f"{format_loc(p.callee)} (inlined from {format_loc(p.caller)})"
    if isinstance(p, FusedLoc):
        return "{" + ", ".join(format_loc(x) for x in p.locs) + "}"
    return "?"


def _hash_parts(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(4, "little"))
        h.update(p)
    return h.digest()


def _obj_bytes(x: object) -> bytes:
    return f"{x!r}#{hash(x)}".encode()  # repr for readability, hash to disambiguate


@dataclass(frozen=True, slots=True)
class Node:
    """A node IS its SSA value: exactly one result type, no names, no CFG."""

    op: str  # dialect-namespaced: "core.add", "abi.slot", "wgsl.frag_coord"
    type: Type
    args: tuple[Node, ...] = ()
    regions: tuple[Region, ...] = ()
    attrs: tuple[Attr, ...] = ()  # compile-time constants — INSIDE identity
    loc: Provenance | None = field(default=None, compare=False)  # EXCLUDED from identity
    _key: bytes | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def key(self) -> bytes:
        if self._key is None:
            parts = [self.op.encode(), _obj_bytes(self.type), _obj_bytes(self.attrs)]
            parts += [a.key for a in self.args] + [r.key for r in self.regions]
            object.__setattr__(self, "_key", _hash_parts(*parts))
        return self._key


@dataclass(frozen=True, slots=True)
class Region:
    """Ordered body of a structured op (or the program itself); pure, with
    typed ``core.param`` binders and a terminating ``core.yield``."""

    params: tuple[Node, ...] = ()
    body: tuple[Node, ...] = ()
    _key: bytes | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def key(self) -> bytes:
        if self._key is None:
            parts = [b"region"] + [p.key for p in self.params] + [n.key for n in self.body]
            object.__setattr__(self, "_key", _hash_parts(*parts))
        return self._key


class Builder:
    """The blessed construction path: sorts attrs canonically, runs the op's
    type rule (or demands an explicit ``type=`` where none can exist, e.g.
    ``core.env``), and validates region arity."""

    def __init__(self, ops: dict):
        self._ops = ops
        self.default_loc: Provenance | None = None  # the rewrite driver's inherit-default

    def emit(self, op: str, *args: Node, regions: tuple = (), loc=None, type: Type | None = None, **attrs) -> Node:
        opdef = self._ops.get(op)
        if opdef is None:
            raise VerifyError(f"unknown op {op!r}; register it (defop) before emitting")
        if len(regions) != opdef.nregions:
            raise VerifyError(f"{op} takes {opdef.nregions} region(s), got {len(regions)}")
        canon = tuple(sorted(attrs.items()))
        loc = loc if loc is not None else self.default_loc
        if type is None:
            if opdef.type_rule is None:
                raise VerifyError(f"{op} has no type rule; pass an explicit type=")
            try:
                type = opdef.type_rule(tuple(a.type for a in args), dict(canon), regions)
            except TypeError as exc:
                points = [format_loc(x) for x in (loc, *(a.loc for a in args)) if x is not None]
                suffix = f" [{'; '.join(points)}]" if points else ""
                raise TypeError(f"{exc}{suffix}") from None
        return Node(op, type, tuple(args), tuple(regions), canon, loc)

    def param(self, index: int, type: Type) -> Node:
        return Node("core.param", type, attrs=(("index", index),))


def verify(region: Region, ops: dict) -> None:
    """Structural checks: params are params, bodies terminate in a yield,
    region arities match the op table. Cheap; run at stage boundaries."""
    for p in region.params:
        if p.op != "core.param":
            raise VerifyError(f"region param must be core.param, got {p.op!r}")
    if not region.body or region.body[-1].op != "core.yield":
        raise VerifyError("region body must end with core.yield")
    seen: set[int] = set()

    def walk(n: Node) -> None:
        if id(n) in seen:
            return
        seen.add(id(n))
        opdef = ops.get(n.op)
        if opdef is not None and len(n.regions) != opdef.nregions:
            raise VerifyError(f"{n.op} carries {len(n.regions)} region(s), expected {opdef.nregions}")
        for a in n.args:
            walk(a)
        for r in n.regions:
            verify(r, ops)

    for n in region.body:
        walk(n)
