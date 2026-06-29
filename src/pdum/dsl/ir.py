"""A small typed IR for shader bodies.

Shader bodies are mostly pure expressions, so the IR is an expression/statement
tree rather than full SSA. ``ast_lower`` builds it untyped (``type=None``); the
``infer`` pass fills every node's ``type``; the WGSL backend emits from it. Keeping
``emit`` a pure ``IR -> str`` function (no GPU) is what makes it unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Type


class Node:
    """Base for all IR nodes. ``type`` is filled in by the infer pass."""

    type: Type | None = None


# --- expressions ------------------------------------------------------------


@dataclass
class Lit(Node):
    value: object
    type: Type | None = None


@dataclass
class Name(Node):
    """A variable reference. ``scope`` ∈ {"uniform", "local", "arg"}."""

    name: str
    scope: str
    type: Type | None = None


@dataclass
class Intrinsic(Node):
    """A backend builtin, e.g. ``frag_coord`` (WGSL ``@builtin(position)``)."""

    name: str
    type: Type | None = None


@dataclass
class Swizzle(Node):
    base: Node
    comps: str  # any of "xyzw", length 1..4
    type: Type | None = None


@dataclass
class Unary(Node):
    op: str  # "-"
    operand: Node
    type: Type | None = None


@dataclass
class BinOp(Node):
    op: str  # + - * / **
    left: Node
    right: Node
    type: Type | None = None


@dataclass
class Compare(Node):
    op: str  # < > <= >= == !=
    left: Node
    right: Node
    type: Type | None = None


@dataclass
class Select(Node):
    """Ternary: ``if_true if cond else if_false`` (Python ``IfExp``)."""

    cond: Node
    if_true: Node
    if_false: Node
    type: Type | None = None


@dataclass
class MakeVec(Node):
    """A vector constructor, e.g. a returned ``(r, g, b)`` tuple → ``vec3``."""

    elems: list[Node]
    type: Type | None = None


@dataclass
class Call(Node):
    """A call to a builtin (``sqrt``, ``length``, ...) or, later, a device fn."""

    func: str
    args: list[Node]
    type: Type | None = None


# --- statements -------------------------------------------------------------


@dataclass
class Let(Node):
    name: str
    value: Node
    type: Type | None = None


@dataclass
class Return(Node):
    value: Node
    type: Type | None = None


@dataclass
class Function:
    name: str
    params: list[str]
    body: list[Node]
    ret_type: Type | None = None
    locals: dict[str, Type] = field(default_factory=dict)
