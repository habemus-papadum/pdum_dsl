"""The DSL type system: a small, hashable, structural type lattice plus ``typeof``.

This is the use-case-independent core that the caching layer keys on. Types are
*pure values* (frozen, hashable) so they can go straight into a cache key. The
defining property (see ``docs/dsl_caching_layer.md``): specialization keys on
*types*, never on captured *values* — so ``closure(5)`` and ``closure(6)`` share
one ``FnType`` and therefore one compiled artifact.

The lattice here is honest about Python values (a Python ``int`` is 64-bit, a
``float`` is 64-bit). Narrowing to a backend's concrete types (e.g. WGSL only has
32-bit scalars) is the *backend's* decision, not the type system's.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import CodeType


@dataclass(frozen=True)
class Type:
    """Base class for all DSL types. Subclasses are frozen → hashable by value."""


@dataclass(frozen=True)
class IntType(Type):
    bits: int
    signed: bool

    def __repr__(self) -> str:
        return f"{'i' if self.signed else 'u'}{self.bits}"


@dataclass(frozen=True)
class FloatType(Type):
    bits: int

    def __repr__(self) -> str:
        return f"f{self.bits}"


@dataclass(frozen=True)
class BoolType(Type):
    def __repr__(self) -> str:
        return "bool"


@dataclass(frozen=True)
class NoneType(Type):
    def __repr__(self) -> str:
        return "none"


@dataclass(frozen=True)
class VecType(Type):
    """A fixed-length vector of a scalar type (the WGSL ``vecN<T>`` family)."""

    elem: Type
    n: int

    def __repr__(self) -> str:
        return f"vec{self.n}<{self.elem!r}>"


@dataclass(frozen=True)
class TupleType(Type):
    """A heterogeneous tuple. Arity is part of the type: ``(1,2) != (1,2,3)``."""

    elems: tuple[Type, ...]

    def __repr__(self) -> str:
        return f"({', '.join(repr(e) for e in self.elems)})"


@dataclass(frozen=True)
class FnType(Type):
    """The structural function type of a DSL closure.

    ``template`` is the function's code object — compared *by value* (CPython
    code objects implement value equality over ``co_code``/``co_consts``/...), so
    an unchanged re-run hits the same key while an edited body misses. ``env_types``
    are the types of the captured free variables (a capture that is itself a DSL
    closure contributes its own ``FnType`` — nested-closure recursion).
    """

    template: CodeType
    env_types: tuple[Type, ...]

    def __repr__(self) -> str:
        return f"Fn<{self.template.co_qualname}>({', '.join(repr(t) for t in self.env_types)})"


# --- singletons for the common scalars -------------------------------------

i32 = IntType(32, True)
u32 = IntType(32, False)
i64 = IntType(64, True)
u64 = IntType(64, False)
f32 = FloatType(32)
f64 = FloatType(64)
boolean = BoolType()
none = NoneType()


class BigIntError(TypeError):
    """A captured int does not fit any supported fixed-width integer type."""


def typeof(value: object) -> Type:
    """Map a runtime Python value to its DSL :class:`Type`.

    On the hot path *and* correctness-defining: too coarse and the wrong
    specialization is silently reused (see the caching doc). Ints are
    range-bucketed; ``bool`` is checked before ``int`` (it is a subclass).
    """
    # NOTE: order matters — bool is a subclass of int.
    if isinstance(value, bool):
        return boolean
    if isinstance(value, int):
        if -(2**63) <= value < 2**63:
            return i64
        if 0 <= value < 2**64:
            return u64
        raise BigIntError(f"int {value!r} does not fit in 64 bits")
    if isinstance(value, float):
        return f64
    if value is None:
        return none
    if isinstance(value, tuple):
        return TupleType(tuple(typeof(v) for v in value))

    # A DSL closure presents its structural function type. Imported lazily to
    # avoid a module import cycle (jit imports types).
    from .jit import Handle

    if isinstance(value, Handle):
        return value.fntype

    raise TypeError(f"unsupported value for typeof: {value!r} ({type(value).__name__})")


def typeof_tuple(values: tuple[object, ...]) -> tuple[Type, ...]:
    """``typeof`` over a tuple of values (env captures, or call arguments)."""
    return tuple(typeof(v) for v in values)
