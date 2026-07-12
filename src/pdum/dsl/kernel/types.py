"""The structural type lattice and template identity.

Types are **frozen, hashable, structural values** — they go straight into
cache keys, and equality means "same structural summary", never object
identity. A ``Type`` is not "the Python class of a value": it is whatever
summary a ``ValueKind.typeof`` chose to extract (see ``valuekind.py`` and
architecture §13). The lattice is honest about Python widths (a Python int is
64-bit); narrowing to a backend's widths is a backend ``type_map`` decision,
never recorded here.

``TemplateId`` is code identity as a sum type: a ``Base`` wraps a code object
(CPython compares code objects **by value**, so an unchanged notebook re-run
produces an *equal* identity and an edited body a different one — the
live-coding invalidation story), and a ``Derived`` is a transform-minted
identity (``grad(f)``) that can never collide with its base.

Book: ``docs/book/ch01-types-are-values.ipynb``. Architecture: §2.1–§2.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import CodeType
from typing import Hashable

SCALAR_KINDS = frozenset({"f64", "f32", "i64", "i32", "u64", "u32", "bool"})  # the closed scalar lattice


@dataclass(frozen=True, slots=True)
class Type:
    """Base of the lattice. Subclasses are frozen dataclasses: structural eq/hash."""


@dataclass(frozen=True, slots=True)
class Scalar(Type):
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in SCALAR_KINDS:
            raise ValueError(f"unknown scalar kind {self.kind!r}; expected one of {sorted(SCALAR_KINDS)}")

    def __repr__(self) -> str:
        return self.kind


@dataclass(frozen=True, slots=True)
class Vec(Type):
    """An IR-level vector type, produced by dialect lowering rules (e.g. a
    3-tuple literal in a shader's return position becoming ``core.vec``).
    ``typeof`` NEVER produces ``Vec`` — captures summarize as ``Tuple``."""

    elem: Scalar
    n: int

    def __post_init__(self) -> None:
        if not isinstance(self.elem, Scalar):
            raise TypeError(f"Vec element must be a Scalar, got {self.elem!r}")
        if not 2 <= self.n <= 4:
            raise ValueError(f"Vec length must be 2..4, got {self.n}")

    def __repr__(self) -> str:
        return f"vec{self.n}<{self.elem!r}>"


@dataclass(frozen=True, slots=True)
class Tuple(Type):
    """The honest summary of a Python tuple: element-wise, arity in the
    identity. Whether a backend packs one as a GPU vec, a struct, or N scalar
    slots is dialect/backend business — never recorded here."""

    elems: tuple[Type, ...]

    def __repr__(self) -> str:
        return f"({', '.join(repr(e) for e in self.elems)})"


@dataclass(frozen=True, slots=True)
class Array(Type):
    """An array *summary*: rank but (by default) no shape — see architecture §13."""

    dtype: Type
    ndim: int
    layout: str  # "C" | "F" | "A"
    byteorder: str  # "<" | ">" | "="
    writeable: bool

    def __repr__(self) -> str:
        return f"array<{self.dtype!r},{self.ndim}d,{self.layout}{'' if self.writeable else ',ro'}>"


@dataclass(frozen=True, slots=True)
class Record(Type):
    name: str
    fields: tuple[tuple[str, Type], ...]

    def __repr__(self) -> str:
        inner = ", ".join(f"{n}: {t!r}" for n, t in self.fields)
        return f"{self.name}{{{inner}}}"


def _canon(v: Hashable) -> Hashable:
    """Type-aware identity for lifted values. Python's ``==`` is cross-type
    (``1 == 1.0 == True``, ``0.0 == -0.0``) — under plain dataclass equality
    those would share one cache key, i.e. a silent wrong hit. ``float.hex``
    is exact and keeps the sign of zero (and makes nan self-equal, which is
    the right semantics for a key)."""
    if isinstance(v, tuple):
        return ("t", tuple(_canon(x) for x in v))
    if isinstance(v, float):
        return ("f", v.hex())
    return (type(v).__qualname__, v)


@dataclass(frozen=True, slots=True, eq=False)
class LiteralType(Type):
    """The ONE value-in-type opt-in: an explicit ``Literal`` lift (architecture §2.1)."""

    base: Type
    value: Hashable

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LiteralType) and (self.base, _canon(self.value)) == (other.base, _canon(other.value))

    def __hash__(self) -> int:
        return hash((self.base, _canon(self.value)))

    def __repr__(self) -> str:
        return f"Literal[{self.base!r} = {self.value!r}]"


# --- template identity -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemplateId:
    """Code identity as a sum type; see ``Base`` and ``Derived``."""

    @property
    def label(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Base(TemplateId):
    """A ``def`` site. Dataclass eq/hash delegate to the code object's CPython
    *value* semantics (over co_code/co_consts/co_firstlineno/...)."""

    code: CodeType

    @property
    def label(self) -> str:
        return self.code.co_qualname

    def __repr__(self) -> str:
        return f"Base<{self.label}>"


@dataclass(frozen=True, slots=True)
class Derived(TemplateId):
    """A transform-minted identity, e.g. ``Derived("grad", Base<f>, (("wrt", 0),))``."""

    tag: str
    base: TemplateId
    static_params: tuple[tuple[str, Hashable], ...] = ()

    @property
    def label(self) -> str:
        return f"{self.tag}({self.base.label})"

    def __repr__(self) -> str:
        return f"Derived<{self.label}>"


@dataclass(frozen=True, slots=True)
class FnType(Type):
    """The structural type of a DSL closure: ``(template identity, env types)``.

    Two closures share an ``FnType`` iff they come from value-equal code AND
    their captured values have equal types — the caching thesis in one value.
    """

    template: TemplateId
    env_types: tuple[Type, ...]

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.env_types)
        return f"Fn<{self.template.label}>({inner})"


# --- singletons for the common scalars ----------------------------------------

f64, f32, i64, i32, u64, u32, boolean = (Scalar(k) for k in ("f64", "f32", "i64", "i32", "u64", "u32", "bool"))
