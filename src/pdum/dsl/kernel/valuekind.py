"""Value -> Type summaries: the ``typeof`` and ``fingerprint`` views.

A ``ValueKind`` is one registration per Python type yielding (eventually) four
views of a value. This module ships the two *identity* views:

- ``typeof(v)``      — the full structural summary (the sole artifact-key
  vocabulary). Arbitrarily rich per kind — int range-bucketing here, shapes in
  an opt-in array kind later (architecture §13) — but always a frozen,
  hashable ``Type``, never a predicate.
- ``fingerprint(v)`` — a cheap structural tag for the hot path, governed by
  the **soundness law**: ``fingerprint(a) == fingerprint(b)`` must imply
  ``typeof(a) == typeof(b)`` (equal-or-both-raise). A fingerprint collision is
  a silent wrong cache hit — the worst failure class — so the law is enforced
  by a property fuzz in CI (``tests/test_valuekind.py``).

Kinds receive the **dispatching table** on every call, so composite kinds
(tuples, later records/arrays) recurse through whatever table — base or
extended — the call entered by. A kind must never capture a table at
construction: that would freeze layered overrides out of nested elements.

Step 7 adds the two *marshaling* views. ``flatten(v)`` (dynamic, hot path)
lives on the kind, dispatched by value class like the identity views. The
static view — leaf entries — dispatches on the **Type**, not the value
(``FnType`` is produced by both Handles and Pipelines), so it registers per
Type class via ``register_leaves``; the walkers themselves live in
``pack.py`` with the leaf vocabulary. The alignment law (fuzz-enforced, like
soundness): ``table.flatten(v)`` yields exactly one value per entry of
``table.leaf_entries(table.typeof(v))``, in order — a drifted kind would
otherwise corrupt packed bytes silently. ``BUILTINS``
and the module-level ``typeof``/``fingerprint`` conveniences are the *staged
seed* of surface C: in step 8 they fold into the explicit ``Registry``, and
code should migrate to ``registry.typeof``. Unregistered types fail loudly —
silently treating an unknown object as its Python class would put an unsound
key in the cache.

Book: ``docs/book/ch01-types-are-values.ipynb``. Architecture: §2.9, §13.
"""

from __future__ import annotations

from typing import Hashable, Protocol

from .types import Scalar, Tuple, Type, boolean, f64, i64, u64


class BigIntError(TypeError):
    """A captured int fits no supported fixed-width bucket (hazard: a 5 and a
    2**70 must never share a type, or the packed value silently corrupts)."""


class ValueKind(Protocol):
    def typeof(self, v: object, table: KindTable) -> Type: ...

    def fingerprint(self, v: object, table: KindTable) -> Hashable: ...

    def flatten(self, v: object, table: KindTable) -> tuple: ...  # aligned with leaf_entries(typeof(v))


class KindTable:
    """Python type -> ValueKind dispatch (MRO order, loud on a miss)."""

    def __init__(self) -> None:
        self._kinds: dict[type, ValueKind] = {}
        self._aspects: dict[str, dict[type, object]] = {}  # aspect -> Type class -> rule

    def register(self, pytype: type, kind: ValueKind) -> None:
        """Register ``kind`` as the summarizer for ``pytype`` (and, via MRO,
        its unregistered subclasses).

        Parameters
        ----------
        pytype : type
            The Python class whose instances this kind summarizes.
        kind : ValueKind
            The summarizer; called as ``kind.typeof(v, table)``.

        Raises
        ------
        TypeError
            If the kind is missing a protocol view — loud HERE, at the
            registration, not later inside a packer's per-value loop.
        """
        missing = [v for v in ("typeof", "fingerprint", "flatten") if not callable(getattr(kind, v, None))]
        if missing:
            raise TypeError(f"{type(kind).__name__} is not a ValueKind: missing {', '.join(missing)}")
        self._kinds[pytype] = kind

    def extend(self) -> KindTable:
        """A child table seeded with this table's registrations — the layered
        extension shape (stdlib -> user -> session) surface C will formalize.

        Returns
        -------
        KindTable
            An independent table; registrations on it never touch the parent,
            and composite kinds recurse through the *child* (kinds get the
            dispatching table per call).
        """
        child = KindTable()
        child._kinds.update(self._kinds)
        child._aspects = {name: dict(rules) for name, rules in self._aspects.items()}
        return child

    def register_aspect(self, aspect: str, type_cls: type, rule) -> None:
        """Register a **Type-keyed** rule. Value-keyed behaviour is a
        ``ValueKind``; behaviour derivable from the *type* alone is an aspect
        (``leaves`` — the static leaf walk; ``child`` — descend one step;
        ``rebuild`` — reassemble from leaves). One registry, one MRO lookup,
        one layering story: ``extend()`` copies aspects too, so a child table
        can override any of them (pack.py owns the marshaling aspects)."""
        self._aspects.setdefault(aspect, {})[type_cls] = rule

    def aspect(self, name: str, t: Type):
        """The registered ``name`` rule for a Type, by MRO like ``kind_for``."""
        rules = self._aspects.get(name, {})
        for cls in type(t).__mro__:
            rule = rules.get(cls)
            if rule is not None:
                return rule
        raise TypeError(f"no {name} rule registered for {type(t).__name__!r} ({t!r})")

    def leaf_entries(self, t: Type) -> tuple:
        """``((sub_path, Leaf), ...)`` for a Type — the plan-building walk."""
        return self.aspect("leaves", t)(t, self)

    def kind_for(self, v: object) -> ValueKind:
        """The registered kind for ``v``, searching ``type(v).__mro__`` in
        order (exact type first, by MRO definition).

        Raises
        ------
        TypeError
            If no class in the MRO is registered — deliberately loud.
        """
        for cls in type(v).__mro__:
            kind = self._kinds.get(cls)
            if kind is not None:
                return kind
        raise TypeError(
            f"no ValueKind registered for {type(v).__name__!r}; "
            f"register one (or give the class a __dsl_type__) rather than guessing"
        )

    def typeof(self, v: object) -> Type:
        return self.kind_for(v).typeof(v, self)

    def fingerprint(self, v: object) -> Hashable:
        return self.kind_for(v).fingerprint(v, self)

    def flatten(self, v: object) -> tuple:
        return self.kind_for(v).flatten(v, self)


# --- builtin kinds -------------------------------------------------------------


class _ConstKind:
    """A kind whose summary is one fixed scalar regardless of the value."""

    def __init__(self, ty: Scalar) -> None:
        self._ty = ty

    def typeof(self, v: object, table: KindTable) -> Type:
        return self._ty

    def fingerprint(self, v: object, table: KindTable) -> Hashable:
        return self._ty.kind

    def flatten(self, v: object, table: KindTable) -> tuple:
        return (v,)


def _int_scalar(v: int) -> Scalar:
    if -(2**63) <= v < 2**63:
        return i64
    if 2**63 <= v < 2**64:
        return u64
    raise BigIntError(f"int {v!r} does not fit in 64 bits")


class _IntKind:
    def typeof(self, v: int, table: KindTable) -> Type:
        return _int_scalar(v)

    def fingerprint(self, v: int, table: KindTable) -> Hashable:
        return _int_scalar(v).kind

    def flatten(self, v: int, table: KindTable) -> tuple:
        return (v,)


class _TupleKind:
    """Python tuples summarize element-wise to ``Tuple`` — the honest identity
    (the ``center = (cx, cy)`` capture, M0's top gap). Vec-ness is a dialect
    interpretation applied later, never here; errors only propagate from
    elements."""

    def typeof(self, v: tuple, table: KindTable) -> Type:
        return Tuple(tuple(table.typeof(x) for x in v))

    def fingerprint(self, v: tuple, table: KindTable) -> Hashable:
        return ("t", tuple(table.fingerprint(x) for x in v))

    def flatten(self, v: tuple, table: KindTable) -> tuple:
        return tuple(leaf for x in v for leaf in table.flatten(x))


BUILTINS = KindTable()
BUILTINS.register(bool, _ConstKind(boolean))
BUILTINS.register(int, _IntKind())
BUILTINS.register(float, _ConstKind(f64))
BUILTINS.register(tuple, _TupleKind())


def typeof(v: object) -> Type:
    """Summarize ``v`` against the builtin kind table.

    Parameters
    ----------
    v : object
        Any value of a registered Python type.

    Returns
    -------
    Type
        The frozen structural summary (cache-key vocabulary).

    Raises
    ------
    TypeError
        For unregistered types (loud by design).
    BigIntError
        For ints outside every 64-bit bucket.
    """
    return BUILTINS.typeof(v)


def fingerprint(v: object) -> Hashable:
    """Cheap structural tag for ``v``, bound to ``typeof`` by the soundness law.

    Parameters
    ----------
    v : object
        Any value of a registered Python type.

    Returns
    -------
    Hashable
        A tag safe to use as the hot-path stand-in for the full ``Type``.

    Raises
    ------
    TypeError
        For unregistered types.
    BigIntError
        For ints outside every 64-bit bucket (the tag must bucket identically).
    """
    return BUILTINS.fingerprint(v)
