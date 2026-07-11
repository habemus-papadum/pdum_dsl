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

The marshaling views (``leaf_types``/``flatten``) land in step 7. ``BUILTINS``
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

    # step 7 adds: leaf_types(t: Type) -> tuple[Leaf, ...]; flatten(v) -> tuple


class KindTable:
    """Python type -> ValueKind dispatch (MRO order, loud on a miss)."""

    def __init__(self) -> None:
        self._kinds: dict[type, ValueKind] = {}

    def register(self, pytype: type, kind: ValueKind) -> None:
        """Register ``kind`` as the summarizer for ``pytype`` (and, via MRO,
        its unregistered subclasses).

        Parameters
        ----------
        pytype : type
            The Python class whose instances this kind summarizes.
        kind : ValueKind
            The summarizer; called as ``kind.typeof(v, table)``.
        """
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
        return child

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


# --- builtin kinds -------------------------------------------------------------


class _ConstKind:
    """A kind whose summary is one fixed scalar regardless of the value."""

    def __init__(self, ty: Scalar) -> None:
        self._ty = ty

    def typeof(self, v: object, table: KindTable) -> Type:
        return self._ty

    def fingerprint(self, v: object, table: KindTable) -> Hashable:
        return self._ty.kind


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


class _TupleKind:
    """Python tuples summarize element-wise to ``Tuple`` — the honest identity
    (the ``center = (cx, cy)`` capture, M0's top gap). Vec-ness is a dialect
    interpretation applied later, never here; errors only propagate from
    elements."""

    def typeof(self, v: tuple, table: KindTable) -> Type:
        return Tuple(tuple(table.typeof(x) for x in v))

    def fingerprint(self, v: tuple, table: KindTable) -> Hashable:
        return ("t", tuple(table.fingerprint(x) for x in v))


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
