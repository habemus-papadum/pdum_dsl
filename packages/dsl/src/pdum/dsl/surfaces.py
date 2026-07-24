"""The five registration doors as friendly decorators — SATELLITE sugar.

Every function here is a thin wrapper over the Registry's plain dicts
(surface E): ``defop`` fills ``registry.ops`` (surface A), ``intrinsic`` /
``overload`` / ``overload_method`` fill ``registry.overloads`` (surface B),
``record`` fills the kind table + overloads (surface C), ``spell`` fills a
backend's ``code_for_op`` (surface D), and ``Dialect`` bundles any of them
for one-call installation. Nothing here is machinery — deleting this module
would cost convenience, not capability.

Battery discipline (surface B): a DSL-written overload must be CAPTURE-FREE
— it is inlined at every call site with env prefix 0, so a capture would
collide with the caller's env paths. Enforced loudly at registration.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import fields as dc_fields
from dataclasses import is_dataclass

from .capture import make_handle
from .ops import PURE, OpDef
from .registry import Registry
from .types import Record, Type, boolean, f64, i64

_PYTYPES = {float: f64, int: i64, bool: boolean}


def _invalidate(registry: Registry) -> None:
    """Registrations change what a build MEANS: both tiers must miss (the
    two-tier law's first question, answered coarsely — bump tier 1, fresh
    tier 2). Cheap at install time (caches are empty); correct afterwards."""
    from .cache import ArtifactCache

    registry.specializations.bump_generation()
    registry.artifacts = ArtifactCache()


def defop(registry: Registry, name: str, type_rule) -> None:
    """Surface A: a new op, typed by its rule, legal in every later build."""
    registry.ops[name] = OpDef(name, type_rule, PURE)
    _invalidate(registry)


def spell(registry: Registry, backend_name: str, op: str, template: str | None) -> None:
    """Surface D: how ``backend_name`` writes ``op`` down (``{0}``-style
    template), or ``None`` for 'native — handled by the renderer itself'.
    Presence of the KEY is the capability declaration that gates shared
    decompositions (§2.10)."""
    registry.backends[backend_name].code_for_op[op] = template
    _invalidate(registry)


def intrinsic(registry: Registry, name: str, op: str) -> None:
    """Surface B, op-flavored: calling ``name(...)`` emits ``op``."""
    registry.overloads[name] = op
    _invalidate(registry)


def overload(registry: Registry, name: str):
    """Surface B, DSL-flavored: the decorated (capture-free) function IS the
    implementation, inlined at every call site — numba's portable-batteries
    economics, without a second compiler."""

    def deco(fn):
        handle = make_handle(fn, "device")
        if handle.env:
            raise TypeError(f"overload {name!r} must be capture-free; it captured {list(handle.env)}")
        registry.overloads[name] = handle
        _invalidate(registry)
        return fn

    return deco


def record(registry: Registry, cls):
    """Surface C: a frozen dataclass becomes a first-class captured value —
    ``typeof`` → a ``Record``, fields flatten positionally (the Record leaf
    walker and ``child`` aspect already exist, ch08), methods become
    overload_methods inlined with ``self`` as the first argument."""
    if not is_dataclass(cls):
        raise TypeError(f"@record wants a dataclass, got {cls.__name__}")

    def _field_type(f):  # annotations may be strings under `from __future__ import annotations`
        py = f.type
        if not isinstance(py, type):
            module = vars(sys.modules.get(cls.__module__, None) or object())
            py = {"float": float, "int": int, "bool": bool}.get(f.type) or module.get(f.type)
        nested = getattr(py, "__dsl_record__", None)
        if nested is not None:  # value-type expansion (200 §S.2): records NEST
            return nested
        if py not in _PYTYPES:
            raise TypeError(
                f"record field {f.name!r}: float/int/bool or an @record class (got {f.type!r}); "
                f"register the nested record first — tuples are a later step"
            )
        return _PYTYPES[py]

    rec_type = Record(cls.__name__, tuple((f.name, _field_type(f)) for f in dc_fields(cls)))

    class _Kind:
        def typeof(self, v, table) -> Type:
            return rec_type

        def fingerprint(self, v, table):
            return ("rec", rec_type.name)

        def flatten(self, v, table) -> tuple:
            return tuple(leaf for f in dc_fields(cls) for leaf in table.flatten(getattr(v, f.name)))

    registry.table.register(cls, _Kind())
    for name, fn in vars(cls).items():
        if inspect.isfunction(fn) and not name.startswith("_"):  # plain defs only:
            # staticmethod/classmethod/nested classes have no `self` to prepend
            handle = make_handle(fn, "device")
            if handle.env:
                raise TypeError(f"record method {cls.__name__}.{name} must be capture-free")
            registry.overloads[(cls.__name__, name)] = handle
    cls.__dsl_record__ = rec_type  # the introspection door: `SomeRecord.__dsl_record__` -> its Record type
    _invalidate(registry)
    return cls


class Dialect:
    """Pure bundling sugar: a named pile of registrations, installed in one
    call. No behavior of its own — a Dialect IS its install list."""

    def __init__(self, name: str, installers: tuple = ()):
        self.name, self.installers = name, tuple(installers)

    def install(self, registry: Registry) -> Registry:
        for fn in self.installers:
            fn(registry)
        return registry
