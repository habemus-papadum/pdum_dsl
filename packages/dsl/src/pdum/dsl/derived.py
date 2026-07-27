"""First-class derived values (design 130 §7): the ONE wrapper protocol.

A ``DerivedValue`` is anything FnType-shaped that is not a source-backed
``Handle``: a fused pipeline, a transformed kernel (`over`, `jvp`, `grad`),
a future scheduled composite. Dispatch needs exactly four things from it —
``fntype`` (the cache-key half), ``fp`` (the precomputed fingerprint),
``captures`` (the runtime values, aligned with ``env_types``), ``kind``
(the routing role) — and marshaling descends it through the same ``FnType``
child aspect as a Handle. Before this class, Pipeline and each transform
wrapper re-implemented the protocol by hand and each needed its own
ValueKind registration; the step-12 review found the one that was missing.
One base class, one MRO-covered kind, no repeats.
"""

from __future__ import annotations

from .types import Derived, FnType
from .valuekind import BUILTINS, KindTable


class DerivedValue:
    """Subclasses set fntype/fp/captures (and may override ``kind``)."""

    __slots__ = ("fntype", "fp", "captures")

    def __init__(self, tag: str, base_template, env_types: tuple, static: tuple, fp, captures: tuple):
        self.fntype = FnType(Derived(tag, base_template, static), env_types)
        self.fp = fp
        self.captures = captures

    @property
    def kind(self) -> str:  # default: the first capture's role travels up
        return self.captures[0].kind

    @property
    def env_types(self) -> tuple:
        return self.fntype.env_types

    def __call__(self, *args, out=None):
        from .registry import DEFAULT  # lazy, like Handle.__call__: phase A stays runtime-free

        return DEFAULT.dispatch(self, args, out)


class _DerivedKind:
    """Capturable like a Handle: type = its FnType, fingerprint precomputed,
    leaves = its captures' leaves (the EnvLeaf recursion, unchanged)."""

    def typeof(self, v: DerivedValue, table: KindTable):
        return v.fntype

    def fingerprint(self, v: DerivedValue, table: KindTable):
        return v.fp

    def flatten(self, v: DerivedValue, table: KindTable) -> tuple:
        return tuple(leaf for val in v.captures for leaf in table.flatten(val))


BUILTINS.register(DerivedValue, _DerivedKind())
