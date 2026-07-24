"""Provisioning (200 §1.7, §6.4): materialization joins on contract names.

The resting state is VIRTUAL — the builder is the single source of truth,
and running it collects the spec. Materialization is a separate, pluggable
act with a no-waste law: no allocate-then-overwrite, no gratuitous copies.

- ``init(key, default=..., overrides={glob: strategy})``: each leaf's values
  are the closed-form random field ``normal(fold_in(init_key, leaf_name),
  leaf_layout)`` (§1.8) — materialized directly into the leaf's one
  allocation, scaled in place. Same key → same init, forever, on any
  device. Strategies match by name glob, first match wins; ``zeros``,
  ``ones``, and ``normal(std=)`` cover the worked example.
- ``safetensors(path, translate=None)``: checkpoint entries become
  descriptors over the mmap'd file directly — zero host copies; foreign
  naming schemes are translation TABLES (data, not code); shapes validate
  against the declared extents and a mismatch refuses.

The cache dividend (gate 10) needs no code here: provisioning never
touches build identity, so virtual and provisioned builds share one
fingerprint — analyze first, provision later, hit warm.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass

import numpy as np

from .random import fold_in
from .random import normal as _normal_field
from .scope import Scope
from .tensor import Tensor


@dataclass(frozen=True)
class Strategy:
    kind: str  # "normal" | "zeros" | "ones"
    std: float = 1.0


def normal(std: float = 1.0) -> Strategy:
    return Strategy("normal", std)


zeros = Strategy("zeros")
ones = Strategy("ones")


@dataclass(frozen=True)
class init:  # noqa: N801 — the §6.4 spelling: provision(root, source=init(...))
    key: int
    default: Strategy | None = None
    overrides: tuple = ()  # ((glob, Strategy), ...) — first match wins

    def __post_init__(self):
        if isinstance(self.overrides, dict):
            object.__setattr__(self, "overrides", tuple(self.overrides.items()))

    def materialize(self, name: str, layout) -> Tensor:
        strat = next(
            (s for pat, s in self.overrides if fnmatch.fnmatch(name, pat)),
            self.default,
        )
        if strat is None:
            raise KeyError(f"no init strategy matches leaf {name!r} — pass default= or add an override glob")
        names = tuple(d.name for d in layout.dims)
        shape = tuple(d.size for d in layout.dims)
        if strat.kind == "zeros":
            return Tensor.from_numpy(np.zeros(shape), names)
        if strat.kind == "ones":
            return Tensor.from_numpy(np.ones(shape), names)
        arr = _normal_field(fold_in(self.key, name), layout).to_numpy()
        if len(names) == 0:
            arr = np.asarray(arr)
        arr *= strat.std  # in place: the field materialization IS the one allocation
        return Tensor.from_numpy(arr, names)


_ST_DTYPES = {"F64": "<f8", "F32": "<f4", "I64": "<i8", "I32": "<i4", "U8": "|u1", "BOOL": "|b1"}


@dataclass(frozen=True)
class safetensors:  # noqa: N801 — the §6.4 spelling
    path: str
    translate: tuple = ()  # ((leaf name, file name), ...) — a table, not code

    def __post_init__(self):
        if isinstance(self.translate, dict):
            object.__setattr__(self, "translate", tuple(self.translate.items()))

    def materialize(self, name: str, layout) -> Tensor:
        mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        n = int.from_bytes(bytes(mm[:8]), "little")
        header = json.loads(bytes(mm[8 : 8 + n]).decode())
        start = 8 + n
        fname = dict(self.translate).get(name, name)
        entry = header.get(fname)
        if entry is None:
            have = sorted(k for k in header if k != "__metadata__")
            raise KeyError(f"checkpoint has no entry {fname!r} for leaf {name!r} (entries: {have})")
        shape = tuple(d.size for d in layout.dims)
        if tuple(entry["shape"]) != shape:
            raise ValueError(
                f"leaf {name!r} declares extents {shape} but the checkpoint carries "
                f"{tuple(entry['shape'])} — provisioning joins on contract names AND shapes"
            )
        o0, o1 = entry["data_offsets"]
        arr = np.frombuffer(mm[start + o0 : start + o1], dtype=_ST_DTYPES[entry["dtype"]]).reshape(shape)
        return Tensor.from_numpy(arr.astype(np.float64, copy=False), tuple(d.name for d in layout.dims))


def provision(root: Scope, *, source) -> dict[str, Tensor]:
    """Materialize every declared leaf through ``source``, joined on the
    flat contract names. Returns the name-keyed weights dict — runtime
    state is plain dicts; everything joins on names (§1.7)."""
    return {name: source.materialize(name, p.layout) for name, p in sorted(root.coll.leaves.items())}
