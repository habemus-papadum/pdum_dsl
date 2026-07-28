"""The marker vocabulary — re-exported from its home at the language tier.

The numpy-authority ruling (200 §S.2 amendment, P8) moved ``Marker`` and the
pointwise vocabulary DOWN to ``pdum.dsl.markers``: numpy names the
primitives, the marker object is the identity at every tier, and the same
declaration feeds the value-tier ops, the reference spellings, and this
tensor tier's eager numpy execution. This module re-exports the SAME
objects (no consumer rewrites) and keeps what is genuinely tensor-tier:
the REDUCERS — declared monoids for ``reduce``, which the value language
does not have.

Application at the tensor tier is ALWAYS through the compute operators
(owner ruling, P6): calling a marker on a tensor refuses, pointing at
``pointwise(...)``; a HOST-SCALAR call applies the numpy fn directly (the
numpy-authority sharpening — markers are ordinary math on plain numbers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pdum.dsl.markers import (  # noqa: F401 — THE vocabulary, one home
    Marker,
    add,
    cos,
    div,
    eq,
    exp,
    ge,
    gt,
    le,
    log,
    lt,
    maximum,
    minimum,
    mul,
    ne,
    neg,
    pw,
    sin,
    sqrt,
    stop_gradient,
    sub,
    tanh,
    where,
)

from .registry import MARKERS, REDUCERS


@dataclass(frozen=True)
class Reducer:
    """A reduction primitive: a declared monoid (plus numpy ufunc for the
    reference layer). `identity` is None when no dtype-independent identity
    exists (max/min); pass `zero=` to reduce() in that case if the reduced
    extent can be empty."""

    name: str
    fn: object  # numpy ufunc; .reduce(arr, axis=...) is used
    identity: object = None
    associative: bool = True
    commutative: bool = True
    normalize: bool = False  # divide by the (static) reduced numel


class red:
    """The initial reducer set."""

    sum = Reducer("sum", np.add, identity=0)
    prod = Reducer("prod", np.multiply, identity=1)
    max = Reducer("max", np.maximum)
    min = Reducer("min", np.minimum)
    mean = Reducer("mean", np.add, identity=0, normalize=True)


PW = {m.name: m for m in vars(pw).values() if isinstance(m, Marker)}
RED = {m.name: m for m in vars(red).values() if isinstance(m, Reducer)}


def pw_marker(name: str):
    # resolve a pointwise marker name: primitives, then registered composites
    if name in PW:
        return PW[name]
    if name in MARKERS:
        return MARKERS[name]
    raise KeyError(f"unknown pointwise marker {name!r}")


def reducer(name: str):
    if name in RED:
        return RED[name]
    if name in REDUCERS:
        return REDUCERS[name]
    raise KeyError(f"unknown reducer {name!r}")
