"""The analysis cache (330 §4) — analysis as a cached, composable citizen.

The content-door discipline, extended from artifacts to FACTS: an
analysis result is keyed by the content keys of its inputs plus the
analysis's own fingerprint (name@version) plus its parameters. First run
computes — slow is fine; every later run in the hot loop is a lookup and
skips. The cache is the one Memo discipline (cache.py), so the seam is
instrumented by construction: ``analysis.miss`` and an
``analysis.compile`` span ride the events seam, and ``no_reanalysis()``
extends the warmth law — a warm loop asserts ZERO misses exactly as it
asserts zero recompiles today.

Composition is the point: a ``Fact`` passed as an input contributes its
KEY to the new key (the analysis DAG — a refinement starts from the
coarse pass it refines, and the prefix never recomputes) while its VALUE
unwraps for the function. There is no invalidation machinery ANYWHERE in
this file, by construction: regions are immutable and content-addressed,
so a changed program is a different key and staleness cannot be spelled.

A MEASUREMENT is an analysis whose evaluator is the machine — declared
``ledger=True``: its value must be plain JSON data (enforced at build,
loudly — measurements are records, not objects) and it round-trips
through ``save_ledger``/``load_ledger``, so nothing keyed identically is
ever measured twice, across sessions included. Loading is a cold act.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from pdum.dsl import events
from pdum.dsl.cache import Memo

ANALYSES = Memo("analysis", capacity=1 << 30)
LEDGER: dict[str, "Fact"] = {}  # the measurement subset, save/load's subject


@dataclass(frozen=True)
class Fact:
    """One cached analysis result: the key is the identity, the analysis
    field is provenance (name@version), the value is the payload."""

    key: str
    analysis: str
    value: object


def _canon(x):
    """A stable, content-addressed shape for one analysis input. Facts
    contribute their KEY (the DAG edge); IR objects contribute their
    content key; plain data canonicalizes structurally; anything else
    refuses — an unkeyable input would silently void the cache's honesty."""
    if isinstance(x, Fact):
        return ("fact", x.key)
    k = getattr(x, "key", None)
    if isinstance(k, bytes):
        return ("ir", k.hex())
    if isinstance(x, bool) or x is None:
        return ("lit", x)
    if isinstance(x, (str, int, float)):
        return ("lit", x)
    if isinstance(x, bytes):
        return ("bytes", x.hex())
    if isinstance(x, (tuple, list)):
        return ("seq", tuple(_canon(e) for e in x))
    if isinstance(x, (set, frozenset)):
        return ("set", tuple(sorted(repr(_canon(e)) for e in x)))
    if isinstance(x, dict):
        return ("map", tuple(sorted((str(kk), _canon(v)) for kk, v in x.items())))
    if isinstance(x, np.ndarray):
        arr = np.ascontiguousarray(x)
        return ("arr", str(arr.dtype), arr.shape, hashlib.sha256(arr.tobytes()).hexdigest())
    raise TypeError(
        f"analysis inputs must be content-keyable (Fact, Region/Node, plain data, ndarray) — "
        f"got {type(x).__name__}: an unkeyable input has no honest cache identity (330 §4)"
    )


@dataclass(frozen=True)
class Analysis:
    """A declared analysis: name@version IS the fingerprint — bump the
    version when the computation's meaning changes and every downstream
    key changes with it (the DAG re-derives exactly what depended on it)."""

    name: str
    version: int
    fn: Callable
    ledger: bool = False

    def __call__(self, *inputs, **params) -> Fact:
        payload = ("analysis", self.name, self.version, _canon(inputs), _canon(params))
        key = hashlib.sha256(repr(payload).encode()).hexdigest()

        def build():
            args = tuple(x.value if isinstance(x, Fact) else x for x in inputs)
            value = self.fn(*args, **params)
            if self.ledger:
                # measurements are RECORDS, not objects: normalize through
                # JSON now so in-session and reloaded values are identical
                try:
                    value = json.loads(json.dumps(value))
                except TypeError as exc:
                    raise TypeError(
                        f"ledger analysis {self.name!r} produced a non-JSON value "
                        f"({type(value).__name__}) — measurements are plain data by law (330 §4)"
                    ) from exc
            fact = Fact(key, f"{self.name}@v{self.version}", value)
            if self.ledger:
                LEDGER[key] = fact
            return fact

        return ANALYSES.get_or_compile(key, build)


def defanalysis(name: str, version: int = 1, *, ledger: bool = False):
    """Declare an analysis: ``@defanalysis("l1.peak", 2)`` over a function
    of unwrapped inputs. ``ledger=True`` marks a measurement."""

    def deco(fn: Callable) -> Analysis:
        return Analysis(name, version, fn, ledger)

    return deco


def no_reanalysis():
    """The warmth law over facts: everything inside must be a cache hit
    (``no_compile``'s sibling — a warm loop never re-analyzes)."""
    return events.forbid("analysis.miss")


def save_ledger(path) -> int:
    """Write the measurement facts to ``path`` (JSON, sorted). Returns the
    count. In-memory (non-ledger) facts never persist — plans and regions
    are rebuilt from their keys, measurements are irreplaceable."""
    data = {k: {"analysis": f.analysis, "value": f.value} for k, f in sorted(LEDGER.items())}
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=True))
    return len(data)


def load_ledger(path) -> int:
    """Seed the cache from a saved ledger: a reloaded measurement HITS and
    is never re-run. Loading is a cold act (it emits the misses it seeds
    through) — do it outside warm loops."""
    data = json.loads(Path(path).read_text())
    for key, rec in data.items():
        fact = Fact(key, rec["analysis"], rec["value"])
        LEDGER[key] = fact
        ANALYSES.get_or_compile(key, lambda fact=fact: fact)
    return len(data)
