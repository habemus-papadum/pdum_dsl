"""The event seam (kernel side) — design 120 §3.

Expensive, should-be-rare occurrences announce themselves here. The kernel
knows nothing about sampling, stacks, or aggregation — a satellite installs a
sink. Dark by default: an empty ``SINKS`` costs one list truthiness test.

``SINKS`` is a module-global ON PURPOSE (not a ``ContextVar``): a compile can
legally run on another thread (the ``_Slot`` machinery exists for exactly
that), and a context-scoped sink would attribute it to nobody. ``forbid`` IS
a ``ContextVar``, preserving ``no_compile``'s scoped/nestable/async semantics
exactly. ``detail`` is a lazy callable evaluated only in the forbidden raise —
it is how a miss keeps its "nearest entry differs in: …" diagnostic without
the seam knowing what a cache is.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from time import perf_counter_ns

SINKS: list = []  # satellites append a callable (name, key, dur_ns, depth, detail) -> None
# `detail` is None or a LAZY zero-arg callable; a sink may invoke it (ideally sampled).
_FORBID = contextvars.ContextVar("pdum_forbid", default=())
_DEPTH = contextvars.ContextVar("pdum_depth", default=0)


class EventForbidden(RuntimeError):
    """An event fired inside ``forbid()`` — the loop is not as hot as claimed."""


def _check(name: str, key: object, detail=None) -> None:
    for pat in _FORBID.get():
        if name == pat or (pat.endswith("*") and name.startswith(pat[:-1])):
            text = detail() if detail is not None else ""
            suffix = f": {text}" if text else ""
            raise EventForbidden(f"{name} fired under forbid({pat!r}); key={key!r}{suffix}")


def emit(name: str, key: object = None, dur_ns: int = 0, detail=None) -> None:
    """A point event: it happened, once, here."""
    if _FORBID.get():
        _check(name, key, detail)
    for sink in SINKS:
        sink(name, key, dur_ns, _DEPTH.get(), detail)


@contextmanager
def span(name: str, key: object = None):
    """A timed event. Nests: ``depth`` gives the phase tree for free."""
    if _FORBID.get():
        _check(name, key)
    if not SINKS:
        yield
        return
    depth = _DEPTH.get()
    token = _DEPTH.set(depth + 1)
    t0 = perf_counter_ns()
    try:
        yield
    finally:
        _DEPTH.reset(token)
        dur = perf_counter_ns() - t0
        for sink in SINKS:
            sink(name, key, dur, depth, None)


@contextmanager
def forbid(*patterns: str):
    """Assert that none of ``patterns`` (exact, or ``prefix*``) fires inside."""
    token = _FORBID.set(_FORBID.get() + patterns)
    try:
        yield
    finally:
        _FORBID.reset(token)
