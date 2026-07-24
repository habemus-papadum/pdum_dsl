"""The recorder satellite (design 120 §6): what the event seam's data means.

The kernel emits ``(name, key, dur_ns, depth, detail)`` tuples into a dark
sink list; this module turns them into a debugging instrument:

- **Counts are exact; detail is sampled.** Every event increments its bucket;
  the sampling policy gates only the expensive extras (the stack walk, the
  lazy ``detail()`` call). Default ``first=8, then=100``: immediate exemplars
  for the common few-events case, survival under a firehose.
- **Traces are structured and interned.** Frames are ``(code, lineno)`` pairs
  — no eager source reads (that is ``traceback.extract_stack``'s hidden
  cost); ten thousand drift events from one loop collapse to ONE ``Trace``
  and a count. Fingerprinting, applied to stacks.
- **The span tree is per-thread.** Depth alone interleaves wrongly under the
  cross-thread compiles the global sink exists to see (120 §8, amended);
  buckets remember the smallest depth seen per (thread, name) and the report
  indents by it.
- ``record()`` refuses to nest (one global sink list — merging two
  recordings silently would lie); ``expect()`` turns exact per-event budgets
  into assertions; ``forbid`` is re-exported from the seam.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import CodeType

from . import events as _seam
from .events import EventForbidden, forbid  # noqa: F401 — re-exported API

_PKG = __file__.rsplit("/", 1)[0]  # .../pdum/dsl — frames under here are internals


@dataclass(frozen=True, slots=True)
class Frame:
    code: CodeType
    lineno: int

    @property
    def filename(self) -> str:
        return self.code.co_filename

    @property
    def qualname(self) -> str:
        return self.code.co_qualname

    @property
    def line(self) -> str:  # LAZY: linecache touches disk only if someone renders
        import linecache

        return linecache.getline(self.filename, self.lineno).strip()

    def __repr__(self) -> str:
        return f"{self.qualname} ({self.filename}:{self.lineno})"


@dataclass(frozen=True, slots=True)
class Trace:
    frames: tuple

    @property
    def user_frames(self) -> tuple:
        """Frames outside pdum/dsl — *your* loop, not our internals."""
        return tuple(f for f in self.frames if not f.filename.startswith(_PKG))


@dataclass(frozen=True, slots=True)
class Exemplar:
    key: object
    dur_ns: int
    trace: Trace | None
    explain: str = ""


@dataclass
class Bucket:
    name: str
    count: int = 0
    total_ns: int = 0
    min_ns: int = 0
    max_ns: int = 0
    depth: int = 0
    traces: list = field(default_factory=list)  # interned Trace objects, first-seen order
    trace_counts: dict = field(default_factory=dict)  # Trace -> count
    exemplars: list = field(default_factory=list)
    keys: dict = field(default_factory=dict)  # key -> count

    def by_key(self) -> list:
        return sorted(self.keys.items(), key=lambda kv: -kv[1])

    def add(self, dur_ns: int) -> None:
        self.count += 1
        self.total_ns += dur_ns
        self.min_ns = dur_ns if self.count == 1 else min(self.min_ns, dur_ns)
        self.max_ns = max(self.max_ns, dur_ns)


@dataclass(frozen=True)
class Sampling:
    """Per-name detail policy. ``first`` exemplars always; then 1-in-``then``.
    ``every=N`` is the deterministic override tests want."""

    first: int = 8
    then: int = 100
    every: int | None = None
    exemplars: int = 8

    def admits(self, seen: int) -> bool:
        if self.every is not None:
            return seen % self.every == 0
        return seen < self.first or (self.then > 0 and seen % self.then == 0)


def _capture_trace(depth_cap: int = 32) -> Trace:
    frames, f = [], sys._getframe(2)
    while f is not None and len(frames) < depth_cap:
        frames.append(Frame(f.f_code, f.f_lineno))
        f = f.f_back
    return Trace(tuple(frames))


class Recording:
    """The armed sink: exact counts, sampled traces, a report table."""

    def __init__(self, policy: dict | None = None, default: Sampling = Sampling()):
        self._policy = policy or {}
        self._default = default
        self._lock = threading.Lock()
        self._buckets: dict[str, Bucket] = {}
        self._interned: dict[tuple, Trace] = {}
        self._depths: dict[tuple, int] = {}  # (thread, name) -> min depth (amendment: per-thread)

    # -- the sink -----------------------------------------------------------
    def _sink(self, name, key, dur_ns, depth, detail) -> None:
        tid = threading.get_ident()
        with self._lock:
            b = self._buckets.get(name)
            if b is None:
                b = self._buckets[name] = Bucket(name, depth=depth)
            seen = b.count
            b.add(dur_ns)
            tkey = (tid, name)
            self._depths[tkey] = min(self._depths.get(tkey, depth), depth)
            b.depth = min(b.depth, depth)
            if key is not None:
                b.keys[key] = b.keys.get(key, 0) + 1
            pol = self._policy.get(name, self._default)
            sampled = pol.admits(seen)
        if not sampled:
            return
        trace = _capture_trace()
        sig = tuple((f.code, f.lineno) for f in trace.frames)
        with self._lock:
            trace = self._interned.setdefault(sig, trace)
            if trace not in b.trace_counts:
                b.traces.append(trace)
            b.trace_counts[trace] = b.trace_counts.get(trace, 0) + 1
            if len(b.exemplars) < pol.exemplars:
                explain = detail() if detail is not None else ""
                b.exemplars.append(Exemplar(key, dur_ns, trace, explain))

    # -- the API ------------------------------------------------------------
    def __getitem__(self, name: str) -> Bucket:
        return self._buckets.get(name) or Bucket(name)

    def __contains__(self, name: str) -> bool:
        return name in self._buckets

    def names(self) -> list:
        return list(self._buckets)

    def top(self, n: int = 5) -> list:
        return sorted(self._buckets.values(), key=lambda b: -b.total_ns)[:n]

    def __str__(self) -> str:
        rows = [f"{'event':<28} {'count':>7} {'total':>9} {'mean':>9} {'traces':>7}"]
        for b in sorted(self._buckets.values(), key=lambda b: (b.depth, -b.total_ns)):
            mean = b.total_ns / b.count if b.count else 0
            rows.append(
                f"{'  ' * b.depth + b.name:<28} {b.count:>7} {_fmt(b.total_ns):>9} "
                f"{_fmt(mean):>9} {len(b.traces) or '—':>7}"
            )
        return "\n".join(rows)

    def timeline(self, title: str = "events"):
        from .bench import Timeline

        tl, cursor = Timeline(title=title), 0.0
        for b in sorted(self._buckets.values(), key=lambda b: -b.total_ns):
            if b.total_ns:
                tl.add(b.name, cursor, b.total_ns / 1e9, "host")
                cursor += b.total_ns / 1e9
        return tl


def _fmt(ns: float) -> str:
    for scale, unit in ((1e9, "s"), (1e6, "ms"), (1e3, "µs")):
        if ns >= scale:
            return f"{ns / scale:.1f}{unit}"
    return f"{ns:.0f}ns"


_ARMED = threading.Lock()


@contextmanager
def record(policy: dict | None = None, default: Sampling = Sampling()):
    """Arm a recording around a block. NON-reentrant by design (120 §8): two
    concurrent recordings would interleave into one global sink and lie."""
    if not _ARMED.acquire(blocking=False):
        raise RuntimeError("events.record() is already armed (recordings do not nest — 120 §8)")
    rec = Recording(policy, default)
    _seam.SINKS.append(rec._sink)
    try:
        yield rec
    finally:
        _seam.SINKS.remove(rec._sink)
        _ARMED.release()


@contextmanager
def expect(**counts: int):
    """Exact per-event budgets, asserted on exit: ``expect(**{"spec.compile": 1,
    "guard.drift": 0})``."""
    with record() as rec:
        yield rec
    bad = {n: (rec[n].count, want) for n, want in counts.items() if rec[n].count != want}
    if bad:
        lines = ", ".join(f"{n}: got {got}, expected {want}" for n, (got, want) in bad.items())
        raise AssertionError(f"event budget violated — {lines}")
