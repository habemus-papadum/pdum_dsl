"""The two-tier cache: the thesis made executable.

Tier 1 — the **specialization cache** — maps
``(fp_head, arg_fp, backend_fp, generation) -> FastRecord``, where ``fp_head``
is a Handle's precomputed ``("H", code, env_fp)`` digest. Types and identity
only; no component of any key ever derives from a captured *value*.

Tier 2 — the **artifact cache** — maps
``(content_key, backend_token, flags) -> artifact``. Content-addressed and
**generation-free**: two templates that lower to identical IR share one
artifact, and a generation bump (which clears tier 1) cannot orphan it.

Bookkeeping this module owns, per the architecture (§2.12, §4.4) and the
hazard doc: per-key futures so concurrent misses compile once (same-thread
re-entry is a loud error until recursion is really needed); **guards** —
precomputed identity checks against dependency drift, refuse-or-recompile,
never stale; **LRU eviction** and **superseded-template retirement** (an
edited-and-rerun template retires its predecessor's entries — the L-cache
leak fix); per-tier counters plus ``explain_miss`` so a miss can *name the
differing key component*; and the ``no_compile()`` context, which turns "this
loop must not recompile" into an assertion.

The hot path (step 8) probes ``_ready`` and checks guards inline; this class
is the miss-path engine and the bookkeeping.

Book: ``docs/book/ch03-one-compile-per-signature.ipynb``.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

_MISSING = object()
_KEY_PARTS = ("template", "env_types", "arg_types", "backend", "generation")

_NO_COMPILE = contextvars.ContextVar("pdum_no_compile", default=False)


class CompileForbidden(RuntimeError):
    """A compile was reached inside ``no_compile()`` — the loop is not hot."""


class ReentrantCompile(RuntimeError):
    """A compile re-entered its own key (recursion slots arrive when needed)."""


@dataclass
class FastRecord:
    """The tier-1 value. Step 3 fills artifact+guards; extract/plan (step 7)
    and staging/launch (step 8) complete the precompiled hit path."""

    artifact: Any
    guards: tuple = ()  # ((holder, name, expected_object), ...) — identity compares
    extract: Callable | None = None
    plan: Any = None
    staging: Any = None
    launch: Callable | None = None


def guards_ok(guards: tuple) -> bool:
    for holder, name, expected in guards:
        current = holder.get(name, _MISSING) if isinstance(holder, dict) else getattr(holder, name, _MISSING)
        if current is not expected:  # identity: rebinding to an equal value still drifts
            return False
    return True


@contextmanager
def no_compile():
    """Assert that everything inside is a cache hit (the thesis, testable)."""
    token = _NO_COMPILE.set(True)
    try:
        yield
    finally:
        _NO_COMPILE.reset(token)


class _Slot:
    __slots__ = ("event", "value", "error", "owner")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None
        self.owner = threading.get_ident()


class _TierCache:
    """One tier: insertion-ordered dict as LRU + per-key futures + counters."""

    def __init__(self, name: str = "artifact", capacity: int = 1024) -> None:
        self.name = name
        self.capacity = capacity
        self._lock = threading.Lock()
        self._ready: dict[tuple, Any] = {}
        self._building: dict[tuple, _Slot] = {}
        self.hits = self.misses = self.compiles = self.evictions = 0

    def __len__(self) -> int:
        return len(self._ready)

    def _explain(self, key: tuple) -> str:
        return ""

    def get_or_compile(self, key: tuple, compile_fn: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._ready:
                self.hits += 1
                value = self._ready.pop(key)
                self._ready[key] = value  # move-to-end: LRU touch
                return value
            slot = self._building.get(key)
            if slot is None:
                slot = self._building[key] = _Slot()
                mine = True
            else:
                mine = False
        if not mine:
            if slot.owner == threading.get_ident():
                raise ReentrantCompile(f"{self.name}: compile re-entered its own key {key!r}")
            slot.event.wait()
            if slot.error is not None:
                raise slot.error
            return slot.value
        self.misses += 1
        try:
            if _NO_COMPILE.get():
                raise CompileForbidden(f"{self.name}-tier miss under no_compile(): {self._explain(key)}")
            slot.value = compile_fn()
        except BaseException as exc:
            slot.error = exc
            with self._lock:
                self._building.pop(key, None)
            slot.event.set()
            raise
        self.compiles += 1
        with self._lock:
            self._building.pop(key, None)
            self._ready[key] = slot.value
            while len(self._ready) > self.capacity:
                self._ready.pop(next(iter(self._ready)))
                self.evictions += 1
        slot.event.set()
        return slot.value


# Tier 2, ``(content_key, backend_token, flags) -> artifact``: the base tier as-is.
ArtifactCache = _TierCache


class SpecializationCache(_TierCache):
    """Tier 1. Owns the generation counter, guards, and template retirement."""

    def __init__(self, capacity: int = 1024) -> None:
        super().__init__("specialization", capacity)
        self.generation = 0
        self.guard_misses = 0
        self.retirements = 0
        self._current_code: dict[tuple[str, str], Any] = {}

    def key_for(self, handle, arg_fp: tuple = (), backend_fp: tuple = ()) -> tuple:
        return (handle.fp, tuple(arg_fp), backend_fp, self.generation)

    def bump_generation(self) -> int:
        """The coarse invalidation knob: clears tier 1 (tier 2 is content-
        addressed and untouched — identical IR recompiles for free)."""
        with self._lock:
            self.generation += 1
            self.retirements += len(self._ready)
            self._ready.clear()
            return self.generation

    def _retire_superseded(self, key: tuple) -> None:
        head = key[0]
        if not (isinstance(head, tuple) and len(head) == 3 and head[0] == "H"):
            return
        code = head[1]
        loc = (code.co_filename, code.co_qualname)
        with self._lock:
            current = self._current_code.get(loc)
            self._current_code[loc] = code
            if current is None or current == code:
                return
            dead = [k for k in self._ready if isinstance(k[0], tuple) and k[0][0] == "H" and k[0][1] == current]
            for k in dead:
                self._ready.pop(k)
                self.retirements += 1

    def get_or_compile(self, key: tuple, compile_fn: Callable[[], FastRecord]) -> FastRecord:
        with self._lock:
            record = self._ready.get(key)
        if record is not None and record.guards and not guards_ok(record.guards):
            self.guard_misses += 1  # drift: refuse the stale entry, recompile
            with self._lock:
                self._ready.pop(key, None)
        self._retire_superseded(key)
        return super().get_or_compile(key, compile_fn)

    def _explain(self, key: tuple) -> str:
        with self._lock:
            candidates = list(self._ready)
        best: tuple[int, list[str]] | None = None
        for k in candidates:
            parts = [*_split_head(k[0], key[0]), k[1] != key[1], k[2] != key[2], k[3] != key[3]]
            names = [n for n, differs in zip(_KEY_PARTS, parts) if differs]
            if names and (best is None or len(names) < len(best[1])):
                best = (0, names)
        if best is None:
            return "first sight (no comparable entry cached)"
        return f"nearest entry differs in: {', '.join(best[1])}"


def _split_head(a: object, b: object) -> tuple[bool, bool]:
    """Compare two fp heads as (template_differs, env_differs)."""
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == 3 and len(b) == 3:
        return (a[1] != b[1], a[2] != b[2])
    return (a != b, False)
