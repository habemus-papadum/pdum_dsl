"""The type-keyed specialization cache + the generation counter.

This is the heart of the "Julia model, not numba model" policy: one compiled
artifact per ``(FnType, arg_types, generation)`` triple, reused across all capture
*values*. The cache is **generic over the artifact type** — it never inspects what
it stores, it just calls a ``compile_fn`` on a miss. For WebGPU the artifact is a
compiled WGSL pipeline bundle; for a future LLVM/PTX backend it would be native
code. That genericity is what keeps the cache use-case-independent.

See ``docs/dsl_caching_layer.md`` → "Dispatch flow", "Correctness and invalidation"
(thread-safety via per-key futures; generation as a live-coding sledgehammer).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .types import FnType, Type

# --- generation: the live-coding invalidation knob -------------------------
# A single global counter folded into every cache key. Bumping it invalidates
# *all* specializations (the doc's acknowledged sledgehammer; precise
# dependency-graph world-age is future work). Note it lives in the *key*, not in
# any held Handle — a closure created before a bump adopts the new world on its
# next compile.

_GENERATION = 0
_GEN_LOCK = threading.Lock()


def current_generation() -> int:
    return _GENERATION


def bump_generation() -> int:
    """Invalidate all cached specializations (e.g. after redefining a function)."""
    global _GENERATION
    with _GEN_LOCK:
        _GENERATION += 1
        return _GENERATION


SpecKey = tuple[FnType, tuple[Type, ...], int]


class _Entry:
    """A cache slot, possibly still being compiled by another thread."""

    __slots__ = ("event", "artifact", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.artifact: Any = None
        self.error: BaseException | None = None


class SpecCache:
    """A thread-safe ``(FnType, arg_types, generation) -> artifact`` cache.

    ``compile_count`` / ``hit_count`` are exposed so the render loop can assert the
    thesis: over N frames that only change capture *values*, ``compile_count == 1``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[SpecKey, _Entry] = {}
        self.compile_count = 0
        self.hit_count = 0

    def key_for(self, fntype: FnType, arg_types: tuple[Type, ...]) -> SpecKey:
        return (fntype, arg_types, current_generation())

    def get_or_compile(
        self,
        fntype: FnType,
        arg_types: tuple[Type, ...],
        compile_fn: Callable[[], Any],
    ) -> Any:
        """Return the cached artifact for the key, compiling it once on a miss.

        Uses a per-key future so concurrent callers that miss the same key block
        on the first compile rather than racing to compile twice.
        """
        key = self.key_for(fntype, arg_types)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self.hit_count += 1
                mine = False
            else:
                entry = _Entry()
                self._entries[key] = entry
                mine = True

        if not mine:
            entry.event.wait()
            if entry.error is not None:
                raise entry.error
            return entry.artifact

        try:
            artifact = compile_fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised after publishing
            entry.error = exc
            entry.event.set()
            with self._lock:
                # Drop the failed slot so a later call can retry.
                self._entries.pop(key, None)
            raise
        entry.artifact = artifact
        with self._lock:
            self.compile_count += 1
        entry.event.set()
        return artifact

    def __len__(self) -> int:
        return len(self._entries)
