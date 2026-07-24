"""Cache-backed registries — the process dicts died at P3 (200 §1.5, §3.2).

A registry here is a name-keyed cache on the core's ``Memo`` — instrumented
by construction (``<kind>.miss`` events, compile spans, per-key futures) —
with registry semantics on top:

- entries are **program vocabulary**, never evicted: an IR program references
  its markers and reducers by name, so eviction would orphan programs
  (capacity is effectively infinite; the cache discipline wanted here is
  idempotence and derivation-under-cache, not memory bounding);
- **re-registering identical content is a hit** — one entry, no event (the
  idempotence pin); a conflicting registration under a taken name REFUSES
  with a designed message, never overwrites;
- **derivations are cache entries computed on demand from cache entries**:
  partials, component markers, and adjoint scanners go through ``derive``,
  so the rewrite runs once per name and ``events.forbid("marker.miss")``
  can pin "this loop derives nothing new".
"""

from __future__ import annotations

from typing import Callable

from pdum.dsl.cache import Memo


class RegistryConflict(ValueError):
    """A name was re-registered with different content."""


class CacheRegistry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._memo = Memo(kind, capacity=1 << 30)  # vocabulary: no eviction, ever

    def __contains__(self, name: str) -> bool:
        return self._memo.peek(name) is not None

    def __getitem__(self, name: str):
        entry = self._memo.peek(name)
        if entry is None:
            raise KeyError(name)
        return entry

    def get(self, name: str, default=None):
        entry = self._memo.peek(name)
        return default if entry is None else entry

    def register(self, name: str, value, content: Callable):
        """Idempotent registration: identical content lands on the one
        existing entry (a hit); different content under a taken name refuses."""
        got = self._memo.get_or_compile(name, lambda: value)
        if got is not value and content(got) != content(value):
            raise RegistryConflict(
                f"{self.kind} {name!r} is already registered with different "
                f"content — names are program vocabulary and entries are "
                f"immutable; pick a fresh name (or derive one from the "
                f"content digest by passing name=None)"
            )
        return got

    def derive(self, name: str, make: Callable):
        """Derivation-under-cache: ``make`` runs only on the first request;
        every later request for ``name`` is a hit on the derived entry."""
        return self._memo.get_or_compile(name, make)


MARKERS = CacheRegistry("marker")
REDUCERS = CacheRegistry("reducer")
