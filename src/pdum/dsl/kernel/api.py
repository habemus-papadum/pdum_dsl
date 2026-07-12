"""The user-facing surface: ``@jit``.

Calling a Handle dispatches through the DEFAULT registry (hit = extract →
pack → launch; miss = the §4.3 pipeline). ``Handle.__call__`` lives on the
class itself in ``capture.py`` — real to static tooling, no import-order
dependence — with a lazy registry import so capture stays reflection-only.

Batteries note: the kernel registers nothing into DEFAULT itself — but
importing ANY ``pdum.dsl.*`` module executes the package ``__init__``, which
wires the base dialect and the Python backend (Python import semantics: there
is no "bare" kernel import from outside). ``NoBackend`` and an empty rule
table are therefore the experience of a hand-built ``Registry()``, which is
exactly the object the chapters use to show each seam; populate one
explicitly via ``stdlib.install(reg)`` / ``backends.python.install(reg)``.
"""

from __future__ import annotations

from collections.abc import Callable

from .capture import Handle, make_handle


def jit(kind: str = "device") -> Callable[[Callable], Handle]:
    """Decorator: phase-A capture, compile-free. ``@jit(kind="fragment")``
    returns a :class:`~pdum.dsl.kernel.capture.Handle`; calling the Handle
    enters the two-tier dispatch.

    No ``table=`` parameter by design: dispatch fingerprints through the
    REGISTRY's table, so a per-handle table here would be silently ignored —
    a split-brain the API must not invite. Custom tables enter via a custom
    Registry (surface E completes at step 10); ``make_handle`` remains the
    low-level door."""

    def deco(fn: Callable) -> Handle:
        return make_handle(fn, kind)

    return deco
