"""The user-facing surface. Step 2 ships ``@jit`` (capture only); the
``Handle.__call__`` hot path, ``NoSourceError``, and ``MissingRule`` land in
step 8 with the first backend."""

from __future__ import annotations

from collections.abc import Callable

from .capture import Handle, KindTable, make_handle
from .valuekind import BUILTINS


def jit(kind: str = "device", table: KindTable = BUILTINS) -> Callable[[Callable], Handle]:
    """Decorator: phase-A capture, compile-free. ``@jit(kind="fragment")``
    returns a :class:`~pdum.dsl.kernel.capture.Handle`."""

    def deco(fn: Callable) -> Handle:
        return make_handle(fn, kind, table)

    return deco
