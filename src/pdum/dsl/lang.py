"""User-facing language sentinels.

Users write ``from pdum.dsl import builtins`` and reference ``builtins.FragCoord.xy``
in a jitted body. The body is never executed in Python (it is lowered from its
AST), so these objects only need to *exist* for name resolution; lowering keys on
the attribute names structurally. They also return marker objects so experimenting
in plain Python doesn't crash.
"""

from __future__ import annotations


class _Intrinsic:
    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str) -> _Intrinsic:
        return _Intrinsic(f"{self._name}.{attr}")

    def __iter__(self):
        # The body is lowered from AST, not executed, but making swizzles
        # unpackable (``x, y = builtins.FragCoord.xy``) keeps editors/type-checkers
        # happy and lets snippets be poked at in plain Python.
        while True:
            yield _Intrinsic(f"{self._name}.component")

    def __repr__(self) -> str:
        return f"<intrinsic {self._name}>"


class _Builtins:
    """The ``builtins`` namespace recognized by the WGSL frontend."""

    FragCoord = _Intrinsic("FragCoord")  # @builtin(position): pixel coords (top-left origin)
    resolution = _Intrinsic("resolution")  # reserved uniform (future)
    time = _Intrinsic("time")  # reserved uniform (future)


builtins = _Builtins()
