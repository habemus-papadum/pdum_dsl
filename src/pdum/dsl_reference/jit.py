"""``@jit`` and phase-A capture: turning a decorated Python function into a
:class:`Handle` = ``(FnType, Env)``.

Phase A happens at *decoration* time (inside the demo's ``for`` loop, that is when
the closure's cells are already bound). It is **compile-free**: it only reads the
function's code object and captured cell values, derives ``env_types``, and builds
the structural ``FnType`` plus an ``Env`` of runtime values. Compilation is phase
B and happens later, in the backend/runtime, keyed on the ``FnType``.

See ``docs/dsl_caching_layer.md`` → "Dispatch flow" / "Practical extraction".
"""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable

from .types import FnType, Type, typeof

# Roles a jitted function can play. The backend interprets these; the capture
# machinery is agnostic.
KINDS = frozenset({"device", "fragment", "vertex", "compute"})


_EMPTY_CELL = object()


def safe_cell(cell: object) -> object:
    """Read ``cell.cell_contents``, guarding the empty-cell case.

    A self-referential recursive closure has a not-yet-bound self-cell at
    construction time; reading it raises ``ValueError: Cell is empty``.
    """
    try:
        return cell.cell_contents  # type: ignore[attr-defined]
    except ValueError:
        return _EMPTY_CELL


class Handle:
    """A first-class DSL closure value: structural type + runtime environment.

    Attributes
    ----------
    fntype : FnType
        The structural function type ``(code, env_types)`` — the cache key.
    env : dict[str, object]
        Captured free-variable *values*, keyed by name (``co_freevars`` order).
        Never part of any cache key; for WebGPU these become uniform contents.
    kind : str
        The role (``"device"``, ``"fragment"``, ...).
    pyfunc : Callable
        The original Python function (kept for source/AST lowering).
    source : str
        Dedented source snapshot, taken at decoration time.
    """

    def __init__(self, fntype: FnType, env: dict[str, object], kind: str, pyfunc: Callable):
        self.fntype = fntype
        self.env = env
        self.kind = kind
        self.pyfunc = pyfunc
        self.source = _safe_getsource(pyfunc)

    @property
    def env_types(self) -> tuple[Type, ...]:
        return self.fntype.env_types

    @property
    def freevars(self) -> tuple[str, ...]:
        return self.pyfunc.__code__.co_freevars

    def __call__(self, *args: object) -> Program:
        """Calling a Handle does NOT run the Python body — it builds a deferred
        :class:`Program` (the partial application). The body is only ever lowered
        from its AST and executed on the GPU."""
        return Program(self, args)

    def __repr__(self) -> str:
        return f"Handle[{self.kind}]({self.fntype!r}, env={list(self.env)})"


class Program:
    """A jitted entry point applied to arguments — the thing a runtime draws.

    For a no-argument fragment shader this just wraps the entry Handle. For the
    higher-order case (``shader(img)``) the args carry the device Handles whose
    ``FnType`` participate in the program's specialization key.
    """

    def __init__(self, entry: Handle, args: tuple[object, ...]):
        self.entry = entry
        self.args = args

    @property
    def arg_types(self) -> tuple[Type, ...]:
        return tuple(typeof(a) for a in self.args)

    def __repr__(self) -> str:
        return f"Program({self.entry!r}, args={self.args})"


def make_handle(func: Callable, kind: str) -> Handle:
    """Phase A: extract ``(FnType, Env)`` from a freshly-defined function.

    ``co_freevars`` is the compiler's *sorted* order and aligns positionally with
    ``__closure__``; we zip them into the env. ``env_types`` is derived in that
    same order so the ``FnType`` is stable across capture-value changes.
    """
    code = func.__code__
    freevars = code.co_freevars
    cells = func.__closure__ or ()
    values = tuple(safe_cell(c) for c in cells)

    env = {name: val for name, val in zip(freevars, values) if val is not _EMPTY_CELL}
    env_types = tuple(typeof(env[name]) for name in freevars if name in env)

    fntype = FnType(template=code, env_types=env_types)
    return Handle(fntype, env, kind, func)


def jit(kind: str = "device") -> Callable[[Callable], Handle]:
    """Decorator: ``@jit(kind="fragment")`` → returns a :class:`Handle`.

    Compile-free. Compilation is deferred to the backend at draw/call time.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown jit kind {kind!r}; expected one of {sorted(KINDS)}")

    def deco(func: Callable) -> Handle:
        return make_handle(func, kind)

    return deco


def _safe_getsource(func: Callable) -> str:
    try:
        return textwrap.dedent(inspect.getsource(func))
    except OSError, TypeError:
        # Source not available (e.g. some REPL contexts). Bytecode lowering is the
        # documented fallback; not needed for M0.
        return ""
