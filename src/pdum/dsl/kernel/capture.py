"""Phase A: turning a decorated Python function into a ``Handle``.

Capture happens at decoration time, every time a closure is rebuilt — which
in the target workflows is *every loop iteration*, so this path is reflection
only: read the code object, read the closure cells, summarize capture types
via memoized fingerprints. **No parse, no IR, no compile, ever.** Missing
source does not fail here (the snapshot is simply ``None``; phase B raises a
loud ``NoSourceError`` when lowering actually needs it, step 6). What *does*
fail loudly, immediately, is an untypeable capture — better at the def site
than inside a cache key.

A ``Handle`` is a first-class DSL closure: ``(FnType, env values, snapshot)``.
Nesting is structural — a capture that is itself a ``Handle`` contributes its
``FnType`` to the parent's ``env_types``, so composed programs ("the program
is the parameter container") have tree-shaped, value-free identities that are
stable across rebuilds.

Memos: snapshots live in a ``WeakKeyDictionary`` keyed by code object (dies
with the code; preserves the decoration-time text so a later on-disk edit
cannot silently supply stale source — phase B's coherence check completes the
defense). The ``FnType`` memo is a plain dict for now; eviction policy lands
with the cache tier (step 3).

Book: ``docs/book/ch02-what-a-closure-is.ipynb``. Architecture: §2.3–§2.4, §4.1.
"""

from __future__ import annotations

import inspect
import textwrap
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from types import CodeType
from typing import Hashable

from .types import Base, FnType, Type
from .valuekind import BUILTINS, KindTable

_EMPTY = object()  # a not-yet-bound cell (self-referential closure under construction)


def safe_cell(cell: object) -> object:
    try:
        return cell.cell_contents  # type: ignore[attr-defined]
    except ValueError:
        return _EMPTY


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    text: str
    filename: str
    firstlineno: int
    qualname: str


_SNAPSHOTS: weakref.WeakKeyDictionary[CodeType, SourceSnapshot | None] = weakref.WeakKeyDictionary()
_FNTYPES: dict[tuple[CodeType, tuple[Hashable, ...]], FnType] = {}


def _take_snapshot(fn: Callable) -> SourceSnapshot | None:
    code = fn.__code__
    try:
        lines, start = inspect.getsourcelines(fn)  # start = the TEXT's first line (decorators included)
    except OSError, TypeError:
        return None  # phase B's NoSourceError, not ours
    # filename and firstlineno must name the SAME file: getsourcelines unwraps
    # (@wraps) but co_filename does not, so a wrapped fn would pair the
    # wrapper's file with the wrappee's line — every later Loc would lie.
    filename = getattr(inspect.unwrap(fn), "__code__", code).co_filename
    return SourceSnapshot(textwrap.dedent("".join(lines)), filename, start, code.co_qualname)


class Handle:
    """A first-class DSL closure: structural identity + runtime environment.

    ``fntype`` is the cache-key half (never touches values); ``env`` holds the
    captured *values* (never in any key). ``fp`` is the precomputed handle
    fingerprint — code object (value-compared) + env fingerprints — so nested
    handles fingerprint in O(1) at the parent.
    """

    __slots__ = ("fntype", "env", "env_fp", "fp", "snapshot", "kind", "table", "pyfunc", "captures", "__weakref__")

    def __init__(self, fntype, env, env_fp, snapshot, kind, table, pyfunc):
        self.fntype = fntype
        self.env = env
        # Frozen at construction, from the same order `env_types` was built in:
        # the marshaling law is that `captures[i]` is a value of `env_types[i]`.
        # Deriving it per call from `env.values()` would let a later mutation of
        # the dict silently transpose the two — swapped uniforms, no error.
        self.captures = tuple(env.values())
        self.env_fp = env_fp
        self.fp = ("H", pyfunc.__code__, env_fp)
        self.snapshot = snapshot
        self.kind = kind
        self.table = table
        self.pyfunc = pyfunc

    @property
    def env_types(self) -> tuple[Type, ...]:
        return self.fntype.env_types

    @property
    def freevars(self) -> tuple[str, ...]:
        return self.pyfunc.__code__.co_freevars

    def __call__(self, *args: object, out: object = None) -> object:
        """Phase B's door: dispatch through the DEFAULT registry (hit = extract
        → pack → launch; miss = compile). Defined ON the class so a Handle is
        honestly callable — to users and to static tooling alike — but the
        import is lazy: capture stays reflection-only at import time, and
        nothing about phase A touches the runtime. ``out`` is launcher data
        (destinations / launch domain), never identity."""
        from .registry import DEFAULT

        return DEFAULT.dispatch(self, args, out)

    def __repr__(self) -> str:
        return f"Handle[{self.kind}]({self.fntype!r}, env={list(self.env)})"


class _HandleKind:
    """A captured DSL closure presents its own structural ``FnType``."""

    def typeof(self, v: Handle, table: KindTable) -> Type:
        return v.fntype

    def fingerprint(self, v: Handle, table: KindTable) -> Hashable:
        return v.fp  # precomputed at construction

    def flatten(self, v: Handle, table: KindTable) -> tuple:
        return tuple(leaf for val in v.captures for leaf in table.flatten(val))


BUILTINS.register(Handle, _HandleKind())


def make_handle(fn: Callable, kind: str, table: KindTable = BUILTINS) -> Handle:
    """Phase A: reflect ``(FnType, env)`` out of a freshly defined function.

    Parameters
    ----------
    fn : Callable
        The just-defined function; its ``co_freevars`` align positionally
        with ``__closure__`` (CPython sorts freevars).
    kind : str
        The role, interpreted by dialects/backends ("device", "fragment", …);
        capture is agnostic and does not validate it.
    table : KindTable
        The kind table that summarizes captures (the session registry, later).

    Raises
    ------
    TypeError
        If a captured value has no registered kind (loud at the def site).
    """
    code = fn.__code__
    # Read the memo ONCE: `in` could be satisfied by a value-equal twin code
    # object whose weak entry dies before a second lookup (rare GC race).
    snap = _SNAPSHOTS.get(code, _EMPTY)
    if snap is _EMPTY:
        snap = _SNAPSHOTS[code] = _take_snapshot(fn)
    cells = fn.__closure__ or ()
    bound = [(name, v) for name, c in zip(code.co_freevars, cells) if (v := safe_cell(c)) is not _EMPTY]
    env_fp = tuple(table.fingerprint(v) for _, v in bound)
    fntype = _FNTYPES.get((code, env_fp))
    if fntype is None:
        fntype = _FNTYPES.setdefault((code, env_fp), FnType(Base(code), tuple(table.typeof(v) for _, v in bound)))
    return Handle(fntype, dict(bound), env_fp, snap, kind, table, fn)
