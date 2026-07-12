"""The blessed combinator library: plumbum-style pipelines over kernels.

**Satellite** — attaches to the kernel with zero kernel edits (the first
extension-locality proof, ahead of its CI gate). Semantics follow
`pdum_plumbum` (credit: habemus-papadum/pdum_plumbum): stage constructors are
curried, ``|`` composes stages into an **inert** pipeline value, and ``>``
threads a value through it, ideally once. Design record:
``design/combinators-notes.md``.

This is the *definition layer* (step 3b). What is real today: identities
(flattened ``Derived("pipe", …)`` templates that hit the specialization cache across
rebuilds), Roles and the composition-rule registry (loud
``IncompatibleRoles`` with explanations), configured-stage syntax
(``stage[config]`` — recorded **conservatively as static**: changing config
changes identity; the static/runtime split lands with execution), and
materializer terminals (recorded, not yet executing). What is stubbed:
application — ``value > pipeline`` needs the compiler chain (fusion build
rule at lowering; DPS/ResultPlan at marshaling; launch at the backends), so
it dispatches through an installable hook and is loud without one.

The module-level role/rule/dispatcher state is the staged seed of the
explicit Registry (surface E), same as ``valuekind.BUILTINS``; it folds in at
step 8.

Book: ``docs/book/ch04-pipelines-are-values.ipynb``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Hashable

from .kernel.capture import Handle
from .kernel.types import Derived, FnType, Type
from .kernel.valuekind import BUILTINS, KindTable


class IncompatibleRoles(TypeError):
    """Composition refused: no rule for this (op, role, role) — the message says why."""


class NotYetExecutable(RuntimeError):
    """``>`` reached a pipeline with no dispatcher installed (backends land later)."""


@dataclass(frozen=True)
class Role:
    """What a stage *is* for composition purposes: ``Handle.kind``, grown up."""

    name: str
    terminal: bool = False  # terminals (materializers) may only end a pipeline


_ROLES: dict[str, Role] = {}
_HINTS: dict[str, str] = {}
_RULES: dict[tuple[str, str, str], str] = {}


def register_role(name: str, *, terminal: bool = False, hint: str = "") -> Role:
    role = _ROLES[name] = Role(name, terminal)
    if hint:
        _HINTS[name] = hint
    return role


def register_composition(op: str, left: str, right: str, semantics: str) -> None:
    """Record that ``left-role op right-role`` composes, and *how* — the
    semantics tag ("fuse", "terminal", later "orchestrate") is consumed by
    build rules once lowering exists; the definition layer only gates."""
    _RULES[(op, left, right)] = semantics


# This library owns MACHINERY plus exactly one concept: the materializer
# (needed by its own `collect`). Everything else ships with its owner —
# "device" + the fuse rule belong to the base-language (stdlib) package once
# lowering exists; "fragment" to the WGSL package; "audio_node" to an audio
# package. Never pre-enumerated here (ch04 walkthrough, 2026-07-11).
register_role("materializer", terminal=True, hint="a materializer ends a pipeline")


def _role_of(part) -> Role:
    role = _ROLES.get(part.kind)
    if role is None:
        raise IncompatibleRoles(f"no Role registered for kind {part.kind!r}; register_role({part.kind!r}) first")
    return role


def _check(op: str, left, right) -> None:
    a, b = _role_of(left), _role_of(right)
    if a.terminal:  # terminality is STRUCTURAL, not a pair rule:
        raise IncompatibleRoles(_HINTS.get(a.name, f"{a.name} is terminal") + "; nothing may follow it")
    if b.terminal:
        return  # a terminal may end any pipeline (semantics tag: "terminal")
    if (op, a.name, b.name) not in _RULES:
        hints = "; ".join(h for h in (_HINTS.get(a.name), _HINTS.get(b.name)) if h)
        detail = f" — {hints}" if hints else ""
        raise IncompatibleRoles(f"no composition rule for ({op}, {a.name}, {b.name}){detail}")


class Stage:
    """One pipeline-composable kernel: a Handle plus (static, for now) config."""

    __slots__ = ("handle", "config")

    def __init__(self, handle: Handle, config: tuple = ()):
        self.handle = handle
        self.config = config

    @property
    def kind(self) -> str:
        return self.handle.kind

    @property
    def fp(self) -> Hashable:
        return ("S", self.handle.fp, self.config)

    def __getitem__(self, config) -> Stage:
        return Stage(self.handle, config if isinstance(config, tuple) else (config,))

    def __or__(self, other) -> Pipeline:
        return _compose(self, other)

    def __lt__(self, value):  # ``value > stage`` (reflected comparison, plumbum-style)
        return _compose(self).apply(value)

    def __repr__(self) -> str:
        cfg = f"[{', '.join(map(repr, self.config))}]" if self.config else ""
        return f"{self.handle.fntype.template.label}{cfg}"


class Terminal:
    """A boundary stage (materializer): ends a pipeline, executes nothing yet."""

    __slots__ = ("name",)
    kind = "materializer"

    def __init__(self, name: str):
        self.name = name

    @property
    def fp(self) -> Hashable:
        return ("T", self.name)

    def __or__(self, other) -> Pipeline:
        return _compose(self, other)  # role rules refuse, with the explanation

    def __repr__(self) -> str:
        return self.name


collect = Terminal("collect")  # materialize to the host at the boundary (stub)


def _parts(x) -> tuple:
    if isinstance(x, Pipeline):
        return x.parts  # splicing = flattening: associativity vanishes syntactically
    if isinstance(x, (Stage, Terminal)):
        return (x,)
    if isinstance(x, Handle):
        return (Stage(x),)
    raise TypeError(f"cannot pipe {x!r}; stages are @op(...) constructions, Handles, or terminals")


def _compose(*items) -> Pipeline:
    parts = tuple(p for item in items for p in _parts(item))
    for left, right in zip(parts, parts[1:]):
        _check("pipe", left, right)
    return Pipeline(parts)


class Pipeline:
    """An inert, first-class composition. Identity is a flattened
    ``Derived("pipe", …)`` over the stages — value-free, rebuild-stable."""

    __slots__ = ("parts", "fntype", "fp")

    def __init__(self, parts: tuple):
        stages = [p for p in parts if isinstance(p, Stage)]
        if not stages:
            raise IncompatibleRoles("a pipeline needs at least one kernel stage")
        self.parts = parts
        env_types: tuple[Type, ...] = tuple(s.handle.fntype for s in stages)
        static = (
            ("config", tuple(s.config for s in stages)),
            ("terminals", tuple(p.name for p in parts if isinstance(p, Terminal))),
        )
        self.fntype = FnType(Derived("pipe", stages[0].handle.fntype.template, static), env_types)
        self.fp = ("P", tuple(p.fp for p in parts))

    @property
    def kind(self) -> str:  # a pipeline's role is its last non-terminal stage's role
        last = [p for p in self.parts if isinstance(p, Stage)][-1]
        return last.kind

    def __or__(self, other) -> Pipeline:
        return _compose(self, other)

    def __lt__(self, value):  # ``value > pipeline``
        return self.apply(value)

    def apply(self, value):
        if _DISPATCHER is None:
            raise NotYetExecutable(
                "nothing can execute a pipeline yet — fusion lands with lowering, launch with the "
                "backends; install set_dispatcher(fn) to experiment (as ch04 does with dummy artifacts)"
            )
        return _DISPATCHER(self, value)

    def __repr__(self) -> str:
        return " | ".join(repr(p) for p in self.parts)


def op(factory: Callable) -> Callable:
    """plumbum's ``@pb``, for kernels: the decorated closure factory becomes a
    curried stage constructor — ``add(1)`` configures a Stage, runs nothing."""

    @functools.wraps(factory)
    def construct(*args, **kwargs) -> Stage:
        built = factory(*args, **kwargs)
        if not isinstance(built, Handle):
            raise TypeError(f"@op factory {factory.__name__!r} must return a jitted Handle, got {type(built).__name__}")
        return Stage(built)

    return construct


_DISPATCHER: Callable | None = None


def set_dispatcher(fn: Callable | None):
    """Install the application hook (the runtime's job at step 8); returns the
    previous dispatcher so callers can restore it."""
    global _DISPATCHER
    previous, _DISPATCHER = _DISPATCHER, fn
    return previous


class _PipelineKind:
    """Pipelines are first-class values of the type system (capturable, composable)."""

    def typeof(self, v: Pipeline, table: KindTable) -> Type:
        return v.fntype

    def fingerprint(self, v: Pipeline, table: KindTable) -> Hashable:
        return v.fp


BUILTINS.register(Pipeline, _PipelineKind())
