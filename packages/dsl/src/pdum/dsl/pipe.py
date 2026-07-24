"""The fuse pipe: plumbum-style ``|`` composition of device functions.

``|`` means fuse-inline and nothing else (design 200 §1.1): stage
constructors are curried (``@op``), ``|`` composes stages into an inert
Pipeline value (a flattened ``Derived("pipe")`` identity — value-free,
rebuild-stable, warm across rebuilds), and application threads one value
through the fused artifact. The role/composition vocabulary lives ON the
Registry (never module globals); composition-time checking is a convenience
against DEFAULT — the authoritative gates are the kind vocabulary at
dispatch and the build rule at lowering.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Hashable

from .capture import Handle
from .derived import DerivedValue


class IncompatibleRoles(TypeError):
    """Composition refused: no rule for this (op, role, role) — the message says why."""


class NotYetExecutable(RuntimeError):
    """``>`` reached a pipeline with no dispatcher installed (backends land later)."""


def _vocab():
    """Compose-time checks read DEFAULT's vocabulary (a documented convenience;
    hand-built registries gate at dispatch/build, which hold the registry)."""
    from .registry import DEFAULT

    return DEFAULT


def _check(op: str, left, right) -> None:
    reg = _vocab()
    for part in (left, right):
        if part.kind not in reg.kinds:
            raise IncompatibleRoles(
                f"no kind registered for {part.kind!r}; register_kind({part.kind!r}) in the owning package"
            )
    a, b = left.kind, right.kind
    if a in reg.terminal_kinds:  # terminality is STRUCTURAL, not a pair rule:
        raise IncompatibleRoles((reg.kinds.get(a) or f"{a} is terminal") + "; nothing may follow it")
    if b in reg.terminal_kinds:
        return  # a terminal may end any pipeline (semantics tag: "terminal")
    if (op, a, b) not in reg.compositions:
        hints = "; ".join(h for h in (reg.kinds.get(a), reg.kinds.get(b)) if h)
        detail = f" — {hints}" if hints else ""
        raise IncompatibleRoles(f"no composition rule for ({op}, {a}, {b}){detail}")


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
        if isinstance(config, dict):  # named config: canonicalize like Node.attrs
            config = tuple(sorted(config.items()))
        elif not isinstance(config, tuple):
            config = (config,)
        try:
            hash(config)
        except TypeError:
            raise TypeError(f"config must be hashable (got {config!r}); wrap collections as tuples") from None
        return Stage(self.handle, config)

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


class Pipeline(DerivedValue):
    """An inert, first-class composition. Identity is a flattened
    ``Derived("pipe", …)`` over the stages — value-free, rebuild-stable.
    Protocol (fntype/fp/captures + the ValueKind) comes from ``DerivedValue``
    (130 §7) — one wrapper protocol, no hand-rolled copies."""

    __slots__ = ("parts",)

    def __init__(self, parts: tuple):
        stages = [p for p in parts if isinstance(p, Stage)]
        if not stages:
            raise IncompatibleRoles("a pipeline needs at least one kernel stage")
        self.parts = parts
        static = (
            ("config", tuple(s.config for s in stages)),
            ("terminals", tuple(p.name for p in parts if isinstance(p, Terminal))),
        )
        super().__init__(
            "pipe",
            stages[0].handle.fntype.template,
            tuple(s.handle.fntype for s in stages),
            static,
            ("P", tuple(p.fp for p in parts)),
            tuple(s.handle for s in stages),
        )

    @property
    def kind(self) -> str:  # a pipeline's role is its last non-terminal stage's role
        last = [p for p in self.parts if isinstance(p, Stage)][-1]
        return last.kind

    def __or__(self, other) -> Pipeline:
        return _compose(self, other)

    def __lt__(self, value):  # ``value > pipeline``
        return self.apply(value)

    def apply(self, value):
        dispatcher = _vocab().dispatcher
        if dispatcher is None:
            raise NotYetExecutable(
                "nothing can execute a pipeline here — install a dispatcher on the registry "
                "(registry.dispatcher = fn); reference execution is spelled reference(pipeline)(value)"
            )
        return dispatcher(self, value)

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


def build_pipe(pipeline: Pipeline, rules: dict, ops: dict, arg_types: tuple, derived: dict, *, context=None, prefix=()):
    """The fusion build rule: ``Derived("pipe")`` lowers WITHOUT source — each
    stage's body is inlined in sequence, env paths prefixed by stage index.
    This is the dress rehearsal for transforms (grad/over build the same way).
    Terminals are runtime concerns and vanish here; config awaits schemas."""
    from .ir import Builder, Loc, Region, VerifyError
    from .lower import Lowerer, check_coherence

    if len(arg_types) != 1:
        raise VerifyError(f"pipe threads exactly one value; got {len(arg_types)} arg types")
    builder = Builder(ops)
    param = builder.param(0, arg_types[0])
    current = param
    for i, stage in enumerate(p for p in pipeline.parts if isinstance(p, Stage)):
        check_coherence(stage.handle)
        code = stage.handle.pyfunc.__code__
        names = code.co_varnames[: code.co_argcount]
        if len(names) != 1:
            raise VerifyError(f"pipe stage {stage!r} must take exactly one argument")
        sub = Lowerer(
            stage.handle, rules, ops, derived, prefix=(*prefix, i), wrap=Loc("<pipeline>", i + 1), context=context
        )
        sub.locals[names[0]] = current
        current = sub.run_body()
    return Region(params=(param,), body=(builder.emit("core.yield", current),))


PIPE_BUILDERS = {"pipe": build_pipe}


def install(registry) -> None:
    """The pipe's registrations: the materializer kind + the fusion build
    rule. The "device" kind and the (pipe, device, device) fuse rule ship
    with the value language (role vocabularies belong to their owners)."""
    registry.register_kind("materializer", hint="a materializer ends a pipeline", terminal=True)
    registry.derived.update(PIPE_BUILDERS)


# Pipeline's ValueKind comes from DerivedValue via MRO — no per-class registration.
