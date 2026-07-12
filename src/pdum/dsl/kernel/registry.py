"""The Registry (surface E, v1) and the hot path: the thesis, mechanically.

One explicit object owns what the staged seeds have been accumulating:
the kind table, the lowering rule packs, the Derived build rules, the
backends, and the two cache tiers. Satellites register INTO it at import
(``pdum.dsl.stdlib`` brings the base language, ``pdum.dsl.backends.python``
brings the first backend); the kernel never imports a satellite.

``dispatch`` is the whole runtime:

- **hit** — fingerprint the args (types, never values), build the key,
  probe tier 1, check guards, then run the precompiled ``FastRecord``:
  compiled extract → generic pack into reused staging → ``launch``.
  No parse, no typeof, no IR, no plan — those all happened once, at miss.
- **miss** — §4.3 in order: typeof → lower (coherence inside) → NORMALIZE →
  legalize_params(plan) → tier-2 probe on the *content key* (identical IR
  never renders twice, even from different templates) → render+compile →
  plan/extractor/guards → ``FastRecord`` installed under a per-key future.

Guards are the drift police: for every captured cell, ``(cell,
"cell_contents", captured_value)`` — an *identity* triple. A capture rebound
after decoration is caught at the next call (guard miss, loud counter,
recompile against the same frozen env) rather than silently serving bytes
from a world that no longer exists. Edited-and-rerun templates retire their
predecessors via the cache's code map.

``Backend`` carries name/render/compile/fp plus the step-9 columns
(``plan``, ``param_types``, ``make_launcher``). Still deferred from §2.10:
``code_for_op``-gated decompositions and a real ``params_key`` column —
type_map was absorbed into the wgsl renderer's tables, and the fragment
target format rides ``fp`` (coarse but sound) until multiple formats exist.

Book: ``docs/book/ch09-end-to-end-on-cpu.ipynb``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .cache import ArtifactCache, FastRecord, SpecializationCache
from .capture import Handle
from .ir import VerifyError
from .lower import lower_handle
from .ops import CORE_OPS
from .pack import ABI_OPS, NORMALIZE_ENV, build_extractor, legalize_params, pack_into, plan_from_types
from .rewrite import run_stage
from .valuekind import BUILTINS, KindTable


@dataclass(frozen=True)
class Backend:
    """A capability record (§2.10; the step-9 columns arrive with the second
    backend). The three optional callables let a backend own its layout, its
    parameter convention, and its launcher without the kernel knowing any
    target's shape — the python backend takes every default."""

    name: str
    render: Callable  # (region, plan, name) -> source text
    compile: Callable  # (source, name) -> artifact
    fp: tuple  # enters every specialization key (a backend/version change is a new world)
    plan: Callable | None = None  # (env_types, arg_types, table) -> PackPlan; None = dense staging
    param_types: Callable | None = None  # (target) -> tuple | None; a family that DERIVES params
    #   (compute: params ARE thread coords) returns their types; None = typeof the call args
    make_launcher: Callable | None = None  # (artifact, plan) -> launch(staging, leaves); None = artifact


class NoBackend(RuntimeError):
    """Dispatch reached a registry with no backend — ``import pdum.dsl`` for
    batteries, or ``register_backend`` your own."""


class Out:
    """Tags the ``out=`` payload on the leaves channel, so launchers peel it
    by TYPE, not by tail position — buffer leaves (ch12) cannot collide."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _guards(target) -> tuple:
    if isinstance(target, Handle):
        cells = dict(zip(target.freevars, target.pyfunc.__closure__ or ()))
        own = tuple((cells[name], "cell_contents", expected) for name, expected in target.env.items())
        # Recurse into captured kernels: an INNER handle's cell drifting is the
        # same staleness as an outer one (review-caught hole — inlined callees
        # are baked into the artifact, so their drift must guard the entry).
        return own + tuple(g for v in target.captures if isinstance(v, Handle) for g in _guards(v))
    return tuple(g for h in target.captures for g in _guards(h))  # a Pipeline guards its stages


class Registry:
    """v1: enough for backends + rule packs + the dispatch loop. Layering
    (``extend()``), overloads, and Dialect bundles complete it at step 10."""

    def __init__(self, table: KindTable = BUILTINS):
        self.table = table
        self.lower_rules: dict = {}
        self.derived: dict = {}
        self.backends: dict[str, Backend] = {}
        self.routes: dict[str, str] = {}  # kind -> backend name; absent kinds use the default
        self.default_backend: str | None = None
        self.specializations = SpecializationCache()
        self.artifacts = ArtifactCache()

    def register_backend(self, backend: Backend, *, default: bool = False, kinds: tuple = ()) -> None:
        self.backends[backend.name] = backend
        for kind in kinds:  # per-role routing: role vocabularies ship WITH their backends
            self.routes[kind] = backend.name
        if default or (self.default_backend is None and not kinds):
            # A kind-scoped backend never CLAIMS the default slot: a registry
            # holding only wgsl must refuse a "device" kernel loudly, not
            # silently compile it through the compute backend (review-caught).
            self.default_backend = backend.name

    def backend_for(self, kind: str) -> Backend:
        name = self.routes.get(kind, self.default_backend)
        if name is None:
            raise NoBackend("no backend registered; `import pdum.dsl` wires the batteries")
        return self.backends[name]

    def dispatch(self, target, args: tuple, out=None):
        """THE hot path. `target` is anything FnType-shaped: Handle or Pipeline.
        `out` is launcher data (destinations / launch domain) — it rides the
        leaves channel and never touches any key (070 §3: grid strips)."""
        spec, table = self.specializations, self.table
        backend = self.backend_for(target.kind)
        key = (target.fp, tuple(table.fingerprint(a) for a in args), backend.fp, spec.generation)
        record = spec.probe(key)
        if record is None:  # the miss thunk is only ever built on a miss
            record = spec.get_or_compile(key, lambda: self._build(target, args, backend))
        leaves = pack_into(record.plan, record.staging, record.extract(target.captures, args))
        return record.launch(record.staging, leaves if out is None else (*leaves, Out(out)))

    def _build(self, target, args: tuple, backend: Backend) -> FastRecord:
        """The miss path (§4.3), once per (types, backend, generation)."""
        derived_params = backend.param_types(target) if backend.param_types else None
        if derived_params is not None and args:  # params are DERIVED (thread coords): a positional
            raise VerifyError(  # arg would be silently dead — refuse it loudly instead
                f"{backend.name} derives this kernel's parameters; positional arguments are not "
                f"accepted — pass the launch domain via out="
            )
        arg_types = derived_params if derived_params is not None else tuple(self.table.typeof(a) for a in args)
        env_types = target.env_types
        ops = {**CORE_OPS, **ABI_OPS}
        region = lower_handle(target, self.lower_rules, ops, arg_types=arg_types, derived=self.derived)
        plan = (backend.plan or plan_from_types)(env_types, arg_types, self.table)
        region = run_stage(region, NORMALIZE_ENV, ops)
        region = run_stage(region, legalize_params(plan), ops)
        artifact = self.artifacts.get_or_compile(  # content-addressed: identical IR compiles once
            (region.key, backend.fp),  # fp, not name: a backend VERSION change is a new artifact world
            lambda: backend.compile(backend.render(region, plan)),
        )
        extract = build_extractor(env_types, arg_types, plan, self.table)
        return FastRecord(
            artifact=artifact,
            guards=_guards(target),
            extract=extract,
            plan=plan,
            staging=bytearray(plan.staging_size),
            launch=backend.make_launcher(artifact, plan) if backend.make_launcher else artifact,
        )


DEFAULT = Registry()  # the staged seeds fold in here; satellites register at import
