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

from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import perf_counter_ns

from . import events
from .cache import ArtifactCache, FastRecord, SpecializationCache
from .capture import Handle
from .derived import DerivedValue
from .ir import VerifyError
from .lower import lower_handle
from .ops import CORE_OPS
from .pack import ABI_OPS, NORMALIZE_ENV, build_extractor, legalize_params, pack_into, plan_from_types
from .rewrite import rewrite, run_stage
from .valuekind import BUILTINS, KindTable


@dataclass(frozen=True)
class Backend:
    """A capability record (§2.10; the step-9 columns arrive with the second
    backend). The three optional callables let a backend own its layout, its
    parameter convention, and its launcher without the kernel knowing any
    target's shape — the python backend takes every default."""

    name: str
    render: Callable  # (region, plan, backend) -> source text; backend = THIS record (spelling table)
    compile: Callable  # (source, name) -> artifact
    fp: tuple  # enters every specialization key (a backend/version change is a new world)
    plan: Callable | None = None  # (env_types, arg_types, table) -> PackPlan; None = dense staging
    param_types: Callable | None = None  # (target) -> tuple | None; a family that DERIVES params
    #   (compute: params ARE thread coords) returns their types; None = typeof the call args
    make_launcher: Callable | None = None  # (artifact, plan) -> launch(staging, leaves); None = artifact
    code_for_op: dict = field(default_factory=dict)  # op -> spelling template; keys() = capability set


class NoBackend(RuntimeError):
    """Dispatch reached a registry with no backend — ``import pdum.dsl`` for
    batteries, or ``register_backend`` your own."""


# Out tags the ``out=`` payload on the leaves channel: launchers peel it by
# TYPE, never tail position — buffer leaves (ch12) cannot collide with it.
Out = namedtuple("Out", "value")


def _guards(target) -> tuple:
    if isinstance(target, Handle):
        cells = dict(zip(target.freevars, target.pyfunc.__closure__ or ()))
        own = tuple((cells[name], "cell_contents", expected) for name, expected in target.env.items())
        # Recurse into captured kernels AND derived values: an INNER cell
        # drifting is the same staleness as an outer one, whether it arrives
        # through a Handle or a wrapper (step-11 review noted the wrapper gap).
        deep = (v for v in target.captures if isinstance(v, (Handle, DerivedValue)))
        return own + tuple(g for v in deep for g in _guards(v))
    return tuple(g for h in target.captures for g in _guards(h))  # pipelines/transforms guard their bases


class Registry:
    """v1: enough for backends + rule packs + the dispatch loop. Layering
    (``extend()``), overloads, and Dialect bundles complete it at step 10."""

    def __init__(self, table: KindTable = BUILTINS):
        self.table = table
        self.lower_rules: dict = {}
        self.derived: dict = {}
        self.ops: dict = {}  # surface A: dialect OpDefs, merged into every build
        self.overloads: dict = {}  # surface B: call-name -> impl (satellites interpret)
        self.decompositions: list = []  # (op_name, rule): applied when the backend lacks the op
        self.backends: dict[str, Backend] = {}
        self.routes: dict[str, str] = {}  # kind -> backend name; absent kinds use the default
        self.default_backend: str | None = None
        self.specializations = SpecializationCache()
        self.artifacts = ArtifactCache()

    def register_backend(self, backend: Backend, *, default: bool = False, kinds: tuple = ()) -> None:
        self.backends[backend.name] = backend
        self.routes.update(dict.fromkeys(kinds, backend.name))  # roles ship WITH their backends
        if default or (self.default_backend is None and not kinds):
            # A kind-scoped backend never CLAIMS the default slot: a registry
            # holding only wgsl must refuse a "device" kernel loudly, not
            # silently compile it through the compute backend (review-caught).
            self.default_backend = backend.name

    def backend_for(self, kind: str) -> Backend:
        if (name := self.routes.get(kind, self.default_backend)) is None:
            raise NoBackend("no backend registered; `import pdum.dsl` wires the batteries")
        return self.backends[name]

    def dispatch(self, target, args: tuple, out=None):
        """THE hot path. `target` is anything FnType-shaped: Handle or Pipeline.
        `out` is launcher data (destinations / launch domain) — it rides the
        leaves channel and never touches any key (070 §3: grid strips)."""
        if events.SINKS:  # one branch, ~1.2% of a warm hit (120 §5); dark otherwise
            return self._dispatch_traced(target, args, out)
        spec, table = self.specializations, self.table
        backend = self.backend_for(target.kind)
        key = (target.fp, tuple(table.fingerprint(a) for a in args), backend.fp, spec.generation)
        record = spec.probe(key) or spec.get_or_compile(key, lambda: self._build(target, args, backend))
        leaves = pack_into(record.plan, record.staging, record.extract(target.captures, args))
        return record.launch(record.staging, leaves if out is None else (*leaves, Out(out)))

    def _dispatch_traced(self, target, args: tuple, out=None):
        """dispatch's traced twin (120 §5): the same four steps with a
        timestamp between each, emitted as a batch AFTER launch so sink cost
        never pollutes a later phase. ``test_traced_dispatch_agrees`` pins
        the two bodies together — edit BOTH or CI says so."""
        t0 = perf_counter_ns()
        spec, table = self.specializations, self.table
        backend = self.backend_for(target.kind)
        key = (target.fp, tuple(table.fingerprint(a) for a in args), backend.fp, spec.generation)
        record = spec.probe(key) or spec.get_or_compile(key, lambda: self._build(target, args, backend))
        t1 = perf_counter_ns()
        values = record.extract(target.captures, args)
        t2 = perf_counter_ns()
        leaves = pack_into(record.plan, record.staging, values)
        t3 = perf_counter_ns()
        result = record.launch(record.staging, leaves if out is None else (*leaves, Out(out)))
        t4 = perf_counter_ns()
        for name, dur in (
            ("dispatch.probe", t1 - t0),
            ("dispatch.extract", t2 - t1),
            ("dispatch.pack", t3 - t2),
            ("dispatch.launch", t4 - t3),
        ):
            events.emit(name, key, dur)
        return result

    def _build(self, target, args: tuple, backend: Backend) -> FastRecord:
        """The miss path (§4.3), once per (types, backend, generation)."""
        derived_params = backend.param_types(target) if backend.param_types else None
        if derived_params is not None and args:  # a positional arg would be silently dead: refuse
            raise VerifyError(f"{backend.name} derives the params; pass the launch domain via out=")
        arg_types = derived_params if derived_params is not None else tuple(self.table.typeof(a) for a in args)
        ops = {**CORE_OPS, **ABI_OPS, **self.ops}
        rules = dict(self.lower_rules)
        context = {"registry": self}  # THE build context (130 §7): rule packs read ctx.context, never the rules dict
        with events.span("lower", target.fp):
            region = lower_handle(target, rules, ops, arg_types=arg_types, derived=self.derived, context=context)
        plan = (backend.plan or plan_from_types)(target.env_types, arg_types, self.table)
        with events.span("rewrite", target.fp):  # decompositions + the two ABI stages
            gated = [r for op, r in self.decompositions if op not in backend.code_for_op]
            if gated:  # shared decompositions run only where the target lacks the native op (§2.10)
                region = rewrite(region, gated, ops, name="decompose")
            region = run_stage(region, NORMALIZE_ENV, ops)
            dialects = frozenset(op.split(".", 1)[0] for op in self.ops)
            region = run_stage(region, legalize_params(plan, dialects), ops)

        def _render_then_compile():
            with events.span("render", region.key):
                source = backend.render(region, plan, backend)
            return backend.compile(source)

        artifact = self.artifacts.get_or_compile(  # content-addressed: identical IR compiles once
            (region.key, backend.fp),  # fp, not name: a backend VERSION change is a new artifact world
            _render_then_compile,
        )
        extract = build_extractor(target.env_types, arg_types, plan, self.table)
        return FastRecord(
            artifact=artifact,
            guards=_guards(target),
            extract=extract,
            plan=plan,
            staging=bytearray(plan.staging_size),
            launch=backend.make_launcher(artifact, plan) if backend.make_launcher else artifact,
        )

    def extend(self) -> Registry:
        """Surface-E layering (stdlib -> user -> session): a child registry
        with copied registrations, an extended kind table, FRESH caches."""
        child = Registry(self.table.extend())
        for a in ("lower_rules", "derived", "routes", "ops", "overloads"):
            getattr(child, a).update(getattr(self, a))
        # Backend records are shared-immutable EXCEPT code_for_op: copy it, or a
        # child's spell() would mutate the parent's table (review-caught leak).
        child.backends = {k: replace(b, code_for_op=dict(b.code_for_op)) for k, b in self.backends.items()}
        child.decompositions, child.default_backend = list(self.decompositions), self.default_backend
        return child

    def load_entry_points(self, group: str = "pdum.dsl.backends", entries=None) -> list:
        """Third-party discovery (080 §4): each entry point names an
        ``install(registry)``; explicit, lazy, no import-order dependence."""
        from importlib.metadata import entry_points

        eps = list(entries) if entries is not None else list(entry_points(group=group))
        for ep in eps:
            ep.load()(self)
        return [ep.name for ep in eps]


DEFAULT = Registry()  # the staged seeds fold in here; satellites register at import
