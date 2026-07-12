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

``Backend`` v1 carries what the Python backend needs (token, render,
compile, fp). The §2.10 columns it defers — ``type_map``, ``code_for_op``
gating shared decompositions, ``params_key`` — arrive with the WGSL backend,
which is the first consumer that needs them.

Book: ``docs/book/ch09-end-to-end-on-cpu.ipynb``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .cache import ArtifactCache, FastRecord, SpecializationCache
from .capture import Handle
from .lower import lower_handle
from .ops import CORE_OPS
from .pack import ABI_OPS, NORMALIZE_ENV, build_extractor, legalize_params, pack_into, plan_from_types
from .rewrite import run_stage
from .valuekind import BUILTINS, KindTable


@dataclass(frozen=True)
class Backend:
    """A capability record (§2.10, the v1 columns)."""

    name: str
    render: Callable  # (region, plan, name) -> source text
    compile: Callable  # (source, name) -> artifact; artifact(staging, leaves) runs
    fp: tuple  # enters every specialization key (a backend/version change is a new world)


class NoBackend(RuntimeError):
    """Dispatch reached a registry with no backend — ``import pdum.dsl`` for
    batteries, or ``register_backend`` your own."""


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
        self.default_backend: str | None = None
        self.specializations = SpecializationCache()
        self.artifacts = ArtifactCache()

    def register_backend(self, backend: Backend, *, default: bool = False) -> None:
        self.backends[backend.name] = backend
        if default or self.default_backend is None:
            self.default_backend = backend.name

    def backend_for(self, kind: str) -> Backend:
        # v1: one default backend serves every role; per-role routing lands
        # with the second backend (ch10). The tripwire below makes forgetting
        # that a loud error instead of silent wrong-backend compilation.
        if self.default_backend is None:
            raise NoBackend("no backend registered; `import pdum.dsl` wires the batteries")
        if len(self.backends) > 1:
            raise NotImplementedError("multiple backends registered: per-role routing is the ch10 work")
        return self.backends[self.default_backend]

    def dispatch(self, target, args: tuple):
        """THE hot path. `target` is anything FnType-shaped: Handle or Pipeline."""
        spec, table = self.specializations, self.table
        backend = self.backend_for(target.kind)
        key = (target.fp, tuple(table.fingerprint(a) for a in args), backend.fp, spec.generation)
        record = spec.probe(key)
        if record is None:  # the miss thunk is only ever built on a miss
            record = spec.get_or_compile(key, lambda: self._build(target, args, backend))
        leaves = pack_into(record.plan, record.staging, record.extract(target.captures, args))
        return record.launch(record.staging, leaves)

    def _build(self, target, args: tuple, backend: Backend) -> FastRecord:
        """The miss path (§4.3), once per (types, backend, generation)."""
        arg_types = tuple(self.table.typeof(a) for a in args)
        env_types = target.env_types
        ops = {**CORE_OPS, **ABI_OPS}
        region = lower_handle(target, self.lower_rules, ops, arg_types=arg_types, derived=self.derived)
        plan = plan_from_types(env_types, arg_types, self.table)
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
            launch=artifact,
        )


DEFAULT = Registry()  # the staged seeds fold in here; satellites register at import
