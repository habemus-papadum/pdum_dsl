# 120 — Events & instrumentation: counting the expensive things

**Status:** proposal (not implemented). Supersedes nothing; extends 010 §6
(budgets) and §2.12/§4.4 (the cache bookkeeping).

**The ask.** There is a class of bug this codebase cannot currently see: not a
wrong answer, but a *right answer computed too often*. A specialization that
recompiles every frame. A guard that drifts on every iteration. A memoized
analysis that misses in a tight loop. The output is correct; the machine is
melting. The thesis of this project — *one compile per signature* — is exactly
the claim that these events are **rare**, so the ability to observe them is not
a nice-to-have; it is how the thesis is falsified.

This document proposes a generic **event seam** in the kernel plus a
**recorder satellite** that turns every expensive occurrence into a counted,
timed, optionally stack-traced datum; a **`Memo`** primitive so that future
fingerprint-keyed analyses are instrumented by construction rather than by
discipline; the **line-budget increase** this costs; and the removal of the
`bench.py` monkeypatch that exists today because no seam does.

---

## 1. What exists today (and why it is not enough)

### 1.1 Four ad-hoc counters

`kernel/cache.py` already reaches for this capability, four times:

```python
class _TierCache:
    self.hits = self.misses = self.compiles = self.evictions = 0

class SpecializationCache(_TierCache):
    self.guard_misses = 0
    self.retirements = 0
```

They are bare integers, incremented inline, read in tests as
`DEFAULT.specializations.compiles`. `tests/test_runtime.py` builds a `Delta`
fixture over four of them and asserts `compiles == 1`, `hits == 299`,
`guard_misses == 0` — **the thesis test**. This is good, and it is the seed of
everything below. What it cannot tell you:

- *Which* template recompiled (the counter is global, the key is discarded).
- *Where* the call came from (no stack).
- *How long* it took (no timing).
- *Which phase* of the compile was expensive (lower? render? `cc`?).
- Anything at all about events nobody thought to add a counter for.

Add a fifth counter and you have a fifth integer with the same five holes. The
pattern is asking to be generalized.

### 1.2 `no_compile()` — the right idea, once

```python
@contextmanager
def no_compile():
    """Assert that everything inside is a cache hit (the thesis, testable)."""
```

A `contextvars.ContextVar` that turns a tier-1 miss into `CompileForbidden`,
with `_explain(key)` naming *which key component* differed ("nearest entry
differs in: env_types"). This is a **performance assertion**: a statement that
an expensive event does not occur in this region, enforced in CI.

It is also hard-coded to exactly one event. There is no `no_guard_drift()`, no
`no_eviction()`, no `no_retirement()` — though each of those is the same bug
class, and a drifting guard in a render loop is arguably the *worst* of them
because it is invisible: the answer stays right, the cache silently recompiles
every frame, and no counter you are watching moves.

**The design test for this proposal:** if `no_compile()` cannot be
re-expressed as a one-line call on the generic mechanism, the generic mechanism
is wrong.

### 1.3 `bench.py` — instrumentation by monkeypatch

`bench.instrument()` decomposes the dispatch hot path into `key+probe /
extract / pack / launch`. It does so by **temporarily overwriting fields of a
live cache entry**:

```python
record.extract, record.launch = timed_extract, timed_launch
try:
    for _ in range(frames):
        registry.dispatch(target, args, out)
        ...
finally:
    record.extract, record.launch = orig_extract, orig_launch
```

Its own docstring frames this as a virtue ("zero kernel edits"), and given the
budget pressure of §6 that was a reasonable trade at step 10b. But it is at its
limit, and the seams show:

1. **It can only reach dataclass fields.** `extract` and `launch` happen to be
   `FastRecord` attributes, so they can be swapped. The genuinely expensive
   things — `lower_handle`, the `rewrite` decomposition pass, the two
   `run_stage` passes, `backend.render`, `backend.compile` (which for the C
   target **shells out to `cc`**) — are *locals inside `Registry._build`*. No
   monkeypatch can reach them. The entire miss path, which is where all the
   milliseconds are, is unobservable.

2. **It re-derives the cache key by hand.** `_warm_record()` reconstructs
   `key_for(target, arg_fp, backend.fp)` outside `dispatch`, duplicating the key
   logic. If `dispatch`'s key construction ever changes, this silently
   instruments the wrong record (or none).

3. **It fights the cache and loses.** A guard drift or a generation bump
   mid-loop rebuilds the record, dropping the shims — which the code detects
   only by a sentinel check and reports as
   `RuntimeError("record was rebuilt mid-instrument (guard drift?)")`. The tool
   for finding perf bugs *breaks* on the perf bug it should be reporting.

4. **It mutates shared state under a `try/finally`.** A `KeyboardInterrupt`
   between the two assignments leaves a timing shim wired into a production
   cache entry.

This is the "monkeypatch nonsense" to undo. The fix is not to make the wrapping
cleverer; it is to give the kernel a seam so no wrapping is needed.

### 1.4 What we can reuse

- **`_explain(key)`** already names the differing key component on a miss. It is
  O(cache size), so it must be *sampled*, but it is exactly the diagnostic a
  miss exemplar wants attached.
- **The key head** is `("H", code, env_fp)` — a code object. Any event carrying
  its key can name the offending **template**, by file and qualname, for free.
- **`viz.Timeline`** already renders labelled spans as static HTML. Nested
  spans (below) feed it directly; the flamegraph is nearly free.
- **`SourceSnapshot(text, filename, firstlineno, qualname)`** in `capture.py` is
  already the "structured source location" vocabulary. Traces should speak it.

---

## 2. The design in one paragraph

Two primitives. **(1) An event seam** in the kernel: `emit(name, key)` and a
nesting `span(name, key)` context manager, both dark by default — an empty sink
list costs one truthiness test. The kernel knows nothing about sampling, stacks,
aggregation, or reporting; a **recorder satellite** installs a sink and owns all
of that. **(2) A `Memo`** — a fingerprint-keyed cache that is instrumented *by
construction*, so every future "store the analysis instead of recomputing it"
gets hit/miss counts, timing, and stacks without its author remembering to ask.
The two existing cache tiers become the first two citizens of it.

And one invariant that makes the whole thing affordable:

> **Counts are exact and always on. Detail is sampled and armed.**

The existing integer counters stay exactly as they are — they are an increment,
they are free, `tests/test_runtime.py` depends on them, and they cannot lie.
The event system is a *second, dark layer* that, when armed, attaches timing,
keys, and stacks. `events.record()` never changes what `.compiles` reports.
This is what lets the tier-1 **hit** path keep counting hits without ever
calling into the seam.

---

## 3. The kernel seam

Complete proposed source (**41 tokenized lines**, measured with
`scripts/loc_budget.py`):

```python
"""The event seam (kernel side).

Expensive, should-be-rare occurrences announce themselves here. The kernel
knows nothing about sampling, stacks, or aggregation — a satellite installs a
sink. Dark by default: an empty ``SINKS`` costs one list truthiness test.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from time import perf_counter_ns

SINKS: list = []  # satellites append a callable (name, key, dur_ns, depth) -> None
_FORBID = contextvars.ContextVar("pdum_forbid", default=())
_DEPTH = contextvars.ContextVar("pdum_depth", default=0)


class EventForbidden(RuntimeError):
    """An event fired inside ``forbid()`` — the loop is not as hot as claimed."""


def _check(name: str, key: object) -> None:
    for pat in _FORBID.get():
        if name == pat or (pat.endswith("*") and name.startswith(pat[:-1])):
            raise EventForbidden(f"{name} fired under forbid({pat!r}); key={key!r}")


def emit(name: str, key: object = None, dur_ns: int = 0, depth: int = 0) -> None:
    """A point event: it happened, once, here."""
    if _FORBID.get():
        _check(name, key)
    for sink in SINKS:
        sink(name, key, dur_ns, depth)


@contextmanager
def span(name: str, key: object = None):
    """A timed event. Nests: ``depth`` gives the phase tree for free."""
    if _FORBID.get():
        _check(name, key)
    if not SINKS:
        yield
        return
    depth = _DEPTH.get()
    token = _DEPTH.set(depth + 1)
    t0 = perf_counter_ns()
    try:
        yield
    finally:
        _DEPTH.reset(token)
        dur = perf_counter_ns() - t0
        for sink in SINKS:
            sink(name, key, dur, depth)


@contextmanager
def forbid(*patterns: str):
    """Assert that none of ``patterns`` fires inside. ``no_compile()`` is one call."""
    token = _FORBID.set(_FORBID.get() + patterns)
    try:
        yield
    finally:
        _FORBID.reset(token)
```

Design notes, each of which is a decision:

- **The sink signature is `(name, key, dur_ns, depth)`** — four positional
  scalars, no allocation, no `Event` object in the kernel. The satellite decides
  what an event *is*. A kernel that defines an `Event` dataclass has already
  decided too much (and spent budget doing it).
- **`SINKS` is a plain module-global list, not a `ContextVar`.** This is
  deliberate and is the one place the design breaks symmetry with `no_compile`.
  See §8 (threading) — a compile can legally happen on *another thread* (the
  `_Slot` machinery exists precisely for that), and a `ContextVar` sink would
  silently miss it. The recorder must see cross-thread compiles.
- **`forbid` *is* a `ContextVar`**, preserving today's `no_compile` semantics
  exactly (scoped, nestable, async-safe).
- **`depth` gives the phase tree for free.** `spec.compile` contains `lower`,
  `render`, `artifact.compile`; the recorder reconstructs self-time vs total
  from the depth column without the kernel knowing what a tree is.
- **Wildcards are prefix-only** (`forbid("spec.*")`). One `str.startswith`, no
  regex, no `fnmatch` import.

Then `no_compile()` — the design test from §1.2 — becomes exactly:

```python
def no_compile():
    return forbid("spec.miss", "artifact.miss")
```

with `CompileForbidden = EventForbidden` kept as an alias so the book chapters,
`tests/test_runtime.py`, and any user code keep working unchanged.

---

## 4. The event vocabulary (hook points)

Dotted names, so `forbid("spec.*")` and per-name sampling policy both work.
Every event carries the **key** it pertains to where one exists, which is what
makes attribution ("*which* template?") possible.

| Event | Site | Why it is worth a stack |
| --- | --- | --- |
| `spec.miss` | `SpecializationCache.get_or_compile` | The thesis-violating event. Already counted (`misses`). |
| `spec.compile` | `Registry._build` (span) | Total miss-path cost. Parent of everything below. |
| `spec.hit` | `SpecializationCache.probe` | **Counted only, never emitted** — the hot path. See §5. |
| `guard.drift` | `SpecializationCache.probe`, guard branch | **The invisible one.** A capture rebound after decoration recompiles *every frame* with a correct answer and no symptom. Today: `guard_misses += 1`. |
| `cache.evict` | `_TierCache.get_or_compile`, LRU pop | Capacity thrash. You want the stack of whoever pushed it over. |
| `template.retire` | `_retire_superseded` | An edited-and-rerun template retiring its predecessor *in a loop* silently nukes tier 1. |
| `generation.bump` | `bump_generation` | The sledgehammer (024 §305). In a loop, catastrophic. |
| `lower` | `Registry._build` → `lower_handle` (span) | AST/`inspect` work. |
| `rewrite` | `Registry._build` → `rewrite`/`run_stage` (span) | Decomposition + the two ABI stages. |
| `render` | `Registry._build` → `backend.render` (span) | Source generation. |
| `artifact.miss` / `artifact.compile` | `ArtifactCache` (span) | The real money: for the C backend this **forks `cc`** (tens of ms). |
| `capture.snapshot` | `capture._take_snapshot` | `inspect.getsourcelines` = **file I/O**, memoized in a `WeakKeyDictionary`. A memo miss here in a hot loop is a genuine invisible cliff. |
| `fntype.miss` | `capture.make_handle`, `_FNTYPES` memo | Same story, plain dict. |

**Deliberately NOT hooked:** `KindTable.typeof` / `.fingerprint` (per-arg, per
call — the hot path), and tier-1 `hits`. These stay free forever. If you ever
want to know how much time goes there, that is `benchmark()`'s job, not the
recorder's.

---

## 5. The hot path, and why `bench.instrument` gets deleted

The uncomfortable question: `bench.instrument()` times the **hit** path
(`key+probe / extract / pack / launch`), and §4 just said the hit path must
never call the seam. How do we delete the monkeypatch without taxing dispatch?

Measured on this machine (Apple M-series, Python 3.14, `bench.benchmark`,
minimum-of-samples estimator; the *ratios* are the portable part, not the
absolute nanoseconds):

| | cost | as % of a warm dispatch |
| --- | --- | --- |
| **warm dispatch (tier-1 hit)** | **2.43 µs** | the budget |
| 4× dark `span()` context managers, inline in `dispatch` | 1966 ns | **81 %** |
| 4× no-op `mark()` function calls, inline in `dispatch` | 107 ns | 4.4 % |
| 1× `if SINKS:` branch at the top of `dispatch` | 29 ns | **1.2 %** |

Context managers on the hot path are **dead on arrival** — an 81% tax to
observe a path that is 100% observable by other means. Even no-op function-call
marks cost 4.4%, which on a per-frame path that the whole project exists to keep
tight is not obviously worth it.

**Recommendation: one branch.**

```python
def dispatch(self, target, args, out=None):
    if events.SINKS:                                    # 29 ns, 1.2%
        return self._dispatch_traced(target, args, out)
    ...                                                 # today's body, untouched
```

`_dispatch_traced` is a kernel-owned twin that runs the same four steps with
`perf_counter_ns()` between them and emits `dispatch.probe`, `dispatch.extract`,
`dispatch.pack`, `dispatch.launch`. It is ~8 lines. This buys:

- `bench.instrument()` loses `_warm_record`, both timing shims, the
  restore-in-`finally`, and the rebuilt-mid-instrument sentinel — it becomes
  *"arm a sink, run N frames, aggregate"*, and it is strictly more capable
  because the same run now also yields the miss-path phase tree.
- The instrumented frame is driven through the **real** `dispatch` entry point;
  no hand-replayed key derivation (bug #2 in §1.3 disappears by construction).
- A record rebuilt mid-loop is no longer an error — it is **an event**
  (`guard.drift`), which is the thing you wanted to know.
- `gpu_timeline()` keeps its `timed_call` artifact probe (that is an artifact
  capability, not a monkeypatch) but stops swapping `record.launch`; it reads
  the GPU split off the same sink.

**The one risk this introduces** is exactly the one `bench.py`'s docstring was
right to fear: two dispatch bodies can drift apart, and then you are measuring
code you do not run. Mitigate with a mandatory differential test —
`test_traced_dispatch_agrees`: for a matrix of targets (scalar, tuple-carry,
array-capture, C backend, WGSL), assert `dispatch(...) == _dispatch_traced(...)`
elementwise and that the emitted phase names cover every step the untraced body
performs. If someone edits one body and not the other, CI says so.

*(Alternative, if 1.2% is judged too dear: rebind `registry.dispatch` to the
traced twin at arm time — a kernel-owned method swap, zero cost when dark. It is
strictly faster and marginally less honest. The differential test is mandatory
either way; I would take the branch and the honesty.)*

---

## 6. The recorder satellite

Everything rich lives in `src/pdum/dsl/events.py` (satellite, alongside
`bench.py` and `viz.py`). The kernel imports nothing from it.

### 6.1 Structured tracebacks

The requirement is a **structured object, not a string**, and the budget for it
is generous because these events are expensive by definition.

```python
@dataclass(frozen=True, slots=True)
class Frame:
    code: CodeType          # the code object itself — hashable, already alive
    lineno: int

    @property
    def filename(self) -> str: return self.code.co_filename
    @property
    def qualname(self) -> str: return self.code.co_qualname
    @property
    def line(self) -> str: ...   # linecache — LAZY, only if rendered

@dataclass(frozen=True, slots=True)
class Trace:
    frames: tuple[Frame, ...]

    @property
    def user_frames(self) -> tuple[Frame, ...]:
        """Frames outside pdum/dsl — *your* loop, not our internals."""
```

Three decisions worth stating:

1. **Do not use `traceback.extract_stack()`.** It eagerly reads source files off
   disk to populate `FrameSummary.line`. That is the expensive half, and it is
   the half you almost never look at. Walk `sys._getframe()` yourself, keep
   `(code, lineno)`, and resolve filename/qualname/source lazily on render. A
   code object is hashable and is already kept alive by the function.

2. **Intern the traces.** The tuple of `(code, lineno)` pairs is hashable, so a
   `dict[tuple, Trace]` collapses ten thousand `guard.drift` events fired from
   one loop into **one** `Trace` object plus a count of 10 000. This is the
   fingerprinting idea of the whole project applied to stacks — and it is what
   makes "capture a full stack on every expensive event" cheap enough that
   sampling becomes an optimization rather than a necessity.

3. **Depth-cap at 32 frames** and offer `user_frames`. The internal frames are
   *constant* across every instance of a given event; the signal is the caller.

### 6.2 Sampling

Per **event name**, because the right policy differs by three orders of
magnitude between `artifact.compile` (rare, 50 ms each — trace 100% of them) and
`guard.drift` (potentially 10 000/frame when the bug bites — trace 1%).

- `every=N` — deterministic, every Nth. **The default**, because tests must be
  reproducible.
- `rate=p` — probabilistic, avoids lockstep aliasing with a periodic loop.
- `first=K, then=N` — capture the first K exemplars of a name, then decay to
  1-in-N. **The right default for a debugging tool**: you always get exemplars
  immediately (the common case is a handful of events), and you survive the
  firehose when a perf bug turns a rare event hot.
- `exemplars=8` — a reservoir cap per bucket, so a long run has bounded memory.

Note the interaction that makes this coherent: **counts remain exact under any
sampling policy** (§2), because the count is an increment in the sink and the
sampling decision only gates the stack walk and the `_explain` call.

### 6.3 Aggregation and the report

The sink accumulates `(name, key, depth) -> Bucket(count, total_ns, min_ns,
max_ns, traces, exemplars)`. `depth` reconstructs the tree.

```python
from pdum.dsl import events

with events.record() as ev:
    for i in range(300):
        make_shader(i * 0.001, 1.0)(0.5)

print(ev)
# event                count     total      mean   traces
# guard.drift            299    14.2ms    47.5µs        1   ← the bug
# spec.compile           299     3.10s    10.4ms        1
#   lower                299     412ms     1.4ms        1
#   rewrite              299      88ms     294µs        1
#   render               299      38ms     127µs        1
#   artifact.compile       1      52ms      52ms        1
# dispatch.launch        300     1.2ms     4.1µs        —
```

The indentation *is* the `depth` column. That report is the deliverable: it says
"you recompiled 299 times, here is where the time went, and here is the one
stack that caused all of it."

```python
ev["guard.drift"].count                 # 299 — exact, unsampled
t = ev["guard.drift"].traces[0]         # a Trace, interned, shared by all 299
[(f.qualname, f.filename, f.lineno) for f in t.user_frames]
ev["guard.drift"].by_key()              # → which template (key head = ("H", code, env_fp))
ev["spec.miss"].exemplars[0].explain    # "nearest entry differs in: env_types"  (sampled)
ev.top(3)                               # by total time
ev.timeline()                           # → viz.Timeline → static HTML flamegraph
```

### 6.4 Assertions — the regression gate

This is the part that earns its keep in CI:

```python
with events.forbid("spec.*"):                    # today's no_compile()
    ...
with events.forbid("guard.drift"):               # NEW: captures are not drifting
    ...
with events.expect(**{"spec.compile": 1, "guard.drift": 0}):   # exact budget
    ...
```

`forbid("guard.drift")` is a test you **cannot write today**, and it is the
canary for the entire bug class that motivated this document.

---

## 7. `Memo` — instrumented by construction

The second primitive, and the one aimed at the future: as analyses get
expensive (axis inference, shape propagation, egglog rewrites, provenance
maps), each will want to be keyed by a fingerprint and cached. If each such
cache is hand-rolled, each will hand-roll its own counter, or — far more
likely — none will, and the one that thrashes will be the one nobody
instrumented.

```python
class Memo:
    """A fingerprint-keyed cache that is instrumented by construction.
    Emits <name>.hit / <name>.miss / a <name>.compute span. LRU + capacity."""

    def __init__(self, name: str, capacity: int = 1024): ...
    def get(self, key: Hashable, compute: Callable[[], Any]) -> Any: ...

axes = Memo("axes.analysis")
info = axes.get(region.key, lambda: expensive_axis_analysis(region))
```

The claim that this abstraction carries weight is testable: **`_TierCache`
already *is* this**, and `ArtifactCache = _TierCache` literally. The refactor is
to make `_TierCache` a `Memo` (or a subclass), which means the two existing
tiers stop being special and every future analysis cache inherits their
bookkeeping, their LRU, their per-key futures, and now their observability. The
`WeakKeyDictionary` snapshot memo and the `_FNTYPES` dict in `capture.py` are
two more candidates that should become `Memo`s in the same pass — both are
already caches, neither is currently counted, and a miss in either is a file-I/O
or type-construction cost inside a per-frame decoration path.

*Scope note:* `Memo` needs `__len__`, capacity, and eviction; it does **not**
need the specialization tier's guards, generation, or template retirement, which
stay in `SpecializationCache`. Keep the base primitive small enough that the
budget below holds.

---

## 8. Threading, async, and re-entrancy

- **`SINKS` is global and lock-free on read.** Sinks are appended/removed at
  arm/disarm, not per event; the recorder's own accumulators take a lock. This
  is deliberate: `_Slot` means **thread B can be running the compile thread A is
  waiting on**, and a `ContextVar`-scoped sink would attribute that compile to
  nobody. An internal debugging tool that misses cross-thread compiles is
  useless precisely when concurrency is the bug.
- **`forbid` stays a `ContextVar`** — matching `no_compile` exactly, and
  correctly propagating into `asyncio` tasks (but *not* into threads, which is
  today's semantics and should not silently change).
- **Consequence to document loudly:** two concurrent `events.record()` blocks on
  different threads will interleave into one global sink. `record()` should
  therefore refuse to nest re-entrantly (raise, do not silently merge) — an
  explicit limitation, not an accident.
- **A failed compile is still expensive.** `span` must record on the exception
  path too (`try/finally`, as written), and the exemplar should carry the
  exception. A compile that takes 40 ms and *then* raises is a perf event.

---

## 9. The budget proposal

Architecture §6: "budgets are architecture." The kernel sits at **1147 / 1150
tokenized lines** — three lines of headroom — which is why `bench.py`
monkeypatches instead of asking for a seam. That trade has now been paid for
twice (an unobservable miss path, a tool that breaks on the bug it hunts), and
the honest move is to buy the seam rather than to keep being clever.

### 9.1 Proposed `scripts/loc_budget.py` changes

| Knob | Now | Proposed | Why |
| --- | --- | --- | --- |
| `KERNEL_TOTAL_CAP` | 1150 | **1235** | +85. Projected actual: ~1204 (below), leaving ~30 lines of deliberate headroom. |
| `FILE_CAPS["events.py"]` | — | **55** | New file. Prototype in §3 measures **41**; 55 allows the sink protocol to grow (a second wildcard form, a `disarm` guard) without another negotiation. |
| `FILE_CAPS["cache.py"]` | 165 | **175** | Currently 160. Adds ~10 lines of hooks (`guard.drift`, `cache.evict`, `template.retire`, `generation.bump`, the two miss events) and **removes** ~8 (the `_NO_COMPILE` ContextVar, `no_compile`, and the inline `CompileForbidden` raise all move to the seam). Net ≈ +6. |
| `FILE_CAPS["registry.py"]` | 110 | **125** | Currently 101. Adds the `if SINKS:` branch, `_dispatch_traced` (~8 lines), and the four `span()`s in `_build` (`lower`, `rewrite`, `render`, and the artifact-tier span). Net ≈ +10. |
| `FILE_CAPS["capture.py"]` | 85 | 85 (unchanged) | Currently 75. The two memo events fit in existing headroom. |
| `SATELLITE_CAPS["events.py"]` | — | **300** | The recorder: trace capture + interning, sampling policy, buckets, the report table, `record`/`expect`. Sized against `viz.py` (450) and `bench.py` (350). |
| `SATELLITE_CAPS["bench.py"]` | 350 | **300** | It should **shrink**: `_warm_record`, the two timing shims, the restore-in-`finally`, and the rebuilt-mid-instrument sentinel all die (~40 lines). Lowering the cap makes the removal load-bearing rather than aspirational. |

**Projected kernel total: 1147 + 41 (events.py) + 6 (cache.py) + 10
(registry.py) ≈ 1204 / 1235.**

### 9.2 What the +85 buys

Stated plainly, so it can be refused:

- Every expensive event in the system becomes countable, timeable, and
  attributable to a template and a stack — including the **entire miss path**,
  which is currently 100% dark and holds all of the milliseconds.
- `no_compile()` generalizes from one hard-coded event to *any* event, which
  turns "the captures aren't drifting" and "we aren't thrashing the LRU" into CI
  assertions.
- The `bench.py` monkeypatch — a documented fragility that mutates live cache
  entries and breaks under the exact conditions it is meant to diagnose — is
  deleted, not extended.
- Future fingerprint-keyed analyses (`Memo`) are observable by construction, so
  this negotiation does not recur every time someone adds a cache.

It costs **7.4%** of the kernel budget and **1.2%** of a warm dispatch.

### 9.3 If the budget is refused

The fallback is to keep the monkeypatch and add counters by hand, forever. That
is a real option and it is what the project has done so far. It should be chosen
consciously, with the knowledge that the miss path stays unobservable — because
no amount of external wrapping can reach a local variable inside `_build`.

---

## 10. Implementation plan (for the implementer)

Ordered, each step independently green.

1. **`kernel/events.py`** (§3) + `FILE_CAPS` entry. Unit tests: dark-path
   no-op, sink receives `(name, key, dur_ns, depth)`, `forbid` raises with a
   wildcard, `span` records on the exception path, `depth` nests.
2. **Re-express `no_compile()`** on `forbid` (§3), keeping `CompileForbidden` as
   an alias. **Gate: `tests/test_runtime.py` passes untouched.** If it does not,
   the seam is wrong — stop and revisit.
3. **Hook `cache.py`** (§4): `spec.miss`, `guard.drift`, `cache.evict`,
   `template.retire`, `generation.bump`, `artifact.miss`. Keep the integer
   counters exactly as they are (§2). Test: counters and events agree.
4. **Hook `registry._build`** with the four spans, and add the `if SINKS:`
   branch + `_dispatch_traced`. **Gate: `test_traced_dispatch_agrees`** (§5) —
   the anti-drift test; write it before the twin, not after.
5. **The recorder satellite** (§6): traces, interning, sampling, buckets,
   report. Test against a deliberately-drifting capture: assert
   `guard.drift.count == 299` and that `user_frames[0]` names the test's own
   loop line.
6. **Delete the `bench.py` monkeypatch** (§5): `instrument()` becomes a sink
   reader; lower `SATELLITE_CAPS["bench.py"]` to 300 so it cannot come back.
   Gate: `tests/test_bench.py` passes with the rewritten internals.
7. **`Memo`** (§7): extract from `_TierCache`, migrate the two tiers, then the
   `capture.py` snapshot/`_FNTYPES` memos.
8. **`viz` integration**: `ev.timeline()` → the existing `Timeline` widget.
9. **Book**: this belongs in `ch11b-measuring-the-machine.ipynb`, which already
   owns the "measuring" narrative and currently teaches `bench.instrument()`.

---

## 11. Open questions

1. **One source of truth, or two?** This proposal keeps the integer counters
   *and* adds events (§2), which means two places count `compiles`. The
   alternative — counters derived from the event system — is cleaner but forces
   the tier-1 **hit** path through the seam, which §5 measured at 81% (context
   manager) or 4.4% (mark call). I recommend two sources and a test that asserts
   they agree; the duplication is 6 integers and it keeps the hot path pristine.
2. **`if SINKS:` branch (1.2%) vs. arm-time method swap (0%)** — §5. I recommend
   the branch.
3. **Naming.** `events` vs `observe` vs `probe`. `events` reads best in
   `events.forbid("spec.*")` and `with events.record()`.
4. **Does `record()` need to be re-entrant?** §8 says no, and says so loudly.
5. **Sampling default.** `first=8, then=100` is proposed; it may want to be
   per-name from the start rather than a global default with per-name overrides.
6. **Should `Memo` subsume `SpecializationCache`'s guards/generation** (one
   primitive, more surface) or stay a small base with the tier as a subclass
   (§7)? Recommend the latter, on budget grounds.
