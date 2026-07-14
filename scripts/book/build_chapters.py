"""THE book builder: every chapter notebook is GENERATED from this file.

Run from the repo root:  uv run python scripts/book/build_chapters.py
Then re-execute:         ./scripts/test_notebooks.sh   (required after every regen)

This file is the single source of truth for docs/book/*.ipynb — edit
chapters HERE, never in the .ipynb (a regen overwrites it). It lived in
session scratchpads until step 13, when two parallel sessions edited the
same chapter and the fragmentation showed; now it is repo-owned so every
PR that touches a chapter touches the builder (the ch11b lesson).
"""

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


def gpu(src):
    """A code cell tagged `gpu`: executes only where an adapter answers; the
    harness skips it elsewhere and committed outputs survive (R17)."""
    return nbf.v4.new_code_cell(src, metadata={"tags": ["gpu"]})


def notebook(cells, path):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = cells
    nbf.write(nb, path)
    print("wrote", path)


# ══════════════════════════════════════════ ch00 ══════════════════════════════

ch00 = [
    md("""\
# Chapter 0 — The thesis

`pdum.dsl` is a Python DSL compiler framework: a numba-like `@jit` workflow with
Julia-like **type-keyed caching**. This book is its documentation, written
**bottom-up**: one chapter per implementation step, each chapter exposing the
internals built in that step — not just the API. Run every cell; poke at
everything. Source references point into `src/pdum/dsl/`; the design rationale
lives in `docs/design/010_proposed-architecture.md`, and the vocabulary in
[`GLOSSARY.md`](GLOSSARY.md).

This chapter contains **no new code**. It states the one idea the whole system
is built around, and demonstrates it with the frozen proof of concept
(Milestone 0, preserved at `pdum.dsl_reference` — see `reference/README.md`),
so you can see the destination before we lay the first brick."""),
    md("""\
## The thesis

A closure is **(code identity, typed environment, environment values)** — and
compilation should be keyed on the *types*, never the *values*:

| Phase | When | What happens |
|---|---|---|
| **A — capture** | at `@jit` decoration (every closure rebuild) | read the code object + captured values → summarize their **types**. No compilation, ever. |
| **B — call** | at draw/call time | cache key = (code identity, env types, arg types, …). **Miss**: compile once. **Hit**: re-marshal the current values and launch. |

The payoff is the cost model: building a GPU pipeline costs ~1–10 ms;
rewriting its uniform buffer costs ~0.1 ms. If the cache keys on types, a
render loop that rebuilds its closure every frame with *new parameter values*
pays the 0.1 ms path — moving a knob is a buffer write, not a recompile.
That's the entire product: **the loop stays hot**.

Nobody else's default works this way: numba freezes captured values into the
compiled code as constants, and JAX bakes closure captures into the trace —
in both, `closure(5)` and `closure(6)` are different compiled programs. (The
receipts are in `docs/design/022_closure_specialization.md` and
`docs/design/research/R4-jax.md`.)"""),
    code("""\
# The frozen proof of concept, kept importable precisely so this chapter can run it.
from pdum.dsl_reference import builtins, jit


def disk(cx, cy, radius):
    @jit(kind="fragment")  # M0's own kind vocabulary — the frozen reference keeps its names
    def shader():
        x, y = builtins.FragCoord.xy
        dx = x - cx
        dy = y - cy
        d2 = dx * dx + dy * dy
        return (1.0, 0.5, 0.0) if d2 < radius * radius else (0.05, 0.05, 0.12)

    return shader


h = disk(320.0, 240.0, 70.0)
print("the Handle:          ", h)
print("its structural type: ", h.fntype)
print("its captured values: ", h.env)"""),
    md("""\
`@jit` did **not** compile anything — phase A only read the closure's code
object and cells. The `Handle` holds the two halves the thesis separates: a
structural *type* (into cache keys) and the captured *values* (never in a
key; destined for the uniform buffer)."""),
    code("""\
h5, h6 = disk(100.0, 100.0, 50.0), disk(300.0, 200.0, 50.0)
print("different values?    ", h5.env != h6.env)
print("same TYPE?           ", h5.fntype == h6.fntype, "  <- one compilation, shared")

h_int = disk(100.0, 100.0, 50)  # radius is an int this time
print("int radius same type?", h_int.fntype == h5.fntype, " <- a capture changed TYPE: its own compilation")"""),
    code("""\
import math

HAS_GPU = True
try:
    from pdum.dsl_reference.webgpu import Context

    ctx = Context()
    drawer = ctx.offscreen_drawer(size=(320, 240))
except (ImportError, RuntimeError) as e:  # no adapter / no wgpu — anything else should FAIL the chapter
    HAS_GPU = False
    print(f"no GPU adapter available ({type(e).__name__}); skipping the live demo")

if HAS_GPU:
    for k in range(120):  # rebuild the closure EVERY frame with fresh values
        t = k * 0.05
        drawer.update(disk(160 + 80 * math.cos(t), 120 + 60 * math.sin(t), 45.0))
        drawer.show()
    print(f"frames={drawer.uniform_writes}  compiles={drawer.compile_count}")
    assert drawer.compile_count == 1, "the thesis, asserted"
"""),
    code('''\
def rgba_png(pixels: bytes, w: int, h: int) -> bytes:
    """Raw RGBA bytes -> PNG, stdlib only (15 lines, because this repo adds no deps lightly)."""
    import struct
    import zlib

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    raw = b"".join(b"\\x00" + pixels[y * w * 4 : (y + 1) * w * 4] for y in range(h))
    return b"\\x89PNG\\r\\n\\x1a\\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


if HAS_GPU:
    from IPython.display import Image, display

    w, hh = drawer.target.size
    display(Image(data=rgba_png(drawer.target.read_pixels(), w, hh)))'''),
    md("""\
## What M0 got wrong — and what the rebuild does about it

The proof of concept proved the thesis but earned a redesign
(`docs/design/010_proposed-architecture.md`, built from the research corpus in
`docs/design/research/`). The load-bearing faults, each now a structural fix:

| M0 fault | The rebuild's answer |
|---|---|
| `flatten()` re-parsed and re-lowered ASTs **every frame** just to collect values | a precompiled per-entry extractor; the hot path is key build + pack + launch, no AST/IR objects exist on it |
| captured uniforms were **scalars only** (no `center=(cx, cy)`) | a marshaling layer: one logical value → N physical slots, planned once from types |
| the core **imported the WGSL backend's tables** | five registration surfaces on one explicit Registry; backends are capability records |
| the expression-tree IR **could not grow** (no `if`/`for`, arrays, transforms) | one frozen `Node`/`Region` micro-IR, three region ops, everything else is rewrite rules |
| backend params (target format) **missing from the cache key** | a two-tier cache with a key-completeness perturbation test |
| nothing could hold a capture value *wrongly* — by discipline only | the IR has **no field that can hold a value**; the anti-pattern is unrepresentable |"""),
    md("""\
## The map of the book

| Chapter | Step builds | Kernel modules |
|---|---|---|
| **ch00 — The thesis** *(you are here)* | scaffolding, budgets, this map | — |
| ch01 — Types are values | the type lattice, `typeof`, fingerprints | `types.py`, `valuekind.py` |
| ch02 — What a closure is | phase-A capture | `capture.py` |
| ch03 — One compile per signature | the two-tier cache (thesis proven with dummy artifacts) | `cache.py` |
| ch04 — Pipelines are values | combinator syntax + roles (satellite: zero kernel edits) | `combinators.py` |
| ch05 — Programs are values | the IR | `ir.py`, `ops.py`, `printer.py` |
| ch06 — Everything is a rule | the rewrite engine | `rewrite.py` |
| ch07 — Source to IR | lowering | `lower.py` + rule packs |
| ch08 — One value, N parameters | marshaling | `pack.py` |
| ch09 — End to end on the CPU | the Python backend + the hot path | `backends/python.py`, `api.py` |
| ch10 — The GPU and the seam | the WGSL backend; **M1 complete** | `backends/wgsl/` |
| ch11 — The five surfaces | batteries, records, registry | `stdlib/` |
| ch12+ | arrays/loops/C · `over`/jvp · tensors on CPU · grad · CUDA/Metal + tiles · units (020, re-sequenced at the 130 fork) | per plan |

Budgets are architecture: the kernel is CI-capped at 1150 counted lines
(`scripts/loc_budget.py`), and "a new capability lands with zero kernel edits"
will itself become a CI test. Every chapter reports the running total.

**Next:** [Chapter 1 — Types are values](ch01-types-are-values.ipynb)."""),
]

# ══════════════════════════════════════════ ch01 ══════════════════════════════

ch01 = [
    md("""\
# Chapter 1 — Types are values

Step 1 builds the vocabulary every cache key will be written in.

**Source built this step** (with counted-line budgets from `scripts/loc_budget.py`):

| File | Counted lines / cap | What |
|---|---|---|
| `src/pdum/dsl/kernel/types.py` | 94 / 100 | the structural `Type` lattice + `TemplateId` code identity |
| `src/pdum/dsl/kernel/valuekind.py` | 61 / 95 | `typeof` + `fingerprint` (the marshaling half arrives in ch08) |

Tests: `tests/test_types.py`, `tests/test_valuekind.py` (including the
soundness fuzz), `tests/test_budgets.py`. Glossary terms introduced: *Type,
summary function, typeof, fingerprint, ValueKind, TemplateId, FnType, Literal
lift, Tuple-vs-Vec* — see [GLOSSARY.md](GLOSSARY.md)."""),
    code("""\
from pdum.dsl.kernel import types as T

color = T.Record("Color", (("r", T.f32), ("g", T.f32), ("b", T.f32)))
examples = [
    T.f64,
    T.Tuple((T.f64, T.f64)),
    T.Vec(T.f32, 3),
    T.Array(T.f32, 2, "C", "<", True),
    color,
    T.LiteralType(T.i64, 8),
    T.Array(T.f32, 2, "C", "<", False),
]
for t in examples:
    print(f"{t!r:<40} hash={hash(t) & 0xFFFF:04x}")"""),
    md("""\
A `Type` is a **frozen dataclass compared structurally** — two independently
constructed types are equal, hash equal, and collide in a dict on purpose.
That property is the entire reason they can serve as cache keys. Note
`LiteralType`: the *one* place a value is allowed inside a type, and only by
explicit opt-in (the future `Literal` lift — bake-as-constant, recompile per
value)."""),
    code("""\
import dataclasses

print("equal?     ", T.Scalar("f64") == T.f64)
print("same obj?  ", T.Scalar("f64") is T.f64)

artifact_cache = {T.Vec(T.f32, 3): "<pretend compiled pipeline>"}
print("dict probe:", artifact_cache[T.Vec(T.f32, 3)])

# Literal lifts are TYPE-AWARE about their value: Python's == is cross-type
# (1 == 1.0 == True), which would merge distinct baked constants into one key.
print("Literal 1 vs 1.0:", T.LiteralType(T.f64, 1) == T.LiteralType(T.f64, 1.0))

try:
    T.f64.kind = "f32"
except dataclasses.FrozenInstanceError as e:
    print("frozen:    ", e)"""),
    code("""\
# Validation is loud at construction — a malformed type must never reach a key.
for build in (lambda: T.Scalar("f16"), lambda: T.Vec(T.f32, 5), lambda: T.Vec(T.Vec(T.f32, 2), 2)):
    try:
        build()
    except Exception as e:
        print(f"{type(e).__name__}: {e}")"""),
    md("""\
## `typeof` is a summary function, not a class lookup

The question "what is this value's type" is answered per-Python-type by a
registered **ValueKind**, and the kind *chooses* how much of the value its
summary records (architecture §13). The built-in int kind is the canonical
example: it reads the value's **range** — `5` and `2**63` get different
types, because packing them into the same 8 bytes would corrupt one of them.
Later, an opt-in array kind will put *shapes* in the type the same way. The
dial exists per kind; nothing is global."""),
    code("""\
from pdum.dsl.kernel.valuekind import BigIntError, fingerprint, typeof

print(typeof(True), "|", typeof(5), "|", typeof(1.5))
print(typeof(2**63 - 1), "vs", typeof(2**63), "  <- same Python class, different summaries")
try:
    typeof(2**70)
except BigIntError as e:
    print("BigIntError:", e)

# Unregistered types are LOUD — guessing would put an unsound key in the cache.
try:
    typeof(object())
except TypeError as e:
    print("TypeError:  ", e)"""),
    code("""\
# Python tuples summarize ELEMENT-WISE to Tuple — honestly, including
# heterogeneous and nested ones. This is the `center = (cx, cy)` capture that
# M0 could not express, with no vec interpretation smuggled in.
print(typeof((1.0, 2.0)), "|", typeof((1, 2.0)), "|", typeof(((1.0, 2.0), 3)))
print("arity is part of the identity:", typeof((1, 2)) != typeof((1, 2, 3)))

# Loudness comes only from elements:
try:
    typeof((1.0, 2**70))
except BigIntError as e:
    print("an element error surfaces:", e)"""),
    md("""\
> **Resolved at this chapter's walkthrough.** An earlier draft of this step
> summarized homogeneous scalar tuples as `Vec` — a shader-language
> interpretation leaking into the identity layer. The rule is now: **`typeof`
> produces `Tuple`, never `Vec`.** `Vec` stays in the lattice purely as an
> *IR-level* type that dialect lowering rules produce (a 3-tuple literal in a
> shader's return position becomes `core.vec`), and whether a captured
> `Tuple((f64, f64))` is *packed* as one `vec2<f32>` uniform or two scalar
> slots is the backend's PackPlan decision (ch08/ch10). This is the same split
> M0 documented as "two type levels" — the ledger entry lives in
> `docs/design/010_proposed-architecture.md` §10, and this note stays in the book as
> provenance of the correction."""),
    code("""\
import timeit

v = (1.0, 2.5, 3.5)
print("fingerprint:", fingerprint(v))
print("typeof:     ", typeof(v))

n = 100_000
def best(fn):  # min-of-repeats is standard timeit practice; single runs have ±30% noise
    return min(timeit.repeat(lambda: fn(v), number=n, repeat=5)) / n * 1e9
print(f"fingerprint {best(fingerprint):6.0f} ns/call   typeof {best(typeof):6.0f} ns/call   (best of 5)")"""),
    md("""\
The **fingerprint** is the hot-path stand-in for `typeof`: it must be cheap
(it runs on every call in the future dispatch path) and it is bound by the
**soundness law** — `fingerprint(a) == fingerprint(b)` must imply the same
`typeof` outcome. A fingerprint collision would be a *silent wrong cache
hit*: the worst failure class in the whole system, worse than a crash.

Notice the two calls above measure **nearly the same** today — expected, not
alarming. For scalar-family kinds the summary *is* cheap: interned singletons
plus one frozen-dataclass wrap. The fingerprint earns its keep on kinds where
building the full `Type` is real work — the array kind (`typeof` reads
dtype/flags/layout and constructs an `Array`; the fingerprint is a small
precomputed tag) and the `Handle` kind (`typeof` builds an `FnType`) — which
arrive in later chapters. Capture fingerprints are also computed once at
phase A and memoized on the `Handle`; only *argument* fingerprints run per
call. The step-9 microbenchmark gate decides these trade-offs with data — one
live option is `fingerprint := the Type itself` for cheap kinds, making
soundness trivially true.

The soundness law is enforced by a property fuzz in CI; here it is, live:"""),
    code("""\
import random

from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.valuekind import BUILTINS


def soundness_fuzz(table, n=2000, seed=20260711):
    rng = random.Random(seed)

    def value(depth=0):
        r = rng.random()
        if r < 0.15:
            return rng.choice([True, False])
        if r < 0.45:
            span = rng.choice([2**8, 2**62, 2**63, 2**64, 2**66])
            return rng.randrange(-span, span)
        if r < 0.7 or depth >= 2:
            return rng.uniform(-1e9, 1e9)
        return tuple(value(depth + 1) for _ in range(rng.randrange(0, 6)))

    # (This generator has a CI twin in tests/test_valuekind.py — grow BOTH
    # when the value universe grows.)
    def outcome(fn, v):
        try:
            return ("ok", fn(v))
        except TypeError as e:  # BigIntError subclasses TypeError
            return ("raise", type(e).__name__)

    seen = {}
    for _ in range(n):
        v = value()
        fp, ty = outcome(table.fingerprint, v), outcome(table.typeof, v)
        if fp[0] == "raise":
            assert ty[0] == "raise", f"fingerprint raised but typeof didn't for {v!r}"
            continue
        prior = seen.setdefault(fp, (v, ty))
        if prior[1] != ty:
            return f"SOUNDNESS VIOLATION: {prior[0]!r} and {v!r} share {fp} but differ: {prior[1]} vs {ty}"
    return f"sound over {n} random values"


print(soundness_fuzz(BUILTINS))"""),
    code("""\
# Now break the law on purpose: a float kind whose fingerprint collides with
# ints. Kinds receive the dispatching TABLE on every call — that is what lets
# BUILTINS.extend() overrides reach elements nested inside tuples too.
class SloppyFloatKind:
    def typeof(self, v, table):
        return T.f64

    def fingerprint(self, v, table):
        return "i64"  # deliberately wrong: collides with the int bucket tag

    def flatten(self, v, table):  # the marshaling view (ch08); part of the contract
        return (v,)


sloppy = BUILTINS.extend()  # a layered child table; BUILTINS itself is untouched
sloppy.register(float, SloppyFloatKind())
print("override reaches nested floats:", sloppy.fingerprint((1.5, 2.5)))

print(soundness_fuzz(sloppy))"""),
    md("""\
That violation, in the real system, would have been `closure(1.5)` silently
reusing the artifact compiled for `closure(1)` — packing a float into an i32
uniform slot. The fuzzer exists so that enriching a type (say, adding shapes)
while forgetting the fingerprint fails CI instead of corrupting a buffer."""),
    code("""\
# Code identity: CPython compares code objects BY VALUE. This single fact is
# the live-coding story — an unchanged re-run hits the cache, an edit misses.
SRC = "def f(x):\\n    return x + k\\n"


def code_of(src):
    ns = {}
    exec(compile(src, "<notebook-cell>", "exec"), ns)
    return ns["f"].__code__


a, b = code_of(SRC), code_of(SRC)
print("same source, two compiles:   a is b ->", a is b, "  a == b ->", a == b)
print("after a one-token edit:      ", a == code_of("def f(x):\\n    return x + k + 1\\n"))

base = T.Base(a)
grad = T.Derived("grad", base, (("wrt", 0),))
print(base, "|", grad, "| collide? ->", base == grad)

# FnType — the thesis in one value (ch02 will build these from real closures):
print(T.FnType(base, (T.i64,)), "  rebuilt-equal ->", T.FnType(T.Base(b), (T.i64,)) == T.FnType(base, (T.i64,)))"""),
    md("""\
## What we can't do yet

- `typeof(None)`, arrays, records, `Handle`s — kinds arrive with their users
  (arrays/records in the stdlib around ch11–11, `Handle` in ch02).
- Nothing *produces* an `FnType` from a real closure yet — that is phase-A
  capture, **ch02**.
- `leaf_types`/`flatten` (the marshaling views of a `ValueKind`) — ch08.
- The kind table is a module-level builtin seed; it folds into the explicit
  `Registry` (surface C) in ch09.

**Budget after step 1:** kernel 155 / 1150 counted lines.

**Next:** ch02 — what a closure is: `make_handle`, snapshots, and `FnType`s
built from live closure cells."""),
]


# ══════════════════════════════════════════ ch02 ══════════════════════════════

ch02 = [
    md("""\
# Chapter 2 — What a closure is

Step 2 builds **phase A**: the reflection that turns a just-defined Python
function into a `Handle` — the thesis's `(code identity, typed environment,
environment values)` made concrete. Phase A runs on *every closure rebuild*,
i.e. every iteration of your hot loop, so its contract is strict: read the
code object, read the cells, summarize types via memoized fingerprints —
**no parse, no IR, no compile, ever** — and never fail on missing source
(that's phase B's loud error, ch07). It *does* fail loudly on an untypeable
capture: better at the def site than inside a cache key.

| File | Counted lines / cap | What |
|---|---|---|
| `src/pdum/dsl/kernel/capture.py` | 67 / 85 | `make_handle`, `Handle`, `SourceSnapshot`, the Handle ValueKind |
| `src/pdum/dsl/kernel/api.py` | 8 / 50 | `@jit` (capture only; the call path arrives in ch09) |

Glossary terms settled this chapter: *Handle, SourceSnapshot* — see
[GLOSSARY.md](GLOSSARY.md)."""),
    code("""\
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.capture import make_handle


def make_closure(x):
    @jit(kind="device")
    def inner(y):
        return x + y

    return inner


f5, f6 = make_closure(5), make_closure(6)
print("a Handle:            ", f5)
print("identities equal:    ", f5.fntype == f6.fntype, "  <- values differ, ONE specialization")
print("envs (runtime data): ", f5.env, "|", f6.env)
print("fingerprints equal:  ", f5.fp == f6.fp)
print("float capture:       ", make_closure(3.0).fntype == f5.fntype, " <- type changed: its own identity")"""),
    code("""\
# Kernel objects also render RICHLY in Jupyter and marimo — the viz satellite
# (zero kernel edits; docs/design/060_rendering-notes.md). Hover the pills;
# expand the source. Static HTML only: interactive like a diagram, not an app.
from pdum.dsl import viz

viz.install()
f5"""),
    md("""\
What `@jit` actually did: looked up the memoized `SourceSnapshot` for the code
object (taken once, at first decoration, while `linecache` is coherent),
read the closure cells in `co_freevars` order, fingerprinted each capture,
and looked up the memoized `FnType`. Everything after the first decoration
of a given code object is dictionary probes. Let's price it:"""),
    code("""\
import timeit

n = 50_000
t = min(timeit.repeat(lambda: make_closure(5), number=n, repeat=5)) / n * 1e9
print(f"phase A, memo-warm: {t:6.0f} ns per closure rebuild")
# Rebuilding a 300-module network every training step at this price is
# sub-millisecond — the claim from the NN detour, now measurable."""),
    code("""\
def make_shader(center):
    @jit(kind="simple_shader.fragment")
    def shader():
        return center

    return shader


h = make_shader((320.0, 240.0))
print(h.fntype, "  <- a Tuple inside the identity: M0's top gap, closed at capture")

# CAREFUL: only enclosing-FUNCTION names become closure cells. A module- or
# cell-level name is a GLOBAL — phase A does not capture it:
center = (320.0, 240.0)


@jit(kind="simple_shader.fragment")
def shader_global():
    return center


print("global 'center' captured?", shader_global.freevars, shader_global.fntype)
# Globals are a different mechanism: classify_names decides their fate at
# lowering (ch07), and per-call guards defend against drift (ch09)."""),
    code("""\
# Live-coding invalidation, now through real Handles: an unchanged re-run
# produces a VALUE-EQUAL code object (hit); a one-token edit misses.
SRC = "def f(k):\\n    def g():\\n        return k\\n    return g\\n"


def build(source):
    ns = {}
    exec(compile(source, "<cell>", "exec"), ns)
    return make_handle(ns["f"](7), "device")


print("re-run unchanged:", build(SRC).fntype == build(SRC).fntype)
print("after an edit:   ", build(SRC).fntype == build(SRC.replace("return k", "return k + 1")).fntype)"""),
    md("""\
## The program is the parameter container

Composition is structural: a capture that is itself a `Handle` contributes
its `FnType` to the parent's `env_types`. Build a network out of small
closures each owning its own weights, rebuild the whole object graph every
step with fresh values — the root identity never moves. (Neither JAX nor
PyTorch can offer this: one retraces on rebuilt closures, the other needs
stateful modules. See the desiderata §2.1 note.)"""),
    code("""\
def dense(w, b):
    @jit()
    def layer(x):
        return w * x + b

    return layer


def net(w1, b1, w2, b2):
    inner = dense(w1, b1)

    @jit()
    def outer(x):
        return inner(x) * w2 + b2

    return outer


a = net(1.0, 2.0, 3.0, 4.0)
print("composed identity: ", a.fntype)
print("env_types:         ", a.env_types, "  <- the child's FnType is INSIDE the identity")
print("new weights, hit?  ", net(0.9, 1.9, 2.9, 3.9).fntype == a.fntype)"""),
    code("""\
from pdum.dsl.kernel.valuekind import typeof

# Handles are first-class values of the type system itself:
h = make_closure(5)
print(typeof(h))
print(typeof((h, make_closure(6))), "  <- a tuple of layers: identities compose")"""),
    md("""\
## Pricing the compositional model: a 100-block mock transformer

The claim from the NN detour (desiderata §2.1, working notes in
`docs/design/030_deep-learning-notes.md`): identity is built **bottom-up**, each
Handle carrying the precomputed digest of its subtree, so a parent
incorporates a child in O(1) — no re-traversal. Two consequences to
*measure*, not assert: per-step rebuild cost scales with module count, and an
**unchanged subtree** (a frozen backbone) can be reused across steps
essentially for free. The "attention" below is a mock — no tensor ops exist
yet; what we are pricing is pure identity construction (phase A), which is
exactly what a training loop pays per step before any math runs.

Watch the **ns-per-handle column** in the output: fingerprint *construction*
is O(1) per child, but Python re-hashes nested fingerprint tuples on dict
probes (tuples don't memoize their hash), so deep spines drift mildly
superlinear. That drift is a known, pre-shaped escalation — `Handle.fp`
becomes a flat memoized digest (the `Node.key` technique) if the step-9
microbench gate ever demands it — and the frozen-backbone pattern below makes
it moot for the common fine-tuning case anyway."""),
    code("""\
import timeit


def attention(wq, wk, wv, wo):
    @jit()
    def attn(x):
        return wq * x + wk * x + wv * x + wo  # mocked; real tensor ops arrive much later

    return attn


def mlp(w1, w2):
    @jit()
    def ff(x):
        return w2 * (w1 * x)

    return ff


def block(p):
    a, m = attention(p, p, p, p), mlp(p, p)

    @jit()
    def blk(x):
        return m(a(x))

    return blk


def compose(f, g):
    @jit()
    def seq(x):
        return g(f(x))

    return seq


def transformer(weights):
    model = block(weights[0])
    for p in weights[1:]:
        model = compose(model, block(p))
    return model


weights = [float(i) for i in range(100)]
model = transformer(weights)
print("root identity (truncated):", repr(model.fntype)[:90], "...")
print("rebuilt with new weights, same identity:",
      transformer([w + 0.5 for w in weights]).fntype == model.fntype)
print()
for n in (10, 50, 100):
    handles = 4 * n - 1  # attn + ff + blk per block, plus n-1 composes
    t = min(timeit.repeat(lambda: transformer(weights[:n]), number=5, repeat=3)) / 5
    print(f"{n:4d} blocks: {t * 1e6:7.0f} µs per full rebuild   ({t / handles * 1e9:5.0f} ns per handle)")"""),
    code("""\
# The unchanged-subtree trick: a frozen backbone is built ONCE, outside the
# loop; each "training step" rebuilds only the trainable head and one compose.
# Identity is bit-identical either way — phase A cost collapses to the spine
# that actually changed. (Fine-tuning and adapters fall out of the model.)
backbone = transformer(weights[:99])


def step_full_rebuild():
    return compose(transformer(weights[:99]), block(weights[99]))


def step_frozen_backbone():
    return compose(backbone, block(weights[99]))


assert step_full_rebuild().fntype == step_frozen_backbone().fntype

t_full = min(timeit.repeat(step_full_rebuild, number=5, repeat=3)) / 5
t_frozen = min(timeit.repeat(step_frozen_backbone, number=5, repeat=3)) / 5
print(f"rebuild all 100 blocks: {t_full * 1e6:7.0f} µs/step")
print(f"frozen backbone:        {t_frozen * 1e6:7.0f} µs/step   ({t_full / t_frozen:4.0f}x cheaper, same identity)")"""),
    code("""\
# Two edge cases phase A must survive:

# 1. A self-referential closure's own cell is EMPTY at decoration time.
captured = {}


def grab(fn):
    captured["h"] = make_handle(fn, "device")
    return fn


def factory():
    @grab
    def rec(n):
        return 1 if n == 0 else rec(n - 1)

    return rec


factory()
h = captured["h"]
print("self-cell skipped:", h.env, "| template still knows the name:", h.freevars)

# 2. Source unavailable (some REPLs): phase A succeeds; the snapshot is None
#    and phase B (ch07) raises the loud NoSourceError if lowering is attempted.
ns = {}
exec(compile(SRC, "<no-file>", "exec"), ns)
print("no source -> snapshot:", make_handle(ns["f"](1), "device").snapshot)"""),
    code("""\
# The snapshot: decoration-time source, memoized per code object. Taking it
# EAGERLY is the first half of the stale-source defense (a later on-disk edit
# cannot retroactively change what we captured); phase B's coherence check
# (ch07) is the second half.
snap = f5.snapshot
print(snap.qualname, "@", snap.filename, "line", snap.firstlineno)
print(snap.text)
print("memoized:", make_closure(1).snapshot is make_closure(2).snapshot)"""),
    md("""\
## What we can't do yet

- **Calling a Handle does nothing yet** — there is no cache to dispatch into
  (ch03) and no backend to run on (ch09). A Handle is pure identity + data.
- Lowering the body (`snap.text` → typed IR) is ch07; the snapshot coherence
  check lives there.
- The `FnType` memo is unbounded — eviction policy arrives with the cache
  tier (ch03), per the hazard doc's L-cache warning.

**Budget after step 2:** kernel 230 / 1150 counted lines.

**Next:** ch03 — one compile per signature: the two-tier cache, proven with
dummy artifacts before any compiler exists."""),
]


# ══════════════════════════════════════════ ch03 ══════════════════════════════

ch03 = [
    md("""\
# Chapter 3 — One compile per signature

Step 3 builds the **two-tier cache** — and with it, Phase I's promise comes
due: the thesis is provable end-to-end **before any compiler exists**. The
"artifacts" in this chapter are fake strings; everything else — keys, guards,
generations, eviction, retirement, thread-safety — is the real machinery the
GPU pipelines will live in from ch09 on.

```
tier 1  SPECIALIZATION    (fp_head, arg_fp, backend_fp, generation) -> FastRecord
                   |  miss only
tier 2  ARTIFACT  (content_key, backend_token, flags) -> compiled artifact
                   |  miss only            (content-addressed, generation-FREE)
                   v
              lower -> rewrite -> render -> backend compile   (chapters 4-8)
```

| File | Counted lines / cap | What |
|---|---|---|
| `src/pdum/dsl/kernel/cache.py` | 147 / 150 | both tiers, futures, guards, LRU, retirement, `no_compile`, `explain_miss` |

Glossary terms settled: *specialization cache, artifact cache, generation, guard,
FastRecord (partially — its hit-path fields fill in ch08/ch09).*"""),
    code("""\
import itertools

from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.cache import ArtifactCache, CompileForbidden, FastRecord, SpecializationCache, no_compile
from pdum.dsl.kernel.valuekind import fingerprint

cache = SpecializationCache()
serial = itertools.count(1)


def fake_gpu_compile(handle):
    # Stand-in for lower -> rewrite -> render -> backend.compile (ch05-ch09).
    return f"<pipeline #{next(serial)} for {handle.fntype!r}>"


def draw(handle, *args, backend=("fake-gpu", "rgba8unorm")):
    key = cache.key_for(handle, tuple(fingerprint(a) for a in args), backend)
    rec = cache.get_or_compile(key, lambda: FastRecord(artifact=fake_gpu_compile(handle)))
    return rec.artifact


def disk(cx, cy, radius):
    @jit(kind="simple_shader.fragment")
    def shader():
        return (cx, cy, radius)  # the body is irrelevant until ch07 — identity is what's real

    return shader


print(draw(disk(100.0, 100.0, 50.0)))
print(draw(disk(300.0, 200.0, 75.0)))
print(f"compiles={cache.compiles}  hits={cache.hits}")"""),
    md("""\
The key is four components, and **no captured value appears in any of them**:
`fp_head` is the Handle's precomputed `("H", code, env_fp)` digest (template
identity + capture *types*), then argument fingerprints, the backend token +
codegen-relevant params (M0's missing-format fault, cured structurally), and
the generation. Values ride in `Handle.env`, headed for uniform buffers —
never through here."""),
    code("""\
# The M0 acceptance test, on the new kernel, with no compiler in existence.
import math

with no_compile():  # from here on, ANY compile raises — the loop must be hot
    for k in range(300):
        t = k * 0.05
        draw(disk(320 + 180 * math.cos(t), 240 + 120 * math.sin(t), 70.0))

print(f"rebuilds=300  compiles={cache.compiles}  hits={cache.hits}")
assert cache.compiles == 1, "one compile, three hundred value-fresh frames"
"""),
    code("""\
from pdum.dsl import viz

viz.install()
cache  # the cache renders its counters as pills — watch them across the chapter"""),
    code("""\
# Perturb each key component; under no_compile() the miss RAISES, and the
# error NAMES the nearest entry's differing component. Key completeness is
# an experience, not a comment.
probes = {
    "a capture turns int": lambda: draw(disk(100, 100.0, 50.0)),
    "an argument appears": lambda: draw(disk(100.0, 100.0, 50.0), 1),
    "backend changes":     lambda: draw(disk(100.0, 100.0, 50.0), backend=("webgpu", "bgra8")),
}
for what, probe in probes.items():
    try:
        with no_compile():
            probe()
    except CompileForbidden as e:
        print(f"{what:22s} -> {e}")"""),
    code("""\
# Live-coding hygiene: an EDIT retires the predecessor's entries (no leak),
# and bump_generation() is the sledgehammer that clears tier 1 wholesale.
from pdum.dsl.kernel.capture import make_handle

SRC_V1 = \'\'\'
def f(k):
    def g():
        return k
    return g
\'\'\'
SRC_V2 = SRC_V1.replace("return k", "return 2 * k")


def handle_from(src):
    ns = {}
    exec(compile(src, "<mymodule.py>", "exec"), ns)  # same def site both times
    return make_handle(ns["f"](7), "device")


draw(handle_from(SRC_V1))
print(f"v1 cached:      entries={len(cache)}  retirements={cache.retirements}")
draw(handle_from(SRC_V2))
print(f"after the edit: entries={len(cache)}  retirements={cache.retirements}  <- v1 retired, not leaked")

cache.bump_generation()
print(f"after bump:     entries={len(cache)}  generation={cache.generation}")
draw(disk(1.0, 2.0, 3.0))
print(f"same closure, new world: compiles={cache.compiles}")"""),
    code("""\
# Guards: a rebound dependency (think: frozen global) must never serve stale.
FROZEN = {"palette": object()}  # object identity stands in for a folded global


def draw_guarded(handle):
    key = cache.key_for(handle, (), ("fake-gpu",))
    rec = cache.get_or_compile(
        key,
        lambda: FastRecord(artifact=fake_gpu_compile(handle), guards=((FROZEN, "palette", FROZEN["palette"]),)),
    )
    return rec.artifact


a = draw_guarded(disk(9.0, 9.0, 9.0))
print("hit while the dependency is stable:", draw_guarded(disk(8.0, 8.0, 8.0)) is a)
FROZEN["palette"] = object()  # drift!
b = draw_guarded(disk(7.0, 7.0, 7.0))
print(f"refused + recompiled after drift:  {b is not a}   guard_misses={cache.guard_misses}")"""),
    code("""\
import threading

race, results = SpecializationCache(), []
barrier = threading.Barrier(8)


def worker():
    barrier.wait()
    key = race.key_for(disk(1.0, 1.0, 1.0))
    rec = race.get_or_compile(key, lambda: FastRecord(artifact=f"<built by {threading.current_thread().name}>"))
    results.append(rec.artifact)


threads = [threading.Thread(target=worker) for _ in range(8)]
[t.start() for t in threads]
[t.join() for t in threads]
print(f"8 threads raced one key: compiles={race.compiles}, distinct artifacts={len(set(results))}")"""),
    code("""\
# Tier 2 is CONTENT-addressed and generation-free: once lowering exists,
# two templates producing identical IR share one compiled artifact.
art, n = ArtifactCache(), itertools.count(1)


def build():
    return f"<compiled blob #{next(n)}>"


print(art.get_or_compile(("sha256:9f3a...", "wgsl", ()), build))
print(art.get_or_compile(("sha256:9f3a...", "wgsl", ()), build), " <- same content: same blob")
print(art.get_or_compile(("sha256:9f3a...", "c", ()), build), "    <- same content, other backend")
print(f"artifact-tier compiles={art.compiles}")"""),
    md("""\
## The two-tier law, experienced

| When this changes… | …this misses |
|---|---|
| a captured **value** (the hot loop) | **nothing** — pure hit |
| a capture or argument **type** | specialization tier |
| the function **body** (edit + rerun) | specialization tier (+ predecessor retired) |
| a **backend** or codegen flag | specialization tier (+ artifact tier on first sight) |
| a frozen **dependency** (guard drift) | specialization tier, by refusal — never stale |
| `bump_generation()` | all of tier 1; tier 2 untouched (content-addressed) |

**Phase I is complete: the thesis is now a passing test suite on the new
kernel — and no compiler exists yet.**

## What we can't do yet

- `FastRecord.extract/plan/staging/launch` are `None` — the precompiled hit
  path fills in at ch08/ch09; today the "artifact" is a string.
- Guards are synthetic — real dependency tags come from `classify_names`
  at lowering (ch07).
- Calling a Handle still does nothing; `draw()` here is the shape of the
  ch09 dispatch, written by hand.

**Budget after step 3:** kernel 378 / 1150 counted lines.

**Next:** ch04 — pipelines are values: the blessed combinator layer,
attached with zero kernel edits."""),
]


# ══════════════════════════════════════════ ch04 ══════════════════════════════

ch04 = [
    md("""\
# Chapter 4 — Pipelines are values

Step 3b builds the **blessed combinator library** — plumbum-style pipeline
syntax over kernels (semantics after
[`pdum_plumbum`](https://github.com/habemus-papadum/pdum_plumbum); design
record: `docs/design/040_combinators-notes.md`). Two things set this chapter apart:

- It is a **satellite**: `src/pdum/dsl/combinators.py` (~140 counted
  lines) attaches with **zero kernel edits** — the kernel sits untouched at
  378/1150. That is the extension-locality claim, demonstrated ahead of its
  CI gate. It ships composition *machinery* plus the single concept it
  owns (`materializer`, the host boundary). Every role vocabulary — even
  `device`, the base language's neutral composable — arrives with its owning
  package, registered live below as stand-ins.
- It is the *definition layer only*. Pipelines are identities today;
  execution arrives with the compiler chain (fusion at lowering, DPS +
  launch at the backends — see combinators-notes §3b). Every stub is loud.

| Piece | What |
|---|---|
| `@op` | plumbum's `@pb` for kernels: curried stage constructors |
| `\\|` | composes stages into an **inert** pipeline value |
| `>` | threads a value through, once (`value > pipeline`) |
| `stage[config]` | the bracket contract: an opaque config payload (040_combinators-notes §3c) |
| Roles + composition rules | who may compose, and what composing *means* |

Glossary terms settled: *Role, composition rule, materializer,
Stage/Pipeline.*"""),
    code("""\
from pdum.dsl.combinators import collect, op, register_composition, register_role
from pdum.dsl.kernel.api import jit

# The satellite ships MACHINERY plus its one own concept (the materializer).
# Even "device" — the base language's neutral composable, @jit's default —
# arrives from its owner: the stdlib/core-dialect package, once lowering
# exists (ch07). Until then, the chapter registers the stand-in:
register_role("device")
register_composition("pipe", "device", "device", "fuse")  # function composition fuses


@op
def add(k):
    @jit()
    def go(x):
        return x + k

    return go


@op
def mul(k):
    @jit()
    def go(x):
        return x * k

    return go


pipeline = add(1) | mul(2) | collect
print("the pipeline: ", pipeline)
print("its identity: ", pipeline.fntype)
print("nothing ran — a pipeline is a VALUE; application is a separate act.")"""),
    code("""\
from pdum.dsl import viz

viz.install()
pipeline  # stages as chips; hover a chip for its structural identity"""),
    code("""\
print("rebuilt each frame, fresh values:  ", (add(9) | mul(7) | collect).fp == pipeline.fp)
print("a stage param changes TYPE:        ", (add(1.0) | mul(2) | collect).fp == pipeline.fp)
print(
    "configured stage add(1)[64]:       ",
    (add(1)[64] | mul(2) | collect).fp == pipeline.fp,
    " <- no schemas yet: every config component value-specializes (040 §3c)",
)"""),
    code("""\
# The blessed operators FLATTEN, so grouping never even reaches the cache.
# (Compositions built outside this library still get unified by the
# content-addressed artifact tier, once lowering exists.)
a, b, c = add(1), mul(2), add(3)
print("(a|b)|c == a|(b|c):", ((a | b) | c).fp == (a | (b | c)).fp)
print("flattened parts:   ", a | b | c)"""),
    code("""\
from pdum.dsl.combinators import IncompatibleRoles, Stage, register_role


def holo():  # a kind NO package owns — so this lesson stays true forever
    @jit(kind="holo_display")
    def shader():
        return (0.0, 0.0)

    return shader


# The library owns only "device" and "materializer"; every other role ships
# WITH its domain package. (When this chapter was written, "fragment" was the
# example of a not-yet-existing vocabulary — the demo shader package now
# registers its own kinds at import, which is this lesson HAVING COME TRUE.
# The demonstration therefore uses a kind that stays unowned.)
try:
    add(1) | holo()
except IncompatibleRoles as e:
    print("before any holo package exists ->", e)

# Stand-in for what a holo-display package would do at import time:
register_role(
    "holo_display",
    hint="an entry point cannot be fused mid-pipeline; orchestration arrives with the pass runtime",
)

attempts = {
    "device | holo": lambda: add(1) | holo(),
    "holo | device": lambda: Stage(holo()) | add(1),
    "materializer mid-pipe": lambda: add(1) | collect | mul(2),
}
print()
for what, build in attempts.items():
    try:
        build()
    except IncompatibleRoles as e:
        print(f"{what:22s} -> {e}")"""),
    md("""\
Refusals are **rule lookups, not hardcoded checks**: a composition rule maps
`(op, role, role)` to the *semantics* of composing — `fuse` (inline into one
kernel), `terminal` (materialize at the boundary), later `orchestrate`
(render graphs, buffer chains). "Compatible" returns *how*, not just
*whether* — the fusion-vs-orchestration table is in
`docs/design/040_combinators-notes.md` §2.3.

And note what this chapter has now demonstrated twice: **role vocabularies
ship with their domains**. The satellite pre-enumerates nothing; `device` +
the fuse rule were registered as stand-ins for the base-language (stdlib)
package, and `fragment` for the WGSL backend package (ch10) — the same
discipline as `FragCoord` being a backend-package intrinsic, never a core
one. Terminality, by contrast, is *structural* (a `Role.terminal` flag, no
pair rules): nothing may follow a terminal; a terminal may end anything."""),
    code("""\
import itertools

from pdum.dsl.combinators import NotYetExecutable, set_dispatcher
from pdum.dsl.kernel.cache import FastRecord, SpecializationCache, no_compile
from pdum.dsl.kernel.valuekind import fingerprint

# From step 8 on, importing pdum.dsl installs a REAL dispatcher (batteries) —
# uninstall it for this cell so the definition/application split stays visible:
_batteries = set_dispatcher(None)
try:
    6 > (add(1) | mul(2))
except NotYetExecutable as e:
    print("without a backend:", e)

cache, serial = SpecializationCache(), itertools.count(1)


def dummy_dispatcher(pipeline, value):
    key = cache.key_for(pipeline, (fingerprint(value),), ("dummy",))
    rec = cache.get_or_compile(key, lambda: FastRecord(artifact=f"<fused kernel #{next(serial)}>"))
    return ("DeviceValue", rec.artifact)


prev = set_dispatcher(dummy_dispatcher)
print()
print("6 > pipeline ->", 6 > (add(1) | mul(2)))
with no_compile():
    for k in range(300):  # rebuild the WHOLE pipeline with fresh stage values, every step
        out = k > (add(k) | mul(k + 1))
print(f"300 rebuilt applications: compiles={cache.compiles}  hits={cache.hits}")
set_dispatcher(prev if prev is not None else _batteries)"""),
    code("""\
from pdum.dsl.kernel.valuekind import typeof

p = add(1) | mul(2)
print(typeof(p))
print(typeof((p, add(3) | mul(4))), " <- pipelines compose as values, like any Handle")"""),
    md("""\
## What we can't do yet

- **Fusion is a promise, not a fact**: the `Derived("pipe", …)` build rule —
  emit the composed body as IR — lands with lowering (ch07), and `value >
  pipeline` computes for real at the CPU backend (ch09). From ch09 on,
  combinator style is the **house style** for examples.
- **Outputs**: `DeviceValue`, `ResultPlan`, and destination-passing arrive
  with bidirectional marshaling (ch08); materializers execute then.
- **Config** has its own specialization regime (040_combinators-notes §3c),
  deliberately NOT the capture thesis: per component, strip →
  value-specialize (default) → type-specialize (rare), schema-declared and
  kernel-overridable. Today, with no schemas, everything value-specializes —
  the general model's degenerate case.
- **Orchestration** (fragment-pass graphs, buffer chains) waits for the pass
  runtime, after the GPU chapter.

**Budget after step 3b:** kernel *unchanged* at 378 / 1150; the satellite is
133 counted lines outside the cap — that is the point.

**Next:** ch05 — programs are values: the `Node`/`Region` IR, content
hashing, and the invariant that makes the anti-pattern unrepresentable."""),
]


# ══════════════════════════════════════════ ch05 ══════════════════════════════

ch05 = [
    md("""\
# Chapter 5 — Programs are values

Step 4 builds the IR — and with it, the system's deepest structural claim:
**the anti-pattern is unrepresentable.** A `Node` has no field that can hold
a runtime value; `attrs` is the single, deliberate carve-out for compile-time
constants, inside identity and visible in the printed text. A captured value
is a slot number here, never a value.

| File | Counted lines / cap | What |
|---|---|---|
| `src/pdum/dsl/kernel/ir.py` | 84 / 150 | `Node`, `Region`, `Loc`, content key, `Builder`, `verify` |
| `src/pdum/dsl/kernel/ops.py` | 88 / 110 | `OpDef` + the ~22-op core dialect table (a dialect IS a dict) |
| `src/pdum/dsl/kernel/printer.py` | 45 / 80 | the MLIR-flavored textual form (golden tests; migration insurance) |

Glossary terms settled: *Node/Region, content key, dialect (as a dict of
OpDefs).*"""),
    code("""\
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.ir import Builder, Loc, Node, Region
from pdum.dsl.kernel.ops import CORE_OPS
from pdum.dsl.kernel.printer import print_program

b = Builder(CORE_OPS)


def disk_body():
    \"\"\"The disk shader's decision, built by hand: (x-cx)^2 + (y-cy)^2 < r^2.\"\"\"
    x, y = b.param(0, T.f64), b.param(1, T.f64)
    cx = b.emit("core.env", type=T.f64, slot=0)
    cy = b.emit("core.env", type=T.f64, slot=1)
    r = b.emit("core.env", type=T.f64, slot=2)
    dx, dy = b.emit("core.sub", x, cx), b.emit("core.sub", y, cy)
    d2 = b.emit("core.add", b.emit("core.mul", dx, dx), b.emit("core.mul", dy, dy))
    hit = b.emit("core.cmp", d2, b.emit("core.mul", r, r), pred="lt")
    return Region(params=(x, y), body=(b.emit("core.yield", hit),))


prog = disk_body()
print(print_program(prog, name="disk"))"""),
    code("""\
from pdum.dsl import viz

viz.install()
prog  # the same program, rendered: hover any %ref for its type and provenance"""),
    md("""\
Read the text like MLIR: every node is an SSA value with exactly one result
type; captures appear as `core.env {slot = k}` — a *slot*, never a value;
types are the honest widths (`f64` — narrowing to `f32` is a backend
`type_map` decision at render, chapters away)."""),
    code("""\
# Content identity: a memoized sha256 over structure — the artifact-tier key.
print("key:              ", prog.key.hex()[:16], "…")
print("rebuilt:          ", disk_body().key == prog.key)

b2 = Builder(CORE_OPS)
x, y = b2.param(0, T.f64), b2.param(1, T.f64)
tweaked = Region((x, y), (b2.emit("core.yield", b2.emit("core.env", type=T.f64, slot=7)),))
print("tweaked shares key?", tweaked.key == prog.key, " <- different structure, different key")

# `loc` — where code came from — is EXCLUDED from identity:
n1 = b.emit("core.add", x, y, loc=Loc("notebook.py", 1))
n2 = b.emit("core.add", x, y, loc=Loc("elsewhere.py", 99))
print("loc excluded:     ", n1 == n2 and n1.key == n2.key)"""),
    code("""\
import dataclasses

# THE invariant, inspected: no field on Node can hold a runtime value.
for f in dataclasses.fields(Node):
    if not f.name.startswith("_"):
        print(f"  Node.{f.name:<8} : {f.type}")

try:
    Node(op="core.env", type=T.f64, value=5)  # there is no such field
except TypeError as e:
    print("smuggling a value ->", type(e).__name__, "-", e)"""),
    code("""\
# The one value-shaped slot, used deliberately — and visibly:
env = b.emit("core.env", type=T.f64, slot=0)  # runtime capture: re-marshaled per call
const = b.emit("core.const", type=T.f64, value=8.0)  # Literal lift: value IN identity
print(env.attrs, " vs ", const.attrs)
print("their keys differ per value?",
      b.emit("core.const", type=T.f64, value=9.0).key != const.key,
      " <- const recompiles per value; env never does. That difference IS the thesis.")"""),
    code("""\
# Structural sharing is real (a DAG, not a tree) — and the printer shows it:
e = b.emit("core.env", type=T.f64, slot=0)
sq = b.emit("core.mul", e, e)
total = b.emit("core.add", sq, sq)  # the SAME node object, used twice
print(print_program(Region(body=(b.emit("core.yield", total),)), name="shared"))"""),
    code("""\
# Structured control flow: exactly three region ops (if / for / call), forever
# until forced — a fourth is priced at ~180 lines x live transform columns.
f = b.emit("core.env", type=T.f64, slot=0)
cond = b.emit("core.cmp", f, b.emit("core.const", type=T.f64, value=0.5), pred="lt")
then = Region(body=(b.emit("core.yield", f),))
other = Region(body=(b.emit("core.yield", b.emit("core.neg", f)),))
branch = b.emit("core.if", cond, regions=(then, other))
print(print_program(Region(body=(b.emit("core.yield", branch),)), name="abs_ish"))

try:  # branches must agree — checked at CONSTRUCTION, not at render
    i = b.emit("core.env", type=T.i64, slot=1)
    b.emit("core.if", cond, regions=(then, Region(body=(b.emit("core.yield", i),))))
except TypeError as e:
    print("\\nrefused:", e)"""),
    code("""\
# Core arithmetic is STRICT (settled at this chapter's walkthrough): operands
# must share a type — there is NO promotion in the kernel. Promotion, where a
# language wants it, is a DIALECT's lowering policy (auto-insert casts), or
# absent (write float(i) yourself — the Julia model: cast, then same-type add).
from pdum.dsl.kernel.ir import VerifyError

i, fl = b.emit("core.env", type=T.i64, slot=0), b.emit("core.env", type=T.f64, slot=1)
print("i64 + i64 ->", b.emit("core.add", i, i).type)
try:
    b.emit("core.add", i, fl)
except TypeError as e:
    print("i64 + f64 ->", e)
w = b.emit("core.cast", i, to=T.f64)
print("cast(i) + f64 ->", b.emit("core.add", w, fl).type, "  <- the cast is IN the IR, in the hash")

v = b.emit("core.vec", fl, fl, fl)
print(v.type, "->", b.emit("core.extract", v, index=2).type, "|", b.emit("core.cast", fl, to=T.f32).type)

for bad in (lambda: b.emit("core.frobnicate", i), lambda: b.emit("core.env", slot=0)):
    try:
        bad()
    except VerifyError as e:
        print("VerifyError:", e)"""),
    md("""\
## What we can't do yet

- Nothing *produces* IR but our hands: rewriting arrives in ch06, lowering
  (source → IR, the combinator build rule) in ch07. Every program in this
  chapter was built with the `Builder`, on purpose — the IR is a public,
  inspectable value, not compiler-internal state.
- Nothing consumes it either: legalization (ch08) and rendering (ch09) are
  where `core.env` becomes physical slots and text.
- The artifact tier can now key on `Region.key` for real — ch03's fake
  `"sha256:9f3a..."` strings retire when lowering lands.

**Gates now armed:** the anti-pattern field check and the printer golden test
(`tests/test_ir.py`).

**Budget after step 4:** kernel 595 / 1150 counted lines.

**Next:** ch06 — everything is a rule: `Pat`/`RuleSet`, the one rewrite
driver, and stage legality."""),
]


# ══════════════════════════════════════════ ch06 ══════════════════════════════

ch06 = [
    md("""\
# Chapter 6 — Everything is a rule

Step 5 builds the **one pass mechanism**. From here on, every compiler
activity — simplification, backend decompositions, param legalization,
transform columns, even rendering — is `(pattern, fn)` **data**, run by a
single greedy driver. The review question this chapter arms: *any proposed
new mechanism must first prove it cannot be a rule.*

| File | Counted lines / cap | What |
|---|---|---|
| `src/pdum/dsl/kernel/rewrite.py` | 113 / 150 | `Pat`, rule sets, the driver, `Stage` legality, `MatchLog`, the non-termination budget |

The driver's character, on purpose: **greedy, directional, deterministic** —
bottom-up, first-matching-rule-wins (order is priority), DAG sharing
preserved, loud budget guard. The end of this chapter measures the road not
taken (equality saturation as the core) with a real e-graph engine.

Glossary terms settled: *rule / RuleSet / Stage (legality), match log.*"""),
    code("""\
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.ir import Builder, Region
from pdum.dsl.kernel.ops import CORE_OPS
from pdum.dsl.kernel.printer import print_program
from pdum.dsl.kernel.rewrite import MatchLog, Pat, Stage, rewrite, run_stage

b = Builder(CORE_OPS)


def const(v):
    return b.emit("core.const", type=T.f64, value=v)


def is_const(m, name, value=None):
    n = m[name]
    return n.op == "core.const" and (value is None or dict(n.attrs)["value"] == value)


# Rules are DATA: a pattern and a replacement function. This is the entire
# extension mechanism for compiler behavior.
SIMPLIFY = [
    (Pat("core.add", ("x", "z"), guard=lambda m: is_const(m, "z", 0.0)), lambda bl, m: m["x"]),
    (Pat("core.mul", ("x", "z"), guard=lambda m: is_const(m, "z", 1.0)), lambda bl, m: m["x"]),
    (
        Pat("core.add", ("x", "y"), guard=lambda m: is_const(m, "x") and is_const(m, "y")),
        lambda bl, m: bl.emit(
            "core.const", type=m["root"].type, value=dict(m["x"].attrs)["value"] + dict(m["y"].attrs)["value"]
        ),
    ),
    (Pat("core.neg", (Pat("core.neg", ("x",)),)), lambda bl, m: m["x"]),
    (Pat("core.sub", ("x", "x")), lambda bl, m: bl.emit("core.const", type=m["root"].type, value=0.0)),
]

x = b.emit("core.env", type=T.f64, slot=0)
noisy = b.emit("core.add", b.emit("core.mul", b.emit("core.add", x, const(0.0)), const(1.0)), const(0.0))
before = Region(body=(b.emit("core.yield", noisy),))
print(print_program(before, name="before"))

log = MatchLog()
after = rewrite(before, SIMPLIFY, CORE_OPS, name="simplify", log=log)
print()
print(print_program(after, name="after"))
print()
for stage, old, new in log.entries:
    print(f"  [{stage}] {old.op}  ->  {new.op}")"""),
    code("""\
from IPython.display import display

from pdum.dsl import viz

viz.install()
display(log)  # every firing, with hover provenance
after"""),
    md("""\
`((x+0)*1)+0` collapsed to `x` in one bottom-up pass — each child's rewrite
exposed the parent's opportunity, and the match log shows every firing. Two
disciplines worth noticing in the driver: **DAG sharing is preserved** (a
shared node is rewritten once, and stays shared), and **binders are
untouchable** (`core.param` is never rewritten — the one structural
exemption)."""),
    code("""\
# Sharing: the SAME node object used twice is rewritten once, and the result
# is still shared. (This is what keeps content keys and artifact reuse sane.)
shared = b.emit("core.add", b.emit("core.env", type=T.f64, slot=1), const(0.0))
root = b.emit("core.mul", shared, shared)
log2 = MatchLog()
out = rewrite(Region(body=(b.emit("core.yield", root),)), SIMPLIFY, CORE_OPS, log=log2)
res = out.body[-1].args[0]
print("still shared:", res.args[0] is res.args[1], "| rule firings:", len(log2.entries))

# Nonlinear patterns: a repeated capture name demands structural equality.
print("x - x -> ", rewrite(
    Region(body=(b.emit("core.yield", b.emit("core.sub", b.emit("core.env", type=T.f64, slot=2),
                                             b.emit("core.env", type=T.f64, slot=2))),)),
    SIMPLIFY, CORE_OPS).body[-1].args[0].attrs)"""),
    code("""\
# Provenance rides along for free (050_provenance_tracking.md): fresh
# replacement nodes inherit the replaced node's loc via the builder default;
# survivors keep their own story; and type errors render the source points.
from pdum.dsl.kernel.ir import CallLoc, Loc, format_loc

lit = b.emit("core.add", const(3.0), const(4.0), loc=Loc("art.py", 7))
folded = rewrite(Region(body=(b.emit("core.yield", lit),)), SIMPLIFY, CORE_OPS).body[-1].args[0]
print("folded const:", dict(folded.attrs), "carries", format_loc(folded.loc))
print("an inlining chain renders as:", format_loc(CallLoc(Loc("wave.py", 5), Loc("art.py", 40))))

i3 = b.emit("core.env", type=T.i64, slot=3, loc=Loc("art.py", 11))
f4 = b.emit("core.env", type=T.f64, slot=4, loc=Loc("art.py", 12))
try:
    b.emit("core.add", i3, f4, loc=Loc("art.py", 13))
except TypeError as terr:
    print("a typed error is a starting region:")
    print("  ", terr)"""),
    code("""\
# Rules reach inside regions — branches of a core.if are cleaned in place:
e = b.emit("core.env", type=T.f64, slot=0)
cond = b.emit("core.cmp", e, e, pred="lt")
branch = b.emit("core.if", cond, regions=(
    Region(body=(b.emit("core.yield", b.emit("core.add", e, const(0.0))),)),
    Region(body=(b.emit("core.yield", b.emit("core.neg", b.emit("core.neg", e))),)),
))
cleaned = rewrite(Region(body=(b.emit("core.yield", branch),)), SIMPLIFY, CORE_OPS)
print(print_program(cleaned, name="cleaned"))"""),
    code("""\
# Stages: rules to fixpoint, THEN conversion-target legality — "which
# dialects may exist after stage N" is machine-checked, not folklore.
from pdum.dsl.kernel.ir import VerifyError
from pdum.dsl.kernel.ops import OpDef

toy_ops = dict(CORE_OPS)
toy_ops["toy.blit"] = OpDef("toy.blit", lambda a, at, r: T.f64)
tb = Builder(toy_ops)
prog_with_toy = Region(body=(tb.emit("core.yield", tb.emit("toy.blit")),))

try:
    run_stage(prog_with_toy, Stage("mid", [], legal=frozenset({"core"})), toy_ops)
except VerifyError as verr:  # NB: `as e` would DELETE our env node named e — Python quirk
    print("refused:", verr)

# And the non-termination budget — a depth-growing rule fails LOUDLY, not
# with a blown stack:
ping = [(Pat("core.neg", ("x",)), lambda bl, m: bl.emit("core.neg", bl.emit("core.neg", m["root"])))]
try:
    rewrite(Region(body=(b.emit("core.yield", b.emit("core.neg", e)),)), ping, CORE_OPS, name="ping")
except VerifyError as err:
    print("guarded:", err)"""),
    md("""\
## The detour, measured: should equality saturation be the core?

The honest question (asked at this step's walkthrough): e-graphs solve the
**phase-ordering problem** — instead of greedily committing to each rewrite,
an e-graph holds *all* equivalent forms at once and extracts the best by
cost. Is that technology strong enough to BE the engine, rather than a
satellite? We installed `egglog` (the state-of-the-art e-graph engine, Rust
core) and measured, rather than argued:"""),
    code("""\
import time

try:
    t0 = time.perf_counter()
    from egglog import (  # noqa: I001
        EGraph, Expr, StringLike, birewrite, i64, i64Like, rewrite as eg_rw, ruleset, var, vars_,
    )

    IMPORT_MS = 1000 * (time.perf_counter() - t0)

    class Num(Expr):
        def __init__(self, value: i64Like) -> None: ...
        @classmethod
        def var(cls, name: StringLike) -> Num: ...
        def __add__(self, other: Num) -> Num: ...
        def __mul__(self, other: Num) -> Num: ...

    a2, b2, c2 = vars_("a b c", Num)
    i2, j2 = var("i", i64), var("j", i64)
    algebra = ruleset(
        eg_rw(a2 + Num(0)).to(a2),
        eg_rw(a2 * Num(1)).to(a2),
        eg_rw(Num(i2) + Num(j2)).to(Num(i2 + j2)),
        eg_rw(Num(i2) * Num(j2)).to(Num(i2 * j2)),
        birewrite(a2 * (b2 + c2)).to(a2 * b2 + a2 * c2),  # distribution, BOTH ways
        birewrite(a2 + b2).to(b2 + a2),
        birewrite(a2 * b2).to(b2 * a2),
    )

    # The phase-ordering win our greedy driver CANNOT get: x*2 + x*3 -> x*5
    # requires FACTORING (reverse distribution) — a greedy engine with
    # distribute-forward walks away from it. The e-graph holds both.
    eg = EGraph()
    xx = Num.var("x")
    expr = eg.let("e", xx * Num(2) + xx * Num(3))
    t0 = time.perf_counter()
    eg.run(algebra * 8)
    t1 = time.perf_counter()
    print("extracted:", eg.extract(expr))
    print(f"import={IMPORT_MS:.0f}ms  saturate={1000 * (t1 - t0):.1f}ms")
except ImportError:
    print("egglog not installed on this platform — see the measured summary below")"""),
    md("""\
**The verdict** (recorded in 010 §10, confirming research verdicts V2/R7 —
now with our own numbers instead of citations):

| Axis | Greedy driver (ours) | Equality saturation (egglog) |
|---|---|---|
| Phase-ordering | loses (commits greedily) | **wins** (`x*2+x*3 → x*5`, ~1 ms) |
| Kernel-sized cost | microseconds | **~20 ms** saturate+extract — our whole miss budget |
| Determinism | total order, golden-testable | bounded iterations ⇒ heuristic output |
| Non-equational passes (slot numbering, AD, render) | natural | not equalities — poor fit |
| Binders (`core.for`/`core.call` params) | untouched by design | the classic e-graph hard case |
| Dependency | zero (stdlib) | Rust wheel, **~1.5 s import** |

**Decision: the greedy driver is the core; equality saturation is an
opt-in optimizer pass** at the seam that already exists (a pass is any
`Region → Region` function — architecture §12). Where factoring-class
optimizations matter, an egglog-backed pass exports a pure expression
subregion, saturates, extracts by cost, and rebuilds — paying its ~20 ms
only where a domain asked for it. What we deliberately give up in the core:
the phase-ordering wins. What we keep: determinism, zero dependencies, and
one engine a reader can hold in their head — all 113 lines of it."""),
    md("""\
## What we can't do yet

- Nobody *feeds* the engine: `lower_ast` rules and the compile driver's
  stage ladder arrive with lowering (ch07); today stages are assembled by
  hand.
- Decompositions gated on backend op sets and param legalization (the
  `abi.slot` stage) arrive with marshaling and the first backends
  (ch08–ch10).
- Transform columns (jvp/batch) are the same rule shape, much later.

**Gates armed:** golden-printed-IR-after-stage (`tests/test_rewrite.py`).

**Budget after step 5:** kernel 707 / 1150 counted lines.

**Next:** ch07 — source to IR: `classify_names`, the fused typing+lowering
pass, the combinator build rule, and the dialect cast-insertion decision."""),
]


# ══════════════════════════════════════════ ch07 ══════════════════════════════

ch07 = [
    md("""\
# Chapter 7 — Source to IR

Step 6 builds **phase B's front half**: the fused typing+lowering pass. From
a Handle's decoration-time snapshot to verified, typed core IR — in one
forward walk, with the *language itself living outside the kernel* as
`lower_ast` rules.

| Piece | Counted lines / cap | What |
|---|---|---|
| `kernel/lower.py` | 94 / 170 | the DRIVER: coherence, rebased locs, name fates, inlining, Derived build-rule dispatch |
| `stdlib/base_lang.py` (satellite) | ~80 | the base dialect's rule pack — the accepted Python subset, as a dict |
| `combinators.build_pipe` (satellite) | ~25 | the `Derived("pipe")` fusion build rule |

Two decisions live in this chapter: **the base dialect is strict** (no
auto-cast insertion — `float(i)` is the user's job, per the ch05 settlement;
a friendlier dialect may choose otherwise in ITS pack), and **the language
is a registration** (widening Python support = adding a dict entry in a
satellite, never editing the driver — numba's `typeinfer.py` failure mode,
structurally excluded).

Glossary terms settled: *name fates, lower_ast rule, build rule,
NoSourceError/StaleSourceError.*"""),
    code("""\
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.lower import lower_handle
from pdum.dsl.kernel.ops import CORE_OPS
from pdum.dsl.kernel.printer import print_program
from pdum.dsl.stdlib.base_lang import LOWER_RULES


def dist2(cx, cy):
    @jit()
    def go(px, py):
        dx = px - cx
        dy = py - cy
        return dx * dx + dy * dy

    return go


h = dist2(320.0, 240.0)
region = lower_handle(h, LOWER_RULES, CORE_OPS, arg_types=(T.f64, T.f64))
print(print_program(region, name="dist2"))"""),
    md("""\
Source became typed IR: parameters are `%p`s, captures are `core.env` with
**path** attrs (`slot = (0,)` — paths, not flat ints, because inlining nests
environments; marshaling flattens the tree in ch08), and every node carries
an absolute source location, rebased through the snapshot."""),
    code("""\
from pdum.dsl.kernel.ir import format_loc

sub = next(n for n in region.body[-1].args[0].args[0].args for n in [n] if n.op == "core.sub")
print("this core.sub came from:", format_loc(sub.loc))

# THE STRICT EXPERIENCE, end to end from real source — the base dialect does
# not auto-cast; the error is a starting region with your file and lines:
@jit()
def mixed(x):
    return x + 1  # an i64 literal against an f64 parameter


try:
    lower_handle(mixed, LOWER_RULES, CORE_OPS, arg_types=(T.f64,))
except TypeError as terr:
    print("\\nrefused:", terr)


@jit()
def fixed(x):
    return x + float(1)  # the Julia way — and the cast is IN the IR, IN the hash


print()
print(print_program(lower_handle(fixed, LOWER_RULES, CORE_OPS, arg_types=(T.f64,)), name="fixed"))"""),
    code("""\
# Name fates are a CLOSED taxonomy: param, local, capture — anything else is
# loud. Globals have no sanctioned fate yet (no silent freezing — the M0-era
# hazard the guards exist for).
from pdum.dsl.kernel.lower import MissingRule, NameFateError

SOME_GLOBAL = 42


@jit()
def leaky(x):
    return x + SOME_GLOBAL


try:
    lower_handle(leaky, LOWER_RULES, CORE_OPS, arg_types=(T.f64,))
except NameFateError as e:
    print("fate error: ", e)


@jit()
def tup(x):
    return (x, x)


try:
    lower_handle(tup, LOWER_RULES, CORE_OPS, arg_types=(T.f64,))
except MissingRule as e:
    print("missing rule:", e, " <- widening = a dict entry in a rule pack")"""),
    code("""\
# Inlining: a captured kernel called in the body is fused in — multi-statement
# bodies included (M0 demanded single-return; that restriction is gone).
def make_inner(k):
    @jit()
    def inner(y):
        t = y * k
        return t + t

    return inner


def make_outer(inner, c):
    @jit()
    def outer(x):
        return inner(x) + c

    return outer


fused = lower_handle(make_outer(make_inner(2.0), 10.0), LOWER_RULES, CORE_OPS, arg_types=(T.f64,))
print(print_program(fused, name="fused"))
print()
mul_node = next(n for n in fused.body[-1].args[0].args[0].args if n.op == "core.mul")
print("inlined provenance:", format_loc(mul_node.loc))"""),
    md("""\
Read the env paths: `c` is the outer's capture `(0,)`; the inner kernel sits
at slot 1, so *its* capture `k` is `(1, 0)` — the environment TREE from ch02,
now syntactic in the IR, ready for `EnvLeaf` flattening at ch08. And the
provenance chains: the `core.mul` knows it lives in `inner`'s source,
*inlined from* the call site in `outer`."""),
    code("""\
# ch04's promise, kept: the pipe build rule lowers Derived("pipe") WITHOUT
# source — pure structure, each stage inlined in sequence. Fusion is now a
# fact at the IR level (execution needs ch09's backend).
from pdum.dsl.combinators import PIPE_BUILDERS, op, register_composition, register_role

register_role("device")  # stand-in, as in ch04: the stdlib package owns this
register_composition("pipe", "device", "device", "fuse")


@op
def padd(k):
    @jit()
    def go(x):
        return x + k

    return go


@op
def pmul(k):
    @jit()
    def go(x):
        return x * k

    return go


pipeline = padd(1.0) | pmul(2.0)
body = lower_handle(pipeline, LOWER_RULES, CORE_OPS, arg_types=(T.f64,), derived=PIPE_BUILDERS)
print(print_program(body, name="fused_pipe"))
print()
print("stage provenance:", format_loc(body.body[-1].args[0].loc))"""),
    code("""\
from pdum.dsl import viz

viz.install()
body  # the fused pipeline, rendered — hover %refs to see <pipeline> provenance"""),
    md("""\
## What we can't do yet

- **Nothing calls `lower_handle` automatically** — dispatch (typeof the
  arguments, build the key, probe the cache, lower on miss) is ch09's hot
  path. Today we pass `arg_types` by hand.
- **Overloads** (`sqrt`, methods) await the Registry (step 8's surfaces);
  calls resolve only to casts and captured kernels.
- **Tuples, attributes, `if`/`for` statements** await their dialect packs.
- **Guards are still synthetic** — the base pack refuses globals outright,
  so there is nothing to guard yet; folded-global fates arrive with a
  policy, and their guards with them.

**Budget after step 6:** kernel 829 / 1150; satellites: combinators 163/250,
stdlib 83/1500 (the honesty-clause bucket, now separately counted).

**Next:** ch08 — one value, N parameters: bidirectional marshaling
(PackPlan + ResultPlan), leaves, and the `abi.slot` stage."""),
]


# ══════════════════════════════════════════ appendix A ═══════════════════════

ch07a = [
    md("""\
# 7a — The lay of the land

Seven chapters in — capture, the two-tier cache, the IR, rewriting,
lowering — this interlude surveys the language itself: what Python the DSL
accepts today, what it refuses and why, what "type inference" means here,
and how we compare to numba and the kernel DSLs.

It is the first of a recurring series. After later rule-pack milestones, a
short `chNNa` interlude records the **deltas** — what just became possible
— rather than restating this baseline. The cells here run live against the
current rule pack (the supported list is printed *from* it), so if an
output ever disagrees with the prose, a later chapter has already moved
the frontier and the next interlude names the change.

Statuses used throughout:

- ✅ **supported today** — lowers, typed, tested.
- 🔧 **a registration away** — an entry in a rule pack; zero new machinery.
- 🏗️ **needs planned machinery** — named chapter; real design content.
- 🚫 **out by design** — with the reason.
"""),
    code("""\
# The language IS data: the supported surface, printed from the rule pack.
from pdum.dsl.stdlib.base_lang import _CASTS, LOWER_RULES

GLOSS = {
    "Expr": "docstrings / stray constants (ignored)",
    "Assign": "single-name assignment (SSA rebinding allowed)",
    "Return": "return <expr> (must be reached)",
    "Constant": "int / float / bool literals (honest types: 1 is i64)",
    "Name": "parameters, locals, captures",
    "BinOp": "+ - * / % **  (STRICT: operands must share a type)",
    "UnaryOp": "unary -",
    "Compare": "single < > <= >= == !=",
    "IfExp": "a if cond else b  (branch types must match)",
    "Call": "float()/int()/bool() casts · captured-kernel calls (inlined)",
}
print("the base pack accepts", len(LOWER_RULES), "syntax forms:\\n")
for node_type in LOWER_RULES:
    print(f"  ✅ ast.{node_type.__name__:<10} — {GLOSS.get(node_type.__name__, '')}")
print("\\n  casts:", ", ".join(sorted(_CASTS)))"""),
    code("""\
# What the language REFUSES, it refuses loudly — the errors are the index.
# Every function below is real source in this cell; each shows its actual error.
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.lower import lower_handle
from pdum.dsl.kernel.ops import CORE_OPS
from pdum.dsl.stdlib.base_lang import LOWER_RULES


@jit()
def wants_tuple(x):
    return (x, x)  # tuples: a dialect decision (vec? record?) — registration away


@jit()
def wants_attr(x):
    return x.real  # attributes: records/swizzles land with their dialects


@jit()
def wants_index(a):
    return a[10]  # subscripts: the arrays chapter


@jit()
def wants_for(x):
    total = x
    for i in range(4):  # statements need region lowering + the strict join policy
        total = total + x
    return total


@jit()
def wants_boolop(x):
    return x > 0.0 and x < 1.0  # and/or: registration + a short-circuit decision


@jit()
def wants_augassign(x):
    x += 1.0  # pure sugar for x = x + 1.0
    return x


# As of chapter 7 every one of these refuses. A ✅ line means a later
# chapter's rule pack widened the base language — the next interlude has it.
for h in (wants_tuple, wants_attr, wants_index, wants_for, wants_boolop, wants_augassign):
    try:
        lower_handle(h, LOWER_RULES, CORE_OPS, arg_types=(T.f64,))
        print(f"  ✅ {h.fntype.template.label}: lowers now")
    except Exception as e:
        print(f"  {type(e).__name__:12s} {e}")"""),
    md("""\
## Case study: `x = a[10].b`

The question this example really asks: *what does type inference have to do
here?* Answer: **nothing exotic** — at specialization time `a` arrives with
a concrete type, and every step is one local op rule reading it. The gap
today is vocabulary (no subscript rule, no index op), not machinery:"""),
    code("""\
# Today: refused, with the exact gap named.
@jit()
def foo(a):
    x = a[10].b
    return x


try:
    lower_handle(foo, LOWER_RULES, CORE_OPS, arg_types=(T.f64,))
except Exception as e:
    print(type(e).__name__, "-", e)"""),
    code("""\
# The same computation, hand-built — a five-line TOY dialect adds the index
# op with its type rule, and the types chain with zero inference machinery:
from pdum.dsl.kernel.ir import Builder, Region
from pdum.dsl.kernel.ops import OpDef
from pdum.dsl.kernel.printer import print_program

P = T.Record("P", (("b", T.f32), ("pad", T.f32)))
ArrP = T.Array(P, 1, "C", "<", True)


def _index_rule(args, attrs, regions):
    return args[0].dtype  # Array(dtype=...) -> its element type. That's it.


TOY = dict(CORE_OPS)
TOY["tensor.index"] = OpDef("tensor.index", _index_rule)

tb = Builder(TOY)
a = tb.emit("core.env", type=ArrP, slot=(0,))
ten = tb.emit("core.const", type=T.i64, value=10)
elem = tb.emit("tensor.index", a, ten)  # : P        (rule read Array.dtype)
x = tb.emit("core.field", elem, name="b")  # : f32   (rule read Record.fields)
print(print_program(Region(body=(tb.emit("core.yield", x),)), name="foo"))
print()
print("a[10]   :", elem.type)
print("a[10].b :", x.type, "  <- two local rules; 'inference' never appears")"""),
    md("""\
## What "type inference" is here — and is it sufficient?

**Forward-only typing inside the fused lowering pass.** Types enter through
exactly three doors — argument types at specialization, capture types via
`typeof`, literal types via `typeof` — and propagate through per-op
`type_rule`s. No unification, no fixpoint, no backward flow.

Why that is *sufficient* rather than immature: **specialization concretizes
everything at entry** (the Julia/numba insight — inference with concrete
argument types is just forward evaluation over types), and the strict-core
decision extends to joins: when `if`/`for` *statements* land, a name bound
in both branches must have the SAME type (loud otherwise), and `core.for`
carries are explicitly typed. Same-type-or-loud at every join is exactly
what eliminates numba's `typeinfer.py` fixpoint machinery and phi-node
unification — the single most complex component of that compiler, deleted
by a language decision.

**How dialects strengthen it** — four doors, no fifth:
1. **op type rules** — a dialect's ops carry their own typing (the toy
   `tensor.index` above);
2. **lower_ast rules** — a dialect decides what syntax *means* (and whether
   it auto-inserts casts);
3. **overloads** (step 8) — type-driven selection per target;
4. **post-IR analysis passes** — shapes, units, ranges, solver-backed
   checks (§12 of the architecture) run over the frozen Region, where
   multi-pass is cheap.
"""),
    md("""\
## The cross-library matrix

Construct-level comparison against the systems that define user expectations
(sources: current official docs, empirically spot-checked against numba
0.66 — full surveys in `docs/design/research/R10`/`R11`). Our column uses the
four statuses; the point of each row is the *decision* it surfaces, not the
checkmark.

| construct | **pdum.dsl** | numba | Triton | Taichi | cupy.jit |
|---|---|---|---|---|---|
| ternary `a if c else b` | ✅ | ✅ | ✅ | ✅ | ✅ |
| aug-assign `x += 1` | 🔧 pure desugar | ✅ | ✅ | ✅ (atomic on fields) | ✅ |
| chained `0 < x < 1` | 🔧 pure desugar | ✅ | scalars only | ✅ | 🚫 |
| `and` / `or` | 🔧 + one policy call | ✅ short-circuit | scalars only | bitwise, no short-circuit | ✅ |
| tuples + unpacking | 🔧 (`Tuple` type exists; ops are a dialect addition like `tensor.index` above) | ✅ | ✅ | partial | ✅ |
| `if`/`else` *statement* | 🏗️ ch12 — region lowering + strict joins | ✅ | ✅ | ✅ | ✅ |
| `for` + `range` | 🏗️ ch12 (`core.for` op already exists) | ✅ | `range` only | ✅ auto-parallelized | `range` only |
| early / multiple `return` | 🚫 settled 2026-07-12: one return, at the tail | ✅ must unify | top-level only | 🚫 one return, at the bottom | ✅ |
| `while` | 🏗️ needs a fourth region op; `for` covers kernel targets first | ✅ | ✅ | serial only | ✅ |
| `a[i]` subscript | 🏗️ arrays chapter | ✅ | load-only | read+write | element only |
| `.field` attribute | 🔧 once `@record` kinds land (`core.field` works today) | `@jitclass` | 🚫 | ✅ structs | 🚫 |
| closures / nested defs | ✅ **the founding abstraction** — captured kernels inline, typed env in the key | partial: cannot be returned | 🚫 module-level only | 🚫 module-level only | 🚫 |
| global reads | 🚫 loud `NameFateError` | frozen silently at compile | constexpr-only, else error | frozen silently | frozen **silently**, never invalidated |
| runtime recursion | 🚫 (inlining model) | ✅ limited | 🚫 | 🚫 | 🚫 |
| comprehensions | 🚫 in-kernel — host Python at capture covers the compile-time uses | partial | tuple-only | compile-time only | 🚫 |
| `try` / `with` / `match` / `yield` | 🚫 by design — no exception objects or effects inside kernels | partial / objmode | 🚫 | 🚫 | 🚫 |
| kwargs / defaults | 🏗️ dispatch layer (step 8) | ✅ | calls only | partial | 🚫 |
"""),
    md("""\
### What the matrix decides

**The consensus core is tiny.** Arithmetic, `if`, `for`+`range`, and
calls-that-inline is the entire intersection of the three GPU DSLs. We are
four *registrations* (aug-assign, bool ops, chained compares, tuples) and
one *machinery step* (ch12 statement lowering: `if`/`for`/early-return over
the region ops that already exist) away from it. That ordering goes into
the plan.

**Three rows are policy, not capability** — and land in *dialects*, never
the kernel:

1. `and`/`or` — the field disagrees (numba short-circuits, Taichi lowers to
   bitwise, Triton refuses tensors): the base pack will pick short-circuit
   via `core.if` and say so; a SIMD dialect may pick bitwise.
2. early return — numba unifies return paths; Taichi refuses all but one.
   **Settled at this walkthrough (2026-07-12): we take the strict end — one
   `return`, at the tail; `core.yield` IS the return.** No return-path
   unification machinery, ever (010 ledger).
3. globals — everyone else freezes silently (cupy never even invalidates);
   we keep the loud `NameFateError`. A silent snapshot is the staleness bug
   this project exists to kill: **capture it or pass it.**

**Specialization is where we quietly win.** Triton's `tl.constexpr` +
`do_not_specialize`, Taichi's `ti.template()` + `ti.static`, cupy's
implicit per-signature keying — three ad-hoc syntaxes for one need — all
map onto machinery this design already has: per-kind fingerprinting
(value- vs type-specialization as a `ValueKind` decision), the config
bracket's strip → value → type pipeline, and plain host Python at capture
time standing in for `ti.static`.
"""),
    md("""\
## Classes and dataclasses as structs

numba's `@jitclass` declares typed fields and compiles methods; its modern
successor `structref` is the acknowledged precedent (research: R8, R10). Our
equivalents are **already designed, scheduled at the surfaces chapter**:

- a `@record`-style decorator (surface C) registers a frozen dataclass as a
  `ValueKind`: `typeof` → the `Record` type below, `flatten` → its fields;
- field access is `ast.Attribute` → `core.field` (the op exists today);
- methods are `@overload_method` — ordinary DSL code, erased to free
  functions (which the GPU targets require anyway);
- numpy structured dtypes converge on the same `Record` through the ndarray
  kind — one struct story for both roads.
"""),
    code("""\
# The Record machinery, previewed today (only the *decorator sugar* is pending):
Color = T.Record("Color", (("r", T.f32), ("g", T.f32), ("b", T.f32)))
cb = Builder(CORE_OPS)
c = cb.emit("core.env", type=Color, slot=(0,))
g = cb.emit("core.field", c, name="g")
print("c.g :", g.type, " — field access types today; the @record sugar is ch11's")"""),
]

ch08 = [
    md("""\
# 8 — One value, N parameters

Chapter 3 proved the thesis at the cache: change a captured *value* and
`compiles == 1`. This chapter builds the machinery that makes it true at the
**byte** level. M0's disease was the per-frame flatten — every call re-walked
the environment hunting for its uniforms. The cure is structural, and it is
**bidirectional** (design 040 §3b — kernels are destination-passing at the
ABI):

- a **PackPlan** is built once per cache entry, from **types alone** —
  values never shape a plan;
- per call, one generic loop writes leaf values into a reused **staging**
  buffer (byte slots) or out the **leaves channel** (buffer-class leaves,
  when ndarrays arrive);
- a **ResultPlan** mirrors it outbound: destinations allocated from the
  result type, result bytes unflattened back into the logical value;
- and the decision is **printable IR**: logical `core.env` ops legalize into
  physical `abi.slot` ops, so marshaling is golden-testable, never folklore.

Source: `src/pdum/dsl/kernel/pack.py`. New words for the glossary: *leaf*,
*slot*, *plan*, *staging*, *leaves channel*."""),
    code("""\
import ast
import struct

from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.lower import lower_handle
from pdum.dsl.kernel.ops import CORE_OPS
from pdum.dsl.kernel.pack import (
    ABI_OPS,
    NORMALIZE_ENV,
    build_extractor,
    legalize_params,
    pack_into,
    plan_from_types,
    result_plan,
    unpack_result,
)
from pdum.dsl.kernel.printer import print_program
from pdum.dsl.kernel.rewrite import MatchLog, run_stage
from pdum.dsl.kernel.valuekind import BUILTINS
from pdum.dsl.stdlib.base_lang import LOWER_RULES

ALL_OPS = {**CORE_OPS, **ABI_OPS}


def _subscript(ctx, node):  # stand-in: const-index tuple access (tuples land at step 10)
    return ctx.emit("core.extract", ctx.lower(node.value), node=node, index=node.slice.value)


RULES = dict(LOWER_RULES)
RULES[ast.Subscript] = _subscript


def make_shader(cx, cy, gain):
    center = (cx, cy)

    @jit()
    def bright(v):
        return v * gain

    @jit()
    def shader(x):
        return bright(x - center[0]) + center[1]

    return shader


sh = make_shader(0.5, 0.25, 2.0)
region = lower_handle(sh, RULES, ALL_OPS, arg_types=(T.f64,))
print(print_program(region, name="shader"))"""),
    md("""\
A capture worth marshaling: a **tuple** (`center`) and a **nested kernel**
(`bright`, which itself captures `gain`). Note the `core.extract` ops eating
the tuple env, and the `(0, 0)`-style env paths from ch07's inlining.

## The leaf walk — from types alone

Before any value is touched, the *types* already determine every physical
parameter. `leaf_entries` is the static walk; the `FnType` walker is the
architecture's **EnvLeaf recursion** — a captured kernel's leaves are its
env's leaves, paths prefixed. These are the same paths lowering stamps on
`core.env`."""),
    code("""\
for sub, leaf in BUILTINS.leaf_entries(sh.fntype):
    print(f"  env path {str(sub):<8} {leaf}")
print()
print("(0, 0) is `gain` — reached THROUGH the captured `bright` kernel.")"""),
    md("""\
## Normalizing: composite captures become leaf paths

`core.extract`-of-an-env folds into the env's *path* — after this stage,
every surviving capture is leaf-typed. This is a rewrite Stage like any
other (ch06's driver, ch06's MatchLog); marshaling adds no new mechanism."""),
    code("""\
log = MatchLog()
normed = run_stage(region, NORMALIZE_ENV, ALL_OPS, log=log)
for stage, old, new in log.entries:
    print(f"  [{stage}] {old.op} {dict(old.attrs)}  ->  {new.op} {dict(new.attrs)}")
print()
print(print_program(normed, name="shader"))"""),
    md("""\
## The plan: types in, offsets out

`plan_from_types` walks env types then arg types and asks the layout policy
(`packed_dest`, the reference dense layout — real backends bring std140 and
friends at step 9) for a destination per leaf. **No value was consulted.**
The argument travels the same road as the captures — it is just the `arg`
root."""),
    code("""\
plan = plan_from_types(sh.env_types, (T.f64,), BUILTINS)
print(f"  {'source':<18} {'offset':>6}  fmt")
for spec in plan.slots:
    src = f"{spec.source.root}[{spec.source.index}]{list(spec.source.sub)}"
    print(f"  {src:<18} {spec.dest.offset:>6}  {spec.dest.fmt}")
print(f"  staging = {plan.staging_size} bytes; SlotSpec.convert (all None today) is the units seat")"""),
    md("""\
## Legalizing: the marshaling decision becomes IR

`legalize_params(plan)` rewrites every logical `core.env` into a physical
`abi.slot` carrying its byte offset and format.

The stage then proves its own point. A conversion target (`legal={core, abi}`
— which *dialects* may remain) **cannot** express "no capture survived":
`core.env` **is** `core`, so it would sail straight through. That is why the
stage also declares `forbid={"core.env"}` — an op-level check, run after the
rules reach fixpoint. Per-frame flattening is structurally impossible because
a machine checks it, not because the rule set happens to be total today. (The
first draft of this chapter claimed the namespace set did the work; the step-7
review caught the lie, and `Stage.forbid` is the fix.)"""),
    code("""\
final = run_stage(normed, legalize_params(plan), ALL_OPS)
print(print_program(final, name="shader"))
print()
print("artifact key:", final.key.hex()[:16], "… (offsets are in it: layout is identity)")"""),
    md("""\
## The payoff, in hex

Build the extractor once, then move only bytes. `build_extractor` compiles
the plan's leaf paths into **one getter per slot** — a chain of pure
index/attribute reads resolved against the *types* at build time — so the
per-call path does no kind dispatch and no recursive walk. (That is the
other half of killing M0's per-frame flatten: the IR no longer hunts for
uniforms, and neither does the extractor.) A second shader instance with
**different values but the same types** reuses the plan, the extractor, and
the artifact key untouched — the difference between the two calls is exactly
the staging bytes."""),
    code("""\
def hexdump(buf):
    return " ".join(f"{b:02x}" for b in buf)


extract = build_extractor(sh.env_types, (T.f64,), plan, BUILTINS)
staging = bytearray(plan.staging_size)
pack_into(plan, staging, extract(sh.captures, (0.0,)))
print("gain=2.0, center=(0.5, 0.25):")
print("  ", hexdump(staging))

sh2 = make_shader(0.5, 0.25, 9.0)  # new VALUES, same types
final2 = run_stage(
    run_stage(lower_handle(sh2, RULES, ALL_OPS, arg_types=(T.f64,)), NORMALIZE_ENV, ALL_OPS),
    legalize_params(plan),
    ALL_OPS,
)
staging2 = bytearray(plan.staging_size)
pack_into(plan, staging2, extract(sh2.captures, (0.0,)))  # SAME compiled extractor
print("gain=9.0, same types:")
print("  ", hexdump(staging2))
print()
print("plan reused:      ", plan == plan_from_types(sh2.env_types, (T.f64,), BUILTINS))
print("same artifact key:", final.key == final2.key)
print("same bytes:       ", bytes(staging) == bytes(staging2))"""),
    md("""\
## The output mirror: results are destinations

Kernels don't *return* values at the ABI — they write into destinations
they were handed (a compute kernel's out-buffer, a fragment's render
target). The functional `y = f(x)` stays the language-level truth; the
bridge is the **ResultPlan**: destinations allocated from the result type,
device bytes unflattened back into the logical value. Bidirectional from
the start — the output half is not bolted on later (040 §3b). Destinations
get reused across calls while types hold, same as staging."""),
    code("""\
rp = result_plan(T.Tuple((T.f64, T.f64)), BUILTINS)
out = bytearray(rp.size)
struct.pack_into("<d", out, 0, 0.125)  # stand-in for the device writing its destinations
struct.pack_into("<d", out, 8, 0.875)
print("destinations:", [(s.dest.offset, s.dest.fmt) for s in rp.slots], "| size", rp.size)
print("unflattened :", unpack_result(rp, out, BUILTINS))"""),
    code("""\
from pdum.dsl import viz

viz.install()
final  # hover a node: type + provenance survived all the way to the ABI stage"""),
    md("""\
## Things to notice

- Every `abi.slot`'s `src` matches an env path lowering stamped in ch07 —
  three layers (capture, lowering, marshaling) agree on one path language.
- The artifact key *changed* at legalization (offsets entered the IR):
  layout is identity at the artifact tier, exactly the two-tier law.
- `build_extractor` never calls `flatten` on the hot path — it compiles the
  plan's paths into getters. `flatten` remains the *reference* semantics, and
  a fuzz asserts the two roads agree; a kind whose `child` and `leaves`
  aspects disagreed would write the wrong value into the right slot, which no
  count check could ever see.
- `SlotSpec.convert` is `None` in every slot today. Step 15 puts an
  `Affine` mm→inch converter there: bytes change per call, plan and
  artifact never do — a unit tweak that cannot recompile.

## What we can't do yet

No `FastRecord` is wired: nothing *runs* the plan on a cache hit until the
Python backend and the hot path land (ch09 — `Registry` v1, guards armed,
`Handle.__call__`, the thesis test under `no_compile`). `core.param`s keep
their logical spelling: the physical calling convention is the renderer's
half of the deal (steps 8/9). And the leaves channel is empty until
ndarrays bring `BufferLeaf` (ch12)."""),
]

ch09 = [
    md("""\
# 9 — End to end, on CPU

Eight chapters of machinery; this is the payoff. A kernel goes from source
to typed IR to legalized ABI to **rendered Python you can read** to an
executed image — and then the loop the whole project exists for: hundreds of
frames of changing capture values, **one compile**, enforced by
`no_compile()` rather than hoped for.

New in this step: the `Registry` (surface E, v1) owning the rule packs,
backends, and both cache tiers; the Python backend (a SATELLITE — it
registers itself, zero kernel edits); and `Handle.__call__`. From this
chapter on, `import pdum.dsl` is batteries-included: the base dialect and
the backend wire themselves in, and the hand-registrations earlier chapters
performed as stand-ins are no longer needed."""),
    code("""\
import pdum.dsl  # batteries: base dialect + python backend register into DEFAULT
from pdum.dsl import jit, no_compile
from pdum.dsl.kernel.registry import DEFAULT


def make_disk(cx, cy, r, brightness):
    @jit()
    def disk(x, y):
        dx = x - cx
        dy = y - cy
        inside = 1.0 if dx * dx + dy * dy < r * r else 0.0
        return brightness * inside

    return disk


disk = make_disk(0.0, 0.0, 0.6, 1.0)
print("disk(0, 0)     =", disk(0.0, 0.0))
print("disk(0.9, 0.9) =", disk(0.9, 0.9))"""),
    md("""\
That is the first kernel this framework has ever *run*. Everything below is
the autopsy of what just happened — each stop on the miss path, printed.

## Stop 1: the typed, legalized IR"""),
    code("""\
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.lower import lower_handle
from pdum.dsl.kernel.ops import CORE_OPS
from pdum.dsl.kernel.pack import ABI_OPS, NORMALIZE_ENV, legalize_params, plan_from_types
from pdum.dsl.kernel.printer import print_program
from pdum.dsl.kernel.rewrite import run_stage
from pdum.dsl.kernel.valuekind import BUILTINS

OPS = {**CORE_OPS, **ABI_OPS}
region = lower_handle(disk, DEFAULT.lower_rules, OPS, arg_types=(T.f64, T.f64))
plan = plan_from_types(disk.env_types, (T.f64, T.f64), BUILTINS)
abi = run_stage(run_stage(region, NORMALIZE_ENV, OPS), legalize_params(plan), OPS)
print(print_program(abi, name="disk"))"""),
    md("""\
## Stop 2: the rendered source — read it

The backend renders that region to Python whose *calling convention is a
uniform buffer*: every input, captures and arguments alike, arrives as bytes
in staging, unpacked at the offsets the plan chose. Deliberately not the
fastest way to call Python from Python — it is the same ABI a WGSL uniform
block uses, proven on CPU first."""),
    code("""\
from pdum.dsl.demo.simple_shader.python import render

print(render(abi, plan, name="disk"))"""),
    md("""\
## Stop 3: an image

No arrays yet (ch12), so the host loops over pixels and calls the kernel
per sample — thousands of dispatches, every one a cache hit."""),
    code("""\
ROWS, COLS = 20, 40
shades = " ░▒▓█"
disk = make_disk(0.1, -0.1, 0.55, 1.0)
for j in range(ROWS):
    y = 1.0 - 2.0 * (j + 0.5) / ROWS
    line = ""
    for i in range(COLS):
        x = 2.0 * (i + 0.5) / COLS - 1.0
        v = disk(x, y)
        line += shades[int(v * (len(shades) - 1))]
    print(line)"""),
    md("""\
## Stop 4: the loop — the thesis, enforced

Every frame builds a **fresh closure** over new values (the factory
pattern), calls it, and must hit. `no_compile()` turns "must" into an
assertion: a cold key raises `CompileForbidden` naming the differing key
component instead of silently compiling.

Note what is already true before the loop starts: the ONE compile happened
at the chapter's very first `disk(0.0, 0.0)`, and the 800-pixel image above
was 800 cache hits. The loop below adds 299 fresh closures — and zero
compiles."""),
    code("""\
import time

spec = DEFAULT.specializations
c0, h0 = spec.compiles, spec.hits
t0 = time.perf_counter()
with no_compile():  # a single cold key in here would RAISE, naming the differing component
    for f in range(1, 300):
        frame_disk = make_disk(0.3 * (f % 7), -0.2, 0.4 + 0.001 * f, 1.0 / f)
        frame_disk(0.25, 0.25)
dt = (time.perf_counter() - t0) / 299
print(f"total compiles for this template, whole chapter: {spec.compiles} (image included)")
print(f"new compiles in the 299-frame loop             : {spec.compiles - c0}")
print(f"hits in the loop                               : {spec.hits - h0}")
print(f"guard misses                                   : {spec.guard_misses}")
print(f"per frame: {dt * 1e6:.1f} µs — capture + fingerprint + probe + guards + pack + run")"""),
    md("""\
## Stop 5: the `FastRecord`, field by field

The tier-1 value is everything the hot path needs, precomputed:"""),
    code("""\
key = DEFAULT.specializations.key_for(
    frame_disk, (BUILTINS.fingerprint(0.25),) * 2, ("demo.simple_shader.python", 1)
)
rec = DEFAULT.specializations._ready[key]
print("artifact :", rec.artifact, "(the exec'd function — it carries its listing)")
print("guards   :", [(type(h).__name__, n) for h, n, _ in rec.guards])
print("extract  :", rec.extract, "(compiled per-slot getters, ch08)")
print("plan     :", len(rec.plan.slots), "slots,", rec.plan.staging_size, "bytes staging")
print("staging  :", rec.staging.hex(), " <- reused every call; the only thing that changes")
print("launch   :", rec.launch is rec.artifact)"""),
    md("""\
## Stop 6: tripping a guard

Rebind a captured local *after* decoration and the cell drifts from the
frozen env. The guard (an identity triple per capture) catches it at the
next probe: the counter climbs and the entry is rebuilt — **refuse or
recompile, never silently stale**. Semantics stay decoration-time (the env
is a snapshot; rebuild the handle — the factory pattern — to move values)."""),
    code("""\
def sloppy():
    k = 2.0

    @jit()
    def f(x):
        return x * k

    k = 3.0  # drift: the cell no longer holds what was captured
    return f


g0 = DEFAULT.specializations.guard_misses
f = sloppy()
print("f(1.0) =", f(1.0), " (decoration-time k=2.0, not 3.0)")
print("f(1.0) =", f(1.0), " (same — and loudly counted, not silently served)")
print("guard misses:", DEFAULT.specializations.guard_misses - g0)"""),
    md("""\
## Stop 7: pipelines execute — and this is the house style now

`value > pipeline` dispatches through the *same* two-tier path: the fused
`Derived("pipe")` lowers via its build rule (no source!), renders as ONE
kernel, and the whole pipeline rebuilt with fresh stage values every frame
is still a single specialization."""),
    code("""\
from pdum.dsl.combinators import collect, op


@op
def tone(gain):
    @jit()
    def go(v):
        return v * gain

    return go


@op
def lift(b):
    @jit()
    def go(v):
        return v + b

    return go


print("0.5 > tone(2.0) | lift(0.1) | collect =", 0.5 > (tone(2.0) | lift(0.1) | collect))
c0 = DEFAULT.specializations.compiles
with no_compile():
    for f in range(1, 200):
        out = f / 200 > (tone(1.0 / f) | lift(0.01 * f) | collect)
print("fused-pipe compiles over 199 fresh pipelines:", DEFAULT.specializations.compiles - c0)"""),
    md("""\
## Stop 8: the content-addressed tier earns its keep

Two different templates with identical bodies are two *specializations* but
ONE *artifact* — and a generation bump (the coarse invalidation knob) clears
tier 1 while tier 2 survives untouched."""),
    code("""\
def site_a(k):
    @jit()
    def go(x):
        return x * k - 0.125

    return go


def site_b(k):  # a different def-site, an identical body
    @jit()
    def go(x):
        return x * k - 0.125

    return go


s0, a0 = DEFAULT.specializations.compiles, DEFAULT.artifacts.compiles
site_a(2.0)(3.0), site_b(5.0)(3.0)
print(f"specializations: +{DEFAULT.specializations.compiles - s0}   artifacts: +{DEFAULT.artifacts.compiles - a0}")
a1 = DEFAULT.artifacts.compiles
DEFAULT.specializations.bump_generation()
site_a(2.0)(3.0)
print(f"after bump_generation(): artifacts recompiled: +{DEFAULT.artifacts.compiles - a1} (tier 2 is generation-free)")"""),
    md("""\
## The kernel budget, in the book

The step-8 exit gate requires the line-budget report *in the notebook*: the
machinery you just watched run — capture, two-tier cache, IR, rewriter,
lowering, marshaling, registry — fits in this many counted lines:"""),
    code("""\
import json
import pathlib
import subprocess
import sys

budget = pathlib.Path(pdum.dsl.__file__).parents[3] / "scripts" / "loc_budget.py"
data = json.loads(subprocess.run([sys.executable, str(budget), "--json"],
                                 capture_output=True, text=True).stdout)
for name, f in data["files"].items():
    print(f"  {name:<14} {f['lines']:>4} / {f['cap']}")
print(f"  {'KERNEL TOTAL':<14} {data['kernel_total']:>4} / {data['kernel_cap']}")"""),
    md("""\
## Things to notice

- The staging hexdump in the `FastRecord` is the SAME buffer, mutated in
  place, every frame — `one value, N parameters` ended as `N values, zero
  allocations`.
- The per-frame cost includes rebuilding the closure from scratch. Phase A
  really is that cheap — that was ch02's bet, now carrying a render loop.
- The pipeline loop compiled once across 199 freshly-built pipelines:
  `Derived` identity (ch04) + the build rule (ch07) + this chapter's
  dispatch compose without any new mechanism.

## What we can't do yet

The image above was 800 Python dispatches — arrays and `core.for` (ch12)
turn it into ONE. The Python backend computes f32 as f64 and returns values
natively (the `ResultPlan` goes physical with a DPS target). Role-based
backend routing waits for a second backend — which is the next step: **WGSL
(ch10), completing M1**, where the staging buffer you have been hexdumping
becomes a real uniform block on a real GPU."""),
]

ch10 = [
    md("""\
# 10 — The GPU and the seam

**This chapter completes M1: the vertical slice.** The same IR that rendered
readable Python in ch09 now renders WGSL and runs on a real GPU — captures
become a uniform block, the staging buffer you hexdumped in ch08 becomes the
uniform's bytes, and the thesis holds with a Metal device on the other side.

Step 9 leads with **compute** (the 070 decision): the compute-family
contract v1 says a compute kernel's parameters ARE its thread coordinates —
`f(i, j)` runs once per grid point — and the call passes the launch DOMAIN:
`k(out=(W, H))`. The domain is runtime data riding the leaves channel, so
**changing resolution never recompiles**. Backend policies live in the
backend: WGSL has no f64, so this backend narrows to f32 (in the text AND
the layout) — the differential test below compares within f32 tolerance.

Cells tagged `gpu` execute only where an adapter answers; elsewhere the
harness skips them and the committed outputs (baked on an M3 Ultra, Metal)
remain."""),
    gpu("""\
import pdum.dsl  # noqa: F401  — batteries: the demo backends route themselves
from pdum.dsl import jit, no_compile
from pdum.dsl.demo.simple_shader import wgsl
from pdum.dsl.kernel.registry import DEFAULT

print("adapter:", wgsl.device().adapter_info if hasattr(wgsl.device(), "adapter_info") else "(ready)")


def make_disk(cx, cy, r, gain):
    @jit(kind="simple_shader.compute")  # demo-scoped kind, routed to the demo WGSL cell
    def disk(i, j):
        x = i / 32.0 - 1.0
        y = j / 32.0 - 1.0
        dx = x - cx
        dy = y - cy
        return gain * (1.0 if dx * dx + dy * dy < r * r else 0.0)

    return disk


img = make_disk(0.1, -0.1, 0.55, 1.0)(out=(64, 64))
shades = " ░▒▓█"
for j in range(0, 64, 4):
    print("".join(shades[int(img[j * 64 + i] * 4)] for i in range(0, 64, 2)))"""),
    md("""\
That image was computed by a WGSL kernel dispatched over a 64×64 grid on
the GPU — captures (`cx, cy, r, gain`) traveled as a uniform block packed
from the same `PackPlan` machinery as every chapter since ch08.

## One IR, two texts

The SAME body, lowered once to the same core IR, rendered by two backends:"""),
    gpu("""\
key = next(k for k in DEFAULT.specializations._ready if k[2][:2] == ("demo.simple_shader.wgsl", "compute"))
print(DEFAULT.specializations._ready[key].artifact.__pdum_source__)"""),
    gpu("""\
def make_cpu_twin(cx, cy, r, gain):
    @jit(kind="device")  # identical body, python backend
    def disk(i, j):
        x = i / 32.0 - 1.0
        y = j / 32.0 - 1.0
        dx = x - cx
        dy = y - cy
        return gain * (1.0 if dx * dx + dy * dy < r * r else 0.0)

    return disk


cpu = make_cpu_twin(0.1, -0.1, 0.55, 1.0)
cpu(0.0, 0.0)
pkey = next(k for k in DEFAULT.specializations._ready if k[2][0] == "demo.simple_shader.python")
print(DEFAULT.specializations._ready[pkey].artifact.__pdum_source__)"""),
    md("""\
## The uniform layout IS the plan

`plan_from_types` with the WGSL backend's `dest_for` policy: the same f64
captures that packed as 8-byte `<d` slots for Python pack as 4-byte `<f`
slots here — **narrowing is a backend layout policy, not a kernel concern**.
(The vec3 align-16 footgun arrives with vec types in the uniform address
space — ch12's problem, M0's `layout.py` documents it.)"""),
    code("""\
from pdum.dsl.demo.simple_shader.wgsl import COMPUTE
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.pack import plan_from_types
from pdum.dsl.kernel.valuekind import BUILTINS

env = (T.f64, T.f64, T.f64, T.f64)  # cx, cy, r, gain
py_plan = plan_from_types(env, (), BUILTINS)
gpu_plan = COMPUTE.plan(env, (), BUILTINS)
print(f"  {'capture':<10} {'python offset:fmt':<20} wgsl offset:fmt")
for name, a, b in zip(("cx", "cy", "r", "gain"), py_plan.slots, gpu_plan.slots):
    print(f"  {name:<10} {a.dest.offset:>3} : {a.dest.fmt:<12} {b.dest.offset:>3} : {b.dest.fmt}")
print(f"  staging: {py_plan.staging_size} bytes on cpu, {gpu_plan.staging_size} on gpu — same values, two ABIs")"""),
    md("""\
## The differential gate

M1's claim is a *seam*: any backend behind it computes the same function.
The same body runs per-pixel on the CPU (f64) and per-thread on the GPU
(f32); they must agree to f32 precision:"""),
    gpu("""\
worst = 0.0
for j in range(64):
    for i in range(64):
        worst = max(worst, abs(img[j * 64 + i] - cpu(float(i), float(j))))
print(f"max |gpu - cpu| over 4096 pixels: {worst:.2e}  (f64 vs f32 seam, branch-exact)")"""),
    md("""\
## The thesis, on a GPU

The loop the project exists for: every frame rebuilds the closure with new
values — and now also changes RESOLUTION mid-loop — under `no_compile()`.
Values move as uniform bytes; the domain moves as launcher data; the
artifact never recompiles."""),
    gpu("""\
import time

make_disk(0.0, 0.0, 0.5, 1.0)(out=(64, 64))  # warm (already warm from above)
spec = DEFAULT.specializations
c0, t0 = spec.compiles, time.perf_counter()
with no_compile():
    for f in range(1, 120):
        frame = make_disk(0.4 * (f % 5) / 5, -0.1, 0.3 + 0.002 * f, 1.0)
        frame(out=(64, 64))
    big = make_disk(0.0, 0.0, 0.7, 1.0)(out=(256, 256))  # resolution: runtime data
dt = (time.perf_counter() - t0) / 120
print(f"new compiles across 119 frames + one resolution change: {spec.compiles - c0}")
print(f"per frame (64x64 dispatch + BLOCKING readback): {dt * 1e3:.2f} ms")
print(f"256x256 output arrived with the same artifact: {len(big)} values")
print()
print("microbench readout: the ~ms is the synchronous READBACK round trip —")
print("a render loop that draws instead of reading back pays only write_buffer")
print("+ encode (µs-scale, R12 cost ranking). Readback is the boundary act.")"""),
    md("""\
## The fragment variant

Same machinery, thinner than you'd think: swap the compute stage for a
fullscreen triangle, dispatch for `draw(3)`, storage-out for a texture.
Params become pixel coordinates; the scalar result broadcasts to grayscale
(colors arrive with tuples at step 10). This is M0's disk, reborn behind
the seam:"""),
    gpu("""\
def make_shader(cx, cy, r):
    @jit(kind="simple_shader.fragment")
    def shader(x, y):
        u = x / 32.0 - 1.0
        v = y / 32.0 - 1.0
        dx = u - cx
        dy = v - cy
        return 1.0 if dx * dx + dy * dy < r * r else 0.0

    return shader


rows = make_shader(-0.2, 0.15, 0.5)(out=(64, 64))
for j in range(0, 64, 4):
    print("".join("█" if rows[j][i] > 0.5 else "·" for i in range(0, 64, 2)))
with no_compile():
    make_shader(0.3, -0.2, 0.35)(out=(64, 64))  # moved disk: uniform bytes only
print("fragment thesis: the moved disk was a cache hit")"""),
    md("""\
## Things to notice

- `@workgroup_size` is baked into the WGSL text — pipeline-creation-time by
  spec — so the workgroup size lives in the artifact key exactly where 040
  §3b's block/grid split said it must. Grid (the domain) strips to the
  leaves channel: the 256×256 dispatch was a cache HIT.
- The Env struct's members are named by byte offset (`env.f8`) — the
  uniform layout is the printable plan, not a convention.
- Routing is role-based, and names are honest about their maturity:
  `kind="simple_shader.compute"` / `"simple_shader.fragment"` are
  DEMO-scoped kinds routed to the demo cells — the plain names `compute`
  and `fragment` stay reserved for the real families (080). `kind="device"`
  (the `@jit` default, owned by stdlib since ch04) has no route at all: it
  falls to the registry's DEFAULT backend, which today is the Python demo
  twin. The kinds registered themselves FROM their owning package — the
  ch04 rule, honored by its second customer.

## What we can't do yet

The `[grid, block, smem, stream]` bracket surface (070 §3) — today block is
a fixed backend constant and the domain rides `out=`. Colors and vec types
(tuples, step 10). Arrays and real out-buffers via DLPack (ch12). The
interactive window demo returns with the graphics `draw(target)` surface —
`examples/webgpu/` holds a terminal animation in the meantime. **M1 is
complete: two backends, one IR, the thesis measured on both.**"""),
]

ch11 = [
    md("""\
# 11 — The five surfaces

Step 10 completes the Registry and proves the extension-locality law: every
capability in this chapter lands through a **registration**, and the kernel
— now at exactly its 1150-line cap, finished — was not edited to make any
of it possible.

The five doors (architecture D3):
**A** ops (`defop`) · **B** overloads/batteries (`intrinsic`, `@overload`,
record methods) · **C** types (`@record`) · **D** backends (`spell`,
`code_for_op`) · **E** the Registry itself (`extend()` layering,
entry-point discovery).

Also in this step: the ch07a **table stakes** graduated — aug-assign,
short-circuit `and`/`or`, chained comparisons, tuples with unpacking, and
subscripts are base-pack registrations now. The next interlude (11a)
records the deltas."""),
    code("""\
import pdum.dsl  # noqa: F401  — batteries: base dialect + demo backends + the battery pack
from pdum.dsl import jit, no_compile
from pdum.dsl.kernel.registry import DEFAULT


def make(lo, hi):
    @jit()
    def f(x):
        x += 1.0                          # aug-assign (was a MissingRule in ch07a)
        a, b = x * 2.0, x + 0.5           # tuple literal + unpack
        a, b = b, a                       # the swap idiom
        inside = 0.0 < x < 10.0 and a > b  # chained compare + short-circuit and
        return clamp(a, lo, hi) if inside else smoothstep(0.0, 1.0, x)

    return f


f = make(0.0, 5.0)
print("f(2.0)  =", f(2.0), "  (clamped upper branch)")
print("f(-3.0) =", f(-3.0), " (smoothstep branch — x+1 < 0)")"""),
    md("""\
`and` compiles to `core.if` with the untaken side owned by its branch — the
dominator-placed renderers keep short-circuit REAL on every backend, no new
mechanism. `clamp` and `smoothstep` are **DSL-written batteries**: ordinary
Python registered through `@overload`, inlined at every call site, portable
to every target that exists or ever will.

## Surface B + D: intrinsics and spellings

`sqrt` is the other battery species: an *op* (`math.sqrt`) with a per-target
*spelling*. The rendered source shows the spelling chosen for this target:"""),
    code("""\
@jit()
def hyp(a, b):
    return sqrt(a * a + b * b)


print("hyp(3, 4) =", hyp(3.0, 4.0))
key = next(iter(DEFAULT.specializations._ready))
print(DEFAULT.specializations._ready[key].artifact.__pdum_source__)"""),
    md("""\
## The plan's promised demo: `sinh`, live, without restarting

Register the op (A), the call name (B) — and NO spelling. The shared
decomposition fires because the target lacks the native op (the §2.10 gate:
`op not in backend.code_for_op`). Then hand the target a native spelling
and watch the SAME kernel shape compile differently:"""),
    code("""\
import math

from pdum.dsl.kernel.rewrite import Pat
from pdum.dsl.stdlib.surfaces import defop, intrinsic, spell

defop(DEFAULT, "math.sinh", lambda args, attrs, regions: args[0])
intrinsic(DEFAULT, "sinh", "math.sinh")
DEFAULT.decompositions.append((
    "math.sinh",
    (Pat("math.sinh"), lambda b, m: b.emit(
        "core.div",
        b.emit("core.sub",
               b.emit("math.exp", m["root"].args[0]),
               b.emit("math.exp", b.emit("core.neg", m["root"].args[0]))),
        b.emit("core.const", type=m["root"].type, value=2.0),
    )),
))


@jit()
def s1(x):
    return sinh(x)


print("decomposed:", s1(0.3), "vs math.sinh:", math.sinh(0.3))
src = list(DEFAULT.specializations._ready.values())[-1].artifact.__pdum_source__
print("  rendered via:", "exp-decomposition" if "math.exp(" in src else "native")

spell(DEFAULT, "demo.simple_shader.python", "math.sinh", "math.sinh({0})")


@jit()
def s2(x):
    return sinh(x)  # same shape, NEW build: the gate now finds the native op


print("native    :", s2(0.3))
src2 = list(DEFAULT.specializations._ready.values())[-1].artifact.__pdum_source__
print("  rendered via:", "native math.sinh" if "math.sinh(" in src2 else "decomposition")"""),
    md("""\
## Surface C: the `Color` record, end to end

A frozen dataclass becomes a first-class captured value: `typeof` gives a
`Record`, its fields flatten to uniform slots (the ch08 walkers already knew
how), and its **methods inline** like any callee — the ch07a jitclass story,
delivered.

One deliberate detail: `Color` lives in `pdum.dsl.demo.graphics`, NOT the
stdlib. Real color modeling — spaces, RGB vs Lab vs OkLab — is a domain
library's ground, and the stdlib must not squat on it (design doc 090). The
whole point of the five surfaces is that domain vocabulary arrives like any
package: one import wires it in."""),
    code("""\
from pdum.dsl.demo.graphics import Color  # demo vocabulary: the import IS the installation

c = Color(0.8, 0.4, 0.2)
c = Color(0.8, 0.4, 0.2)


def tint(color, k):
    @jit()
    def f(x):
        return color.luminance() * x + color.scaled(k)[0]

    return f


print("tint =", tint(c, 0.5)(1.0))
key = next(iter(reversed(DEFAULT.specializations._ready)))
rec = DEFAULT.specializations._ready[key]
print("the Color capture became", len(rec.plan.slots) - 0, "uniform slots:")
for s in rec.plan.slots:
    print("  ", s.source, "->", s.dest)"""),
    gpu("""\
def shade(color, cx, cy, r):
    @jit(kind="simple_shader.compute")
    def disk(i, j):
        x = i / 32.0 - 1.0
        y = j / 32.0 - 1.0
        d = sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy))
        return color.luminance() * (1.0 - smoothstep(r - 0.1, r + 0.1, d))

    return disk


img = shade(Color(0.9, 0.7, 0.3), 0.0, 0.0, 0.6)(out=(64, 64))
shades = " ░▒▓█"
for j in range(0, 64, 4):
    print("".join(shades[min(4, int(img[j * 64 + i] * 6))] for i in range(0, 64, 2)))
c0 = DEFAULT.specializations.compiles
with no_compile():
    shade(Color(0.2, 0.9, 0.5), 0.3, -0.2, 0.4)(out=(64, 64))
print("new Color, new geometry, same types ->", DEFAULT.specializations.compiles - c0, "compiles")
print("(smoothstep INLINED into WGSL; sqrt spelled natively; Color = 3 uniform floats)")"""),
    md("""\
The soft edge is `smoothstep` — the same DSL battery that ran on the CPU —
inlined into the WGSL text. The batteries did not know this backend exists.

## Surface E: layering and discovery

`extend()` gives a child registry with copied registrations and FRESH
caches: a session can add vocabulary without touching the parent — the
stdlib → user → session story:"""),
    code("""\
child = DEFAULT.extend()
defop(child, "math.cbrt", lambda args, attrs, regions: args[0])
intrinsic(child, "cbrt", "math.cbrt")
spell(child, "demo.simple_shader.python", "math.cbrt", "({0} ** (1.0/3.0))")


@jit()
def croot(x):
    return cbrt(x)


print("child :", child.dispatch(croot, (27.0,)))
print("parent knows math.cbrt?", "math.cbrt" in DEFAULT.ops)


class FakeEP:  # what a pip-installed backend distribution provides (080 §4)
    name = "acme-npu"

    def load(self):
        return lambda registry: print("  acme-npu install(registry) called")


print("entry points loaded:", DEFAULT.extend().load_entry_points(entries=[FakeEP()]))"""),
    md("""\
## The exit gate: battery economics

The numba lesson (architecture risk #4): batteries written IN the language
are portable to every target for free; hand-spelled intrinsics cost one
entry per op per target, forever. The count so far:"""),
    code("""\
dsl_written = [n for n, v in DEFAULT.overloads.items() if not isinstance(v, (str, tuple)) or isinstance(n, tuple)]
intrinsics = sorted({v for v in DEFAULT.overloads.values() if isinstance(v, str)})
spellings = sum(len(b.code_for_op) for b in DEFAULT.backends.values())
print(f"DSL-written batteries (portable to ALL targets, free): {len(dsl_written)}")
print(f"hand-spelled intrinsic ops:                            {len(intrinsics)}")
print(f"spelling entries paid across today's targets:          {spellings}")
print()
print("Every new target pays ~", len(intrinsics), "spellings and inherits", len(dsl_written), "batteries free.")
print("The ratio improves monotonically with both axes — that is the numba 2:1")
print("economics, structural instead of aspirational.")"""),
    md("""\
## Things to notice

- The kernel is FINISHED: 1150/1150 counted lines, and this entire chapter
  — new syntax, new ops, new types, new spellings, layered registries —
  landed with **zero kernel diffs**. The extension-locality law is now a
  test (`tests/test_surfaces.py`), not a slogan.
- Tuples exist on the GPU only as a fiction: `extract`-of-`tuple` folds away
  via a decomposition GATED on `core.tuple ∉ code_for_op` — the same gate
  that chose sinh's fate. One mechanism, two jobs.
- Record methods are inlined DSL code — `luminance()` became three
  multiplies in both backends' rendered source.

## What we can't do yet

Vec/color RETURNS (a fragment still yields one scalar — vec types land with
the grown-up WGSL backend); `@overload` target-token MRO (deferred until a
battery's BODY must differ per target — decompositions cover today's cases);
`@overload_attribute`; `to_oklab` (wants vec math). Statements (`if`/`for`,
early return) are step 11, where arrays make them worth having. **Next: the
lay-of-the-land delta (11a), then arrays.**"""),
]

ch11a = [
    md("""\
# 11a — Lay of the land: the step-10 deltas

The second interlude (convention: 020 §The book). NOT a restatement of the
ch07a baseline — only what CHANGED. The base pack grew from 10 to 15 syntax
forms; run the ch07a refusal kernels and watch three of the six graduate:"""),
    code("""\
import pdum.dsl  # noqa: F401
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.lower import lower_handle
from pdum.dsl.kernel.ops import CORE_OPS
from pdum.dsl.kernel.registry import DEFAULT
from pdum.dsl.stdlib.base_lang import LOWER_RULES

print("the base pack now accepts", len([k for k in LOWER_RULES if isinstance(k, type)]), "syntax forms:")
for node_type in LOWER_RULES:
    if isinstance(node_type, type):
        print("  ✅ ast." + node_type.__name__)"""),
    code("""\
# ch07a's refusal kernels, re-run against today's pack:
@jit()
def wants_tuple(x):
    return (x, x)


@jit()
def wants_boolop(x):
    return x > 0.0 and x < 1.0


@jit()
def wants_augassign(x):
    x += 1.0
    return x


@jit()
def wants_for(x):
    total = x
    for i in range(4):
        total = total + x
    return total


rules = dict(LOWER_RULES)
for h in (wants_tuple, wants_boolop, wants_augassign, wants_for):
    try:
        lower_handle(h, rules, {**CORE_OPS, **DEFAULT.ops}, arg_types=(T.f64,), context={"registry": DEFAULT})
        print(f"  ✅ {h.fntype.template.label}: lowers now (step 10)")
    except Exception as e:
        print(f"  🏗️ {type(e).__name__}: {e}")"""),
    md("""\
## The delta table

| construct | ch07a | now |
|---|---|---|
| aug-assign `x += 1` | 🔧 | ✅ base pack |
| `and`/`or` | 🔧 + policy | ✅ short-circuit via `core.if` (policy: strict bool operands; truthiness stays a dialect choice) |
| chained `0 < x < 1` | 🔧 | ✅ operand evaluated once (shared node) |
| tuples + unpacking + `t[0]` | 🔧 | ✅ `core.tuple`; folds away on targets that cannot spell it |
| `.field` attribute | 🔧 pending records | ✅ `@record` dataclasses + inlined methods |
| calls beyond casts/captures | 🚫 | ✅ overloads: intrinsics + DSL batteries (surface B) |
| `if`/`for` statements, early return, `while`, `a[i]` on arrays | 🏗️ | 🏗️ step 11, unchanged |

Still true from ch07a: single tail return (settled), no NaN-as-data, loud
refusals everywhere else. The next delta lands with the arrays step."""),
]

ch11b = [
    md("""\
# 11b — Measuring the machine

ch10 reported "~1.8 ms per frame" and hypothesized where it went. A
hypothesis is not an instrument. This half-step (10b, inserted at the ch10
walkthrough) builds the measurement tools the next steps will lean on —
step 11's ray-march verdict and ch15's four-target comparison must not rest
on un-decomposed wall clock.

Three tools, all satellites (the kernel was finished last chapter and was
not touched):

- **`bench.benchmark`** — BenchmarkTools-style adaptive sampling: warm up,
  TUNE evals-per-sample above the timer's resolution floor, then sample to a
  budget. The **minimum** is the estimator (noise on a quiet machine is
  strictly additive).
- **`bench.instrument`** — phase decomposition of the hot path, riding the
  kernel's EVENT SEAM (design 120): arming a sink routes `dispatch` through
  a traced twin that stamps every phase. (Historical note: this shipped at
  step 10b as a monkeypatch on `FastRecord` fields because the kernel budget
  had 3 lines of headroom; the budget conversation that followed is design
  120 §9 — the seam replaced the surgery.)
- **`bench.gpu_timeline`** — WebGPU `timestamp-query` begin/end-of-pass
  clocks (nanoseconds by spec) splitting `launch` into encode+submit / GPU
  execution / readback. The adapter feature is requested by the demo device
  when present."""),
    code("""\
import pdum.dsl  # noqa: F401
from pdum.dsl import jit, viz
from pdum.dsl.bench import benchmark, instrument, phase_timeline

viz.install()


def make(cx, gain):
    @jit()
    def f(x):
        return gain * (x - cx)

    return f


f = make(0.1, 2.0)
f(1.0)
trial = benchmark(lambda: f(1.0))
print("hot path      :", trial)
rebuild = benchmark(lambda: make(0.1, 2.0)(1.0))
print("rebuild + call:", rebuild)
print()
print(f"the full factory pattern costs {(rebuild.minimum - trial.minimum) * 1e6:.2f} µs of phase A per frame")
print(f"evals/sample was TUNED to {trial.evals} — a naive one-call timing of a")
print(f"~{trial.minimum * 1e6:.0f} µs path would be mostly measuring the clock")"""),
    md("""\
## Where the microseconds go

`instrument` wraps the warm record's `extract` and `launch` with timestamp
shims for a few frames, then restores them. The phases are the hit path's
anatomy from ch09, now with numbers attached — rendered by the new timeline
widget (static HTML, like every widget in this book):"""),
    code("""\
phases = instrument(f, 1.0, frames=200)
for name in ("key+probe", "extract", "pack", "launch"):
    print(f"  {name:<10} {phases[name] * 1e6:6.2f} µs")
print(f"  {'total':<10} {phases['total'] * 1e6:6.2f} µs")
phase_timeline(phases)"""),
    md("""\
## The ch10 question, answered

One 256×256 GPU frame, decomposed. The `gpu` lane is the begin/end-of-pass
timestamp delta — what the silicon actually spent:"""),
    gpu("""\
from pdum.dsl.bench import gpu_timeline


def gmake(cx):
    @jit(kind="simple_shader.compute")
    def k(i, j):
        return (i / 128.0 - cx) * (j / 128.0)

    return k


tl = gpu_timeline(gmake(0.5), out=(256, 256))
for name, _, dur, lane in tl.spans:
    print(f"  {name:<14} {dur * 1e6:8.1f} µs   [{lane}]")
print(f"  {'frame total':<14} {tl.total * 1e3:8.2f} ms")
tl"""),
    md("""\
The verdict, with instruments instead of adjectives: the GPU computes this
kernel in single-digit **micro**seconds. The milliseconds are the blocking
readback (a full submit→wait→map round trip — the *boundary act* ch08
promised would be the only place bytes cross) plus bridge encode/submit.
A render loop that draws instead of reading back never pays the big span.

## Scaling: what actually moves"""),
    gpu("""\
for side in (64, 256, 1024):
    tl = gpu_timeline(gmake(0.5), out=(side, side), frames=5)
    spans = {n: d for n, _, d, _ in tl.spans}
    print(
        f"  {side:>4}²  encode {spans['encode+submit'] * 1e6:7.1f} µs   "
        f"gpu {spans['gpu'] * 1e6:7.1f} µs   readback {spans['readback'] * 1e6:8.1f} µs"
    )
print()
print("the surprise: readback barely moves from 64² (16 KB) to 1024² (4 MB) —")
print("it is dominated by FIXED sync latency (submit→wait→map), not bandwidth.")
print("gpu time is micro-scale and quantizes noisily at one-pass resolution.")
print("host phases from the previous section never appear at this magnification.")"""),
    md("""\
## Things to notice

- The thresholds from the step-9 gate table are now REAL: the suite fails if
  the warm hit path exceeds the fail line (`tests/test_bench.py`), instead
  of a notebook printing a number nobody re-reads.
- `instrument` works on ANY backend and never touches a record — it reads
  the same event stream everything else does. ch15 will point it at four
  targets.
- The timeline widget is ~35 lines of the viz satellite: spans in, static
  HTML out, hover for µs. No scripts, both notebook hosts."""),
    md("""\
## The event seam: counting the expensive things

Design 120's addition. Every should-be-rare occurrence — a miss, a compile,
a guard drift, an eviction — announces itself on a kernel seam that costs
one list check when dark. `events.record()` turns a block into a report:
exact counts always; stacks and timings sampled; ten thousand events from
one loop interned to ONE stack. The bug class that motivated it — a capture
rebound every frame, recompiling forever with correct answers and no
symptom — is now both visible and assertable:"""),
    code("""\
from pdum.dsl import events


def drifty_pair():
    c = 1.0

    @jit()
    def g(x):
        return c * x

    def rebind(v):
        nonlocal c
        c = v

    return g, rebind


g, rebind = drifty_pair()
g(1.0)
with events.record() as ev:
    for i in range(50):
        rebind(float(i))  # the same cell, new contents, every frame
        g(1.0)
print(ev)
print()
print("the ONE stack behind all of it:", ev["guard.drift"].exemplars[0].trace.user_frames[0])"""),
    code("""\
# The assertion you could not write before (120 §6.4): captures are stable.
stable = make(0.5, 2.0)
stable(1.0)
with events.forbid("guard.drift"):
    for _ in range(100):
        stable(1.0)
print("100 frames, zero drift — now a CI-shaped guarantee")


# And the miss path, dark since step 8, is a phase tree:
def novel(c):
    @jit()
    def k(x):
        return sqrt(x * c) + c  # noqa: F821 — a body this notebook has never compiled

    return k


with events.record() as ev2:
    novel(2.5)(1.0)  # one genuinely cold compile
print(ev2)"""),
    md("""\
## What we can't do yet

Per-op GPU attribution (one timestamp pair per PASS is the WebGPU deal;
finer needs vendor tools); async/double-buffered readback (the fix for the
big span is API-shaped — arrives with the graphics `draw(target)` surface);
CUDA events and Metal `gpuStartTime` (their backends, step 14). **Next:
arrays, `core.for`, and the C backend — step 11, with a ray-march verdict
these instruments can now referee.**"""),
]

notebook(ch00, "docs/book/ch00-thesis.ipynb")
notebook(ch01, "docs/book/ch01-types-are-values.ipynb")
notebook(ch02, "docs/book/ch02-what-a-closure-is.ipynb")
notebook(ch03, "docs/book/ch03-one-compile-per-signature.ipynb")
notebook(ch04, "docs/book/ch04-pipelines-are-values.ipynb")
notebook(ch05, "docs/book/ch05-programs-are-values.ipynb")
notebook(ch06, "docs/book/ch06-everything-is-a-rule.ipynb")
notebook(ch07, "docs/book/ch07-source-to-ir.ipynb")
notebook(ch07a, "docs/book/ch07a-lay-of-the-land.ipynb")
notebook(ch08, "docs/book/ch08-one-value-n-parameters.ipynb")
notebook(ch09, "docs/book/ch09-end-to-end-on-cpu.ipynb")
notebook(ch10, "docs/book/ch10-the-gpu-and-the-seam.ipynb")
notebook(ch11, "docs/book/ch11-the-five-surfaces.ipynb")
notebook(ch11a, "docs/book/ch11a-lay-of-the-land.ipynb")
notebook(ch11b, "docs/book/ch11b-measuring-the-machine.ipynb")

# ══════════════════════════════════════════ ch12 ══════════════════════════════

ch12 = [
    md("""\
# Chapter 12 — Data and loops

Step 11, and the language grows a body: statement `if`/`for`, arrays as
captures, a **C backend** (the seam's first non-GPU target), and — a
user-directed exercise — **named axes**, the xarray idea, done pedantically.
Design: `docs/design/100_arrays-and-axes.md`; the punning charter it
implements is `090_core-and-extensions.md`.

The kernel was finished two chapters ago, and it shows: everything in this
chapter is satellite registrations — one kernel line was spent (the `device`
field on `Array`, 090's dispatch axis, waiting for step 14), and nothing
else moved."""),
    code("""\
import numpy as np

import pdum.dsl  # noqa: F401
from pdum.dsl import jit, no_compile
from pdum.dsl.kernel.registry import DEFAULT"""),
    md("""\
## Statements: strict joins, bounded loops

`if`/`else` statements lower to `core.if` yielding the JOIN of rebound
locals — and the join is **strict**: a name assigned on either path must
have the SAME type on both, no unification, no fixpoint (ch07a's "what type
inference is here" promise, kept). A name born in only ONE suite is a
branch-local temporary: it dies with its suite (reading it later is a loud
name error), exactly as loop-locals die with their loop. `for i in range(...)` lowers to
`core.for` with the rebound pre-existing locals as **loop carries**. The
loop variable dies with the loop; `while`, `break`, `continue` are refused
by POLICY — bounded loops are the GPU-honest subset every serious kernel
language settles on (R11)."""),
    code("""\
def make(gain):
    @jit()
    def f(x):
        total = 0.0
        count = 0
        for i in range(10):
            v = float(i) * gain
            if v > 2.0:
                total = total + v
                count = count + 1
            else:
                total = total + x
        return total + float(count)

    return f


f = make(0.5)
print("f(1.5) =", f(1.5))
c0 = DEFAULT.specializations.compiles
with no_compile():  # a fresh closure, same types: the thesis, now with control flow
    make(0.7)(2.5)
print("fresh closure -> compiles:", DEFAULT.specializations.compiles - c0)"""),
    code("""\
key = DEFAULT.specializations.key_for(f, (DEFAULT.table.fingerprint(1.5),), DEFAULT.backend_for(f.kind).fp)
print(DEFAULT.specializations.probe(key).artifact.__pdum_source__)"""),
    md("""\
Read the render: the loop carries `(total, count)` as ONE tuple value, the
branch is a real lazy `if`/`else` (dominator placement — work shared by both
paths hoists, branch-exclusive work stays inside), and `v10 > 2.0` gates a
staging read that only the else-path needs. Guard-then-divide keeps working
inside loops for the same reason it worked in ch09."""),
    code("""\
# The refusals ARE the language definition — each names its policy:
@jit()
def wants_while(x):
    while x > 0.0:
        x = x - 1.0
    return x


@jit()
def wants_break(x):
    acc = 0.0
    for i in range(3):
        acc = acc + x
        break
    return acc


@jit()
def early_return(x):
    if x > 0.0:
        return x
    return -x


@jit()
def one_sided(x):
    if x > 0.0:
        y = x  # noqa: F841 — the refusal below IS the lesson
    else:
        pass
    return x


@jit()
def bad_join(x):
    y = 0.0
    if x > 0.0:
        y = 1
    else:
        y = 1.0
    return float(y)


for k in (wants_while, wants_break, early_return, one_sided, bad_join):
    try:
        k(1.0)
        print(f"  ✅ {k.fntype.template.label}")
    except Exception as e:
        print(f"  🚫 {type(e).__name__}: {str(e)[:100]}")"""),
    md("""\
## Arrays are captures — and shape is a VALUE

A numpy array captured by a kernel is summarized by `typeof` like every
capture: `Array(dtype, rank, layout, device)` — **rank, not shape**. Shape
and strides ride the staging buffer as i64 slots (uniforms, in GPU terms);
the payload travels the leaves channel as a pointer. So the thesis extends
to data: *a new shape is a cache HIT* — only rank, dtype, or device changes
recompile.

The classic use: a color table. Map a scalar through a captured palette —
three loads and a luminance dot:"""),
    code("""\
palette = np.array([[0.267, 0.005, 0.329],
                    [0.229, 0.322, 0.546],
                    [0.128, 0.567, 0.551],
                    [0.993, 0.906, 0.144]])  # viridis, heavily abridged


def colorize(table):
    n = table.shape[0] - 1  # host-side int: a plain i64 capture

    @jit()
    def lum(u):
        k = int(u * float(n) + 0.5)
        return 0.2126 * table[k, 0] + 0.7152 * table[k, 1] + 0.0722 * table[k, 2]

    return lum


lum = colorize(palette)
print("lum(0.0) =", lum(0.0), " lum(1.0) =", lum(1.0))
c0 = DEFAULT.specializations.compiles
with no_compile():  # an 8-row palette: DIFFERENT shape, same rank -> hit
    colorize(np.ones((8, 3)))(0.5)
print("new shape, same rank -> compiles:", DEFAULT.specializations.compiles - c0)
palette[3] = 0.0  # in-place edit: buffers are DATA (the pointer travels per call)
print("after palette edit, lum(1.0) =", lum(1.0))"""),
    code("""\
key = DEFAULT.specializations.key_for(lum, (DEFAULT.table.fingerprint(0.0),), DEFAULT.backend_for(lum.kind).fp)
print(DEFAULT.specializations.probe(key).artifact.__pdum_source__)"""),
    md("""\
`leaves[0]` is the payload pointer; the `_u('<q', staging, ...)` reads are
the STRIDES, packed per call like any uniform. That is what "rank-generic"
costs at runtime: two multiplies. And it is what the hit across shapes buys.

## The dial: shape in the type when you want it

§13 calls `typeof` a *summary function* with a dial. `Shaped(x)` turns it
one notch: the full shape enters the type — one specialization per shape,
strides become compile-time CONSTANTS, and the shape/stride slots leave
staging entirely:"""),
    code("""\
from pdum.dsl.stdlib.arrays import Shaped


def colorize_shaped(table):
    n = table.shape[0] - 1

    @jit()
    def lum(u):
        k = int(u * float(n) + 0.5)
        return 0.2126 * table[k, 0] + 0.7152 * table[k, 1] + 0.0722 * table[k, 2]

    return lum


s = colorize_shaped(Shaped(np.ascontiguousarray(palette)))
s(0.5)
c0 = DEFAULT.specializations.compiles
colorize_shaped(Shaped(np.ones((8, 3))))(0.5)  # new shape IS a new type now
print("Shaped, new shape -> compiles:", DEFAULT.specializations.compiles - c0)
fp = (DEFAULT.table.fingerprint(0.5),)
rec = DEFAULT.specializations.probe(DEFAULT.specializations.key_for(s, fp, DEFAULT.backend_for(s.kind).fp))
print("staging bytes with shape in the type:", rec.plan.staging_size, "(just the f64 argument)")"""),
    md("""\
Neither notch is "right". Rank-generic is the render-loop default (palettes
get resized); shape-in-type is the numerics default (unroll against a fixed
64×64 tile). The dial is per-VALUE, the mechanism is one `typeof` choice,
and both notches were already paid for by the two-tier cache.

## Named axes — the xarray exercise

A user-directed detour (100 §2): xarray names its axes, and *most kernel
languages pretend that idea does not exist*. We take the opposite, pedantic
position. `Named(a, ("y", "x"))` — or any `xarray.DataArray` — summarizes as
`NamedArray(..., dims)`: the names live IN the type. The consequences
follow mechanically:

- `isel(y=…, x=…)` keywords are **mandatory**; keyword ORDER is free.
- Positional indexing on a named array is **refused** — transposition is
  precisely the bug names exist to kill, so there is no back door.
- Renaming axes is a *different type* (recompile — and stale `isel` calls
  refuse loudly at lower time, naming the axes that do exist).
- The names are GONE by codegen. Zero runtime cost — because names live on
  the types-not-values side of the line, like every identity in this
  system."""),
    code("""\
from pdum.dsl.stdlib.arrays import Named

field = Named(np.arange(12.0).reshape(3, 4), ("y", "x"))


def sample(t):
    @jit()
    def at(a, b):
        return t.isel(x=b, y=a)  # note the order: names bind, position is dead

    return at


at = sample(field)
print("isel(y=1, x=2) =", at(1, 2), "== numpy [1,2] =", field.array[1, 2])


def sample_positional(t):
    @jit()
    def at(a, b):
        return t[a, b]

    return at


try:
    sample_positional(field)(0, 0)
except Exception as e:
    print("🚫", str(e)[:110])"""),
    code("""\
import xarray as xr

da = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("lat", "lon"))


def sample_xr(t):
    @jit()
    def at(a, b):
        return t.isel(lon=b, lat=a)

    return at


print("xarray isel:", sample_xr(da)(2, 3))
c0 = DEFAULT.specializations.compiles
with no_compile():  # a FRESH DataArray, same dims/dtype/rank: same type, hit
    sample_xr(xr.DataArray(np.zeros((9, 9)), dims=("lat", "lon")))(0, 0)
print("fresh DataArray -> compiles:", DEFAULT.specializations.compiles - c0)
try:
    sample_xr(xr.DataArray(np.ones((2, 2)), dims=("north", "east")))(0, 0)
except Exception as e:
    print("🚫 renamed dims:", str(e)[:90])"""),
    md("""\
### The free lunch, proven by the artifact cache

The pedantry costs nothing — and the system can PROVE it. A positional
kernel over an anonymous array and an `isel` kernel over a named one lower
to IDENTICAL IR (same stride arithmetic, same env paths; the names never
reach a node). Identical IR means an identical content key, and tier 2 is
content-addressed:"""),
    code("""\
ext = DEFAULT.extend()  # fresh caches, same vocabulary
raw = np.arange(6.0).reshape(2, 3)


def pos(t):
    @jit()
    def k(i, j):
        return t[i, j]

    return k


def named(t):
    @jit()
    def k(i, j):
        return t.isel(y=i, x=j)

    return k


fp2 = (ext.table.fingerprint(0),) * 2
ext.dispatch(pos(raw), (0, 1))
ext.dispatch(named(Named(raw, ("y", "x"))), (0, 1))
print("specializations:", len(ext.specializations._ready), " artifacts:", len(ext.artifacts))
print("two source spellings, two cache-key identities, ONE compiled artifact")"""),
    md("""\
## The C target: the seam beyond GPUs

`backends/c.py` is the contribution point's first *citizen* (080's
namespace package stops being infra-only): rendered C99, compiled by the
system `cc`, loaded with `ctypes` — a third source renderer over the same
IR and a second runtime shape (dlopen vs `exec` vs wgpu) under the ONE
calling convention `launch(staging, leaves)`. It never claims the default
route; choosing it is explicit (a child registry — 080's tier-2 override)
until the `device` axis brings data-driven dispatch at step 14."""),
    code("""\
from pdum.dsl.backends import c

print("cc on PATH:", c.is_available())
cext = DEFAULT.extend()
c.install(cext, default=True)

print("twin :", f(1.5))
print("C    :", cext.dispatch(f, (1.5,)))
print("lum twin/C:", lum(0.75), cext.dispatch(lum, (0.75,)))
print("named on C:", cext.dispatch(at, (1, 2)))"""),
    code("""\
kc = cext.specializations.key_for(f, (cext.table.fingerprint(1.5),), cext.backend_for(f.kind).fp)
print(cext.specializations.probe(kc).artifact.__pdum_source__)"""),
    md("""\
Things to read in the C: the loop carry is SCALARIZED (`v25_0, v25_1` —
tuples become lane variables, no structs), staging reads are `memcpy`
inlines the compiler folds to plain loads, and the buffer parameter is
`bufs[k]` — the leaves channel, exactly as the plan ordered it.

## The ray-march spike

The step's exit question (020): can this language express a real iterative
kernel — a sphere tracer: 32 bounded steps, a branch, a multi-carry, an
intrinsic — and what do the instruments (ch11b) say about running it
per-pixel on the CPU?"""),
    code("""\
from pdum.dsl.bench import benchmark


def make_marcher(cx, cy, cz, radius):
    @jit()
    def march(ox, oy):
        t = 0.0
        hit = 0.0
        for i in range(32):
            dx = ox - cx
            dy = oy - cy
            dz = -3.0 + t - cz
            d = sqrt(dx * dx + dy * dy + dz * dz) - radius  # noqa: F821
            if d < 0.001:
                hit = 1.0
            t = t + max(d, 0.001)  # noqa: F821
        return hit * (1.0 - t / 6.0)

    return march


m = make_marcher(0.0, 0.0, 0.5, 1.0)
print("center ray:", m(0.0, 0.0), " edge ray:", m(2.0, 2.0))
print("C agrees:", abs(cext.dispatch(m, (0.0, 0.0)) - m(0.0, 0.0)) < 1e-9)

t_py = benchmark(lambda: m(0.1, 0.2), budget_s=0.25)
t_c = benchmark(lambda: cext.dispatch(m, (0.1, 0.2)), budget_s=0.25)
print("python twin:", t_py)
print("C target   :", t_c)
u = 1e6
print(f"body speedup ~{t_py.minimum / t_c.minimum:.1f}x; a 64x64 frame at per-ray dispatch: "
      f"~{t_c.minimum * 4096 * 1e3:.1f} ms on C")"""),
    md("""\
## The verdict

**Expressiveness: GO.** A sphere tracer is `for` + `if` + carries +
batteries; the render reads like the shader you would have written.

**Granularity: the honest finding.** On the M3 the C body runs a 32-step
march in well under a microsecond — but ~2.4 µs of every per-ray call is
DISPATCH (ch11b's decomposition: key+probe, extract, pack). Per-pixel ×
per-call is simply the wrong shape for CPU frames: the C target wins 7× on
the body and then drowns in per-call overhead that the GPU path amortizes
over the whole domain (ch10's `out=(W,H)` launches 65k threads on ONE
dispatch). CPU frames want the same move — a domain call or DPS
out-arrays — and that is chaining/step-14 territory (100 §6), not a reason
to contort v1.

## Things to notice

- **One kernel line** this whole step (`Array.device`, the 090 axis).
  Statements, arrays, named axes, and a C backend: all satellites. The
  extension-locality law held through the widest step since the kernel
  froze.
- **Loop binders carry TUPLE indices** — `("loop", *inline-prefix, seq)`.
  `core.param` identity is structural, so reusing integer index 0 for a
  loop variable would make two different programs hash identically (an
  artifact-cache collision); and a shared counter would make content keys
  depend on process HISTORY (review-caught). The tuple is unique across
  inlining and deterministic from source order alone — cache keys cannot
  drift between runs.
- **The pedantic pick is free.** Named axes cost one wrapper (or arrive
  free from xarray), are enforced at lower time, and compile to the SAME
  artifact as positional code. Pedantry priced at zero is the two-tier
  cache doing exactly what ch03 promised.
- Next: transforms — `over` and the jvp precursor (step 12; `over` was
  born as "vmap" and renamed at step 13 — ch13 tells that story), where these
  loops and arrays meet their first IR-to-IR columns."""),
]

# ══════════════════════════════════════════ ch12a ═════════════════════════════

ch12a = [
    md("""\
# 12a — Lay of the land

The third interlude (convention: 020 §The book): only what CHANGED since
11a. Step 11 widened the base pack with statement `if`/`for` and the array
indexing surface — the two biggest holes in ch07a's original refusal list."""),
    code("""\
import numpy as np

import pdum.dsl  # noqa: F401
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.lower import lower_handle
from pdum.dsl.kernel.ops import CORE_OPS
from pdum.dsl.kernel.registry import DEFAULT

rules = dict(DEFAULT.lower_rules)
print("the pack now handles", len([k for k in rules if isinstance(k, type)]), "syntax forms:")
for node_type in rules:
    if isinstance(node_type, type):
        print("  ✅ ast." + node_type.__name__)"""),
    code("""\
# ch07a's remaining refusals, re-run against today's pack:
@jit()
def wants_if_stmt(x):
    y = 0.0
    if x > 0.0:
        y = 1.0 / x
    else:
        y = 0.0
    return y


@jit()
def wants_for(x):
    total = x
    for i in range(4):
        total = total + x
    return total


table = np.ones((2, 2))


@jit()
def wants_array_index(i, j):
    return table[i, j]


@jit()
def wants_while(x):
    while x > 0.0:
        x = x - 1.0
    return x


@jit()
def wants_early_return(x):
    if x > 0.0:
        return x
    return -x


ops = {**CORE_OPS, **DEFAULT.ops}  # the registry rides the CONTEXT seam now (130 §7)
for h, sig in (
    (wants_if_stmt, (T.f64,)),
    (wants_for, (T.f64,)),
    (wants_array_index, (T.i64, T.i64)),
    (wants_while, (T.f64,)),
    (wants_early_return, (T.f64,)),
):
    try:
        lower_handle(h, rules, ops, arg_types=sig, context={"registry": DEFAULT})
        print(f"  ✅ {h.fntype.template.label}: lowers now (step 11)")
    except Exception as e:
        print(f"  🚫 {type(e).__name__}: {str(e)[:95]}")"""),
    md("""\
## The delta table

| construct | 11a | now |
|---|---|---|
| `if`/`else` statements | 🏗️ | ✅ `core.if` joins; STRICT (same type both paths, no fixpoint) |
| `for i in range(...)` | 🏗️ | ✅ `core.for`, loop carries; zero-trip defined; loop var dies |
| `a[i, j]` on arrays | 🏗️ | ✅ captures only; strict i64; every axis exactly once |
| `a.isel(y=i, x=j)` | — | ✅ NAMED axes (xarray exercise); mandatory on named arrays |
| early `return` | 🏗️ | 🚫 **policy**: single tail return (Taichi's side of the line) |
| `while` / `break` / `continue` | 🏗️ | 🚫 **policy**: bounded loops only (R11's line) |

Note the status change: 11a's 🏗️ ("machinery pending") became either ✅ or
🚫-by-policy. Nothing is left pending *by accident* in the statement
surface; what remains out is out on purpose, with the reason in the error
message. Still out as recorded CUTS (100 §6): array arguments and results,
views/slices/partial `isel`, `sel` (labels are host-side values),
comprehensions (the step-8 discussion: sugar over `core.for`, revisit with
transforms)."""),
]

notebook(ch12, "docs/book/ch12-data-and-loops.ipynb")
notebook(ch12a, "docs/book/ch12a-lay-of-the-land.ipynb")

# ══════════════════════════════════════════ ch13 ══════════════════════════════

ch13 = [
    md("""\
# Chapter 13 — Transforms are rules

Step 12: the first IR-to-IR machinery — **`over`** (né *vmap*: renamed when the JAX-prior collision became obvious — ours weaves a named CAPTURE axis into a lane coordinate; JAX's widens argument values) and **jvp**, plus two
user-directed additions: the in-kernel derivative operator **`D`** (GLSL's
`dFdx` idea, done analytically) and **named matrix contraction** with
batching as the payoff. Design: `docs/design/110_transforms-and-derivatives.md`.

The architecture priced transforms carefully and demanded a spike before
commitment (>350 counted lines ⇒ re-hear). The spike's finding reshaped the
step and is worth stating up front: **our `over` is SIMT-shaped, not
SIMD-shaped**. It does not widen values with a batch dimension (JAX's way,
where batched predicates force execute-both-and-select and break lazy
branches); it adds one *coordinate parameter* and weaves it into the
accesses that need it — the same move the compute family made when params
became thread coordinates. Intermediates stay scalar; each lane runs its
own branches and trip counts; `if`/`for` need ZERO new machinery. The
transform satellite landed at 338 counted lines (after the review pass hardened it) — under the threshold, no re-hear."""),
    code("""\
import numpy as np

import pdum.dsl  # noqa: F401
from pdum.dsl import jit, no_compile
from pdum.dsl.kernel.registry import DEFAULT
from pdum.dsl.stdlib.arrays import Named
from pdum.dsl.stdlib.transforms import jvp, over"""),
    md("""\
## jvp: dual numbers, compiled

Forward-mode AD: every value `a` gets a shadow `ȧ`, every primitive a
two-line linearization rule (`mul`: `aḃ + ȧb`; `sqrt`: `ṫ/2√a`). No tape,
no tracing — the tangent chain is just MORE NODES in the same region,
synthesized once at build. `jvp(f)(*args, *tangents)` returns
`(primal, directional derivative)`. Control flow: branches get a parallel
lazy `core.if` on the same condition; loops WIDEN — the carry becomes
`(primal, tangent)` because the tangent recurrence needs the primal carry
each iteration:"""),
    code("""\
def make(c):
    @jit()
    def f(x):
        y = c * x * x + sqrt(x)  # noqa: F821
        if y > 2.0:
            y = y * 2.0
        acc = y
        for i in range(3):
            acc = acc * x
        return acc

    return f


f = make(0.5)
jf = jvp(f)
x0, eps = 1.3, 1e-7
p, t = jf(x0, 1.0)
fd = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)
print(f"primal      : {p:.12f}   f(x): {f(x0):.12f}")
print(f"jvp tangent : {t:.8f}   finite diff: {fd:.8f}   |Δ| = {abs(t - fd):.1e}")
c0 = DEFAULT.specializations.compiles
with no_compile():  # a fresh closure UNDER the transform: still a hit
    jvp(make(0.9))(2.0, 1.0)
print("fresh closure under jvp -> compiles:", DEFAULT.specializations.compiles - c0)"""),
    code("""\
key = DEFAULT.specializations.key_for(jf, tuple(DEFAULT.table.fingerprint(a) for a in (x0, 1.0)),
                                      DEFAULT.backend_for(jf.kind).fp)
print(DEFAULT.specializations.probe(key).artifact.__pdum_source__)"""),
    md("""\
Read the render: ONE loop carrying a `(primal, tangent)` pair, the product
rule inlined in its body, and a `(v.., v..)` pair returned at the end. The
`Derived("jvp", …)` identity means the transform participates in the
two-tier cache like any kernel — the fresh-closure hit above is the ch03
thesis surviving its first transform.

## `D`: the derivative of anything, with respect to *here*

The user-directed operator (110 §3): `D(x)` = the partials of any
intermediate w.r.t. the ENCLOSING KERNEL'S parameters — one per param,
positionally. In the shader world where params ARE pixel coordinates, this
is GLSL's `dFdx`/`dFdy` — except GLSL computes a finite difference across
the 2×2 pixel quad, while `D` differentiates analytically. And compute
shaders have NO quads (WGSL has no `dpdx` outside fragment stage), so
analytic `D` is not a convenience there — it is the only derivative in
town, and it is exact:"""),
    code("""\
def make_ring(cx, cy, r):
    @jit()
    def k(i, j):
        d = sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy)) - r  # noqa: F821
        di, dj = D(d)  # noqa: F821 — analytic screen-space gradient
        return abs(di) + abs(dj)  # noqa: F821 — this IS fwidth(d)

    return k


k = make_ring(0.0, 0.0, 1.0)
print("gradient magnitude proxy at (3,4):", f"{k(3.0, 4.0):.6f}")
print("  (|3/5| + |4/5| = 1.4 — the SDF's unit gradient, split across axes)")

# Structure differentiates structurally: D of a tuple is a tuple per param.
@jit()
def kt(i, j):
    p = (i * j, i + j)
    dp_i, dp_j = D(p)  # noqa: F821
    return dp_i[0] + dp_j[1]  # d(ij)/di + d(i+j)/dj


print("D of a tuple:", kt(3.0, 5.0), " (j + 1 =", 5.0 + 1.0, ")")"""),
    md("""\
### Analytic anti-aliasing (the classic, minus the quads)

The canonical `fwidth` trick: smooth a hard edge over exactly one pixel's
worth of parameter change, whatever the zoom. `demo.graphics` ships the GL
vocabulary (`ddx`/`ddy`/`fwidth`) as one-line batteries over `D` — an
explicit import, per the stdlib-minimalism rule:"""),
    code("""\
from pdum.dsl.demo import graphics  # noqa: F401 — wires ddx/ddy/fwidth (+ Color) into DEFAULT


def make_disk(cx, cy, r, zoom):
    @jit()
    def mask(i, j):
        d = length2(((i - 32.0) * zoom - cx, (j - 32.0) * zoom - cy)) - r  # noqa: F821
        w = fwidth(d)  # noqa: F821 — one pixel's worth of d, ANALYTICALLY
        return 1.0 - smoothstep(-w, w, d)  # noqa: F821

    return mask


for zoom in (0.05, 0.2):
    mask = make_disk(0.0, 0.0, 1.0, zoom)
    shades = " ░▒▓█"
    print(f"zoom {zoom}: edge stays one pixel wide")
    for row in range(8, 24, 2):
        line = "".join(shades[min(4, int(mask(float(row), float(col)) * 4.99))] for col in range(16, 48))
        print("   ", line)"""),
    md("""\
The edge is exactly one pixel of smooth ramp at BOTH zooms — that is
`fwidth` adapting the smoothing width to the local derivative, computed
exactly, with no quad neighbors, on a plain CPU backend. Kernels that never
call `D` pay nothing: the tangent slice is synthesized only where demanded
(tier 2 can verify — D-free kernels mint the same artifacts they always
did).

## over: written ignorant, batched by declaration

The batching-ignorance arc. A kernel written for ONE element, against data
whose batch axis it never mentions — the pedantic axis checking REFUSES it
(an unaccounted axis is an error, not a silent broadcast). `over` is the
declaration that accounts for it:"""),
    code("""\
data = Named(np.arange(12.0).reshape(3, 4), ("batch", "x"))


def make_sum(t):
    @jit()
    def g(k):
        return t.isel(x=k) * 10.0  # batch-IGNORANT: only 'x' is named

    return g


g = make_sum(data)
try:
    g(1)
except Exception as e:
    print("🚫 without over:", str(e)[:95], "…")

vg = over(g, axis="batch")  # the declaration: 'batch' is mine now
print("lanes:", [vg(1, b) for b in range(3)])
c0 = DEFAULT.specializations.compiles
with no_compile():  # 100 lanes instead of 3: same TYPE, zero recompiles
    over(make_sum(Named(np.ones((100, 4)), ("batch", "x"))), axis="batch")(1, 99)
print("new batch size -> compiles:", DEFAULT.specializations.compiles - c0)"""),
    code("""\
key = vg, (DEFAULT.table.fingerprint(1), DEFAULT.table.fingerprint(0))
k2 = DEFAULT.specializations.key_for(vg, key[1], DEFAULT.backend_for(vg.kind).fp)
print(DEFAULT.specializations.probe(k2).artifact.__pdum_source__)"""),
    md("""\
The woven kernel is an ordinary scalar kernel with one extra trailing
parameter — the lane coordinate, folded into the SAME stride arithmetic
every named access already does. Nothing widened; the lazy-branch guarantee
survives untouched (each lane takes its own path — no `where` wart); and on
a GPU this parameter is launch-domain-shaped, which is where the batch axis
always wanted to live.

## Named contraction, and batching for free

The stretch goal: `matmul(A, B, i, j)` pairs the operands' UNIQUE shared
axis name (that is the whole "rules engine" — a dozen lines shaped like a
type rule), refuses ambiguity loudly, and expands to the element loop with
the trip count read from the shape SLOT — rank-generic, so new extents are
cache hits. Woven axes are excluded from pairing FIRST, which is exactly
why batching composes:"""),
    code("""\
A = Named(np.arange(6.0).reshape(2, 3), ("row", "inner"))
B = Named(np.arange(12.0).reshape(3, 4), ("inner", "col"))


def make_mm(A, B):
    @jit()
    def cell(i, j):
        return matmul(A, B, i, j)  # noqa: F821 — 'inner' pairs BY NAME

    return cell


cell = make_mm(A, B)
got = np.array([[cell(i, j) for j in range(4)] for i in range(2)])
print("matmul == numpy @ :", np.allclose(got, A.array @ B.array))

try:
    make_mm(A, Named(np.ones((3, 4)), ("k", "col")))(0, 0)
except Exception as e:
    print("🚫 no shared name:", str(e)[:80])

rng = np.random.default_rng(7)
Ab = Named(rng.standard_normal((5, 2, 3)), ("batch", "row", "inner"))
Bb = Named(rng.standard_normal((5, 3, 4)), ("batch", "inner", "col"))
bcell = over(make_mm(Ab, Bb), axis="batch")  # SAME cell kernel, batch woven in
got3 = np.array([[[bcell(i, j, b) for j in range(4)] for i in range(2)] for b in range(5)])
print("batched matmul == np.matmul:", np.allclose(got3, Ab.array @ Bb.array))"""),
    md("""\
`batch` appears in BOTH operands, and without the woven-axes-first
exclusion it would be a contraction candidate — the pairing rule sees only
what `over` has not claimed. One batch-ignorant cell kernel; `np.matmul`
agreement per (b, i, j); zero new mechanism.

### And it composes

Transform composition is `lower_handle` re-entry with a merged context
(design 130 §7) — the step-13 seams made it a property, not a mechanism.
Two batch axes, two `over`s, lanes trailing outermost-last:"""),
    code("""\
data3 = Named(rng.standard_normal((2, 3, 4)), ("b1", "b2", "x"))


def make_g(t):
    @jit()
    def g(k):
        return t.isel(x=k) * 10.0

    return g


composed = over(over(make_g(data3), axis="b2"), axis="b1")
print("composed(k=1, b2=2, b1=0):", composed(1, 2, 0), "== numpy:", data3.array[0, 2, 1] * 10.0)
try:
    over(over(make_g(data3), axis="b1"), axis="b1")(0, 0, 0)
except Exception as e:
    print("🚫 duplicate axis:", str(e)[:80])"""),
    md("""\
## Things to notice

- **The spike verdict**: SIMT weaving makes `over` across `if`/`for` a
  non-event — the 180-lines-per-region-op price was for value-widening,
  which we do not do. What is LOST: cross-lane collectives (no `psum` over
  a woven axis) — recorded, GPU-shaped, deferred.
- **One tangent engine, two doors**: `jvp` seeds tangent params; `D` seeds
  basis vectors and shares memoized tangents across calls in the same
  kernel. Custom ops join by registering a jvp rule — the transform column
  is surface-A-shaped.
- **`D` is why forward mode came first**: two inputs, many outputs is the
  shader's economics. `grad` (step 13) inverts it: transpose of this
  column, with jvp and finite differences as its oracles.
- Everything here is satellite code: 338 counted lines, zero kernel edits,
  and the walker/renderers learned nothing new — transformed kernels are
  ordinary kernels by the time a backend sees them."""),
]

notebook(ch13, "docs/book/ch13-transforms-are-rules.ipynb")
