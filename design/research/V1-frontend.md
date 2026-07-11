# V1 — Verdict: frontend strategy for the pdum.dsl redesign

*Consolidation verdict, July 2026. Inputs: R1 (numba), R2 (DaCe), R4 (JAX), R5
(kernel DSLs: cupy.jit / Triton / MLX / Taichi), R6 (torch.compile), plus
`docs/desiderata.md`, `design/dsl_caching_layer.md`, and the frozen reference
asset at `src/pdum/dsl_reference/`.*

---

## Recommendation

**Adopt a deliberate two-layer hybrid, which is a hardened version of what the
reference asset already does:**

1. **Identity & capture layer (phase A): trace-free object reflection — no
   source, no bytecode analysis, no tracing.** Read `fn.__code__` (value-compared,
   the cache identity), `co_freevars` + `__closure__` (typed environment),
   `__globals__` (name classification), and snapshot the source text. ~100–150
   lines, CPython-version-stable, compile-free.

2. **Body lowering layer (phase B): AST analysis of the snapshotted source.**
   `ast.parse` → name-classification pass → one forward abstract-interpretation
   pass that types and lowers simultaneously into the (small, typed) IR, with
   dialect/intrinsic tables supplied as an input.

**Bytecode analysis is rejected everywhere. Tracing is rejected for program
acquisition** (it is retained only in the JAX sense of *interpreters over our own
IR* for transformations — that is not a Python frontend strategy).

**Fallback when source is unavailable:** snapshot source eagerly at decoration
time so the common notebook hazards never arise; if the snapshot is empty, fail
**loudly at phase B** (never phase A) with an error naming the function and the
two remedies — put the code in a file/IPython cell, or drop to the per-backend
`raw_kernel` escape hatch (MLX-shaped: kernel body as a string + declared
inputs/outputs), which needs no Python frontend at all. No bytecode-decompilation
fallback, ever.

---

## Rationale

### 1. The family consensus is unanimous for our problem shape

Every system that compiles a *small explicit kernel subset* uses
`inspect.getsource` + `ast.parse`: cupy.jit, Triton, Taichi, and DaCe (R5 §5,
R2 §1). **No member of the kernel-DSL family uses bytecode.** The two bytecode
systems, numba and TorchDynamo, both need bytecode for a reason we do not have:
they must handle *arbitrary* Python — numba any live function object, Dynamo
foreign libraries mid-frame via PEP 523. pdum.dsl owns its input language: users
decorate a `def` in a supported subset. That is exactly the population for which
the AST route is proven at every scale from cupy.jit's 1.8k lines to DaCe's 18k.

### 2. The maintenance tax of bytecode is measured, large, and permanent

- numba: **268 `PYVERSION` conditionals in `numba/core`** (225 in
  byteflow.py + interpreter.py alone), 287 opcode handlers across two parallel
  interpreters, ~1,300 lines of peephole passes that *re-derive AST-level
  structure*, 2–4 months of lag per CPython minor, and breakage from CPython
  **patch** release 3.13.4 (issue #10101) (R1 §1.3).
- Dynamo: 71 `sys.version_info` branches in `bytecode_transformation.py`,
  copy-pasted CPython frame-allocation internals for 3.12, maintainer-described
  "brittle" heuristics, 3–12 month lag per release, a dedicated engineer per
  port (R6 §1.4).

Against this: the `ast` module for our subset (expressions, `if`/`for`/`while`,
assignments, calls, subscripts) changed essentially not at all from 3.9→3.14
(R6 §1.4). DaCe handles 3.14 with one `__annotate__` special case (R2). For a
project whose end state is a ~1000-line kernel maintained by approximately one
person, the bytecode tax alone is disqualifying: numba's frontend (7.7k lines +
version forks) is several times our entire kernel budget.

### 3. Cache identity is orthogonal to lowering strategy — and the hybrid exploits that

The thesis needs a stable code identity regardless of frontend. The code object
(value-compared) delivers it *without any bytecode analysis*: we hold the object
and use its `__eq__`/`__hash__` over `co_code`/`co_consts`/`co_firstlineno`
(`dsl_caching_layer.md`). This gives the exact notebook semantics we want —
re-running an **unchanged** cell produces a value-equal code object → cache
**hit**; editing the body → value-unequal → miss → recompile. No AST-based
system in the survey has this: DaCe and Taichi anchor caches to the decorated
Python object (re-decoration = cold cache, edits on the same object undetected,
R2 §4); JAX keys on function `id()` (perpetual retrace for rebuilt closures,
~240× measured, R4 §2). Dynamo independently confirms the decomposition — it
guards `__code__` and closure contents explicitly (`CLOSURE_MATCH`), it just
checks them with O(entries × guards) predicate trees where we get O(1) hash
lookup because our types are declared, not discovered (R6 §2.5).

So the hybrid's split of labor is principled: **the code object is the identity;
the AST is only the lowering input.** One coherence hazard follows (see
implication 4): the snapshotted source must actually correspond to the code
object, since `inspect.getsource` reads `linecache`, which can go stale.

### 4. Data-dependent control flow rules out tracing outright

Our domains (shaders, simulation kernels) are statement-heavy: `for` over
neighbors, early-exit `if`s on computed values, accumulation. Tracing cannot see
`if x < 0:` on a traced value (`TracerBoolConversionError`), silently unrolls
static loops, and forces the `lax.cond`/`scan` combinator dialect — precisely
the "operator-overloading shadow language" the desiderata's aesthetics section
forbids (R4 §1). Tracing additionally bakes captures into the artifact
(`ClosedJaxpr.consts`) with no re-marshal path — the anti-thesis (R4 §2). An
AST frontend sees `if`/`for` as syntax and lowers them to real structured IR;
the user writes Python.

### 5. Error quality falls out of the AST for free

Every `ast` node carries `lineno`/`col_offset`. Parsing the snapshot with
`filename=co_filename` and offsetting by `co_firstlineno` lets every IR node
carry an exact source location, so type errors and unsupported-syntax errors
point at the user's file and column (Python 3.11+ even gives `end_col_offset`
for caret ranges). Bytecode frontends must reconstruct this from `co_positions`
and lose expression structure; tracing errors point into the tracer's frames.

### 6. Staging lives above the frontend and prefers this split

Phase-A metaprogramming — building `Handle`/`Program` expressions in a Python
loop, `@overload`-style batteries that compute type-time constants and return a
specialized impl (R1 §3.2) — is ordinary Python running *before* any frontend
work. Because capture is reflective and compile-free, staging costs nothing and
needs no tracer. The body-lowering layer only ever sees the final `def`. This is
also why t-string mini-languages (desiderata §4.5) are unaffected by any
source-availability concern: their "source" is the template string itself,
handed to a sub-parser that emits into the same IR.

### 7. Notebook/REPL reality check

IPython/Jupyter register each cell in `linecache` (`<ipython-input-…>` /
`/tmp/ipykernel_…` pseudo-files), so `inspect.getsource` works in the primary
target workflow; the reference asset already relies on this. The genuinely
unsupported cases are the plain `python` REPL, `exec` of raw strings, and
`.pyc`-only distribution — all rare for a live-coding scientific tool, all
detectable at decoration time, and all covered by the fail-loud +
`raw_kernel`-escape-hatch fallback. DaCe ships this exact policy in production
("use IPython/Jupyter or place the source code in a file", R2 §1).

---

## Considered and rejected

### A. Pure bytecode frontend (numba's route) — rejected

*Why it lost:* the measured churn in §2 above, paid forever, for a benefit we
don't need — bytecode's real advantages (no source needed, decorators applied,
closures/globals resolvable) are either recovered by the reflection layer
(`__closure__`, `__globals__`, `__code__`) or waived as a documented constraint
(source availability). Two further strikes: bytecode is where numba's
closure anti-pattern lives (`op_LOAD_DEREF` → `ir.FreeVar(idx, name, value)` —
values consumed at IR-build time, unfixable downstream, R1 §1.4); and new
*syntax* — our prime extensibility axis, including t-string mini-languages — is
an AST-shaped concept with no clean bytecode analogue.

### B. Tracing frontend (JAX's route) — rejected for acquisition

*Why it lost:* fails on data-dependent control flow (the kernel body style of
every target domain), forces a combinator shadow-dialect, bakes capture values
into compiled artifacts with no re-marshal path, and its natural cache identity
(function object / trace observations) is exactly the identity-keying the thesis
exists to displace. What we keep from JAX is everything *behind* the frontend:
the primitive/rule-registry matrix, flat typed IR, pytree-shaped marshaling seam
— all of which run over an AST-derived IR just as well (R4 lessons 1–2).

### C. Guard-based hybrid (Dynamo's route: bytecode capture + value guards) — rejected

*Why it lost:* guards are what a cache key becomes when types are discovered by
observation instead of declared — O(entries × full predicate-tree eval) per
call, a C++ rewrite to make it bearable, recompile limits and unsafe-skip knobs
(R6 §2). Our declared type universe makes the key a total function of the
inputs: `typeof` + one dict probe. Dynamo remains valuable as a *checklist* of
hidden key dimensions (ambient state, backend identity, aliasing — implication 8).

### D. The same hybrid, plus a bytecode-decompilation fallback for missing source — rejected

*Tempting because* it would make the no-source case work transparently. *Why it
lost:* it re-imports the entire per-release treadmill (a decompiler tracks
opcodes just like a frontend) to serve the rarest path, and creates a second
lowering pipeline whose semantics must be proven equivalent to the first —
double maintenance for a case the escape hatch already covers. DaCe, cupy.jit,
Triton, and Taichi all ship without one.

### E. Pure AST with AST-derived identity (hash of source text) — rejected

*Why it lost:* source-text hashing is *less* precise than code-object value
equality (whitespace/comment changes would miss; `co_firstlineno` moves would
hit when they should be examined) and requires source even for identity,
weakening phase A. Code-object equality is free, correct for notebook
semantics, and already validated in the caching design. Source hashing returns
in exactly one place: the cross-process disk key, where code objects don't
survive (`dsl_caching_layer.md` disk-cache section).

---

## Concrete implications for the architecture

### 1. The frontend is two components with a narrow interface

```python
# --- Layer 1: capture (phase A, ~120 lines, no per-version code) -------------
class Handle:
    fntype: FnType              # (code_object, env_types) — the cache identity
    env:    dict[str, object]   # capture VALUES; never in any key
    src:    SourceSnapshot      # taken at decoration; see (4)

def make_handle(fn, kind) -> Handle:
    code   = fn.__code__
    vals   = tuple(safe_cell(c) for c in (fn.__closure__ or ()))
    etypes = typeof_tuple(vals)                       # structural, range-bucketed
    return Handle(FnType(code, etypes), dict(zip(code.co_freevars, vals)),
                  snapshot_source(fn))                # eager, may be EMPTY

# --- Layer 2: lowering (phase B, on cache miss only) --------------------------
def lower(handle, arg_types, dialect) -> IRFunction:
    tree = parse_snapshot(handle.src)                 # raises NoSourceError w/ remedies
    fates = classify_names(tree, handle, dialect)     # see (3)
    return TypedLowerer(dialect, fates,
                        handle.fntype.env_types, arg_types).run(tree)
```

Layer 1 never parses; Layer 2 never looks at values. The lowering entry point
takes the dialect tables as a parameter — fixing the M0 fault line where
`ast_lower`/`infer` import the WGSL intrinsics directly.

### 2. One forward pass, typing fused with lowering

Family consensus (R5 §5.2): no standalone inference pass. Monomorphization per
`(env_types, arg_types)` signature means types flow forward from the signature
through a single `ast.NodeVisitor` that emits typed IR as it goes; loops require
type-stable loop variables (cupy's rule). Budget evidence: a complete
Python→CUDA-C kernel language is 1.8k lines (cupy.jit); our Layer 2 for the M1
subset should land in ~800–1,500 lines. Steal cupy's `Environment` shape
(consts / params / locals / ret_type with defined resolution order) as the
pass's core data structure, emitting IR nodes instead of C text.

### 3. A name-classification pass with an explicit fate taxonomy (DaCe's GlobalResolver, inverted)

Every `Name`/`Attribute` in the body gets exactly one fate, decided *before*
lowering:

| referenced thing | fate | key/marshal consequence |
|---|---|---|
| parameter | typed IR argument | in `arg_types` |
| captured free var | **typed env slot** `EnvVar(name, index, type)` | type in `env_types`; value marshaled per call |
| dialect intrinsic / battery (via registry) | IR op per the replacements table | none |
| other `@jit` Handle | `FnType`-typed callee | callee's `FnType` in the key |
| module-level constant explicitly allowed (e.g. `math.pi`, dtype objects) | folded literal | in dependency-closure hash |
| anything else global | **error** (or explicit `Literal[...]` lift) | — |

The inversion from DaCe: scalar captures are *typed*, never value-frozen; and no
per-call `eval` thunks — capture slots are resolved once, only cell contents are
re-read. The structural impossibility requirement from R1: the IR has **no
constructor that accepts a captured value** — `EnvVar` carries name/index/type
only. Numba's `ir.FreeVar(idx, name, value)` is the one-line mistake this rules
out by type.

### 4. Source snapshot discipline (the coherence hazard of mixed identity)

Because identity is the code object but lowering input is text, they can skew
(stale `linecache` after file edits without re-execution). Policy:

- `snapshot_source(fn)` runs at **decoration** time — the moment the code object
  and `linecache` are guaranteed in sync — storing `(text, co_filename,
  co_firstlineno, co_qualname)`.
- `parse_snapshot` parses with `filename=co_filename` and re-bases line numbers
  by `co_firstlineno`; it takes the single `FunctionDef` and drops
  `decorator_list` (no Triton-style regex stripping — parse, then select).
- Cheap sanity check at phase B: `compile()` the snapshot and compare the
  resulting code object for value-equality with `handle.fntype.template`
  (ignoring `co_filename`/`co_firstlineno`); mismatch → hard error, never a
  silent wrong-source compile. This check runs only on cache miss, so it costs
  nothing in the hot loop.

### 5. The fallback story, exactly

- **Phase A never fails** for missing source — `Handle` creation is reflection
  only, so building/composing programs always works.
- **Phase B raises `NoSourceError`** on first compile of a snapshot-less
  template: names the function, states why (plain REPL / exec'd string /
  pyc-only), and lists remedies: (a) define in a file or IPython cell; (b) use
  the backend-floor escape hatch:

```python
k = raw_kernel(name="myexp", inputs=["inp"], outputs=["out"],
               source="out[i] = exp(inp[i]);", backend="wgsl", ...)
```

  (MLX-shaped, per R5 lesson 5 — the same door the compiler's own output goes
  through, ~100–1,000 lines per backend.)
- t-string mini-languages never hit this path: the template string is the source.

### 6. Error messages as a first-class IR channel

Every IR node carries a `loc: (filename, line, col, end_line, end_col)` taken
from the AST node, threaded through all passes (a `meta`-style side channel per
R6 lesson 5). Unsupported-syntax errors are raised by the classifier/lowerer
with the offending source span quoted — the honest-subset documentation
discipline (DaCe's `python_supported_features.md`) applied to diagnostics.

### 7. CPython policy

Target 3.14 (t-strings) with a documented floor of one or two minors. The
version-sensitive surface is: the `ast` node set for the supported subset
(historically near-frozen), `co_freevars`/`__closure__` semantics (stable), and
code-object equality fields (stable). Budget expectation per new CPython minor:
near zero — vs. numba's measured multi-month ports. This is the single largest
sustainability win of the verdict.

### 8. What stays out of the frontend but was decided by this analysis

- **Dependency-closure hashing (Triton's `DependenciesFinder`)** joins the
  invalidation/generation story: hash referenced-and-folded globals
  (`(name, id(fn.__globals__))`) and transitively-called Handles' keys into the
  template identity; check at call time and refuse/recompile on drift — never
  silently stale (cupy's documented failure).
- **Dynamo's guard inventory as a key checklist**: backend identity + backend
  params, generation, code identity, env_types are the complete key; ambient
  state must be either in the key or provably irrelevant; decide explicitly
  whether capture aliasing is (a) irrelevant per backend, (b) normalized at
  marshal time, or (c) an explicit key component.
- **Transformations are IR interpreters** (JAX rule-matrix over the AST-derived
  IR), not frontend features — the frontend's only obligation to autodiff/vmap
  is typed, flat, multi-result IR with control flow as ops carrying sub-programs.

---

## Confidence and what would change my mind

**Confidence: high.** Every line of evidence points the same way: the four
closest-comparable systems all chose AST; the two bytecode systems document
their own tax in numbers (268 forks / 71 branches, quarter-to-year lags, patch
release breakage); tracing structurally conflicts with both the thesis
(const-baking, identity keys) and the domain (data-dependent control flow); and
the reference asset already validated the trace-free-capture + AST-lowering
split end-to-end on WebGPU. The hybrid's one novel risk — code-object identity
vs. source-text lowering skew — has a cheap, complete mitigation (decoration-time
snapshot + miss-time recompile check).

Evidence that would force reconsideration:

1. **A requirement to compile undecorated/third-party functions** (e.g. jitting
   a callback from someone else's library, no source shipped). That is the one
   job bytecode does that reflection+AST cannot; it would justify a numba-style
   frontend as an *additional* acquisition path, not a replacement.
2. **CPython starts churning the `ast` module for our subset** (e.g. a lazy/new
   AST representation breaking `lineno` semantics or node classes per release).
   Nothing in 3.9→3.14 history suggests this; a PEP moving that way would.
3. **Field data showing source unavailability is common in target workflows** —
   e.g. a significant user base on the plain REPL, marimo/exotic notebooks that
   bypass `linecache`, or pyc-only deployment demands. Would upgrade the
   fallback from fail-loud to something richer (most likely: persistable
   compiled artifacts keyed by the disk-cache scheme, still no decompiler).
4. **The snapshot-coherence check failing in practice** (environments where
   decoration-time `linecache` is already stale). Would force either
   hash-verified source capture at import time or a rethink of the identity/
   lowering split. No surveyed system reports this failure mode.

---

## Design lessons for pdum.dsl

1. **Frontend = reflection for identity, AST for bodies, nothing for values.**
   Phase A reads `__code__`/`__closure__`/`__globals__` and snapshots source;
   phase B parses and lowers. No bytecode analysis, no Python tracing, ever.
2. **Snapshot source at decoration time, verify at compile time.** The
   decoration moment is the only point where code object and `linecache` are
   guaranteed coherent; a miss-time `compile()`-and-compare check makes
   stale-source compiles impossible instead of unlikely.
3. **Make the no-source path a designed experience, not an accident:** phase A
   always succeeds; phase B raises a remediation-bearing `NoSourceError`; the
   `raw_kernel` backend floor is the sanctioned escape hatch. Document the
   supported-environment matrix (file / IPython / plain REPL / exec) honestly.
4. **The IR must be incapable of holding a capture value.** `EnvVar(name, index,
   type)` only — numba's `ir.FreeVar(..., value, ...)` is the one-line
   anti-pattern to exclude by construction.
5. **One fused typing+lowering forward pass over the AST, dialect tables as a
   parameter.** Family-consensus shape, ~800–1,500 lines for the M1 subset,
   cupy's `Environment` as the working data structure, DaCe's fate taxonomy
   (typed, not value-frozen) as the name-resolution front half.
6. **Carry `loc` spans from `ast` nodes through every pass** so all diagnostics
   point at user source with column precision — this is the error-quality
   dividend of the AST choice; don't leak it.
7. **Keep the per-CPython-release surface enumerable and tiny**: `ast` subset
   nodes, closure reflection, code-object equality. Review that list once per
   CPython beta; expect no code changes. That review replacing numba's
   multi-month port cycle is the sustainability argument in one sentence.
