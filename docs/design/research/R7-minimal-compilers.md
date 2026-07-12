# R7 — Minimal-compiler exemplars: how tiny kernels support big systems

Research report for the pdum.dsl redesign. Topic: the concrete disciplines that let a
~10 kloc (or smaller) compiler kernel support many backends and years of feature growth.

**Methodology.** tinygrad was shallow-cloned and measured at commit `afeb5c708f`
(2026-07-11, i.e. today); QBE at its June 2026 head. All line counts below are `wc -l`
on those checkouts unless marked otherwise. Nanopass claims are from the ICFP 2013
paper (fetched PDF) and the framework user guide. egglog status verified against the
live GitHub repo (July 2026).

Sources:
- tinygrad: <https://github.com/tinygrad/tinygrad> (commit `afeb5c708f468f2a7d29bcd4de0e7c7fba1acd9a`)
- Nanopass: Keep & Dybvig, *A Nanopass Framework for Commercial Compiler Development*, ICFP 2013 — <https://www.cs.tufts.edu/comp/150FP/archive/icfp13.pdf>; user guide — <https://github.com/nanopass/nanopass-framework-scheme/blob/main/doc/user-guide.stex>; <https://nanopass.org/documentation.html>
- egglog: <https://github.com/egraphs-good/egglog>
- QBE: <https://c9x.me/compile/> (source: `git://c9x.me/qbe.git`)

---

## 1. tinygrad (deep)

### 1.1 The headline numbers (measured 2026-07-11)

tinygrad enforces a **hard CI line budget**: `.github/workflows/test.yml` runs
`MAX_LINE_COUNT=25000 python sz.py` ("Repo line count < 25000 lines"), where `sz.py`
counts tokenized non-blank/non-comment lines of everything under `tinygrad/`
*excluding* `tinygrad/runtime/autogen` and `tinygrad/viz/assets`. A second workflow
(`szdiff.yml`) posts the **line-count delta of every PR as a bot comment**. The
excluded `runtime/autogen/` (machine-generated ctypes FFI bindings for CUDA, Metal,
AMD, WebGPU, etc.) is **202,952 lines** — generated interface code is treated as free.

Raw `wc -l` by subsystem (excluding autogen):

| Subsystem | LOC | Role |
|---|---:|---|
| `uop/` | 3,131 | the IR + rewrite engine + symbolic math + specs |
| `codegen/` | 2,628 | lowering pipeline (expander, devectorize, decomps, linearize, regalloc) |
| `schedule/` | 1,304 | tensor-graph → kernel ASTs (rangeify, memory planning, multi-device) |
| `engine/` | 607 | realize + TinyJit |
| `mixin/` | 4,565 | Tensor/UOp sugar: elementwise (1,078), movement (607), **gradient (132)**, reduce, rand |
| top-level (`tensor.py`, `device.py`, `dtype.py`, `helpers.py`, …) | 2,221 | |
| **compiler core total** | **~14.5 k** | |
| `renderer/` | 4,417 | **all** code generation for ~14 targets |
| `runtime/` (excl. autogen) | 9,544 | device runtimes for 16 devices, incl. two driver-level GPU runtimes |
| `nn/`, `llm/`, `viz/` | 4,082 | library layer + rewrite-debugger UI |

So: a ~14.5 kloc core drives ~14 code-generation targets whose renderers total 4.4 kloc
— the closest existence proof of "many backends over a small core with tiny
per-backend renderers".

### 1.2 UOp: one small node type

The entire system — tensor graph, kernel IR, symbolic shapes, machine instructions,
even the pattern matcher's own IR — is a single immutable node type
(`tinygrad/uop/ops.py`, 1,734 lines including the rewrite engine and helper methods):

```python
@dataclass(eq=False, slots=True)
class UOp(RandMixin, metaclass=UOpMetaClass):
  op: Ops                      # enum tag
  dtype: DType = dtypes.void
  src: tuple[UOp, ...] = ()    # children
  arg: Any = None              # op-specific payload (permutation, const value, ...)
  tag: Any = None              # scratch annotation used by passes
```

Load-bearing details:

- **Hash consing.** `UOpMetaClass.__call__` interns every node in a weak-value dict
  keyed on `(op, dtype, src, arg, tag)`; identical construction returns the *same
  object*, so identity == structural equality, graph rewriting can use plain dicts,
  and `__del__` evicts dead nodes. `dtype` is inferred from `(op, src, arg)` when
  omitted.
- **Side tables, not fields.** Buffers and metadata live in
  `weakref.WeakKeyDictionary` side tables (`buffers`, `all_metadata`), keeping the
  node itself a pure value (exactly the discipline `dsl_caching_layer.md` recommends
  for `FnType` vs. layout).
- **Content hash built in.** `UOp.key` is a recursive sha256 over
  `(op, dtype, arg) + children keys` — used as the compile-cache key (§1.7).
- **The `Ops` enum is ~90 members** (`uop/__init__.py`, 141 lines), organized in 7
  commented bands: defines/buffers, meta (PROGRAM/LINEAR/SOURCE/BINARY), load/store,
  math, control flow (RANGE/IF/END/BARRIER), scheduler-only ops, and the 6 movement
  ops (RESHAPE/PERMUTE/EXPAND/PAD/FLIP/SHRINK). `GroupOp` defines algebraic property
  sets — `Commutative`, `Associative`, `Idempotent`, `Comparison`, `Elementwise` —
  that generic rewrite rules key on, so an op declared commutative automatically
  participates in commutativity-aware matching.

**Everything is dogfooded into this one IR.** Three examples that show how far this
goes: (a) `upat.py` (171 lines) *compiles pattern matchers themselves* by building an
IR of match predicates out of `CUSTOM`/`STORE`/`AND`/`OR` UOps and emitting Python
source; (b) a compiled kernel is a `PROGRAM` UOp whose `src` grows
`SINK → LINEAR → SOURCE → BINARY` children as compilation proceeds; (c) the
compilation driver itself is a PatternMatcher (§1.4).

### 1.3 PatternMatcher / graph_rewrite: rules as data

A rewrite rule is a `(UPat, callable)` pair. `UPat` mirrors UOp's shape (op set,
dtype set, src patterns, arg, capture name); the callable receives named captures and
returns a replacement UOp or `None`:

```python
symbolic = PatternMatcher([
  (UPat.var("x") + 0, lambda x: x),                                   # identity
  (UPat.var("x", dtypes.bool) * UPat.var("y", dtypes.bool), lambda x,y: x&y),
  (UPat(Ops.RESHAPE, src=(UPat(Ops.RESHAPE, name="x2"), UPat()), name="x"),
   lambda x, x2: x.replace(src=(x2.src[0], x.src[1]))),               # merge reshapes
])
```

Engine mechanics (all in `ops.py`):

- `PatternMatcher.__init__` indexes rules by root op (`pdict`) and precomputes an
  `early_reject` set of required child ops, so `rewrite(uop)` only tries plausible
  rules — this is what lets **915 rules** (measured `grep -c "(UPat"` across the
  package) stay fast.
- Matchers **compose with `+`** (`pm_a + pm_b` concatenates rule lists, memoized), so
  a pipeline stage is assembled like `symbolic_simple + devectorizer + ren.extra_matcher`.
- `graph_rewrite(sink, pm, ctx=..., name=...)` applies rules to fixpoint. The driver
  (`RewriteContext`, ~110 lines) supports top-down, bottom-up, and an MLIR-style
  single-pass `walk_rewrite`; `UPat` lists (vs. tuples) mean "try all permutations of
  children" for commutative matching.
- Patterns are optionally **compiled to Python source** (`UPAT_COMPILE=1` default)
  for speed; the interpreter fallback is 20 lines.
- **Observability is built into the engine**: `VIZ=1` records every match (rule
  source location, before/after node, timing) into `TrackedGraphRewrite` and serves a
  step-through rewrite debugger (`viz/`, 973 lines); `TRACK_MATCH_STATS` prints
  per-rule hit counts; `CAPTURE_PROCESS_REPLAY` pickles every `graph_rewrite` input
  for CI regression replay. This is the price of distributing logic over 900 rules —
  and tinygrad paid it in infrastructure once, centrally.

### 1.4 The pipeline: named nanopasses over one IR

`codegen/__init__.py::full_rewrite_to_sink` (460-line file) is a readable list of
**~25 named `graph_rewrite` calls**: `"early movement ops"`, `"split ranges"`,
`"initial symbolic"`, `"simplify ranges"`, `"expander"`, `"add gpudims"`,
`"devectorize2"`, `"lower all index dtypes"`, `"early/late decompositions"`,
`"final rewrite"`, `"add control flow"`, `"number params"` … Each is a nanopass whose
name shows up in the VIZ debugger. Per-stage well-formedness is checked by
`uop/spec.py` (263 lines): `spec_tensor` and `spec_program` are themselves
PatternMatchers returning `True/False` per node, run under `SPEC=1` — i.e., **IR
grammars as executable, optional asserts rather than static types**.

Then compilation itself is five rewrite rules on the `PROGRAM` node:

```python
pm_to_program = PatternMatcher([
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK),)),                     do_linearize),   # SINK -> +LINEAR
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK), UPat(Ops.LINEAR))),    do_estimates),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.LINEAR, src=UPat(Ops.INS)))), do_assemble),  # ISA path
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.LINEAR))),            do_render),      # +SOURCE (str)
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(Ops.SOURCE))),    do_compile),     # +BINARY (bytes)
])
```

### 1.5 Renderers: a backend is tables + local rewrite rules

`Renderer` (`renderer/__init__.py`, 85 lines) is a *capability record*, not a
framework: `global_max`, `local_max`, `supports_float4`, `shared_max`,
`tensor_cores`, a `code_for_op` dict of format-string lambdas, and an
`extra_matcher: PatternMatcher | None` of backend legalization rewrites, plus
`render(uops) -> str`.

Measured per-backend cost (class deltas within `cstyle.py`, whole files otherwise):

| Target | Where | LOC | Mechanism |
|---|---|---:|---|
| C-style base | `cstyle.py::CStyleLanguage` | ~135 | generic `string_rewrite` PatternMatcher UOp→string |
| C / clang | `ClangRenderer` | ~49 | table overrides |
| OpenCL | `OpenCLRenderer` | ~45 | table overrides |
| **Metal** | `MetalRenderer` | ~52 | table overrides + a few string rules |
| **CUDA** | `CUDARenderer` | ~78 | table overrides + WMMA prefix code |
| HIP/AMD | `HIPRenderer` | ~105 | |
| **WGSL** | `wgsl.py` | **115 (whole file)** | subclass of CStyleLanguage: `type_map`, ~12-rule `string_rewrite`, ~8-rule `extra_matcher` that *legalizes by rewriting* (packed 8/16-bit load/store emulation, shift-dtype fixes, `a != a → is_nan()`), `render_kernel` prologue emitting `@group/@binding` declarations |
| PTX (assembly) | `ptx.py` | 225 | same pattern-driven approach, emits SASS-adjacent text |
| LLVM IR | `llvmir.py` | 283 | |
| Mesa NIR (3 GPUs) | `nir.py` | 310 | |
| x86 machine code | `isa/x86.py` | 844 | direct binary emission incl. isel + regalloc hooks |

Two structural decisions make renderers this small:

1. **Codegen is string-by-pattern.** The final UOp→text step reuses the *same*
   PatternMatcher engine: `string_rewrite` rules map a UOp (with already-rendered
   children available via `ctx[child]`) to a string. New syntax = one more rule.
2. **Legalization is graph rewriting, not renderer logic.** Anything a backend can't
   express is rewritten *before* rendering by its `extra_matcher`, and the shared
   decomposition passes are **parameterized by the renderer's op table**:
   `get_late_rewrite_patterns(tuple(ren.code_for_op.keys()))` — if a backend has no
   native `POW`/`IDIV`/`SIN`, the portable decomposition rules fire; if it has one,
   they don't. This is the "batteries economics" answer: intrinsics are written once
   as portable decomps to a small primitive set, and each backend declares which
   primitives it renders natively.

Separately, the **runtime seam** (`runtime/ops_*.py`) owns allocation, argument
binding, and launch: `ops_webgpu.py` 221 lines, `ops_metal.py` 192, `ops_cuda.py`
133, `ops_cpu.py` 142. Renderer (types → source) and runtime (buffers → dispatch)
never import each other; `device.py` (379 lines) pairs them by device-name string.

### 1.6 ShapeTracker is gone (2025→26): movement ops as rewrites

The famous `ShapeTracker` (movement ops as zero-copy stride-view algebra in
`shape/shapetracker.py` + `view.py`) **no longer exists** — `tinygrad/shape/` is
deleted at this commit. Movement ops are now ordinary graph nodes
(`RESHAPE/PERMUTE/EXPAND/PAD/FLIP/SHRINK` in the `Ops` enum), and
`schedule/rangeify.py` (555 lines, "rangeify") lowers them by rewriting into
`RANGE`/`INDEX` arithmetic on buffer indices, which the symbolic engine
(`uop/symbolic.py`, 464 lines, 127 rules) then simplifies. `uop/movement.py` is 20
lines of cleanup rules (merge adjacent reshapes, cancel identity permutes). Views are
still zero-copy — the movement op never materializes data; it becomes index math in
the consumer kernel — but the *mechanism* is now "the one IR plus rewrite rules"
instead of a bespoke parallel data structure. The lesson for a small system: tinygrad
deleted its most celebrated special-case component to unify on a single IR and rule
engine, and the replacement is *smaller*.

### 1.7 TinyJit and the compile caches: what is captured, what is keyed

`engine/jit.py` (317 lines). `TinyJit` is **capture-and-replay**, not a
specializing JIT:

- Call 0 runs the Python function normally; call 1 re-runs it while `capturing` is
  set, collecting every scheduled kernel batch as `LINEAR` UOps; `jit_lower` then
  **substitutes each input buffer with a `PARAM` UOp carrying a slot index**
  (`linear.substitute({u: UOp.param(i, u.dtype, u.shape, u.device) ...})`), runs
  memory planning, compiles, and batches kernels into device graph executors
  (CUDA Graphs / Metal ICBs) via `graph_split_rewrite`. Calls ≥2 replay
  `CapturedJit` with the *current* buffers bound to the PARAM slots.
- The captured artifact is validated against
  `expected_input_info = [(view-UOp with buffer replaced by NOOP, sorted unbound
  Variables, dtype, device)] `per input plus argument names — i.e. a **structural
  type/shape/view signature** of the inputs. A mismatch **raises `JitError`** rather
  than respecializing; *value* variation is only allowed through explicit symbolic
  `Variable`s (`UOp.variable("i", 1, 10).bind(3)`), whose values flow per-call via
  `var_vals` into kernel args and launch dims (`updated_vars`,
  `updated_launch_dims`). This is precisely the `Literal`/`Val` inversion: tinygrad
  makes *dynamic* the explicit opt-in, pdum makes *static* the explicit opt-in.
- The kernel-level compile cache is exactly the pdum thesis' shape:
  `to_program_cache[(ast.key, type(renderer), renderer.target, *config_flags)]` —
  content hash of the typed IR **plus renderer identity plus every codegen-relevant
  flag** (NOOPT, USE_TC, IMAGE, …). Compare M0's "backend parameters missing from the
  key" fault line. Below it, `compile_cached` adds an on-disk sqlite cache keyed on
  source text.

### 1.8 What growth did to tinygrad (honest reading)

The 2021 "1000 lines" tinygrad is now a ~14.5 kloc core + 4.4 kloc renderers + 9.5
kloc runtimes, with the budget raised stepwise to 25 k counted lines. The kernel
disciplines (one node type, rules as data, renderer=tables) demonstrably held — a new
C-style backend is still a ~50–115 line PR — but the *code style* needed to stay
under budget (dense one-liners, semicolons) costs readability, and the op enum, while
still one enum, now spans tensor-level and instruction-level concerns that only
conventions ("ops that don't exist in programs") keep apart. A redesign copying the
architecture need not copy the compression.

---

## 2. The nanopass discipline

### 2.1 History and the commercial result

Micropass compilers grew out of Dan Friedman's 1999 Indiana compiler course (one
transformation per pass, S-expression IRs); Sarkar/Waddell/Dybvig formalized the
framework (ICFP 2004), and Keep & Dybvig (ICFP 2013) rebuilt the **commercial Chez
Scheme compiler** with it — this became the shipping compiler in Chez 9.0. Verified
claims from the paper's abstract: the new compiler **replaced 5 of the original 10
passes with over 50 nanopasses**, produced code **15–27% faster** than the original
(due to a better register allocator and added optimizations), with compile times
**within a factor of two** — refuting the "too many traversals" objection that had
kept the ICFP 2004 committee skeptical.

### 2.2 Mechanics

Two forms. `define-language` declares an IR grammar (terminals with predicates,
nonterminals with productions), and — the key move — **a new IR is a diff of the
previous one**:

```scheme
(define-language L1
  (extends Lsrc)
  (terminals (- (datum (d)))
             (+ (constant (c))))
  (Expr (e body)
    (- (quote d))
    (+ (quote c))))
```

Everything not mentioned is copied. `define-language` generates record types per
production, an `unparse-L1`, and predicates — so terms are checked-well-formed *data*
and every IR can be printed for free.

`define-pass` declares input and output languages and only the interesting clauses;
the framework **auto-generates transformers for every unchanged production**
(structure-preserving recursion), and `,[e0]` catamorphism syntax inserts recursive
calls:

```scheme
(define-pass remove-one-armed-if : Lsrc (e) -> L1 ()
  (Expr : Expr (e) -> Expr ()
    [(if ,[e0] ,[e1]) `(if ,e0 ,e1 (void))]))   ; every other form: auto-generated
```

### 2.3 Why it was productive, and the honest Python translation

Productive because: (a) the cost of a pass is proportional to *what it changes*, not
to the size of the IR; (b) each pass's output grammar is machine-checked, so a broken
pass is localized immediately; (c) 50 five-line passes are individually reviewable
where 10 fused passes are not; (d) pass order is an explicit, rearrangeable list.

A literal Python port (per-pass dataclass hierarchies + generated visitors) fights
the language: without macros, N languages × M productions of record types is real
code, and Python's startup/HW costs punish 50 full traversals of object trees.
The **workable Python equivalent is what tinygrad converged on**, and it preserves
every benefit except static exhaustiveness:

- one shared node type instead of per-language record types;
- *grammar-as-predicate*: a per-stage spec (PatternMatcher returning bool, cf.
  `uop/spec.py`) checked in debug mode = `define-language`'s well-formedness check;
- *auto-generated boilerplate* = the rewrite driver: a pass only states rules for
  nodes it changes; `graph_rewrite` recurses structure-preservingly through the rest;
- *pass = named `graph_rewrite` call* in an explicit pipeline list.

The delta discipline ("IR n+1 = IR n minus these forms plus those") survives as
documentation + spec diffs: e.g. "after `devectorize`, no vector dtypes remain";
encode it as `spec_after_devectorize = spec - vector_rules + scalar_only_rule`.

---

## 3. e-graphs / egglog (brief)

**What.** Equality saturation keeps *all* discovered-equal versions of a term in a
congruence-closed e-graph, applies rules non-destructively to saturation, then
extracts the cheapest term by a cost model — eliminating phase-ordering problems for
equational theories. **egglog** fuses this with Datalog (analyses and rewrites in one
fixpoint language). Status July 2026: Rust, v2.x, very active (2,500+ commits,
continuous benchmarking), official Python bindings (`egglog` on PyPI, builds typed
EGraphs from Python expressions). Production precedent: Cranelift's mid-end uses an
"aegraph" (acyclic e-graph) for its instruction-simplification rules.

**Cost/benefit for pdum.dsl.** Benefits: algebraic domains pdum wants later — unit
canonicalization (`m/s · s = m`), einops-style rearrangement algebra, index-math
simplification — are pure equational theories, exactly e-graphs' sweet spot; no rule
ordering bugs. Costs: a Rust dependency; extraction requires a cost model; e-graphs
handle context/effects (loads, stores, control flow) poorly, so they fit *pure
expression regions* only; saturation time is unbounded without limits. tinygrad gets
by with 915 *destructive* rules and no e-graph.

**Verdict:** not a kernel component. Keep the kernel's optimizer seam at
"`sink UOp → sink UOp` function"; an egglog-backed optimizer for pure regions
(export region → egglog terms → saturate → extract → reimport) is then a later
plug-in of a few hundred lines, adoptable without touching anything else.

---

## 4. Other tiny multi-backend exemplars

**QBE** (<https://c9x.me/compile/>, active, last commit June 2026). Measured:
**18,210 lines of C total**; the shared core is **9,082 lines in ~21 files, one pass
per file** (`ssa.c`, `gvn.c`, `gcm.c`, `fold.c`, `copy.c`, `alias.c`, `load.c`,
`spill.c`, `rega.c`, `emit.c`, …); per-target directories: **amd64 3,458 · arm64
1,970 · rv64 1,596** (each = instruction selection + ABI + emit). Stated goal: "70%
of the performance of industrial optimizing compilers in 10% of the code." Design
choices that keep it small: **one SSA IL used at every stage** (no separate MIR/LIR),
a fixed pass list (no pass manager), text in/text out, full C ABI per target and
nothing else. QBE independently confirms both tinygrad ratios: shared-core ~9–15 k,
per-backend ~1.5–3.5 k for *native ISA* targets (vs. ~50–300 lines when the target is
a high-level language like WGSL/CUDA-C — emitting source instead of machine code is
a 10–30× LOC discount, which is exactly pdum's situation).

**chibicc** (Rui Ueyama, ~9 kloc C compiler): notable purely for its *discipline* —
built as ~300 commits, each adding one working, tested language feature; the
incremental-extensibility aesthetic as a development method.

**MLIR/xDSL as contrast.** MLIR has the same conceptual atoms (ops as data,
declarative rewrite patterns, dialects-as-plugins) at 3–4 orders of magnitude more
code, because it pays for genericity: arbitrary regions, unranked types, C++
metaprogramming, stability guarantees. tinygrad is the demonstration that the atoms
survive extreme miniaturization if you fix *one* node shape and skip the generality.
xDSL (Python MLIR) inherits MLIR's ceremony (ODS-style op definitions, typed
dialects); adopting it buys ecosystem compatibility, not smallness.

---

## 5. Synthesis: the disciplines that keep a kernel small *and* extensible

1. **One node type; closed op enum; open rule set.** Extensibility lives in *rules
   over* the IR, not in *subclasses of* the IR. New analyses/optimizations/backends
   add rules and tables; the node type and engine never change. (tinygrad UOp; QBE's
   single IL.)
2. **Hash-consing + content hash on the node.** Identity = structural equality makes
   rewriting dict-cheap and gives a correct compile-cache key (`UOp.key`) for free.
3. **Rewrite engine as the single algorithm.** ~150 lines of
   `UPat`/`PatternMatcher`/`graph_rewrite` service simplification, legalization,
   lowering, *autodiff* (pm_gradient: per-op VJP rules, 132 lines), rendering-to-string,
   and even the compile driver. Every one of those is "a list of (pattern, fn) pairs",
   composable with `+`.
4. **Nanopass discipline without nanopass machinery**: many small *named* passes in
   one explicit pipeline function; per-stage executable specs checked in debug mode;
   pass cost proportional to what it changes because the driver auto-recurses.
5. **Renderer-per-backend = capability record + tables + local legalization rules**;
   shared decomposition passes parameterized by each backend's declared primitive set.
   Keep codegen (types→source) and runtime (buffers→launch) as two separate seams.
6. **Line-budget culture enforced by CI**, with a per-PR delta bot, and generated
   code excluded from the count.
7. **Observability paid for centrally**: when logic is 900 distributed rules, a
   rewrite-trace debugger and per-rule stats are the debugger. Build it into the
   engine (a `name=` on every pass, a match log), not as an afterthought.
8. **Kill parallel data structures.** When a special mechanism (ShapeTracker) can be
   re-expressed as ops + rules, the unified version tends to be smaller and
   composes with everything else for free.

---

## Design lessons for pdum.dsl

1. **Adopt a UOp-shaped IR to escape M0's expression-tree ceiling.** One immutable,
   hash-consed dataclass `(op, dtype, src, arg)` with a ~40-member op enum covers
   scalars, control flow (`RANGE/IF/END`), memory (`INDEX/LOAD/STORE`), and
   multi-statement bodies (`SINK`/ordering edges). Budget: node type + interning +
   toposort ≈ 150 lines; `UPat + PatternMatcher + graph_rewrite` ≈ 200 lines. That
   is the entire kernel engine, and it directly answers desiderata Q2 (own IR, not
   xDSL — xDSL buys MLIR ceremony, not smallness).
2. **Make the dialect an input via renderer capability records.** Fix the
   core→WGSL-intrinsics dependency the way tinygrad does: the backend object carries
   `type_map`, `code_for_op`, and an `extra_matcher` of legalization rewrites; the
   frontend/lowering passes take the renderer as a *parameter* and select
   decomposition rules by `supported_ops = renderer.code_for_op.keys()`. This is
   also the batteries answer (Q4): implement `mean`, `clip`, color-space methods
   once as portable decomposition rules to a ~30-op primitive set; a backend only
   ever implements the primitive set.
3. **Budget backends at 50–300 lines and hold to it.** Evidence: WGSL 115, Metal
   ~52, CUDA ~78, C ~49 (renderer side), plus 130–220 lines of runtime
   (alloc/launch) each. If a pdum backend needs more, the missing abstraction
   belongs in shared rules, not the backend. Python backend ≈ a `string_rewrite`
   that prints Python, or direct UOp interpretation (tinygrad's `ops_python.py`,
   231 lines, does exactly this as its spec backend).
4. **Steal PARAM/BIND slot-numbering as the marshaling abstraction (Q6).** Lower
   every capture/argument to `PARAM(slot, dtype, …)` nodes via a rewrite pass
   (`pm_number_params`); the logical→physical split (array → base pointer + shape
   ints; scalar → uniform-buffer offset) is *itself a rewrite* from one PARAM to
   several, done per-backend before numbering. Per-call marshaling is then "write
   values into numbered slots" — tinygrad's `updated_vars`/`updated_launch_dims`
   shows the hot-loop form: precomputed `(call_index, slot) → value` lists, no graph
   traversal per call. This also fixes M0's per-frame `flatten`: value extraction is
   a slot-table fill, separated from structure compilation. Units auto-conversion
   (Q7) slots in here: a conversion is a rewrite inserted at the PARAM boundary at
   phase-B key time.
5. **Copy the `to_program` cache key shape, then put the type-keyed cache above
   it.** Kernel-level key = `(IR content hash, renderer class, target, all codegen
   flags)` — this cures every M0 cache-hygiene fault (backend params in key, no
   generation sledgehammer needed at this layer, disk layer keyed on source text
   below it). pdum's `(FnType, arg_types)` cache sits *above* and maps to IR; note
   tinygrad *raises* on signature drift where pdum must respecialize — that
   difference is pdum's entire value proposition, and it's confined to the layer
   above the pipeline.
6. **Do nanopass the Python way: named passes + executable per-stage specs.** A
   pipeline is a literal list of `graph_rewrite(sink, rules, name="…")` calls; each
   stage boundary gets a spec-PatternMatcher (à la `uop/spec.py`) asserting its
   grammar under a debug flag, with later specs written as diffs of earlier ones.
   Do not build per-pass class hierarchies or visitor codegen.
7. **Transformations are rule sets over the same IR (Q5).** tinygrad's reverse-mode
   autodiff is a 132-line PatternMatcher of per-op VJPs applied at graph level; vmap
   is likewise a rewrite (tinygrad's expander turns RANGE-indexed scalars into
   vectors). Design pdum's autodiff/vmap as `PatternMatcher`s applied IR→IR before
   backend lowering — they then compose with every backend for free, and new ops
   declare their VJP as one rule.
8. **Defer e-graphs; keep the seam.** Optimizer = `sink → sink` function. Ship with
   destructive rules (sufficient at tinygrad scale: 915 rules); revisit egglog
   (Rust+Python, active, v2.x) when units/einops algebras arrive, as an optional
   plug-in for pure expression regions.
9. **Institute the line budget and rewrite-tracing on day one.** A CI check
   (`sz.py`-style, generated marshaling code excluded) with a PR delta bot, and a
   `name=`d rewrite trace viewer, are each <300 lines and are the two cultural
   mechanisms that demonstrably kept tinygrad's kernel disciplined for five years.
10. **Accept the mixin/sugar split.** tinygrad keeps the IR core lean by pushing
    user-facing convenience (operator overloads, broadcasting, method sugar — 4.5
    kloc) into mixins over the same node type. pdum's "struct method syntax" and
    t-string mini-languages are the analogous layer: front-end sugar that only ever
    *constructs* kernel IR, so mini-language plug-ins (einops via PEP 750) never
    touch the kernel.
