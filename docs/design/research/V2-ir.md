# V2 — Verdict: the IR for pdum.dsl

Consolidation verdict for the redesign's IR question. Inputs: R3 (xDSL), R7
(tinygrad/nanopass/QBE/egglog), R2 (DaCe), R4 (JAX), R6 (torch.compile), plus
`docs/desiderata.md` and `design/dsl_caching_layer.md`. Decision date: 2026-07-11.

---

## 1. Recommendation

**Build a purpose-built micro-IR — no xDSL dependency in the kernel.** Its shape is
a deliberate synthesis of the three strongest exemplars:

- **One immutable node type** with a memoized structural content hash
  (tinygrad's UOp economy — R7),
- **structured control flow as regions with typed parameters and value yields**,
  carried *on* the node, never as basic blocks or a CFG (MLIR/scf shape — R3;
  jaxpr's "sub-programs in op params" — R4),
- **the value/attribute split as constitutional law**: operands are runtime data,
  attributes are compile-time constants, and attributes participate in structural
  identity — so "what is in the cache key" has a syntactic answer (R3),
- **all logic as declarative rule sets over this one node type**: simplification,
  legalization, lowering, autodiff/vmap, and rendering-to-string are lists of
  `(pattern, fn)` pairs run by one ~180-line rewrite driver (R7, R4),
- **an MLIR-flavored textual printer from day one**, so golden-file tests work and
  a later migration to xDSL/MLIR is a refactor, not a rewrite (R3).

Budget: **~650 lines** for the complete IR kernel (types, node, op table,
verifier, pattern engine, printer), **~850** including the reference Python
interpreter backend. Concrete node model in §3.

xDSL is retained in exactly one role: a **pinned, optional dev-time oracle** — a
small pdum-IR→xDSL translator plus its interpreter for differential-testing
backends — never on any user-facing path.

---

## 2. Rationale

**Why purpose-built.** Four independent lines of evidence agree that an IR at
pdum's scale is cheap and a framework is not worth its costs:

1. *IRs are measured-cheap.* Autodidax's typed flat IR with builder, typechecker,
   and evaluator is 288 code lines (R4 §5). tinygrad's entire IR + rewrite engine +
   symbolic math is 3.1 kloc and drives ~14 backends (R7 §1.1). R3's own estimate
   for a concept-faithful MLIR-shaped IR — made *after* building a working xDSL
   prototype — is 300–600 lines. The "IRs are expensive, adopt a framework"
   premise is empirically false at this scale.
2. *xDSL's costs land exactly where pdum is sensitive.* It is 0.x with ~biweekly
   releases and 4 breaking releases in the last 10 (R3 §2.1); pdum's
   cache-correctness invariants would depend on a dependency's `__eq__`/`__hash__`
   and printing semantics that it reserves the right to change (v0.59 changed the
   textual format; v0.64 the mutation model). The cache key is the project's crown
   jewel; it must be sovereign.
3. *xDSL buys generality pdum will never use.* Its 18 kloc core funds unstructured
   CFG, variadics, a constraint solver, and MLIR-syntax fidelity. pdum needs ~15–40
   ops, one region construct, a verifier, a printer, a fixpoint rewriter. And xDSL
   contributes nothing to the novel layer — no typeof, no type-keyed cache, no
   capture extraction, no marshaling, and its WGSL backend is a 249-line PoC
   (R3 §4).
4. *Identity.* The desiderata's stated aesthetic is a tiny kernel readable in a
   sitting. "1000 lines of ours on top of an 18 kloc core inside a 146 kloc
   package" (R3) is not that; 650 owned lines are.

**Why this shape and not the others.** The requirements matrix decides it:

- *Five textual, structured backends (WGSL/CUDA/Metal/C/Python).* None of these
  languages has goto. A structured region IR renders to them 1:1 — a region is a
  `{ ... }` block, a `core.if` is an `if`, a `core.for` is a `for`. An
  unstructured CFG (QBE-style) would require a relooper pass per backend, pure
  cost. tinygrad's graph encoding (RANGE/IF/END as ordering edges) requires a
  linearization pass before every render (R7 §1.4).
- *Frontend is an AST of user-written statements.* Shader-style code is
  statement-heavy: `for` over neighbors, early-exit `if`, stores. An AST lowers
  directly to ordered ops in regions. tinygrad never faces this problem — its
  control flow is compiler-*generated* from tensor expressions, which is why the
  pure-graph encoding works there and would be an impedance mismatch here.
- *Autodiff and vmap as rule-level transformations.* R4 (autodidax) and R7
  (pm_gradient, 132 lines) prove both shapes work; but AD of *structured* if/for
  with value yields is the well-understood JAX cond/scan pattern, whereas AD over
  a CFG is the hard version (R6: AOTAutograd only works because a legacy eager AD
  engine existed). Each structured control-flow op costs ~150–200 lines across
  all transformations (R4 §5), which is why the set is capped at three: `if`,
  `for`, `call`.
- *Declarative rewrite passes.* The `(pattern, fn)` rule-set engine is
  representation-agnostic and costs ~180 lines (R7 §1.3). It works identically
  over region-structured nodes; tinygrad's own renderers prove the same engine
  handles string emission.
- *Smallness that survives growth.* tinygrad's discipline held for five years
  under a CI line budget because extensibility lives in *rules over* the IR, not
  *subclasses of* it (R7 §5.1). The node type below is closed; the op namespace,
  rule sets, and renderer tables are open. That is the prime directive
  (incremental extensibility) expressed structurally.

**Why not hash-consing.** tinygrad interns every node globally (identity =
structural equality). With regions and typed binders, interning is subtle
(alpha-equivalence, scope discipline by convention), and pdum kernels are tens to
hundreds of ops, not millions. We take the *benefit* of consing — a structural
content hash usable as the compile-cache key — as a memoized `Node.key` (sha256,
tinygrad's `UOp.key`) without the interning machinery. Ordered region bodies also
carry effect order (stores) for free, avoiding effect-token plumbing.

---

## 3. The core node model

The entire IR data model — concrete field lists:

```python
# ir/types.py (~80 lines) — open set of frozen, structurally-hashable types
class Type: ...                                   # base; all subclasses frozen dataclasses
@dataclass(frozen=True)
class Scalar(Type):  kind: str                    # "f32" "i32" "i64" "u32" "bool"
@dataclass(frozen=True)
class Vec(Type):     elem: Scalar; n: int         # vec2/3/4
@dataclass(frozen=True)
class Array(Type):   dtype: Type; ndim: int; layout: str   # shapes are runtime data, per thesis
@dataclass(frozen=True)
class Record(Type):  name: str; fields: tuple[tuple[str, Type], ...]
@dataclass(frozen=True)
class FnRef(Type):   fntype_id: bytes             # link to the caching layer's FnType
# future, no schema change needed: Quantity(Type) wrapping (elem, dimension, unit)

# ir/core.py (~150 lines) — the whole node model
Attr  = tuple[str, Hashable]                      # attr values: ints/str/Type/tuples thereof

@dataclass(frozen=True, slots=True, weakref_slot=True)
class Node:
    op:      str                       # dialect-namespaced: "core.add", "wgsl.sample"
    type:    Type                      # the node IS its SSA value; exactly one type
    args:    tuple["Node", ...] = ()   # operands — RUNTIME data; never in the cache key's value part
    attrs:   tuple[Attr, ...]   = ()   # compile-time constants — in structural identity by construction
    regions: tuple["Region", ...] = () # nonempty only for core.if / core.for / core.func

    @cached_property
    def key(self) -> bytes:            # recursive sha256 over (op, type, attrs,
        ...                            #   args' keys, regions' keys) — the content hash
                                       #   the artifact cache is keyed on (tinygrad UOp.key)

@dataclass(frozen=True, slots=True)
class Region:
    params: tuple[Node, ...]           # "core.param" nodes: attrs=(("index", i),), typed binders
    body:   tuple[Node, ...]           # ORDERED; order is authoritative for effectful ops;
                                       # last element is the "core.yield" terminator
```

Notes on the shape:

- **SSA without Value objects.** An operand is a reference to a producer `Node`
  (earlier in the body, or a region param). Single result per node; tuples/records
  return one `Record`-typed value projected by `core.field`/`core.item` ops — this
  keeps one node type while satisfying AD's multiple-results requirement (R4
  lesson 6) via projections.
- **Multi-value structured control flow** (scf-shaped, R3 lesson 3):

  ```
  %r = core.if %cond : f32 {            # region 0: then, region 1: else
         ... core.yield %a } { ... core.yield %b }
  %c = core.for %lo, %hi, %step, %init : f32 {   # region params: (%iv: i32, %carry: f32)
         ... core.yield %next_carry }
  ```
- **Captures appear as ops, not consts.** A runtime capture lowers to
  `core.env slot=k : T` (operand-free, slot attr assigned by the numbering pass);
  a `Literal`-lifted capture lowers to `core.const value=v : T`. The first is
  invisible to `Node.key`'s value dimension; the second is in the key by
  construction. This is jaxpr's `constvars` with the substitution R4 lesson 4
  demands — captures as parameters with marshaling annotations, never baked
  values.
- **Types are intrinsic; everything else is a side channel.** `type` is a field
  because type correctness *is* cache correctness and every pass/renderer needs
  it. Non-identity annotations (source locations, diagnostics, analysis results,
  rewrite provenance) live in `WeakKeyDictionary` side tables (hence
  `weakref_slot`) — FX's `meta` lesson (R6 lesson 5) without letting metadata
  leak into structural identity.
- **Op definitions are data, not classes** (IRDL-lite, R3 lesson 4):

  ```python
  # ir/opdefs.py (~100 lines incl. the generic verifier)
  @dataclass(frozen=True)
  class OpDef:
      arity: int | str                 # 2, "variadic", "3+carries"
      n_regions: int = 0
      attrs: tuple[tuple[str, type], ...] = ()
      traits: frozenset[str] = frozenset()   # "pure", "commutative", "terminator", ...
      verify: Callable | None = None         # extra semantic check
  CORE_OPS: dict[str, OpDef] = { "core.add": OpDef(2, traits={"pure","commutative"}), ... }
  ```
  A dialect is a `dict[str, OpDef]` merged into a `Context`; per-op cost stays at
  1–3 lines. Traits let passes query capability instead of switching on names.
- **The rewrite engine** (~180 lines, `ir/pattern.py`): `Pat` mirrors `Node`
  (op set, type, arg patterns, capture name); a `RuleSet` is a list of
  `(Pat, fn)` indexed by root op with early-reject sets, composable with `+`;
  `rewrite(region, rules, name=...)` runs bottom-up to fixpoint, recursing into
  regions, rebuilding ordered bodies, logging every match under a debug flag
  (R7's VIZ discipline, paid once centrally).
- **The printer** (~120 lines): MLIR-flavored text, one op per line,
  `%n = core.add %a, %b : f32`, regions as indented braces. Print-only at first;
  golden-file tests at every pipeline stage. Keeping the flavor MLIR-compatible
  is the deliberate escape hatch: if pdum ever outgrows this kernel, re-hosting
  these dialects on xDSL is mechanical (R3 §6).

### LOC budget

| Component | LOC |
|---|---:|
| `types.py` | ~80 |
| `core.py` (Node/Region/key/builder) | ~150 |
| `opdefs.py` (OpDef table + generic verifier) | ~100 |
| `pattern.py` (Pat/RuleSet/rewrite driver + match log) | ~180 |
| `printer.py` (MLIR-flavored text) | ~120 |
| **IR kernel total** | **~630** |
| `interp.py` (reference Python backend, walks regions) | ~220 |
| per-stage spec RuleSets (grammar checks, debug-only) | ~100 |

For calibration: M0's expression tree was ~130 lines and could not grow; this is
~5× that for control flow, records, declarative rewriting, printing, and a
transformation-ready substrate — versus xDSL's 18 kloc usable core.

---

## 4. Considered and rejected

| Alternative | Why it lost |
|---|---|
| **Adopt xDSL** as the IR framework | 0.x churn (4 breaking releases in ~10, biweekly cadence) under pdum's most correctness-critical layer; a dependency would own the equality/hash semantics the cache key rests on; 146 kloc identity cost against a "readable in a sitting" kernel; contributes zero to the novel layer (caching/typeof/marshaling/WGSL). Its concept set is adopted wholesale; the dependency is not. Retained as pinned dev-time differential-testing oracle and as the named migration target. (R3) |
| **Grow the M0 expression tree** | Cannot carry statements, control flow, records, or transformations — every planned capability presses on it; this is the fault line the redesign exists to fix. No report supports it. |
| **UOp-style pure graph with RANGE/IF/END** (tinygrad-literal) | Proven brilliantly — but for compiler-*generated* control flow. pdum's input is user-written statement-heavy AST code and its outputs are five structured languages; a pure-graph encoding forces linearization before every render and effect-ordering machinery, and global interning is subtle under binders. We take its engine (one node type, content hash, rule sets, renderer economics) and reject only its control-flow encoding. (R7) |
| **Flat SSA/CFG with basic blocks** (QBE-shape) | Unstructured CFG must be re-structured (relooper) to emit WGSL/C — pure cost since the frontend never produces irreducible flow; AD over a CFG is the hard version of AD. QBE's ratios were still instructive (source-emitting backends are 10–30× cheaper than ISA backends). (R7 §4) |
| **jaxpr-literal** (flat ANF, consts baked, trace acquisition) | Two of its three signature moves are the anti-model: tracing cannot see `if`/`for` (combinator shadow-language, forbidden by the aesthetics) and `ClosedJaxpr.consts` bakes captures into artifacts — the documented ~240× retrace pathology pdum exists to fix. Its third move — flat typed eqn lists with sub-programs carried by ops — is adopted as the region-body shape. (R4) |
| **SDFG-style dataflow IR** | Buys graph-rewrite legality proofs pdum doesn't need at ~160 kloc and seconds-scale compiles; sympy on the hot path; value-keyed scalar captures. Its separable pieces (replacements registry, two-layer batteries) plug into this IR unchanged. (R2) |
| **FX-style six-opcode graph** | Minimal vocabulary is adopted, but FX's defining omission — no control flow in the IR — is exactly what forced Dynamo's graph-break/resume-function machinery. pdum puts `if`/`for` in the IR and never needs any of it. (R6) |
| **egglog / e-graph substrate** | Healthy project, wrong layer: fits pure expression regions with a cost model, handles effects/control flow poorly. The optimizer seam stays `Region -> Region`; an egglog plug-in for units/einops algebras can arrive later without touching the kernel. (R7 §3) |

---

## 5. Implications for the architecture

1. **The cache stack becomes two clean layers.** Above: the thesis cache,
   `(FnType, arg_types, generation) -> fastpath record`. Below: the kernel compile
   cache, `(Node.key of the lowered func, renderer.name, renderer flags) ->
   artifact` — tinygrad's proven key shape, curing M0's
   backend-params-missing-from-key fault, with an optional disk layer keyed on
   emitted source text. The IR is touched only between miss and artifact; the hit
   path never sees it (M0's per-frame `flatten` is structurally impossible).
2. **The value/attribute split is the `Literal`/uniform mechanism.** Capture
   classification (phase A) decides `core.env` (operand-ish, runtime, re-marshaled)
   vs `core.const` (attr, in `Node.key`, recompiles per value). The caching
   layer's deliberate exception is now one lowering decision, auditable in
   printed IR.
3. **Marshaling is a rewrite pass, not a runtime walk.** A per-backend
   `legalize_params` RuleSet rewrites one logical `core.env` into N physical ones
   (array → base-pointer env + ndim shape envs; scalar → uniform-buffer slot),
   then `number_params` assigns slots. The emitted slot table *is* the marshaling
   plan: per-call work is "write current values into numbered slots." Units
   auto-conversion later inserts a conversion rewrite at this boundary.
4. **A backend is a capability record + tables + local rules, budgeted at 50–300
   lines.** `Renderer(name, type_map, code_for_op, extra_rules, render)`; shared
   decomposition passes are parameterized by `renderer.code_for_op.keys()`, so
   batteries are written once against the core dialect and decompose only where a
   backend lacks a primitive. Evidence this budget is real: tinygrad WGSL 115 /
   Metal ~52 / CUDA ~78 / C ~49 lines. Runtime (alloc/launch) is a separate seam.
5. **Transformations are rule registries over the same nodes** (JAX matrix): per-op
   `jvp`/`transpose`/`batch` rules in dicts keyed by op name; a transformation is
   a `Region -> Region` function plus its registry. Ship order by measured cost:
   type rules → vmap (~80 lines) → jvp (~80) → reverse mode (~440, last). Every
   new structured op costs one rule per existing transformation — hence the hard
   cap at `if`/`for`/`call`.
6. **Dialects are dict merges.** `Context = {**CORE_OPS, **WGSL_OPS, ...}` plus
   their rule sets; the frontend targets only `core.*`. T-string mini-languages
   (einops) are front-end sugar that constructs core-dialect nodes — they never
   touch the kernel.
7. **Day-one infrastructure, all cheap:** MLIR-flavored printer + golden files at
   every stage; per-stage spec RuleSets under a debug flag (nanopass grammars,
   Python-style); named passes with a rewrite-match log; a `sz.py`-style CI line
   budget with a PR delta bot. These four are the culture that kept tinygrad's
   kernel disciplined for five years.
8. **xDSL's slot:** an optional `dev` extra, pinned, with a ~200-line
   pdum-IR→xDSL translator, used only to differentially test backends against
   its interpreter. It may never be imported from the kernel.

---

## 6. Confidence, and what would change my mind

**Confidence: high** on purpose-built over xDSL (every report that touched the
question landed the same way, including the xDSL report itself, with measured
evidence on both sides). **Medium-high** on region-structured over UOp-pure-graph
— both are proven; regions win on fit to AST input, structured-source output,
and AD tractability, not on any absolute superiority.

Evidence that would flip the decisions:

- *Reopen xDSL adoption* if: the IR kernel drifts past ~2–3 kloc or the op-def/
  verifier layer starts reinventing IRDL's constraint solver; or autodiff over
  regions proves so hard that MLIR-ecosystem tooling (Enzyme-style AD, existing
  lowering stacks) becomes the cheapest path; or xDSL ships a 1.0 with a
  stability promise. The MLIR-flavored printed form and declarative op table
  exist precisely to keep this a refactor.
- *Reopen the pure-graph shape* if: fixpoint rewriting over ordered region bodies
  proves clumsy in practice (excessive body-rebuild churn, ordering bugs in
  pure-op motion), or a scheduling/fusion layer ever becomes a goal — at which
  point tinygrad's RANGE encoding and linearizer is the tested design.
- *Reopen e-graphs* when units algebra or einops rearrangement land and
  destructive rules visibly fight phase-ordering; the `Region -> Region` seam is
  the pre-committed insertion point.

---

## Design lessons for pdum.dsl

1. **Own the IR; it costs ~650 lines.** One frozen `Node(op, type, args, attrs,
   regions)` dataclass + `Region(params, body)` is the whole data model. Never
   let a dependency own the equality/hash semantics the cache key rests on.
2. **Value/attribute split = the caching thesis at the IR level.** Runtime
   captures are `core.env` ops (never in the key's value part); `Literal`-lifted
   captures are attrs (in `Node.key` by construction). Auditable by printing.
3. **Structured regions with value yields; exactly three region ops** (`if`,
   `for`, `call`/`func`). No blocks, no successors, no CFG, ever. Each new
   higher-order op taxes every transformation ~150–200 lines — treat additions
   as constitutional amendments.
4. **Memoized content hash instead of hash-consing.** `Node.key` (recursive
   sha256) gives tinygrad's compile-cache key without global interning under
   binders. Kernel compile cache key = `(Node.key, renderer, flags)`.
5. **One rewrite engine, everything is rules:** simplification, legalization,
   backend decompositions selected by renderer capability, autodiff (per-op VJP
   rules), vmap, and UOp→string rendering are all `RuleSet`s run by the same
   ~180-line driver. New feature = new rules; the node type never changes.
6. **Types are an intrinsic node field; all other annotations are weak side
   tables.** Type correctness is cache correctness. The `Type` set is open
   (frozen dataclasses), so `Quantity`/units arrive later without IR schema
   change.
7. **Marshaling is the `core.env` → physical-params rewrite + slot numbering**;
   the per-call hot path is a slot-table fill, and the IR is never walked after
   compile.
8. **Print MLIR-flavored text and golden-test every stage from day one** — it is
   the debugging story, the spec-check substrate, and the pre-paid escape hatch
   to xDSL/MLIR.
9. **Enforce the budget culturally:** CI line count with per-PR delta, named
   passes, rewrite-match logging, per-stage grammar specs under a debug flag.
   These are what made the exemplar kernels stay small.
10. **Use xDSL where churn can't hurt:** pinned dev-time interpreter oracle for
    differential-testing backends; nothing more.
