# Desiderata

A living document. It records the **desires, influences, aesthetics, and open
questions** that should drive the redesign of `pdum.dsl` — the inputs a research
team needs to evaluate architecture candidates. It is deliberately *not* a design,
a roadmap, or a research plan: it says what we want and what we don't yet know,
not how to build it or how to investigate it.

Context: the current codebase (Milestone 0) is a **reference asset** — a complete,
working, end-to-end proof of the core idea, kept as an artifact to learn from. The
next step is a careful redesign of key infrastructure pieces, informed by research
into the projects and questions collected here. This document will be built up
incrementally as thinking develops.

---

## 1. The core idea (invariant under redesign)

The one idea the reference asset proved, which any redesign must preserve:

- A closure is **(code identity, typed environment, environment values)**.
- Compilation is keyed on the **types** of the environment and arguments — never
  on the values. (`design/closure_specialization.md` documents why numba cannot do
  this; `design/dsl_caching_layer.md` is the full hazard analysis.)
- Capture (phase A, at decoration) is compile-free; compilation (phase B, at first
  call) happens once per type signature.

The driving usage pattern:

> An expression is built up in Python and rendered into a **tight Python loop**.
> Inside the loop, each iteration builds up a *program* — an expression with
> closed-over values. Most iterations require **no compilation**; when the program
> is called, the infrastructure knows how to **pass the current values** to the
> compiled artifact.

Two consequences of that pattern worth stating explicitly:

- **The loop must stay hot.** Re-building the closure each iteration with new
  values must cost roughly a parameter write, not a compile.
- **Value marshaling is a first-class concern.** Passing "the current values" is
  not always one-slot-per-variable: a single logical closed-over value may be
  represented by **multiple physical parameters** — e.g. an array capture becomes
  a data pointer *plus* size/shape information; a WebGPU capture becomes a slot in
  a packed uniform buffer. The infrastructure owns this logical-value →
  physical-parameters mapping, per backend.

## 2. Domains

The domains that motivate the project. Their common shape: a human in the loop,
moving parameters in real time over a compiled kernel.

- **Scientific simulation.**
- **Design optimization** — a simulation is running and the user tweaks values in
  real time, trying to optimize a design.
- **Art** — generative graphics (the shader use case).
- **Synthetic music.**
- Related territory as it appears; this list is not closed.

### 2.1 Differentiable programming (the NN-shaped case, recorded 2026-07-11)

A JAX-style functional NN workflow is a natural satellite of this design, and
exercises it without new kernel mechanisms: networks are **compositions of
small jitted closures, each lexically capturing its own weights** — the
program *is* the parameter container (a Handle tree), something neither JAX
(captures bake into the trace; closure rebuilds retrace) nor PyTorch
(parameters must live in stateful modules) can offer. The flat labeled
parameter tree the optimizer and checkpointing need is **derived** from the
composition by the marshaling layer (LeafPaths through nested envs), not
declared by the user; labels come mostly free from structure (factory
qualnames + capture names) with light explicit annotation where needed.
Train/eval mode is a `Literal` lift (two artifacts, dropout folded away);
dropout RNG is a counter-based generator seeded by an ordinary runtime
capture; gradient clipping lives optimizer-side (optax-shaped chains) or as a
post-transpose IR pass; `grad(loss, wrt="encoder.*")` is a `Derived` identity
matching derived label paths. **Honest scope:** tensor-op performance is
delegated to a mature tensor backend (MLX-class) — this wins on workflow for
interactive/medium-scale differentiable programs (the design-optimization
domain wearing an NN costume), and does not contest LLM-scale training
(XLA/sharding/kernel-ecosystem moats). Known warts: batch-norm-style running
statistics (functional-state threading); reverse-mode memory management
(remat) is real AD engineering beyond the M4 milestone. Working notes:
`design/deep-learning-notes.md`.

## 3. Backends

The framework must support multiple backends behind one core. Currently of
interest:

| Backend | Target | Platform | Notes |
|---|---|---|---|
| **WebGPU** | WGSL | cross-platform | exists in the reference asset |
| **CuPy** | CUDA (via `cupy.RawKernel`) | Linux/NVIDIA | render target is CUDA source rather than WGSL |
| **MLX** | Metal (via MLX custom kernels) | macOS | |
| **Python** | compile *to* Python, or run the Python directly | everywhere | zero-dependency floor; debugging and reference semantics |
| **C** | C source | everywhere | portable native path |

Other backends not yet thought of should be assumed to arrive later; the backend
seam should not encode assumptions specific to the first few.

## 4. Language surface and capabilities

### 4.1 Numba-likeness

The desired feel is **numba-like**: decorate an ordinary-looking Python function,
write a reasonable subset of Python inside it, get a fast compiled kernel.

### 4.2 The "batteries" question

Numba makes NumPy core functions (`mean`, etc.) usable inside jitted code — they
are **recognized intrinsics** the infrastructure knows how to lower, which is
different from merely supporting Python operators. These batteries are a large
part of numba's ergonomic value, but implementing them is real work **per
backend**. How much of this battery set to commit to — and how to make batteries
cheap to add and share across backends — is an open question (§7).

### 4.3 Structured data

- **Arrays/tensors with structured element types** (numba-style records) matter.
  This part of numba's capability set seems important, not optional.
- **Clean method syntax on struct types.** Given, say, a `Color` type, converting
  it to another color space should be a convenient method-like call, not a
  ceremony. Convenience functions attached to user types should feel first-class
  in the DSL.

### 4.4 Units (eventually)

Physical **units and dimensions** should eventually participate: two values of the
same physical dimension expressed in different units should be **auto-converted by
the compiler** — at the level where it arranges the arguments passed to compiled
functions. This is not a day-one feature, but the type-system and marshaling
design should not preclude it.

### 4.5 Embedded mini-languages

Python template literals (t-strings, PEP 750 — available on 3.14, which this
project already targets) could host **many small expression languages** — e.g. an
einops-like notation for tensor rearrangement — as sub-DSLs that are **pluggable
into the same infrastructure** (typed, cached, compiled per backend like anything
else).

## 5. Program transformations

Transformations over programs should be **first-class, with nice syntax** — not
bolt-ons:

- **Automatic differentiation** — the primary one.
- **Auto-vectorization** (vmap-style).
- Other transformations of the kind available in MLX or JAX.

## 6. The prime directive: incremental extensibility

The system does **not** need all capabilities from day one. What it must have from
day one is the property that capabilities can be **added incrementally**:

- new **syntax** (widening the accepted Python subset; new mini-languages),
- new **program analyses** (type rules, unit checking, transformation passes),
- new **backends**.

This is the main criterion for judging architecture candidates. A design that
delivers more features but makes the next feature harder to add is worse than a
smaller design with clean extension seams.

## 7. Open questions

Things genuinely not yet known or decided; prime targets for the research phase.

1. **Frontend analysis: AST or bytecode?** Two candidate approaches to program
   analysis: (a) work on the **Python AST**, with tooling to resolve globals and
   other names against the AST; (b) work on **Python bytecode**, which is what
   numba does. What are the real trade-offs (fidelity, version churn, source
   availability, decorator/closure handling, error messages)?
2. **What intermediate representation?** Build our own small IR (the reference
   asset's expression tree cannot grow to control flow, arrays, or
   transformations), or adopt a toolkit like **xDSL**? Is xDSL mature enough, and
   does an MLIR-style dialect stack fit a system this small?
3. **How does DaCe do it?** DaCe compiles Python-like code to multiple backends —
   an end use case adjacent to ours. What is their actual frontend technique (AST
   or bytecode)? Their program-analysis style (dataflow-centric) is *not*
   necessarily what we want — the question is what's reusable from their approach
   versus what's specific to their model.
4. **Batteries economics.** How do we get numba-style NumPy conveniences without
   hand-implementing every intrinsic × every backend? Is there a layered design
   (portable definitions lowered to backend primitives) that keeps batteries
   cheap?
5. **Where do transformations live?** For autodiff/vmap to be first-class, at what
   level do they operate — source, IR, or backend? What does that imply for the IR
   choice (question 2)?
6. **The marshaling/ABI layer.** What is the right abstraction for "one logical
   value → N physical parameters" that covers uniform buffers (WebGPU), kernel
   arguments (CUDA/Metal), C ABIs, and plain Python calls — including future units
   auto-conversion at the argument-arranging step?
7. **Units in the type system.** How do dimensions/units interact with the
   type-keyed cache (do units live in the type? in the marshaling layer? both)?

### Carried over from the reference asset

Known fault lines identified in the M0 review, expected inputs to the redesign
rather than patches to the current code:

- The **IR is a single-expression tree** — no `if`/`for` statements,
  multi-statement device functions, indexing, or arrays. Nearly every planned
  capability presses on this.
- The **core imports the WGSL dialect tables** (`ast_lower`, `infer` →
  `backends/wgsl/intrinsics`); the dialect must become an input to the frontend,
  not a dependency of it.
- **Per-frame `flatten`** re-lowers ASTs to collect current values; value
  extraction should be separable from structure compilation.
- **Captured uniforms are scalars only**; the capture→layout path for
  vectors/tuples/structs doesn't exist.
- **Cache hygiene**: per-drawer caches, backend parameters (e.g. target format)
  missing from the key, no eviction, generation counter as a global sledgehammer.
- **Two type levels** (honest capture types vs. backend-narrowed types) are
  handled implicitly and will need an explicit story as the lattice grows.

## 8. Influences and reference projects

Projects the research phase should draw from. "Take the best ideas from all of
these" is the brief; the list is open and the team should add comparables it
discovers.

| Project | Why it's interesting | What to look at |
|---|---|---|
| **The M0 reference asset** (this repo) | proves the type-keyed caching thesis end-to-end for WebGPU | `docs/m0/theory/`, `reference/REVIEW.md`, `design/` notes; code frozen at `src/pdum/dsl_reference/`, tests at `reference/tests/` |
| **numba** | the ergonomic north star (decorator workflow, NumPy batteries, structured arrays/records) *and* the documented anti-pattern (identity-keyed caches, captures frozen as constants) | intrinsic/typing/lowering architecture; `types.Record`/`structref`; bytecode frontend; `design/closure_specialization.md` for the caching critique |
| **Julia** | the specialization model being adopted: structural function types, compile-per-type-signature, `Val{}` value lifting, world-age invalidation | via `design/dsl_caching_layer.md` (GPUCompiler comparison included) |
| **cupy.jit** | takes a Python function and produces a high-performance CUDA kernel — directly adjacent to the CuPy backend we want | their frontend/lowering technique |
| **DaCe** — <https://github.com/spcl/dace> | Python-like code compiled to many backends; adjacent end use case, but a specific dataflow-centric analysis style that isn't necessarily ours | frontend technique (AST vs bytecode?); how backends plug in; what's separable from their dataflow model |
| **xDSL** — <https://github.com/xdslproject/xdsl> | a Python compiler-design toolkit (MLIR-family): dialects, rewrites, extensible IR — a candidate answer to the IR question | maturity; fit for a small embedded DSL; cost of adoption vs. own IR |
| **JAX** | the model for first-class transformations (`grad`, `vmap`) with clean composition | how transformations are exposed syntactically and staged |
| **MLX** | both a candidate backend (custom Metal kernels) and a transformations reference | custom-kernel API; their transformation set |
| **einops** | exemplar of a small, beloved mini-language worth hosting via t-strings | the notation, not the implementation |

## 9. Aesthetics

The sensibilities that should survive contact with any architecture:

- **Types, not values.** The cache-on-types thesis is the identity of the project.
  Any feature that would silently key on values (or silently reuse across
  type-relevant changes) is wrong by definition; value-dependent specialization
  must be an explicit opt-in (`Literal`/`Val`-style).
- **The loop stays hot.** Interactive tweaking is the product. Per-iteration work
  should approach the theoretical floor (a value write), and anything expensive
  must be cacheable and cached.
- **User code looks like Python.** A decorated function that reads naturally, not
  a builder API or an operator-overloading shadow language. Mini-languages, where
  they appear, should be small and self-evident (einops, not regex).
- **Incremental over complete.** A narrow language subset honestly documented
  beats a wide one half-working. The reference asset's docs discipline —
  implemented vs. planned called out inline, limitations flagged, not hidden — is
  the standard to keep.
- **Small and readable.** The reference asset is ~15 modules readable in a
  sitting. Growth is inevitable; opacity is not. Prefer designs a newcomer can
  trace end-to-end.
- **Clean seams.** Core / dialect / backend / runtime should be genuinely
  separable: the core must not know what WGSL is. Backends should be addable
  without touching the frontend; syntax without touching backends.
- **Honest types.** The type system tells the truth about Python values (64-bit
  ints, range bucketing); narrowing to backend widths is the backend's explicit
  decision. Correctness of `typeof` defines correctness of caching.
- **Transformations are not afterthoughts.** Autodiff and friends should feel like
  part of the language, which means the IR and the surface syntax must be designed
  with them in mind even if they ship later.

## 10. Not required on day one

Stated to keep scope honest — these are desires with a time dimension, not
omissions:

- the full NumPy-style battery set,
- units/dimensions,
- feature parity across all backends,
- autodiff/vmap shipping with the first redesign milestone (but the design must
  have a place for them).
