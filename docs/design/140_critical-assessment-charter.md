# 140 — Critical assessment charter: concepts, syntax, direction

**Status:** charter — this is a *prompt* for a future team-of-agents run, not
as-built documentation. Nothing in this document changes code. The run it
describes produces a report; humans decide what happens after.

**Why now.** Steps 10–14 landed features quickly: named axes, `over`, `jvp`/`D`,
matmul-by-name, argument arrays, the C grid family. Velocity is exactly the
condition under which mistaken concepts get institutionalized. Before the next
installments (tensor dialect, grad, CUDA/Metal/tiles), we stop and check the
foundations — not by re-reading our own tests, which encode our own
assumptions, but by writing the programs we ultimately want to exist and
seeing where today's concepts carry them and where they buckle.

---

## 1. The governing principle

> We do **not** convert arbitrary programs into GPU kernels. We support a
> **subset** of programs — those that can run *fast* on a GPU — and we refuse
> the rest, loudly and early.

The founding example: a transformation that naively broadcasts a kernel
across each pixel of an image, issuing a host-side GPU kernel call per pixel.
**This is an instance of something we should never have allowed in the first
place.** Not "discouraged," not "documented as slow" — the language's shapes
should make the naive form inexpressible (or confine it to an explicitly
debug-grade path), because the efficient form (one dispatch over a domain)
is the only form that belongs in the subset.

Corollaries the assessment must hold itself to:

- **The boundary is a design artifact.** Where the subset ends is as much a
  deliverable as what's inside it. A probe that concludes "this workload (or
  this part of it) belongs *outside* the subset, and here is the boundary and
  the refusal the user should see" has **succeeded**, not failed.
- **Refusal-first is load-bearing.** Every flaw claim of the form "X silently
  does the wrong thing" outranks "X is missing." Missing is fine; wrong is not.
- **Efficiency is the admission criterion.** A proposed primitive that cannot
  compile to efficient target code — even a beautiful one — does not enter the
  subset. (This was the standing test for named-axis indexing: maximally
  pedantic semantics were acceptable *because* the machine code stayed tight.)
- **Face the workhorse honestly.** Per-element host dispatch of scalar
  kernels exists today, is the test suite's cell-by-cell oracle, and is the
  baseline in the step-14 gate. The assessment must assign it an explicit
  status — first-class, debug-only, or removed — rather than leave it as an
  unexamined violation of the founding example.

## 2. The working hypothesis: a hierarchy of languages

This section states the conceptual model the project *believes* it is
building — including parts already shipped that may not be sound. The
assessment's central job is to **test this hierarchy**: map every existing
concept into it, treat every misfit as a finding, and affirm, amend, or
reject each level.

### Level 0 — device functions

The lowest level. Principles:

- Written in Python syntax, annotated, like everything else.
- Callable from other device functions and from host-visible kernels.
- **Universal by design:** one device-function syntax and semantics, usable
  unchanged inside *any* host-visible kernel type (compute, vertex,
  fragment, tiling). Deviations from universality need justification.
- A typing system and **struct definitions** shared across all kernel types
  travel with device functions.
- Device functions may *touch* arrays but are **not** whole-array
  processors: no `for` loops over an array's extent inside a device
  function. Stronger: there are probably very few places in the whole
  language where a user should ever write a raw loop over an array's size —
  extent iteration belongs to higher-level primitives (map/reduce/contract)
  that a scheduler can own.

### Level 1 — host-visible kernels

- Executable from the host; compiled on demand when a specialization is
  needed.
- They correspond to **concrete kernel types that exist in mainstream use**:
  vertex shaders, fragment shaders, and compute kernels (WebGPU, Metal,
  CUDA). We are not inventing kernel kinds; we are giving good syntax to
  real ones.
- Maximal common syntax across kernel types and backends, with
  backend-specific escapes where needed (the 090 punning charter).
- The full machinery is in scope at this level: shared memory, barriers,
  warp-level intrinsics.
- **Tiling kernels** are a distinct host-visible kind with Triton-like
  semantics: tile-granular operations, shared-memory staging, operations
  that map onto tensor cores, precision control, and element-wise epilogues.
  (Open within the hypothesis: whether tiling semantics also exist at the
  device-function level, or are host-visible only.)

### Level 2 — assemblage languages (existence uncertain)

Above host-visible kernels there *may* be a third level: domain-specific
languages for assembling kernel invocations. The motivating example: an ML
toolkit where the expression the user builds up **is the model**; one
invocation of the high-level object yields **multiple** GPU kernel
dispatches (one per network component, say). The claimed payoff: because of
how we treat closures, values, and types, **the parameters are part of the
program** — so the gradient targets are discoverable from the program
itself, not registered on the side.

The hypothesis explicitly flags this level as *not clearly necessary*. Probe
D (§5) is its existence test, and there are three candidate verdicts it must
decide between:

- **A real DSL** — level 2 earns its own syntax and semantics.
- **A very inert tensor library** — the strongest concrete sketch so far:
  simple indexing operations and maybe named axes; a set of framework
  functions carrying our annotation that perform reshapes, views, and other
  common tensor manipulations, maybe reductions — and **swappable
  backends**, of which one could be PyTorch and another could lower onto our
  own host-visible kernels. (Thinking-out-loud status: a shape to evaluate,
  not a commitment.)
- **Dissolution** — level 2 is ordinary Python orchestrating host-visible
  kernels plus the capture/pipeline mechanic, and we should say so plainly.

### Mapping obligations (where today's code must be placed)

The inventory (§3 Q1) must locate at least these, and record every misfit:

- A `@jit` scalar kernel is today **both** a device function (inlinable via
  call rules) and a host-visible kernel (dispatchable). Is that conflation a
  feature (roles assigned by context) or a flaw (host-visibility should be a
  declared property)?
- Kernel *kind* today is a property of the **dispatching registry**
  (`install_grid` turns the same kernel into a grid kernel), not of the
  kernel. The hierarchy suggests kind may belong on the kernel. Which is
  right?
- `over`, `jvp`, `D`, `matmul`, `Pipeline`: each transforms or composes
  things at *some* level — name the level(s), and note that `over` in
  particular converts a device-function-shaped thing into a host-visible-
  shaped thing (a level-crossing operator may be its true identity).
- The step-14 attention sample uses raw `for s in range(S)` over an axis
  extent — a direct violation of the level-0 loop principle, sitting in our
  flagship chapter. Judge it: is the principle right (and the sample must be
  rewritten with reduce/contract), or is the principle too strong?

## 3. The four questions

The assessment answers, in order:

1. **What does the code look like now?** A concept inventory: every
   user-facing concept (jit, capture, typeof, Named, isel, over, jvp, D,
   matmul, pipeline, grid, registry/dialect, refusals) with a one-paragraph
   statement of what it *claims* to be, verified against source and canon
   docs — with `file:line` citations, not memory — and its position (or
   misfit) in the §2 hierarchy.
2. **What primitives do we actually want to exist?** Derived from the probes
   (§5), not brainstormed in the abstract: a primitive earns a place on this
   list only by appearing in a probe program.
3. **What must be fixed?** Concepts that the probes reveal to be misnamed,
   mis-factored, at the wrong level, or wrong. Each fix proposal must name the
   right-level abstraction, not a band-aid (standing project rule).
4. **What does the syntax look like?** For every probe: both the **defining**
   side (kernel/shader/model source) and the **using** side (call sites,
   loops, host orchestration) — where "kernel" ranges over all three levels:
   device functions, host-visible kernels, and assemblage languages. The
   using side is historically where DSLs rot; give it equal weight.

## 4. Probe method — rules that keep the assessment honest

Each probe writes complete, aspirational programs in our DSL. Rules:

- **Three-way marking.** Every construct in every sample is tagged
  `[exists]` (works today — cite the test that proves it), `[planned]` (in
  020/130 — cite the step), or `[proposed]` (new — this is a finding). The
  tags are the assessment's raw data; without them, "flaw in what exists"
  and "feature not yet built" blur together.
- **Both sides of every kernel.** Definition and invocation. If the
  invocation is ugly, the definition doesn't get credit for being pretty.
- **A stance on named axes.** Each probe states explicitly where names exist,
  where they stop, and why — against the placement priors in §6.4.
- **Composition via capture + pipeline.** The load-bearing value proposition
  of this project is: *functions capture values, get passed around as
  arguments, and are inlined into compiled code*. The pipeline operator is the
  syntax that makes this pleasant. Probes should actively express their
  programs in this style — and report where it fails to fit rather than
  silently falling back to another idiom.
- **Exclusion is a valid verdict** (per §1). But it must come with the
  boundary drawn and the refusal message sketched.
- **Flaw claims are verified claims.** Every "this is broken/misconceived"
  finding is checked by an adversarial verifier agent against actual source
  before entering the report (PR #2 / design 120 methodology).

## 5. The probes

### Probe A — vertex + fragment shaders inside someone else's render loop

The founding thesis (see 010, 060, 090). Write the pair of shaders for a
non-trivial draw and the host code that uses them **inside an existing
render loop that we do not own** (WebGPU is the primary target).

Must cover, on the defining side:
- a vertex shader's two input kinds — **vertex arrays** and **uniforms** —
  are different types of objects; how does the syntax and type system keep
  them distinct, and how much is inferred from capture vs declared?
- **uniforms** can be essentially normal Python objects — living on CPU or
  GPU matters little, because they get copied to the device on every draw
  anyway. The per-draw copy *is* the contract; making it efficient is the
  runtime's job.
- **vertex arrays** should be things **allocated by the graphics backend**
  we are working with, not by us; assume a per-backend high-level
  convenience layer (third-party where possible, ours where necessary) for
  creating them, and show the seam where its objects enter our types. In
  general we hope to rely on third-party libraries for allocation.
- a vertex shader **closing over runtime values and GPU-allocated arrays**,
  specialized and compiled on demand — the project's core mechanic,
  exercised in shader-land.
- vertex → fragment interface: how varyings/interpolated values are expressed
  in the *syntax* of the two kernels — return values, named records, a
  shared signature?
- uniform sharing: the fragment shader may or may not share uniforms with
  the vertex shader; the syntax should make sharing visible and the runtime
  story should make uniform *copying* efficient.
- fragment shader with **multiple outputs** (render to multiple targets):
  what does MRT look like in the syntax?
- instanced drawing, and other machinery we may be forgetting (depth state,
  blending — enumerate what the syntax must *at least* not preclude).

Must cover, on the using side:
- what using the vertex shader **in a WebGPU draw command** looks like,
  concretely.
- the render loop belongs to foreign code: an imgui overlay (or equivalent)
  runs in the same loop and must keep working. We contribute *dispatches into*
  a frame; we never own the frame, the swap chain, or GPU global state.
- what our API hands the host per frame (a bind-and-draw callable? recorded
  commands?) and what the host hands us (time, viewport, its own buffers).

### Probe B — PDE by operator splitting: ping-ponged buffers across kernels

A compute workload with *multiple* kernels and *reused, mutated* buffers.
Write a small PDE solver (e.g. advection–diffusion) split into sub-step
kernels, ping-ponging state buffers A→B→A across a host time-stepping loop.

Must cover:
- who allocates and owns the state buffers; how a kernel declares "I read
  U and write U_next" (today the grid family has exactly one `out` and arrays
  are read-only — this probe *will* exceed what exists; the tags make that
  productive).
- the ping-pong swap in host syntax: is it two dispatch calls with swapped
  arguments, a recorded two-kernel sequence, a pipeline of kernels?
- named axes: does the grid domain have names (`x`, `y`) end-to-end, and do
  boundary conditions read naturally against them?
- what re-JITs and what is reused as buffers swap and steps iterate (the
  type-keyed cache should make the whole loop compile exactly once — verify
  the syntax makes that true, not just possible).

### Probe C — the tile language: tiled GEMM and friends

Design 130 §5 sketched this; the probe pressure-tests it as *user syntax*,
now under the §2 hypothesis that tiling kernels are a distinct host-visible
kind with Triton-like semantics. Write tiled GEMM with: shared-memory
staging tiles (padded sizes — the prime-number trick — to dodge bank
conflicts), barrier synchronization, a tensor-core-shaped inner contraction,
precision control (low-precision variants), sparsity, and an element-wise
epilogue. Also: a warp-level primitive example (reduction or sort step).

Must cover:
- how tile shapes enter the type system (are they `Literal`s? part of the
  kernel's specialization key?) and how the same source retargets when tile
  sizes change.
- whether tiling semantics are host-visible only or also make sense at the
  device-function level (the open question in §2), and whether tiles have
  named axes or deliberately do not (§6.4 prior: they do).
- invocation: what the launch looks like, and what is refused on hardware
  without the required features.
- the standing constraint from the fork discussion: **we do not lower too far
  and then un-lower** — the syntax must keep contraction/tile structure
  visible until target selection, so tensor cores are a *selection*, not a
  pattern-match rescue.

### Probe D — a transformer: model, training loop, optimizer, inference

The maximal stress test, and the **existence test for level 2** (§2): the
model is a built-up expression whose single invocation yields multiple
kernel dispatches, with parameters discoverable *from the program* because
they are captured values. Write, in our world end-to-end: model definition
(attention reusing matmul-by-name; layers as captured functions composed by
pipeline), parameter initialization, the loss, reverse-mode gradients,
gradient clipping, an optimizer (SGD and Adam — Adam's moment buffers are
*state*), the training step with a learning-rate schedule (a host-side
live-knob value feeding kernels every step — the case our cache polarity is
*designed* for), and inference (including the KV-cache question).

Must cover:
- **the state model** — the deepest open question in the project (see §6.1).
  Parameters that update every step, optimizer moments, KV caches: functional
  threading (JAX-style: step takes state, returns new state) vs mutable
  buffers (PyTorch-style). The probe must write the training loop both ways,
  or argue decisively for one.
- **the params-as-captures unification** (§6.5): if gradients can be taken
  with respect to *captured values*, then "loss is a kernel capturing the
  parameters" plus "grad w.r.t. captures" may be the entire training story.
  Evaluate this hypothesis explicitly — including what per-step parameter
  updates do to capture guards and cache identity.
- initialization and RNG: seeds are *values*; our cache keys *types*. How
  does randomness enter without poisoning the cache? (Prior art: JAX's
  explicit PRNG keys exist precisely because of this collision.)
- derivative-operator syntax at every level it appears (§6.5), including
  freezing parts of the computation, selecting which tensors get gradients,
  and which compositions (`grad`-of-`over`, `over`-of-`grad`) mean something
  and which refuse. Activation checkpointing and distributed training are in
  scope as *boundary questions* — scope them, don't design them.
- this probe is a **forcing function, not a commitment**: it is allowed to
  conclude that full training belongs outside the subset (or inside only in
  functional style) — but it must draw that boundary explicitly. Its verdict
  on level 2 must pick among the three candidates in §2 (real DSL, inert
  tensor library, dissolution) with the probe programs as evidence.

## 6. Cross-cutting questions

Answered once, informed by all four probes:

1. **State and mutation.** Everything today is immutable captures/args plus
   one `out`. Probes B and D both demand more. What is the *one* state story,
   at what level does it live (kernel? runtime? stdlib?), and what does it do
   to the type-keyed cache thesis? (Note: mutation is *compatible* with
   type-keyed caching — types don't change when values do — but guards,
   aliasing, and artifact reuse all need explicit answers.)
2. **Memory residency and ownership.** Where do arrays live (host, device,
   unified), who moves them, and what does the syntax show? Backend-allocated
   objects (Probe A's vertex arrays) are the sharpest case: objects we did
   not create entering our type system. (090's punning charter governs.)
3. **Pipelines: which levels, and value- vs dispatch-level.** Stated priors:
   the pipe operator is useful (a) in the assemblage language, (b) *inside*
   device functions, and (c) at host level to build a device function out of
   device functions. Today `|` composes functions *inside* one compiled
   artifact (inlining). Probes A and B additionally want to sequence
   *dispatches* (frames, sub-steps). Are these one concept — a pipeline that
   fuses when it can and records a command sequence when it can't — or two
   concepts that should not share syntax?
4. **Named axes: placement and genericity.** Stated priors: names clearly
   belong in the assemblage tier; they make sense in tiling kernels;
   thread/lane IDs addressable by axis name would be pleasant but is not
   critical. Two hard requirements: (a) **no dual universes** — we must
   never end up writing two versions of every function, one with names and
   one without; state precisely how a kernel written with names is used in a
   nameless context and vice versa. (b) **name-genericity** — axis names are
   domain-specific, so a function must never work only for one concrete
   name. (Promising direction to evaluate: names already live in *types* via
   `Named`, so a name-parameter is just one more specialization axis —
   name-generic functions specialize per name the way shape-generic ones
   specialize per rank.)
5. **The derivative-operator family.** Two kinds of derivative operation,
   living at different levels:

   **Backward (reverse mode):** the traditional gradient of a scalar loss
   with respect to tensors/parameters — assemblage-level. Requirements: it
   must be easy to **freeze** parts of the computation, and to compute
   gradients with respect to a *selected subset* of tensors only. Open
   questions the assessment should scope (boundary verdicts welcome): memory
   management of the backward pass, activation checkpointing, distributed
   training. Possibly also gradients "with respect to a specific kernel"
   (**[OPEN 2]** — meaning needs clarification: per-kernel custom adjoints?
   stopping gradients at a kernel boundary?).

   **Forward:** consumed *inside* kernels — `fwidth` is the canonical
   example. Today's in-kernel `D` differentiates with respect to *the
   function's arguments* — and it is not clear that is the right design.
   Candidate replacement: `D(foo, wrt(x))`, where `x` may be an argument of
   the surrounding function **or any other value in scope** (including
   captures); the analysis finds `x` and substitutes a dual number from that
   point forward. `x` should be allowed to be a *structure*, not just a
   scalar (layout ×N per 110). A second forward-mode consumer: constructing
   optimization algorithms (e.g. convex formulations) where the operator
   wanted is not the derivative but a **subgradient** — note that today's
   jvp rules already return *a* subgradient at kinks (abs/min/max pick a
   side); the assessment should decide whether that is an accident or a
   commitment.

   The assessment maps the whole family — which operator lives at which
   level, with what syntax, and how selection (`wrt`, freezing) reads.
6. **Where do tensors come in?** We have deliberately *not* built a
   tensor-focused library. The no-extent-loops principle (§2 level 0) forces
   the question: extent iteration must live in map/reduce/contract
   primitives generic enough to serve many domains, not just ML. State how
   the tensor concept enters the framework without the framework becoming a
   tensor library — the "inert tensor library" candidate for level 2 (§2) is
   one possible answer — and audit existing samples (the step-14 attention
   chapter) against the principle.
7. **Caching identity at the edges.** GPU-resident buffers, mutated state,
   backend-allocated objects, recorded command sequences, per-step parameter
   updates: restate what the specialization key and the guards are in each
   new situation the probes introduce.
8. **Refusal UX.** Collect the refusal messages the probes sketch. Do they
   read as one voice? (The refusal-message contract tests froze today's; the
   probes preview tomorrow's.)

## 7. Required reading for the team

- Canon: 010 (architecture + ledger), 020 (plan), 090 (core/extensions,
  punning), 100 (arrays/axes), 110 (transforms), 120 (events/instrumentation
  — also the *methodology* template), 130 (tensors/tiles/over).
- Source: `src/pdum/dsl/kernel/` (all), `stdlib/{arrays,transforms,base_lang}.py`,
  `combinators.py`, `backends/c.py`, `demo/simple_shader/`.
- Tests as behavior spec: `test_refusal_contract.py`, `test_grid.py`,
  `test_array_args.py`, `test_jvp_rules.py`, `test_traced_dispatch.py`.
- The book chapters (via `scripts/book/build_chapters.py`) for the *taught*
  model, which is the de facto UX contract.
- `pdum.dsl_reference` is read-only context, as always. **This run modifies
  no code anywhere.**

## 8. Deliverables

One report (numbered doc, next free slot at run time) containing:

1. **Concept inventory** (§3 Q1) with citations and hierarchy placement.
2. **Hierarchy verdict**: each level of §2 affirmed, amended, or rejected,
   with the mapping misfits that drove the verdict.
3. **Syntax portfolio**: the probe programs, fully tagged.
4. **Flaw list**: verified findings only, ranked; each names the right-level
   fix or explicitly states "boundary — exclude, refuse with message X."
5. **Direction memo**: what the probes imply for the ordering and content of
   the next installments (tensor dialect, grad, CUDA/Metal/tiles) — including
   anything that should be *removed or renamed* before it calcifies.

## 9. Suggested run shape

Inventory pass (parallel readers over kernel/stdlib/backends/canon) →
four probe agents in parallel, each producing tagged programs + candidate
findings → cross-cutting synthesizer (§6) + hierarchy judge (§2) →
adversarial verification of every flaw claim against source → report
assembly. Findings that fail verification are dropped, not softened.

## 10. Open items to resolve before launching the run

- **[OPEN 2]** "Computing gradients with respect to a specific kernel"
  (§6.5): clarify intent — per-kernel custom adjoint rules (à la
  `jax.custom_vjp`), gradient stopping at kernel boundaries, or something
  else.
- **[OPEN 3]** Probe A treats WebGPU as the primary target (draw commands
  and the compute list — WebGPU/Metal/CUDA — both say so); the spoken prompt
  twice said "WebGL render loop." Confirm WebGPU-first is right and WebGL
  was shorthand.
- **[OPEN 5]** "Gradient clashing" transcribed → interpreted as gradient
  *clipping* in Probe D. Confirm.
