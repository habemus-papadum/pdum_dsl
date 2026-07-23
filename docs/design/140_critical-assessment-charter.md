# 140 — Critical assessment charter: concepts, syntax, direction

**Status:** charter — this is a *prompt* for a team-of-agents run, not
as-built documentation. Nothing in this document changes code. The run it
describes produces a report; humans decide what happens after.

**Why now.** Steps 10–14 landed features quickly: named axes, `over`, `jvp`/`D`,
matmul-by-name, argument arrays, the C grid family. Velocity is exactly the
condition under which mistaken concepts get institutionalized. Before the next
installments, we stop and check the foundations — not by re-reading our own
tests, which encode our own assumptions, but by writing the programs we
ultimately want to exist and seeing where today's concepts carry them and
where they buckle.

Since the first draft of this charter, two things changed the terrain. The
parallel **tensorlib** stream landed (`explorations/tensorlib/`): a white-box
modeling lab covering denotation (L0) through placement (L3-lite), with
kernels (L4) queued — a working prototype of much of what §2's "assemblage"
level speculated about. And NVIDIA's **CuTe DSL** was studied as the closest
mainstream analogue of a Python-syntax kernel-language hierarchy. The
assessment therefore targets the **joint system**: pdum.dsl's syntax,
caching, and kernel machinery *fronting* the tensorlib representation — not
either half in isolation.

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

## 2. The working hypothesis: two axes, one synthesis

This section states the conceptual model the project *believes* it is
building — including parts already shipped that may not be sound. The
assessment's central job is to **test it**: map every existing concept into
it, treat every misfit as a finding, and affirm, amend, or reject each part.
The hypothesis now has two distinct axes, and a synthesis claim about how
they compose.

### 2.1 Axis one — languages and calling conventions

Who writes what syntax, and what can call what.

**Level 0 — device functions.**

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

**Level 1 — host-visible kernels.**

- Executable from the host; compiled on demand when a specialization is
  needed.
- They correspond to **concrete kernel types that exist in mainstream use**:
  vertex shaders, fragment shaders, and compute kernels (WebGPU, Metal,
  CUDA). We are not inventing kernel kinds; we are giving good syntax to
  real ones. The three compute targets are **equal citizens**: CUDA, Metal,
  and WebGPU carry equal weight, and no probe or design may treat one of
  them as the neutral default.
- Maximal common syntax across kernel types and backends, with
  backend-specific escapes where needed (the 090 punning charter).
- The full machinery is in scope at this level: shared memory, barriers,
  warp-level intrinsics.
- **Tiling kernels** are a distinct host-visible kind with Triton-like
  semantics: tile-granular operations, shared-memory staging, operations
  that map onto tensor cores, precision control, and element-wise epilogues.
  (Whether "distinct kind" survives the §2.3 synthesis is itself under
  assessment.)

**Level 2 — the assemblage layer.**

The first draft asked whether this level exists at all, with three candidate
verdicts (a real DSL / a very inert tensor library / dissolution into host
Python). The tensorlib exploration has since **largely answered the
existence question by construction**: a working library of layout algebra,
compute primitives, markers, IR, and derived autodiff exists, with the
"inert" property in its surface (programs are data; `f`s are declared
markers, not callbacks; frontends are explicitly pluggable). What remains
genuinely open — and is now Probe D's charge — is the **surface**: how our
syntax (capture, closures-over-weights, pipelines, named axes) fronts that
library, whether the model-as-expression / one-invocation-many-dispatches
claim holds up in real syntax, and whether any part of the prototype's shape
should be rejected (governance in §8). The three-verdict framing is retired;
the dissolution verdict remains available only for the *syntax layer* ("no
new surface needed beyond ordinary Python + capture + pipeline"), not for
the library itself.

### 2.2 Axis two — the representation ladder

What the program *is* between authoring and machine code. Tensorlib's
LEVELS.md states the stance, and the assessment inherits it as hypothesis:

- **One IR, one denotation.** The ladder (L0 denotation → L1 footprint →
  L2 storage → L3 placement → L4 kernels → L5 schedule → L6 microkernel) is
  "a stack of well-formedness predicates and erasures, **not** a stack of
  dialects." Each level adds a cost semantics (ops, bytes, traffic, time)
  laid over the same program; none changes what the program means.
- **The machine description is data, not representation.** A machine is a
  tree (cluster → node → GPU → SM → warp → lane); the IR binds dims to
  *level names*; swapping the tree retargets the model.
- **Distribution and tiling are the same move**: split a semantic dim, bind
  one part to a machine level, place the buffer in that level's memory — at
  different tree depths. Collectives are not ops; they are read off the
  algebra by a cost pass. A warp shuffle and a cross-node all-gather are the
  same alignment repair on different links.
- **Erasure invariants everywhere.** Charts, units, placement — metadata
  that can be forgotten leaving the denotation intact, making each layer's
  correctness a one-line theorem.

### 2.3 The synthesis to test

**Claim: the two axes compose — a small number of surface languages emit
programs into one laddered IR.** If true, the consequences are large:

- **"The tiling language" may not be a separate language.** Much of what
  axis one calls a distinct tiling kernel kind is the *same* language with
  dims bound to deeper machine levels. The real syntax question becomes:
  what surface *drives* the split/bind/schedule decisions? (§3 proposes an
  answer; Probe C tests it.)
- **Calling conventions decouple from representation.** CuTe DSL is the
  evidence: its whole hierarchy is a small calling-convention matrix —
  Python→`@jit` runs, Python→`@kernel` errors, `@jit`→`@jit` inlines,
  `@jit`→`@kernel` launches, `@kernel`→`@jit` inlines, `@kernel`→`@kernel`
  errors — over one underlying layout algebra. Three lessons: (1) CuTe uses
  *one* decorator for both device-function and host-orchestration roles —
  context assigns the role, and only the **launch boundary** gets an
  explicit marker; that is direct evidence on our "is `@jit`'s dual role a
  feature or a flaw" obligation. (2) Static vs dynamic is distinguished in
  the *type system* with per-invocation specialization — our caching
  polarity, independently arrived at. (3) CuTe is CUDA-only and encodes
  CUDA's model as neutral — the equal-citizens rule (§2.1) is where we
  deliberately part ways.
- The assessment must produce **our calling-convention matrix** (§10): for
  every level of axis one, who may call whom, with the execution semantics
  of each cell (inline / launch / refuse).

### 2.4 Mapping obligations (where today's code must be placed)

The inventory (§4 Q1) must locate at least these, and record every misfit:

- A `@jit` scalar kernel is today **both** a device function (inlinable via
  call rules) and a host-visible kernel (dispatchable). Is that conflation a
  feature (roles assigned by context, as CuTe does) or a flaw
  (host-visibility should be a declared property)? Note CuTe's refinement:
  the *launch boundary* is explicitly marked even though roles are
  contextual.
- Kernel *kind* today is a property of the **dispatching registry**
  (`install_grid` turns the same kernel into a grid kernel), not of the
  kernel. The hierarchy suggests kind may belong on the kernel. Which is
  right?
- **`over`, in ladder terms, is a split+bind**: it binds an axis to the
  launch domain — the lane is a machine-bound coordinate. Is `over` a
  standalone transform, or an early syntax-level shadow of the general
  binding mechanism (§2.2), to be reabsorbed by it? Its 16× gate and the
  weaving mechanics survive either verdict; the *operator* may not.
- `jvp`, `D`, `matmul`, `Pipeline`: each transforms or composes things at
  *some* level — name the level(s).
- **Our JVP rule table vs "derive, don't enumerate."** Tensorlib's
  PHILOSOPHY holds that hand-maintained rule tables are where semantic rot
  begins; its markers derive partials by tree rewriting, its folds derive
  adjoints by self-application. Our `JVP_RULES` is a hand table. A fixed
  table over ~20 scalar primitives may be defensible precisely because
  composites differentiate *through* it rather than joining it — but the
  collision must be adjudicated, not left to coexist by silence.
- The step-14 attention sample uses raw `for s in range(S)` over an axis
  extent — a direct violation of the level-0 loop principle, sitting in our
  flagship chapter. Judge it: is the principle right (and the sample must be
  rewritten with reduce/contract), or is the principle too strong?

## 3. The candidate architecture: progressive lowering through authored DSLs

This section names a concrete architecture hypothesis for how programs
descend the ladder. It enters the assessment as a **defendant, not a
premise** — its known weak points are pre-registered below, and the
assessment's job is to attack it.

### 3.1 The pipeline

1. **Author at L0.** The program is written in tensorlib primitives
   (pointwise/reduce/scan/fold, layout ops, markers), orchestrated by *our*
   syntax: closures capture weights and configuration; pipelines compose
   stages. The L0 program is about correctness; its denotation is the
   contract everything below must match.
2. **Semantic enrichment.** Passes over the IR, in pure value semantics:
   AD (backward generation), requested-gradients DCE, activation
   checkpointing, placement. Each is measure→transform→re-measure against a
   cost simulator.
3. **Partition.** An analysis (algorithm-assisted, human-decided at first)
   identifies chunks of the IR to become single kernels — e.g. the several
   high-level ops that constitute a matmul + streaming-softmax attention
   block.
4. **Author the descent.** Each chunk is *rewritten by a human (or agent)
   one level down*, in a level-appropriate Python DSL — the tiling DSL for
   L4, and recursively again for deeper levels (tiles → warps/intrinsics).
   Every level's language is a Python DSL through our one frontend machine
   (capture, typeof, specialization); descent is always
   syntax → IR → analysis → partition → syntax.
5. **Certify every descent** (see 3.2).

### 3.2 Amendment one — rewrite chains, not monolithic proofs

The naive form of the proof obligation — author the tiled kernel freely,
then prove it equivalent to the high-level chunk — is rejected in advance.
Whole-program equivalence of two independently authored programs is
undecidable in general and brutal in practice, and tensorlib's own assurance
tiers already forswear it ("no monolithic 'prove these two programs equal'
obligation ever exists"). The obligation instead:

- The lowering DSL's **elaboration produces a chain of named, certified
  rewrites** (split, bind-to-level, reorder-under-declared-license,
  fuse-as-elision-of-materialization, pad-to-tile-with-guards, ...) — or the
  checker reconstructs such a chain as a witness. The author's *experience*
  is still "write the tiled kernel in Python syntax"; the *semantics* of
  that syntax is a rewrite chain.
- **Equivalence is license-relative**: bit-exact where no license is used;
  equal-modulo-declared-associativity where reassociation was claimed. (A
  wrong schedule wastes resources; it cannot change meaning — strategy and
  correctness factor.)
- The cheap tier stays always-on: numeric spot-checks against the L0
  denotation. Lean certification of the rewrite rules themselves is the
  destination, per tensorlib's plan, not a launch blocker.
- Precedents: Exo's user-schedulable languages (lowering as applied
  semantics-preserving rewrites), translation validation (works only on
  small deltas — which is the argument *for* chains), tensorlib LEVELS
  assurance tiers 1–3.

### 3.3 Amendment two — AD and partitioning do not commute

The pipeline order "enrich, then partition" is correct for memory passes and
**wrong by default for AD**: flash attention's backward is not the backward
of the unfused forward — fusing the forward changes the saved-tensor set
(recompute from the logsumexp instead of storing scores). Fusions that
change the adjoint must be *visible to AD*. The existing partial answer is
tensorlib's declared-combine mechanism (`defreducer`: the online-softmax
combine declared at L0, its backward **derived**, provably equal to the
naive form). The open question is first-class, not a pipeline detail:
**where does AD run, and what must be declared at L0 for fusion-aware
backwards to derive?**

### 3.4 The payoff if it survives: a library of certified lowerings

A certified descent is a pair: (content-addressed fingerprint of the IR
chunk → verified lower-level implementation). That is exactly the shape of
our artifact cache — so a **registry of certified lowerings** falls out:
every model whose attention chunk normalizes to the same fingerprint hits
the same verified kernel, across models and users. This extends the
type-keyed caching thesis from "specialize on types" to "reuse verified
lowerings by structure," and it dovetails with tensorlib's content-addressed
marker registries (CONCERNS #22), which already adopted the
build-in-a-loop philosophy. It is also the honest agent-era argument for
human-driven compilation: agents author lowerings cheaply; certification is
what makes an agent-authored compiler trustworthy.

## 4. The four questions

The assessment answers, in order:

1. **What does the code look like now?** A concept inventory: every
   user-facing concept (jit, capture, typeof, Named, isel, over, jvp, D,
   matmul, pipeline, grid, registry/dialect, refusals — and tensorlib's
   Layout, Tensor, markers, Program, grad, fold, placement) with a
   one-paragraph statement of what it *claims* to be, verified against
   source and canon docs — with `file:line` citations, not memory — and its
   position (or misfit) on **both axes** of §2.
2. **What primitives do we actually want to exist?** Derived from the probes
   (§6), not brainstormed in the abstract: a primitive earns a place on this
   list only by appearing in a probe program.
3. **What must be fixed?** Concepts that the probes reveal to be misnamed,
   mis-factored, at the wrong level, or wrong. Each fix proposal must name the
   right-level abstraction, not a band-aid (standing project rule).
4. **What does the syntax look like?** For every probe: both the **defining**
   side (kernel/shader/model source) and the **using** side (call sites,
   loops, host orchestration) — where "kernel" ranges over all levels of
   axis one. The using side is historically where DSLs rot; give it equal
   weight.

## 5. Probe method — rules that keep the assessment honest

Each probe writes complete, aspirational programs. Rules:

- **The joint system is the target.** Where a probe touches tensor
  computation, it writes *our* syntax fronting the *tensorlib*
  representation (the Node/Program schema is explicitly designed for
  pluggable frontends — README: "the main repo's syntax tooling can target
  it later without any rewrite"). Assessing pdum.dsl in isolation from the
  layer it will target answers a question we no longer face.
- **Named hypotheses are defendants.** §2's hierarchy and §3's architecture
  are attacked, not assumed. A probe that breaks one of them has done its
  job.
- **Three-way marking.** Every construct in every sample is tagged
  `[exists]` (works today — cite the test that proves it; tensorlib counts),
  `[planned]` (in 020/130/LEVELS — cite the step), or `[proposed]` (new —
  this is a finding). The tags are the assessment's raw data; without them,
  "flaw in what exists" and "feature not yet built" blur together.
- **Both sides of every kernel.** Definition and invocation. If the
  invocation is ugly, the definition doesn't get credit for being pretty.
- **A stance on named axes.** Each probe states explicitly where names exist,
  where they stop, and why — against the placement priors in §7.4.
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

## 6. The probes

### Probe A — vertex + fragment shaders inside someone else's render loop

The founding thesis (see 010, 060, 090). Write the pair of shaders for a
non-trivial draw and the host code that uses them **inside an existing
render loop that we do not own** (the render target is WebGPU).

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
(Tensorlib's zoo already holds the L0 side: heat2d with guards as boundary
conditions, charted staggered FDTD with derived gradients — build on it.)

Must cover:
- who allocates and owns the state buffers; how a kernel declares "I read
  U and write U_next" (today the grid family has exactly one `out` and arrays
  are read-only — this probe *will* exceed what exists; the tags make that
  productive).
- **the incumbent state stance under test**: tensorlib holds that mutation
  is a *storage-level* phenomenon — programs stay pure value semantics, and
  L2 bufferization later assigns ping-ponged values to alternating buffers
  ("transform where it is safe; commit late"). The probe tests whether the
  *syntax* can stay pure while the buffers alternate underneath, or whether
  the user must see the swap.
- the ping-pong swap in host syntax: is it two dispatch calls with swapped
  arguments, a recorded two-kernel sequence, a pipeline of kernels?
- named axes: does the grid domain have names (`x`, `y`) end-to-end, and do
  boundary conditions read naturally against them?
- what re-JITs and what is reused as buffers swap and steps iterate (the
  type-keyed cache should make the whole loop compile exactly once — verify
  the syntax makes that true, not just possible).

### Probe C — the descent: tiled GEMM and fused attention through §3

Design 130 §5 sketched the tile language; §3 now gives it a concrete shape,
and this probe **operationalizes the candidate architecture**: take the
attention chunk (matmul + streaming softmax) and tiled GEMM, author the
tiled rewrite in the tiling DSL, and sketch the rewrite chain and its
witness (§3.2). Include: shared-memory staging tiles (padded sizes — the
prime-number trick — to dodge bank conflicts), barrier synchronization, a
tensor-core-shaped inner contraction, precision control, sparsity, an
element-wise epilogue, and a warp-level primitive example (reduction or
sort step).

Must cover:
- how tile shapes enter the type system (are they `Literal`s? part of the
  kernel's specialization key?) and how the same source retargets when tile
  sizes change.
- **tensorlib's queued L4 questions are inputs**: LEVELS.md K-A (kernel =
  annotation vs region op), K-B (tiling as split+bind one tier down — does
  anything *new* appear at L4?), K-C (legality ≈ convex instruction sets;
  objective = parent-memory traffic under child capacity), K-D (flagships),
  K-E (cost plumbing), K-F (L2 ordering). Answer them from the syntax side;
  the report is the design brief that reopens L4 (§8).
- whether `over`/binding appears *inside* kernels, and whether tiles have
  named axes or deliberately do not (§7.4 prior: they do — machine-bound
  dims carry axis identity but no charts, per LEVELS surface discipline).
- invocation: what the launch looks like, and what is refused on hardware
  without the required features.
- backend parity (§2.1): the tensor-core-shaped contraction examined against
  all three compute targets — CUDA tensor cores, Metal simdgroup matrices,
  WebGPU's subgroup-level limits — so the syntax does not quietly encode
  CUDA's model as the neutral one.
- the standing constraint: **we do not lower too far and then un-lower** —
  independently converged on by both streams (130's "named contract/reduce/
  map are the working representation until target selection"; LEVELS'
  "lowering must PRESERVE reduce/scan structure on machine-bound dims so
  the backend pattern-matches instead of reverse-engineers"). The rewrite
  chain must keep contraction/tile structure visible until target selection,
  so tensor cores are a *selection*, not a pattern-match rescue.

### Probe D — a transformer: model, training loop, optimizer, inference

The maximal stress test, now explicitly a **joint-system** probe: the model
is written in *our* syntax (attention via matmul-by-name; layers as captured
functions composed by pipeline) emitting the *tensorlib* representation, and
one invocation of the high-level object yields multiple kernel dispatches.
Write, end to end: model definition, parameter initialization, the loss,
reverse-mode gradients, gradient clipping, an optimizer (SGD and Adam —
Adam's moment buffers are *state*), the training step with a learning-rate
schedule (a host-side live-knob value feeding kernels every step — the case
our cache polarity is *designed* for), and inference (including the KV-cache
question).

Must cover:
- **the state model** (§7.1): parameters that update every step, optimizer
  moments, KV caches — functional threading vs mutable buffers vs
  tensorlib's storage-level answer. The probe must write the training loop
  in the candidate styles, or argue decisively for one.
- **the params-as-captures unification, now concrete**: captured tensors
  become `input` leaves of the emitted Program (deterministic capture→leaf
  naming — the binder-seam move); `wrt`/freezing maps onto tensorlib's
  requested-gradients DCE keep-set ("freezing a layer is just a smaller
  keep-set — reachability, no special case"). Test this mapping explicitly —
  including what per-step parameter updates do to capture guards and cache
  identity.
- initialization and RNG: seeds are *values*; our cache keys *types*. How
  does randomness enter without poisoning the cache? (Prior art: JAX's
  explicit PRNG keys exist precisely because of this collision.)
- derivative-operator syntax at every level it appears (§7.5), including
  freezing, selection, and which compositions (`grad`-of-`over`,
  `over`-of-`grad`) mean something and which refuse. Activation
  checkpointing and distributed training are in scope as *boundary
  questions* — tensorlib's DCE/checkpointing/revolve and placed-backward
  machinery is the evidence base; scope the syntax, don't redesign the
  passes.
- **KV-cache must be pressed, not inherited as excluded.** Both streams
  have so far excluded mutation/KV-cache decode (tensorlib's recorded
  boundaries). Two independent deferrals do not add up to a decision;
  inference serving is a first-class workload. The probe either gives it a
  syntax + storage story or draws the exclusion boundary *explicitly*.
- this probe is a **forcing function, not a commitment**: it is allowed to
  conclude that full training belongs outside the subset (or inside only in
  functional style) — but it must draw that boundary explicitly. Its
  verdict on the assemblage *surface* (per §2.1: real new syntax vs
  ordinary Python + capture + pipeline over the library) must cite the probe
  programs as evidence.

## 7. Cross-cutting questions

Answered once, informed by all four probes:

1. **State and mutation.** Everything in pdum.dsl today is immutable
   captures/args plus one `out`; tensorlib deliberately excludes mutation
   and defers storage to L2 bufferization ("commit late"). That stance is
   the incumbent: mutation as a storage-level phenomenon, never a language
   feature. Probes B and D test it rather than reinvent it — and the
   assessment must either endorse the KV-cache/dynamic-shape exclusions
   with an explicit boundary, or propose the L2-level answer.
2. **Memory residency and ownership.** Where do arrays live (host, device,
   unified), who moves them, and what does the syntax show? Backend-allocated
   objects (Probe A's vertex arrays) are the sharpest case: objects we did
   not create entering our type system. (090's punning charter governs.)
3. **Pipelines: which levels, and value- vs dispatch-level.** Stated priors:
   the pipe operator is useful (a) in the assemblage layer, (b) *inside*
   device functions, and (c) at host level to build a device function out of
   device functions. Today `|` composes functions *inside* one compiled
   artifact (inlining). Probes A and B additionally want to sequence
   *dispatches* (frames, sub-steps). Are these one concept — a pipeline that
   fuses when it can and records a command sequence when it can't — or two
   concepts that should not share syntax?
4. **Named axes: placement and genericity.** Stated priors: names clearly
   belong in the assemblage tier; they make sense in tiling kernels
   (machine-bound dims keep their axis *identity* while dropping charts —
   LEVELS' surface discipline); thread/lane IDs addressable by axis name
   would be pleasant but is not critical. Two hard requirements: (a) **no
   dual universes** — never two versions of every function, with and
   without names; state precisely how a kernel written with names is used in
   a nameless context and vice versa. (b) **name-genericity** — axis names
   are domain-specific, so a function must never work only for one concrete
   name. (Promising direction: names already live in *types* via `Named`,
   so a name-parameter is one more specialization axis — name-generic
   functions specialize per name the way shape-generic ones specialize per
   rank. Tensorlib's "names first, order never" is the same conviction from
   the other side.)
5. **The derivative-operator family.** Two kinds of derivative operation,
   living at different levels:

   **Backward (reverse mode):** the traditional gradient of a scalar loss
   with respect to tensors/parameters — assemblage-level, and now concretely
   prototyped (tensorlib `grad`: reverse-mode as program transformation,
   adjoints validated by finite differences). Requirements: easy
   **freezing** and gradient-subset selection (→ DCE keep-sets, per Probe
   D). Memory management, activation checkpointing, distributed training:
   scoped as boundary questions with tensorlib's L1/L3 machinery as the
   evidence base.

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

   **Two lenses, orthogonal to the two kinds.** Differentiation can be
   asked for in a *value-centric* way — start from a computed value and ask
   for its gradient with respect to other input values (the PyTorch lens) —
   or in a *function-centric* way — take a function, in **any of the
   languages of the hierarchy**, and ask for its **VJP** or **JVP** with
   respect to some of its arguments or parameters, getting back a new
   function to specify and then to call (the functional lens). Both lenses
   may be wanted at every level, in forward or backward mode. The prompt is
   deliberately imprecise here and the assessment owes the precision: for
   each level, which lens(es) exist, what the syntax is for *specifying*
   the differential computation, and what the syntax is for *using* the
   resulting function or value.

   The assessment maps the whole family — kind × lens × level — with
   concrete syntax, including how selection (`wrt`, freezing) reads, and
   adjudicates the rule-table-vs-derived collision (§2.4).
6. **Where do tensors come in?** We deliberately did *not* build a
   tensor-focused library into the kernel — and the ecosystem answered by
   building tensorlib beside it. The no-extent-loops principle (§2.1 level
   0) forces the question: extent iteration must live in map/reduce/contract
   primitives generic enough to serve many domains, not just ML — which is
   what tensorlib's primitives are. State how the tensor concept enters the
   *joint* framework without the kernel becoming a tensor library — and
   audit existing samples (the step-14 attention chapter) against the
   principle.
7. **Caching identity at the edges.** GPU-resident buffers, mutated state,
   backend-allocated objects, recorded command sequences, per-step parameter
   updates: restate what the specialization key and the guards are in each
   new situation the probes introduce. **The layout question now has a
   sketch to evaluate**: the specialization fingerprint takes the layout's
   *structural skeleton* — rank, dim names/axis tags, repeat/flip structure
   (which strides are zero/negative), guard presence and form, chart
   presence, machine bindings — computed on tensorlib's `canonical()` form
   so presentation order never splits the cache; *numeric* content (extents,
   concrete strides, offsets, chart origins) stages as runtime values (the
   `array.dim` pattern); codegen-relevant numeric facts (divisibility,
   alignment, contiguity class) enter as **opt-in predicates**,
   Literal-style. Evaluate, don't assume.
8. **Refusal UX.** Collect the refusal messages the probes sketch. Do they
   read as one voice — including tensorlib's (D17: refusals that quote the
   fix)? The refusal-message contract tests froze today's; the probes
   preview tomorrow's.

## 8. Tensorlib: standing, convergences, and governance

**What it is.** A white-box modeling lab (`explorations/tensorlib/`):
exact layout algebra (affine map + box + guards + charts + units +
placement), compute primitives with declared markers, a linear-SSA IR with
one structured combinator (`fold`), reverse-mode AD as program
transformation with derived adjoints, cost semantics per level (ops, peak
memory, traffic), a model zoo pinned to numpy denotations, and L1
optimizers (DCE, min-cut checkpointing, revolve) plus L3-lite placement.
Its reference layer is deliberately slow: it is the **denotational oracle**
the compiled world must match — which is its long-term role in the joint
system (differential testing target, spec layer), not a quarry to strip.

**Independent convergences** (two streams that could not see each other
arriving at the same principle — treat these as likely load-bearing):
- D17 "diagnosis, never surgery" ≈ our refusal-first contract.
- "Preserve reduce/scan structure on machine-bound dims" ≈ 130's
  "named forms are the working representation until target selection" ≈
  the no-unlowering constraint.
- Content-addressed marker registries (CONCERNS #22, adopted "the main
  repo's build-in-a-loop philosophy") ≈ the type-keyed/content-addressed
  caching thesis.
- "Names first, order never" ≈ named axes with no permute.

**Governance.** Tensorlib's decided positions (DESIGN D1–D18, PHILOSOPHY,
LEVELS) are **evidence, not canon** — exactly like our own 010–130, which
this assessment also holds to the fire. Where an assessment verdict
collides with a decided position *in either stream*, the collision is a
**finding escalated to the human**, never silently resolved in either
direction.

**Sequencing.** The tensorlib stream is paused at the L3-lite → L4
boundary (its own order-of-attack reached "L4: manual fusion + tiling with
the rewrite checker"). This assessment runs now; its report — Probe C's
answers to K-A…K-F in particular — is the **design brief that reopens L4**,
with both the syntax-down and IR-up perspectives merged. L2 bufferization
stays deferred per tensorlib's own reasoning (needed for exact reuse, not
for placement or kernel reasoning).

## 9. Required reading for the team

- Canon: 010 (architecture + ledger), 020 (plan), 090 (core/extensions,
  punning), 100 (arrays/axes), 110 (transforms), 120 (events/instrumentation
  — also the *methodology* template), 130 (tensors/tiles/over).
- **Tensorlib**: README, LEVELS, PHILOSOPHY, DESIGN (D1–D18), CONCERNS,
  COMPUTE, REPRESENTATIONS, PLACEMENT, PROVENANCE; notebooks 05–13 for the
  autodiff/memory/placement arcs; `ir.py`, `mdsl.py`, `autodiff.py`,
  `layout.py`, `transforms.py`, `placement.py`, `zoo/`.
- **CuTe DSL**: the DSL introduction / calling-conventions page
  (docs.nvidia.com/cutlass — pythonDSL/cute_dsl_general/dsl_introduction),
  for the matrix in §2.3.
- Source: `src/pdum/dsl/kernel/` (all), `stdlib/{arrays,transforms,base_lang}.py`,
  `combinators.py`, `backends/c.py`, `demo/simple_shader/`.
- Tests as behavior spec: `test_refusal_contract.py`, `test_grid.py`,
  `test_array_args.py`, `test_jvp_rules.py`, `test_traced_dispatch.py`;
  tensorlib's `tests/` as its behavior spec.
- The book chapters (via `scripts/book/build_chapters.py`) for the *taught*
  model, which is the de facto UX contract.
- `pdum.dsl_reference` is read-only context, as always. **This run modifies
  no code anywhere.**

## 10. Deliverables

One report (numbered doc, next free slot at run time) containing:

1. **Concept inventory** (§4 Q1) with citations and placement on both axes.
2. **Hierarchy verdict**: each part of §2 affirmed, amended, or rejected,
   with the mapping misfits that drove the verdict — including the fate of
   `over` and the rule-table adjudication.
3. **The calling-convention matrix** (§2.3): per level, who may call whom,
   with execution semantics per cell (inline / launch / refuse).
4. **Architecture verdict** (§3): the progressive-lowering candidate
   sustained, amended, or rejected — with Probe C's rewrite-chain exercise
   as primary evidence, and explicit rulings on the two pre-registered weak
   points (proof-obligation shape; AD × partition ordering).
5. **Syntax portfolio**: the probe programs, fully tagged.
6. **Flaw list**: verified findings only, ranked; each names the right-level
   fix or explicitly states "boundary — exclude, refuse with message X."
   Collisions with decided positions (either stream) flagged for human
   arbitration per §8.
7. **Direction memo**: what the probes imply for the ordering and content of
   the next installments — the L4 design brief (K-A…K-F answered), the
   tensorlib promotion question (when/whether it leaves `explorations/`),
   the frontend→Node-schema integration plan, and anything that should be
   *removed or renamed* before it calcifies.

## 11. Suggested run shape

Inventory pass (parallel readers over kernel/stdlib/backends/canon AND
tensorlib) → four probe agents in parallel, each producing tagged programs +
candidate findings → cross-cutting synthesizer (§7) + hierarchy judge (§2) +
architecture red team (§3) → adversarial verification of every flaw claim
against source → report assembly. Findings that fail verification are
dropped, not softened.

## 12. Resolution log

Open items from earlier drafts, resolved in review:

- **OPEN 1** (truncated founding example) — restored: naive per-pixel host
  dispatch of a kernel across an image; now anchors §1.
- **OPEN 2** ("gradients with respect to a specific kernel") — retracted as
  a misstatement; replaced by the two-lenses framing (value-centric vs
  function-centric VJP/JVP) in §7.5.
- **OPEN 3** (WebGL vs WebGPU) — "WebGL" was a misstatement: the render
  target is WebGPU, and the compute story gives CUDA, Metal, and WebGPU
  equal representation (§2.1, Probe C).
- **OPEN 4** (the "two different types of objects" seam) — text recovered;
  now in Probe A (uniforms as normal Python objects copied per draw;
  backend-allocated vertex arrays; third-party allocation).
- **OPEN 5** ("gradient clashing") — confirmed as gradient *clipping*
  (Probe D).

Second-round decisions (after tensorlib landed and CuTe was studied):

- **Target**: the joint system (our syntax fronting the tensorlib
  representation), per §5.
- **Governance**: decided positions in either stream are evidence, not
  canon; collisions escalate to the human (§8).
- **Sequencing**: tensorlib paused at the L4 boundary; the report is the
  brief that reopens it (§8).
- **Architecture amendments adopted** (§3): rewrite chains instead of
  monolithic equivalence proofs; AD × partitioning non-commutation
  registered as a first-class open question.
- **Level-2 framing updated** (§2.1): the existence question is answered by
  construction; the open question is the surface.
