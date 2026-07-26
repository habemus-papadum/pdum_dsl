# 220 — The principles: mechanisms that enable behaviors

**Status: living canon.** Each entry names a principle, the mechanism that
realizes it, the behavior it enables, and where that behavior is pinned.
None of these is novel in isolation — every one exists somewhere in some
library. The reason this system is built rather than assembled is the
**weaving**: each mechanism assumes the others, and the behaviors below
fall out of their intersection, not from any one of them. When a design
question arises, it is answered here first: a change that breaks one of
these threads pays for the whole cloth.

## 1. Type-keyed identity — dispatch on types, never values

A closure is (code identity, typed environment, environment values), and
compilation keys on the first two ONLY. Live values ride the uniform
channel; a value change is a buffer write, never a recompile. The one
opt-in is `Literal` — value-into-type by declaration (three doors: value
type, call-site wrap, definition-site annotation), and policy maps fold
into build identity the same way, so a train build and an eval build can
never collide.
**Enables:** render loops with hundreds of fresh closures and zero
compiles; train/eval as two cached Programs; config knobs as live values.
**Pinned:** test_runtime (299 frames, 1 compile), test_scope_assemblage
(policy collision unwritable).

## 2. Content-addressed artifacts — reuse is an outcome, never a mechanism

Tier 2 keys on content: identical IR from different templates shares one
artifact; a generation bump cannot orphan it. Incremental compilation is
a **cache phenomenon** — no invalidation logic exists anywhere; a miss is
always the correct answer. Derivations are cache entries computed on
demand from cache entries (marker partials, adjoint scanners, derived
Programs).
**Enables:** analyze-first-provision-later warm hits; two def-sites, one
artifact; derivation-under-cache with forbid() as the proof.
**Pinned:** test_runtime (identical bodies share), test_provisioning
(gate 10), tl test_events (idempotence via forbid).

## 3. Capture identity decides leaf identity

The program is the parameter container: a leaf is the OBJECT the makers
captured, not a name lookup — one Param captured twice is one input with
the summed gradient. Guards are identity checks against drift: rebinding
to an equal value still recompiles, loudly counted.
**Enables:** weight tying for free (GPT-2's wte); no parameter registry
to keep consistent; stale-capture bugs die structurally.
**Pinned:** test_naming_law (the tied-gradient pin), test_runtime
(guard drift counted).

## 4. Names are the contract surface

One flat name space, path-derived (`h.3.attn.wq`), declared at use, never
auto-suffixed. Everything JOINS ON NAMES: grad maps, provisioning,
optimizers, requested taps, and randomness streams
(`fold_in(root, path)` — insertion-stable where positional splitting is
not). Derived names carry law-suffixes (`.d{i}`).
**Enables:** checkpoints as name-joins; refactors that don't churn RNG;
taps selected by pattern and pruned free.
**Pinned:** test_naming_law (literal name set; rebuilt closure, same
names), test_random (stream stability).

## 5. Layout-derived alignment — batching-unaware code

Layouts are affine and data-independent, which makes alignment decidable
and adjoints derivable. Nothing auto-broadcasts: misalignment REFUSES
quoting the fixing view (D17). The alignment primitive `repeat_like(x,
like)` derives its added-dim set FROM THE LAYOUTS at build time — the one
thing an author cannot enumerate, because batch dims are exactly what the
author doesn't know. Contraction axes, by contrast, ARE author knowledge
and are always declared (`contract(a, b, axis="d")` — mandatory).
**Enables:** code written for (t, d) runs untouched on (b, t, d); one IR
instruction per alignment; placement declared at the leaf rides through
every broadcast automatically.
**Pinned:** test_assemblage_vocab (the batching-unawareness pin),
test_refusal_contract (the extent refusal), test_placement (erasure
bit-exact).

## 6. Closed-form fields — zero-memory citizens

iota and the random fields are pure functions of (key, lattice
coordinates): element i computed directly, no memory, exact under every
view op, free in the cost models. Bits are exact rationals; the generator
is frozen (Philox2x32-10, reference vector pinned).
**Enables:** masks that cost nothing; the recompute theorem — checkpoint
and revolve regenerate dropout masks bit-identically, so gradients under
recompute are exact BY CONSTRUCTION; oracle/device bit-parity by pure
integer arithmetic.
**Pinned:** test_random (regeneration, exactness under views, the frozen
vector), test_transforms (checkpoint ≡ store-all).

## 7. Declarations over recognition — the naive floor

Markers and reducers are DECLARED with their algebra (a name, a numpy
function, identity/associativity); partial derivatives are DERIVED by
inspection of small named bodies — the marker-granularity gate — so
flash attention's backward exists without a hand rule. Programs stay
maximally naive (contract is one visible line of repeat + mul + reduce;
no matmul op exists); recognizing patterns and fusing them is the
business of lowering passes over that naive floor, never of the surface.
**Enables:** new activations differentiate automatically; the derivative
table grows only when a primitive joins the core; compilers may fuse
without semantics ever depending on it.
**Pinned:** test_marker_granularity, test_at_kink (the partition law),
zoo flash (derived backward ≡ naive).

## 8. The surface is the IR, executable

The recognized function set IS the primitive set — one surface function
per IR op (pointwise, reduce, scan, iota, const_like, repeat_like), plus
the structural read (extent). Authored bodies are INSPECTED (binding
names become SSA names; straight-line is enforced; Literal doors bind);
every other function is plain Python that inlines. The same code runs
EAGERLY on numpy-backed tensors — the naive backend is not a second
implementation, it is the same library uncompiled.
**Enables:** the eager-vs-lowered differential as a standing gate;
debugging without compiling; helpers as ordinary readable code; markers
applied only through operators (`pointwise(cos, t)` — bare calls refuse).
**Pinned:** test_assemblage_vocab (eager ≡ lowered; bare-call refusals;
binding names read back).

## 9. Refusals are API

One shape everywhere: what happened, the principle violated, the quoted
fix, and refusal wording is FROZEN behavior (the batteries pin it
literally; a drifted message is an API break). Silent fallbacks do not
exist: unrouted calls, unknown kinds, misalignment, name collisions,
structural-slot violations, out-of-range decode — all refuse by design.
**Pinned:** test_refusal_contract (both packages), and every designed
message in the tree.

## 10. Facts at the boundary, choices in the interior

Semantics are carrier-valued end to end; dtype is a property of buffers
and encodings AT THE BOUNDARY, recorded in descriptors with exact decode
(every bit pattern is a specific rational; inf/nan refuse). The one
sanctioned precision op is explicit `round_to(encoding)` with
straight-through AD; the IR cannot mint encodings; byte truth enters at
descriptors and at L2's assignment, nowhere else.
**Enables:** bf16 checkpoints as facts, not semantics; QAT in boundary
terms with trainable masters; byte-blind capacity mistakes impossible.
**Pinned:** test_precision.

## 11. Named axes, exactly labeled

Axes are NAMES, not positions: alignment is by name, presentation order
is never semantics, and weights are born structured (wq is (d, nh, hk) —
heads are dims, never splits). Dims carry exact metadata as first-class
layout content: coordinate charts with exact rational origins and steps
(a Yee grid's half-integer stagger is Fraction(1, 2), not a comment),
categorical labels, value units, and machine levels — physics and
placement ride the same lattice machinery as everything else.
**Enables:** staggered-grid PDEs where recharting is the discretization's
honesty made syntax; unit checking that refuses exp-of-micrometers;
gradients carrying their primal's charts and placement by construction;
einsum-class contractions with zero index bookkeeping.
**Pinned:** test_lifting (FDTD charts survive the fold), test_charts,
test_signatures, test_placement (gradients carry placement).

## 12. The layout/compute split — free views, few costly primitives

A hard line runs through the IR. On one side: the layout algebra —
affine maps + box domains + guards, deliberately NOT piecewise — where
every operation is a zero-cost view requiring ZERO intelligence, and
every layout op's adjoint is again a layout op. On the other: the few
primitives that actually incur compute — pointwise, reduce, scan, fold,
materialize (the one copy; `take` joins at P9) — and everything else is
built from them. The assumed program form is STRAIGHT-LINE; branching is
the host-level doctrine over cached segments (recorded in 200 §1.3,
honestly not yet vetted by use). The pragmatic bet, stated plainly:
high performance over maximal flexibility — a small primitive set is the
leverage, because differentiation, cost oracles, signatures, alignment,
and eventually descent each reason over a handful of cases instead of an
open op zoo.
**Enables:** exact op counts and peak-memory from layouts alone; the one
derivative table staying one page; the BPTT engine deriving flash's
backward because the case analysis is finite; a reference interpreter
small enough to be an oracle.
**Pinned:** test_opcount (exact tallies), test_memory, the layout-adjoint
table realized in autodiff, test_at_kink.

## 13. The seam — observability without instrumentation

Compile-ish acts announce themselves (spec/artifact misses, Program
builds, adjoint derivations, registrations) on one event seam that costs
a truthiness test when dark. `forbid()` turns "this loop is hot" into an
assertion: zero compiles, zero builds, zero derivations — proved, not
hoped.
**Pinned:** no_compile throughout the D battery; tl test_events
("this training loop builds zero Programs").

## 14. One body language — the kernel is a dialect, not a sibling

The kernel language is not a second language: it is the value language
plus exactly three extensions — the thread ambient, token-threaded
stores, and buffer reads. The scalar marker subset is its effect-free
straight-line core, so device functions ARE value-language kernels and
compose by capture-and-call; the tensor lifter becomes a vectorizing
execution strategy for the same programs, not a competing syntax.
The payoff is singular infrastructure: ONE derivative engine (forward
seeding over the one table) serves autodiff at the tensor tier,
`with_respect_to` in a local scope, and `fwidth`-style ambient
derivatives in a fragment shader — three features, one mechanism.
The ambient itself obeys the law: its primitives are the raw
block/thread pair plus the launch grid reified as a layout (forced by
the affine-only algebra — global→raw is div/mod, banned), and
`global_thread_idx(block, thread, g)` is a stdlib device function —
layout evaluation at the scalar tier, name-bindable to a vendor
built-in.
Invocation stays out of identity by law, stated per config component:
launch geometry is launcher data (threads the one value-specialized
carve-out), tap NAMES specialize while tap tensors are invocation data,
shared memory is structural. Claiming itself is TAGLESS — the naming
law is the one claiming mechanism: a uniquely-named binding IS the
site, whether its sink is a tap buffer, the varying interpolator, or a
render target (`tap()` retired; `flat(...)` the sole site-side
annotation, for interpolation). And observability is honest: a claimed
site whose name goes non-unique under inlining is INVALIDATED with the
reason, never silently renamed.
**Enables:** shader combinators as ordinary function composition; the
AA circle whose edge is analytic; kernels gaining derivatives, records,
and randomness the day the value tier does.
**Pinned:** test_kernel (config bracket, taps, honest invalidation);
test_kernel_spec.py — the committed future spellings as skipped tests,
un-skipping per milestone.

## 15. One lowering engine — dialects are rule-pack layers

Entry 14 states the syntax law; this is its machinery, proven by the
P8.5 pivot: there is ONE lowering engine (the dsl Lowerer), and a
tier is a RULE-PACK LAYER over it, never a second machine. The tl
tensor dialect is the base pack plus tl rows; the kernel tier layers
ambient/store rows over that; the step and assemblage tiers are
further thin layers — each contributes only its genuine vocabulary,
and TENSOR-TYPEDNESS selects the dialect path per node (a scalar
subtree flows through the base pack unchanged, all-scalar subtrees of
a spliced kernel ride the value dialect inside tensor regions).
Type identity is the value's OBSERVABLE frame — for tensors, per dim
its name, domain, and labeling frame, with representation
(strides/offset) riding as a non-identity shadow read only through
the one inference authority — which makes tl's alignment law an
ordinary type rule (refusing at emission, with source locations and
the incumbent's own fix recipes) and makes content keys distinguish
what execution distinguishes, nothing else.
The organizing rule for growth (owner-acked): an op's DEFINITION SITE
owns all its columns — type rule, evaluator row, adjoint row,
spelling rows — so a dialect extends by table rows, never by wrapping
another dialect's machinery.
And schedules are EVALUATION STRATEGIES, never IR: store-all and
revolve share every node (the recompute theorem holds bit-identically
with dropout on, because closed-form fields re-select at absolute
coordinates); an L4 certified descent may BAKE a schedule later
without a representation change.
**Enables:** alignment errors at emission instead of launch; one
derivative walker serving fold, general regions, and the incumbent
adjoints through a migration view; the marshaling dialect (`abi.*`)
shared verbatim between tiers; a kernel's fn-arguments inlining
through the same engine that lowered them as citizens.
**Pinned:** test_dialect.py (differentials bit-identical, alignment
with locations, frame-keyed content keys, the fold/revolve theorem);
test_kernel.py (zero-oracle inlining, one executor per content key).
