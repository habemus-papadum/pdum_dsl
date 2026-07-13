# 090 — Core & extensions: the punning charter (dialect + runtime)

**Status:** canon (2026-07-12, the post-10b pause). Companions: 070
(backend research), 080 (backend organization), R12–R17. Consumed by step
11 (§5) and step 14 (§4, §6).

## 1. The question this answers

We interact with similar-but-not-identical systems at **two layers**:
kernel syntax (thread indices, printf, subgroup ops) and runtime (compile,
launch, streams, events, allocation, readback). The desired shape is
OpenGL's core-plus-extensions model, here called **punning**: one common
interface covers most work portably, with escape hatches into
vendor-specific capability at the explicit, *visible* cost of portability.

The decision (argued in §7): most of the punning **mechanism already
exists** in the finished kernel; what this document adds is the
**conventions and contracts** — plus the one genuinely new design surface
(buffers & tensor interop, §5) that step 11 needs immediately. Code-level
runtime abstraction is deliberately deferred to step 14, when three real
runtimes exist to abstract from.

## 2. Stdlib minimalism (policy)

The stdlib is **language machinery plus scalar math, nothing else**:

- the base-language rule pack (AST forms), the five surface doors, casts,
  the dispatcher, the `device` role;
- the intrinsic scalar core (`sqrt exp sin cos floor abs min max`) — ops
  every target must spell, the price list every new backend pays;
- portable scalar helpers that are shader/numeric lingua franca
  (`clamp mix step smoothstep fract`) — DSL-written, free on every target.

**The squatting test** (extends 080's kind-hygiene rule from names to
*packages*): would a third-party library plausibly own this name with
richer semantics? Then it does not enter the stdlib. `Color` is the
canonical example — real color modeling means spaces, RGB vs Lab vs OkLab,
gamma; that is a domain library's ground, and the five surfaces make such
a library an ordinary pip install (`install(registry)` + entry point).
`Color` and the 2D helpers (`dot2 length2 lerp2`) accordingly moved to
`pdum.dsl.demo.graphics` (2026-07-12), which is **not auto-imported**: one
explicit import wires it in, and that import *is* the lesson.

A small stdlib is not a modesty preference — it is evidence. If extension
is as cheap as the architecture claims, the stdlib proves it by needing
almost nothing.

## 3. The dialect layer: core + extensions

Already exists (step-10 mechanisms — no new machinery required):

- **Core profile** = the core dialect + intrinsic ops every backend
  spells. **Portable batteries** = capture-free DSL bodies, inlined per
  call site, free on every target including future ones.
- **Graceful degradation** = shared decompositions gated on
  `op not in backend.code_for_op` (key presence IS the capability bit).
- **Loud refusal** = `MissingRule` / renderer refusal where neither
  spelling nor decomposition exists.

Conventions this charter adds:

- **Vendor namespaces.** Vendor-specific intrinsics are ordinary ops
  registered under the vendor's namespace — `cuda.laneid`,
  `metal.simdgroup_sum`, `wgsl.subgroupBallot` — spelled by exactly one
  backend and *never* given a decomposition. Using one is opting out of
  portability, visibly, in the kernel's source. Nothing else changes:
  dispatch, caching, and marshaling are indifferent to which ops a body
  uses.
- **Capability flags** (R15's list: `subgroups`, `i64`, `f16`,
  `float-atomics`, `printf`). Declared on the Backend record; checked at
  *build* time when a kernel's surviving ops require them, so the error
  names the kernel, the op, and the missing capability — renderer refusal
  remains the backstop, not the diagnostic. Lands at step 14 with the
  first backend pair that actually differs.
- **printf** is a capability-gated op (`debug.print`), not syntax: CUDA
  spells it natively; Metal has only limited os_log; WGSL has nothing.
  Default is loud refusal on non-supporting targets. A strip-debug
  decomposition (rewrite to nothing) may be registered *explicitly* — the
  framework never silently discards a print. Moot until CUDA (step 14).
- **Thread indices.** The compute-family contract (params ARE thread
  coordinates) is the core profile — portable by construction. Vendor
  addressing beyond it (lane ids, block/thread split, workgroup ids) comes
  in as vendor-namespace intrinsics, not as changes to the family
  contract.

## 4. The runtime layer: what our runtime is — and refuses to be

**What it does** (per target):

1. **Compile** source strings → artifacts (content-addressed in *our*
   tier-2 cache; vendor caches unused — the 070 lesson: name-keyed vendor
   caches never re-check source).
2. **Launch** through the ONE calling convention:
   `launch(staging, leaves)` — staging is the byte-packed uniform block,
   leaves is the buffer channel. This is the interface benchmarking,
   chaining, and transforms all sit on.
3. **Allocate** device buffers it owns, and **adopt** external buffers
   zero-copy (§5).
4. **Order** work: streams/queues as config-bracket values (040 §3c:
   stream strips → leaves channel), not as identity.
5. **Time**: events/timestamps behind an artifact capability protocol
   (`timed_call` — the shipped precedent).
6. **Read back** explicitly, with an async path (§5).

**What it refuses to do:**

- No execution graphs or schedulers — chaining is OUR orchestrate-tag
  encode plan (070 decision 4), not a vendor graph API.
- No tensor semantics, no autograd — tensor libraries are interop
  partners (§5), never merged runtimes (the R13/R14 verdict: rides fail
  on caching, marshaling, scheduling — the three things we own).
- No shape inference and no implicit allocation of results — explicit DPS
  `out=` forever.
- No re-implementing vendor toolkits: **cuda.core IS the CUDA runtime**,
  wgpu-py the WebGPU one; Metal is the one thin shim we own
  (ctypes/PyObjC, tinygrad-scale). Our "runtime" is the *contract* these
  are adapted to, not a driver.

**The pun.** The common interface is two-tiered, and both tiers exist:

- **Framework-facing:** the Backend record's columns
  (`render/compile/fp/plan/param_types/make_launcher/code_for_op`) — what
  the registry needs to build and cache.
- **Instance-facing:** protocols on the artifact, discovered per artifact
  (`getattr`), never declared globally. Today's table:
  `__call__(staging, leaves)` (required), `timed_call(staging, leaves) →
  phase dict` (optional — and `bench.gpu_timeline` is already generic over
  any artifact providing it, knowing nothing about WebGPU). Step 14 adds,
  extracted from three real instances: `alloc`, `stream`, `synchronize`,
  event queries.

**The escape hatch** is that `record.artifact` is public: the vendor
object (the wgpu ComputeProgram, the cuda.core module/stream, the Metal
pipeline) is reachable, and touching it means leaving the common interface
on purpose — the runtime-layer mirror of using a `cuda.*` op.

**The rule of three.** No abstract runtime base class is defined now. We
have exactly one real GPU runtime; abstracting from N=1 is speculative
generality (the same reasoning that defers `families/` to step 14). The
protocol is *extracted* at step 14 from wgpu + cuda.core + Metal sitting
side by side, with bench as its first generic consumer.

## 5. Buffers & tensor-library interop (step 11 consumes this NOW)

The contract for the Array era, fixed before the first Array lands:

- **The Array type carries a device axis** (alongside dtype and the
  §13 shape-summary dial). Device-in-type is dispatch tier 1 (080): where
  the data lives IS the backend choice, no annotation.
- **Buffer leaves are OWNED or ADOPTED.** Owned: allocated by our runtime
  (possibly free via cuda.core / wgpu). Adopted: a zero-copy view of an
  external allocation — the adopter holds a reference so the exporter
  outlives every launch that uses it. Ownership is a leaf property, not a
  type property: it never enters the cache key.
- **Zero-copy in BOTH directions.** Import and export speak the ecosystem
  protocols: DLPack as the lingua franca, `__cuda_array_interface__` for
  the CUDA world, the buffer protocol / NumPy for host-visible and
  unified memory. Exact mechanism per backend is verified when that
  backend is built (R13/R14 groundwork); the *contract* — both
  directions, zero-copy, no blessed tensor library — is fixed here.
- **The test-data workflow** this buys: allocate or adopt → view as
  cupy/MLX/NumPy → initialize with *their* toolkit (random, ranges,
  images) → run our kernels → zero-copy back → assert or visualize.
  Tensor libraries as tooling, not as runtime.
- **Readback is explicit and may be nothing.** On unified-memory targets
  (Metal, and MLX arrays generally) "readback" degenerates to a
  synchronization point — the 10b finding (WebGPU readback ≈ 1.6 ms of
  *fixed sync latency*, not bandwidth) is a per-protocol cost, not a law.
  The async path (map-async on WebGPU, streams on CUDA, shared events on
  Metal) is designed in as the escape from that latency; its first real
  implementation rides the graphics `draw` surface work.

## 6. Testing multi-device: the abstraction ladder

Principle: **test as much as possible with no device, and make the
device-required residue small, explicit, and probe-gated.**

- **Layer 0 — hardware-free, every CI run, every backend:**
  - *Golden codegen*: rendered source snapshots (shipped precedent:
    `test_wgsl_renderer`'s CPU goldens caught the `f16` reserved-word bug
    without a GPU).
  - *Twin differential*: the Python twin is the semantics oracle (core
    numeric policy: trunc div/mod, raise on div-zero, wrap on overflow).
  - *Conformance-vs-fake*: a FAKE runtime — an in-memory artifact
    implementing the §4 protocols that records launches and byte-checks
    staging/leaves against the plan. Marshaling, launch-contract, cache,
    and chaining tests all run against it. (Bench already models this
    style: `benchmark(timer=)` is injectable; the fake runtime is the
    same idea one level up.) The conformance suite is *shared*: every
    real backend must pass the identical suite the fake passes.
- **Layer 1 — device-gated, probe-first:** per-target twin differentials,
  timing smoke (`gpu ≥ 0`, readback > 0), ABI spot checks. Gated by
  `pdum.dsl.testing` probes; `PDUM_REQUIRE_*` prevents silent rot where
  hardware is guaranteed.
- **Layer 2 — cross-device differential:** the same kernel on every
  available target, outputs compared (ch15's chapter is this layer as
  pedagogy).

CI matrix (unchanged from 070): ubuntu + lavapipe (`REQUIRE_WEBGPU`),
macos-15 Metal probe-first, CUDA via burst capacity.

**The CUDA box** (user-provided: a Linux + CUDA machine, SSH-able). Three
modes were offered; the recommended split:

1. **Handoff document + parallel agent on the box** — the primary mode
   for *building* `backends/cuda/` at step 14. It matches the 020 plan
   (parallel agents, one per backend) and this charter makes the handoff
   cheap: the document is 090 + 080 + the conformance suite, and the exit
   gate is mechanical (layer 0 green everywhere, layer 1 green on the
   box). Design-for-skip discipline stays: the backend must be fully
   developable to layer 0 with no device at all.
2. **Direct SSH from the driving session** — for short verification
   bursts: validating cuda.core ABI assumptions (the once-per-FastRecord
   pointer-table escalation, R12), running the layer-1 suite, spot
   benchmarks. Right-sized for minutes, not for building a backend.
3. **User-runs-commands** — fallback only.

An optional, cheap de-risking move before step 14: a layer-0-style *spike*
during step 11's C-backend work — render a trivial kernel to CUDA C and
compile-check it via SSH (mode 2) — to catch cuda.core surprises while the
Metal/CUDA steps are still ahead. Not a gate; an hour's insurance.

## 7. What already exists vs what this defers

| Punning need | Mechanism | Status |
|---|---|---|
| Portable kernel core | core dialect + intrinsics + batteries | shipped (step 10) |
| Degrade where unspelled | decompositions gated on `code_for_op` | shipped (step 10) |
| Vendor kernel escape hatch | vendor-namespace ops, no decomposition | convention here; vocabulary at step 14 |
| Capability errors at build | Backend capability flags | step 14 |
| Common runtime interface | Backend record + `launch(staging, leaves)` | shipped (steps 8–9) |
| Optional runtime features | artifact protocols (`timed_call`) | shipped (step 10b); extended step 14 |
| Vendor runtime escape hatch | `record.artifact` is public | shipped |
| Runtime-generic benchmarking | bench wraps seams + discovers protocols | shipped (step 10b) |
| Buffers, device axis, interop | §5 contract | step 11 |
| Async readback | §5; first impl with graphics `draw` | deferred |
| printf | capability-gated `debug.print` | step 14 (needs CUDA) |
| Abstract runtime protocol *in code* | extracted from 3 instances | step 14 (rule of three) |

The through-line: the kernel is finished, so every row above is satellite
work — which is itself the strongest evidence the punning architecture is
already in place.
