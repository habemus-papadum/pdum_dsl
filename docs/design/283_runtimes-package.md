# 283 — The runtimes package: `pdum.rt` (PROPOSAL)

**Status: PROPOSAL.** The owner ratified a third workspace package for
runtimes and backend emitters (282 §4) with the name and layout "proposed
at implementation" — this is that proposal. It also sketches **how a CUDA
column lands**, as a working reference pattern for the L4 team: use it,
bend it, or replace it — **the final judgment on everything CUDA-shaped
is yours**. Evidence behind every non-CUDA claim: the campaign spikes and
the dual-backend demo (`explorations/graphics/` on
`worktree-graphics-design`, origin), distilled in 210's Metal and
warm-loop sections.

## 1. The name and the shape

**`pdum.rt`** — `packages/rt/src/pdum/rt/`, the third workspace member
beside `pdum.dsl` and `pdum.tl`. Two facts argue for a real package
rather than a subpackage of `tl`:

- The Metal spike's runtime imports **no pdum at all**, and its backend
  imports no Metal — the layer boundary is real and already enforced
  physically. Runtimes are about devices, not tensors.
- Both existing packages consume it: `dsl`'s value tier wants executors;
  `tl`'s kernel/graphics tiers want launch and presentation; and the
  ruled relocation of `tl.sample` + the texture objects ("destination
  theirs", 290 companion ruling) needs a home that is neither.

`conformance/` shrinks back to tests only: the batteries import
translators FROM `pdum.rt` and keep the differential discipline; the
three byte-identical copies of the marker tables die.

## 2. Layout

```
packages/rt/src/pdum/rt/
  __init__.py     # the prefab pairs: webgpu, metal, cuda; on()
  select.py       # Generator / Runtime / Pair dataclasses (282 §7 —
                  #   empty for now; names are pre-assembled instances)
  registry.py     # device acquisition (features requested at creation),
                  #   the executor-fp column keyed through the EXISTING
                  #   (region.key, executor fp) content door — never around it
  emit.py         # ONE expression generator; per-target Dialect hooks
                  #   (four lexical rules: cast, literal suffix,
                  #    declaration form, vector spelling)
  contract.py     # LaunchContract — the negotiated clauses (§4)
  staging.py      # the staging plan's DEVICE representation, declared
                  #   (i64 slots stop narrowing silently; stride-16 when
                  #    a true uniform address space wants it)
  mathrows.py     # target numeric contract rows + freeness proofs
                  #   (Metal tanh clamp is row one)
  wgsl/           # generator.py, runtime.py   (from the spikes/demo)
  msl/            # generator.py, runtime.py   (from the spikes/demo)
  cuda/           # generator.py, runtime.py   (the L4 column — §5)
  encode.py       # the encodable: compile once; bind() ≠ encode();
                  #   encode(encoder, target, clear=...) — the demo's
                  #   corrected primitive; in-flight discipline hooks
  present.py      # presenters: rendercanvas (WebGPU), CAMetalLayer
                  #   (Metal) — host-owned loops, shared encode path
  textures.py     # tl.sample's runtime objects move here (ruled);
                  #   the wgpu type-table registration leaves graphics.py
```

What moves, concretely: `graphics._device()` → `registry` (with feature
requests — today timestamp queries are unreachable without
monkeypatching); `wgsl_executor`'s two translators → `wgsl/` (conformance
imports back); the demo's proven glue seeds `emit.py`/`encode.py`;
`WGPU_FP`/`METAL_FP` become real key components via the content door.

## 3. Selection — the ruled spelling, unchanged

```python
from pdum import rt

ripple[rt.on(rt.metal)](src, dst)          # per-launch selection
art = ripple.artifact(src, dst)            # the public artifact door (H2)
art.on(rt.cuda).launch((src, dst))         # 270's incremental chain
dev = rt.acquire(rt.cuda, features=("timestamps",))   # explicit, once
```

`on(...)` names a **Pair** (generator + runtime, separable, prefab
instances); it **keys the artifact cache**; it never implies a transfer.
Under adoption/managed memory a tensor's bytes can be host and device at
once — residency is a set on the buffer, resolved against the registry,
never a string.

## 4. The launch contract — what a generator RETURNS

A backend cannot return only source text (210, supersession bullet). The
contract consolidates the clauses the spikes/demo proved are negotiated
per target, never universal:

```python
LaunchContract(
    thread_size = (64, 1, 1),   # SPECIALIZES — keys the artifact (ruled);
                                #   lands in source (WGSL), in the dispatch
                                #   call (Metal), or BOTH (CUDA launch_bounds)
    guard       = "emitted" | "exact",   # per-runtime dispatch granularity
    bindings    = per_stage(...),        # DATA, one table per stage;
                                         #   degenerates to param order on CUDA
    slots       = staging_plan,          # device repr DECLARED, incl. carriers
    math        = rows(...),             # numeric-contract substitutions
)
```

## 5. The CUDA column — a reference pattern (yours to judge)

Casual-investigation basis, stated so you can re-verify: **CUDA Python
1.0** is out with stable APIs; `cuda.core` carries the pythonic
Device/Stream/Program/Kernel surface (NVRTC JIT — and 13.3's NVRTC
bundles the standard headers, so runtime compilation needs no local
header setup, the same "no Xcode" property Metal's
`newLibraryWithSource:` gave us); `cuda.tile` is a Python DSL for
tile-model kernels and `cuda.coop` carries block/warp-wide primitives.
A `cuda.tools` name did not surface in our look — treat the subpackage
choice as open. None of this is a bet; it is a starting point.

The pattern, clause by clause:

- **Generator (`cuda/generator.py`)**: emit CUDA C through the ONE
  `emit.py` walker with a CUDA dialect hook. Expected diff from the
  measured WGSL↔MSL result (91/91 rows under four lexical rules): casts
  spell `float(x)` (C++), literals take `f` (R2), declarations are C
  (R3), vectors are `float4` (R4 — and this is where the dormant `Vec`
  type meets your coalescing idiom at the cost of ONE row). Shell:
  `extern "C" __global__ void main0(float* buf0, ...)`; the ambient row
  is COMPOSED (`blockIdx.x * blockDim.x + threadIdx.x`) rather than a
  single builtin — still one leaf row. No binding declarations at all:
  the binding table degenerates to parameter order, the trivial case of
  per-stage-bindings-as-data.
- **Runtime (`cuda/runtime.py`)**: `cuda.core` Device/Stream;
  `Program(source).compile()` → kernel → `launch(grid, block, args)`.
  Guard clause = "emitted" (whole blocks, WebGPU-shaped — proven
  bitwise-immaterial where the exact grid divides). Thread sizing
  appears in BOTH the launch call and (optionally) `__launch_bounds__`
  in source — the supersession ruling's cleanest exhibit: it
  specializes, wherever it lands.
- **Timing**: cudaEvent elapsed time is the honest timer (no feature
  request — Metal-shaped, not WebGPU-shaped); NVTX ranges bind the
  existing span seam (200 §S.4 already says this).
- **Memory**: explicit H2D/D2H on discrete hardware (the transfer half
  of the runtime is REAL here, unlike unified Metal); pinned/managed
  memory is the adoption analog — and inherits the in-flight discipline
  verbatim (210 warm-loop: a slot refresh over pinned memory with two
  launches in flight needs rotation or versioning). DLPack is the
  interchange `Buffer` already names.
- **Duties on arrival**: run the math-rows survey (`exp`, `sinh/cosh`,
  `pow`, fast-math flags are LICENSES, never defaults — the tanh
  precedent); add the ledger's CUDA column (rows join
  `ledger.toml`, drift tests extend per target); wide-range battery
  inputs are already yours.
- **The honest-portability milestone**: the first CUDA differential is
  the first cross-VENDOR exactness test — everything to date shares one
  Apple GPU via Naga. Expect the first real divergences at the
  math-library and rasterization edges; the numeric policy (210) is the
  arbiter on both sides.
- **Where the pattern might bend, deliberately yours**: if `cuda.tile`
  becomes the lowering target for the L4 kernel language (tile-model
  kernels rather than emitted CUDA C), the generator column changes
  shape — but the Pair spelling, the contract clauses, the registry,
  and the ledger discipline all survive that swap untouched. That
  seam-stability is the point of the proposal.

## 6. Division of labor and sequencing

We (graphics campaign) build: the package skeleton, `select`/`registry`/
`emit`/`contract`/`staging`/`mathrows`, the `wgsl/` and `msl/` columns,
`encode`/`present`, and the `textures.py` relocation — in that order,
with the encodable + depth (v2) riding on top. The `cuda/` column is
**yours**, on your box, on your schedule, with this document as pattern
and 210/211 as the constraint set. Coordination points: the content-door
keying (we land it; your column inherits it), the ledger's per-target
rows (yours), and anything you want verified on Apple hardware before
you commit to a shape.
