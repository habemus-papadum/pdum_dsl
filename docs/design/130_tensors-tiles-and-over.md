# 130 — Tensors, tiles, and `over`: axes as IR, lowered late

**Status:** PROPOSED (2026-07-13, the fork-point conversation). Companions:
100 (arrays & axes — the types this animates), 110 (transforms — the v1
this supersedes in part), 120 (events — whose tripwire budget policy and
seam methodology this follows), 090 (the punning charter), 040 §3c (the
config bracket that tile sizes have been waiting for), 030 (the
deep-learning satellite this unblocks). Written at the user's direction
after the step-12 walkthrough; the GEMM sketch in §5 is the centerpiece.

---

**The ask.** Three converging pressures, one design. (1) `vmap` (renamed
**`over`**, §2) currently *erases* the mapped axis at lowering time —
weaving it into stride arithmetic — when the axis is exactly the structure
a scheduler wants to see: to fuse, to tile, to assign to a grid dimension,
to feed a batched tensor-core op. Erasure is premature lowering. (2) Every
capability conversation since step 11 has hit the same wall: **arrays are
not IR values** — no array results, no array args, no elementwise tile
ops; GPT-2, named reductions, and `over`-as-IR are all blocked on it.
(3) The step-14 GPU future (tensor cores, shared memory, warp primitives)
must not require lowering to scalar loops and then *un-lowering* to
rediscover that a loop nest was a matmul. ML's structure maps cleanly onto
GPU operations; the representation should keep it stated, not recovered.

The design principle, stated once so it can be refused:

> **Named `contract` / `reduce` / `map` are the working representation
> until target selection. The scalar core is the FLOOR for scalar
> targets — never a mid-level that tensor targets must climb back out of.**

This is MLIR's linalg-on-tensors lesson, Triton's `tl.dot` lesson, and
tinygrad's schedule-late lesson, adapted to a type-keyed caching kernel.

---

## 1. What exists today (and why it is not enough)

- **`Array`/`NamedArray`/`ShapedArray` types** (100): complete, including
  the device axis. But an Array-typed *value* in IR exists only as
  `array.buffer` (a capture's payload pointer) feeding scalar
  `array.load`s. Kernels return scalars; arrays cannot be arguments,
  results, or intermediates.
- **`vmap` (110)**: SIMT-shaped and *correct* — but implemented as
  lowering-time weaving, so the batch axis vanishes into `abi.slot`
  arithmetic. Execution is per-lane host dispatch: the ray-march spike
  measured ~2.4 µs dispatch against a 0.8 µs C body, so per-lane calling
  is the wrong execution shape for exactly the workloads batching serves.
- **`matmul` (110)**: named contraction as a *scalar-cell* special form —
  the pairing logic (unique shared axis name, woven axes excluded first)
  is proven and stays; the op needs to graduate from cell to tile.
- **Loop semantics**: `core.for` means "sequential." There is no way to
  say *this loop's iterations are independent* (map) or *independent up
  to an associative combine* (reduce) — which is the entire difference
  between a loop a backend must run in order and one it may parallelize,
  vectorize, or hand to a matrix unit.
- **Purity**: absolute. Shared-memory staging and barriers — the heart of
  a modern GEMM — have no representable ordering.
- **Transform plumbing** (110 §5, step-12 review): composition refused;
  five string-keyed context doors smuggle state through the rules dict;
  three hand-rolled copies of the FnType wrapper protocol. Under 120's
  policy inversion these are exactly the "monkeypatch-shaped" workarounds
  that should become seams.

## 2. The rename: `over`

`vmap` fights a JAX prior it contradicts on every axis (argument-position
vs capture-name selection, call-once-batched vs per-lane, widened values
vs woven coordinate). The operator's honest meaning is *"this kernel, over
that axis"* — so: **`over(f, axis="batch")`**, with the `Derived` tag,
docs, chapter, and tests renamed while there are no users. `over`'s
contract is the AXIS BINDING; §4 upgrades what it produces without
changing what it means.

## 3. Design in one paragraph

Arrays become IR values (§4.1), carried by a **named tensor dialect** —
`tensor.map` / `tensor.reduce` / `tensor.contract` / `tensor.slice` — that
is a satellite rule pack over kernel-side loop semantics: `core.for` gains
`kind ∈ {seq, map, reduce}` and `axis` attributes (§4.2). `over` becomes
the axis-binding transform at every level: applied to a kernel it wraps
the body in a `map` loop over the named axis (IR-visible, schedulable);
the step-12 weaving demotes to one *lowering* of that IR on scalar
targets, and the C backend gains the in-artifact lane loop that fixes the
dispatch economics (§4.3). Memory spaces and layouts join the Array type;
`stage`/`barrier` thread effect **tokens** so cooperative GPU code stays
SSA-pure (§4.4). Tile-family kernels (§5) program in tiles — `stage`,
`contract`, named comprehensions — and never mention threads; warps are a
backend decomposition plus 090's vendor-namespace escape hatch (§6). The
transform plumbing this requires (context, composition, derived-value
protocol) becomes kernel seams per 120's policy (§7).

## 4. The layers

### 4.1 Tensors as IR values

- `Array`-typed SSA values: produced by `tensor.*` ops, tile literals
  (`zeros(("m","n"), f32)`), and (per family contract) array *arguments*;
  consumed by `tensor.*` ops, `isel`/`slice`, and DPS results.
- **Results are DPS out-arrays** (the 070/090 decision finally cashed):
  a kernel whose result type is an Array writes into a caller-provided
  destination riding the leaves channel (`Out` already exists); the
  ResultPlan mirror already speaks buffer leaves. NO shape inference —
  out shapes come from declared out-shape rules (070 decision 4).
- Elementwise arithmetic on same-axes tensors; broadcast ONLY against
  scalars (axis alignment is by NAME and must be exact — the pedantic
  posture extends: no positional broadcasting, ever).

### 4.2 Loop semantics + the tensor dialect

- `core.for` attrs: `kind` (`"seq"` default — every existing loop is
  unchanged), `"map"`, `"reduce"` (+ `combine` op-name attr); optional
  `axis` name. Type rule: `map` yields Array (carry type × extent axis),
  `reduce` yields the carry type. Attrs are compile-time constants inside
  identity — no new region op, the 4th-region-op price stays unpaid.
- `tensor.map` / `tensor.reduce` / `tensor.contract` / `tensor.slice`:
  satellite ops (surface A) whose *default decomposition* is the
  corresponding `core.for` form — so every existing scalar target runs
  them TODAY via the decomposition gate (`op ∉ code_for_op`), while a
  tensor-capable target spells them natively (mma, warp reductions).
  `contract` generalizes step-12's matmul: pair the unique shared axis
  name, contract it, output the survivors — the pairing function is
  already written.
- In-kernel surface: **named comprehensions** (the step-8 comprehension
  question, answered): `sum(f(k) for k in axis("inner"))` →
  `tensor.reduce`; `array(f(k) for k in axis("inner"))` → `tensor.map`.
  Reads as math, carries the axis name, never mentions batch.

### 4.3 `over` v2 and execution shapes

`over(f, axis="batch")` produces `for {kind: "map", axis: "batch"}`
around the base body. Lowerings, per target:

- **CPU (C backend)**: a real loop *inside the artifact* — one dispatch
  per batch, not per lane; the compiler auto-vectorizes the lane loop.
  (This alone repays the step: the ray-march verdict said per-lane
  dispatch drowns a 7× body win.)
- **GPU compute**: the axis joins the launch domain (the compute family's
  `out=(W,H)` was always this).
- **Python twin**: the honest interpreter loop.
- Scalar-target *weaving* (step 12's mechanism) remains valid for
  captures-only cases and as the reference lowering.

Composition (`over ∘ over`, `jvp ∘ over`) becomes ordinary once transforms
lower through the derived door with merged contexts (§7) — the woven-map
merge that 110 §5 deferred.

### 4.4 Memory spaces, layouts, tokens, precision

- `Array` grows `space` (`"global"|"shared"|"reg"`) beside `device`, and
  layout refinements (padded strides — the bank-conflict "prime sizes"
  answer is a LAYOUT ATTRIBUTE, not user arithmetic; `format` for
  structured sparsity like 2:4). All type-side: staging is a typed
  conversion, visible in IR and plans.
- `stage(x.tile(...), pad="conflict-free")` → `tile.stage` (returns tile
  + token); `barrier()` → `tile.barrier` (token → token); cooperative ops
  consume tokens. **Purity preserved as dataflow**: ordering is SSA, the
  rewrite driver needs no effect system, and a misordered kernel is a
  *type* error (missing token), not a race.
- Precision: `f16`/`bf16`/`f8e4m3`… join `SCALAR_KINDS` gated by 090/R15
  capability flags; `astype` is the existing `core.cast`. Accumulate-in-
  f32-emit-f16 is then just types doing their job.
- Tile sizes (TM/TN/TK) are **config-bracket values** (040 §3c: block
  value-specializes) — the schema seat reserved at step 9, now needed.

## 5. The centerpiece: a modern GEMM, as it should read

```python
def make_gemm(A, B, bias):                    # A: (m,k) f16 · B: (k,n) f16 — named
    @jit(kind="gpu.tile")                     # TILE family: params are tile coords
    def gemm(tm, tn):
        acc = zeros(("m", "n"), f32)          # register tile, axes named, f32 accum
        for kb in tiles("k"):                 # sequential BLOCKED loop over axis k
            a = stage(A.tile(m=tm, k=kb), pad="conflict-free")   # global→shared
            b = stage(B.tile(k=kb, n=tn))
            barrier()
            acc = acc + contract(a, b, axis="k")   # named contraction ≡ tensor cores
        acc = relu(acc + bias.tile(n=tn))     # elementwise epilogue on the tile
        return acc.astype(f16)                # precision cast at the edge

    return gemm

g = over(make_gemm(A3, B3, bias), axis="batch")   # batched GEMM: one more grid axis
```

What the reader should notice: the programming unit is the **tile, never
the thread** (how 256 threads split `stage`, and how warps own mma
fragments, is the backend's decomposition — the ThunderKittens/Triton
lesson); `contract` maps 1:1 onto `mma.sync` with no pattern recovery;
`over` of a tile kernel is *efficient by construction* because every axis
is already launch-domain-shaped; and nothing in the source names a warp, a
bank, or a fragment shape.

## 6. Warp-level primitives, reductions, sorting

Two tiers, nothing between:

1. **Named tile ops**: `reduce(x, axis="n", op=max)`, `scan`, `sort(x,
   axis="n")` — lowered per target to warp-shuffle trees / `simd_sum` /
   subgroup ops / sequential loops. The user never writes a shuffle.
   Sorting and top-k are *library kernels* in this vocabulary (a family
   package, not IR).
2. **The escape hatch** (090, unchanged): `cuda.shfl_down(x, 8)`,
   `cuda.ballot(p)`, `metal.simd_sum(x)` — vendor-namespace ops, spelled
   by one backend, never decomposed; a visible portability opt-out for
   hand-choreographed algorithms.

## 7. The seams this buys from the kernel (120 policy applied)

Measured motivation: 13 string-keyed context-door sites
(`__woven__`, `__root_argc__`, `__tangents__`, `__woven_hits__`,
`__registry__`), one duplicated lower tail (`_lower_base` ≙ `build_pipe`),
three copies of the wrapper protocol — all clustered at ONE seam.
Step-12's review traced four of its nine bug classes to exactly this
cluster. Per 120's inversion ("a satellite needing hooks must ask for a
seam"), the asks:

| Seam | Where | Est. lines | Kills |
| --- | --- | --- | --- |
| `Lowerer.context` (typed dict threaded through `inline`; root params/argcount exposed) | `lower.py` | ~20 | all five string doors, the root-argc gymnastics |
| `lower_handle(prefix=…)` + derived-base re-entry | `lower.py` | ~10 | `_lower_base`/`build_pipe` duplication; UNBLOCKS transform composition |
| `DerivedValue` protocol + one ValueKind | `capture.py` or new | ~25 | Pipeline/VMapped/Jvp triplication; the missed-kind bug class |
| Binder allocation (deterministic tuple indices) | `ir.py` Builder | ~8 | the satellite-owned soundness invariant |
| `core.for` kind/reduce type rules | `ops.py` | ~10 | — (new capability) |
| Array-result ABI (ResultPlan buffer leaves; launcher adoption) | `pack.py`/`registry.py` | ~15 | the array-results cut |
| Token type | `types.py` | ~3 | — (new capability) |

Projected kernel: 1229 + ~90 ≈ **1320 / 1500** — inside the tripwire
budget, each raise ledgered per policy. Tensor dialect, comprehension
lowering, `over` v2, and all tile vocabulary are SATELLITES (stdlib has
500 free lines; `backends/` raises at step 14 as already noted). Shape
and axis analyses use **`Memo`** (120 §7 named them as intended clients),
so they are instrumented by construction; tensor-lowering phases get
`events.span`s so `events.expect` can gate compile-cost regressions.

## 8. Staged plan (each stage independently green, per 020 discipline)

1. **Seams + `over`** — the kernel refactor of §7's first four rows;
   rebuild transform doors on them (string keys die); rename vmap→`over`;
   composition lands (woven-map merge, duplicate-axis refusal); ch13
   updated. *Gate:* step-12 test suite green with the doors deleted;
   composition tests replace the refusal tests.
2. **Tensor values on CPU** — array results (DPS) + array args;
   `core.for` kinds; `tensor.map/reduce/contract` + comprehensions
   lowering to loops on python/C; `over` v2 emits map-loop IR; C gains
   the in-artifact lane loop. *Gate:* GPT-2-shaped attention block
   (unbatched source, `over`-batched) beats the per-lane spike by ≥10×
   on C; softmax via comprehensions matches numpy; chapter.
3. **Grad** (old step 13, resequenced AFTER tensors deliberately) —
   adjoints of `contract`/`reduce`/`map` are textbook, versus
   differentiating hand-tiled loop nests; the jvp column extends;
   transpose + partial eval; `D` through loops gets tangent carries.
4. **CUDA + tiles** (old step 14) — `backends/cuda` on cuda.core (the
   user's box, handoff-doc + parallel agent per 090 §6); the tile family,
   `stage`/`barrier`/tokens, mma-backed `contract`; the §5 GEMM becomes a
   chapter with tensor-core numbers; Metal follows.

## 9. Open questions

1. `map`-kind `core.for` vs a `tensor.map` region op — attrs on `core.for`
   is budget-cheap but overloads one op with three semantics; revisit if
   transform columns (grad over map-loops) get awkward. Priced fallback:
   the 4th region op at its known cost.
2. Comprehension syntax: `sum(f(k) for k in axis("inner"))` needs
   GeneratorExp lowering — is `axis("name")` a good iterator marker, or
   should extents always be named via the arrays (`for k in A.axis("k")`)?
3. Does `over` take multiple axes (`over(f, axes=("batch","head"))`) as
   the primitive, with composition as sugar — or compose only?
4. Token granularity: one token stream per shared allocation, or one per
   block? (Per-allocation is finer than hardware barriers; start coarse.)
5. Where does the epilogue fusion live — `contract` with a fused
   elementwise region (linalg-style), or scheduler-recognized adjacency?
   Start unfused; measure.
6. f8 formats and 2:4 sparsity: gate behind capability flags from day one,
   or defer entirely to the CUDA step? (Types are cheap; recommend flags
   from day one, lowering deferred.)
