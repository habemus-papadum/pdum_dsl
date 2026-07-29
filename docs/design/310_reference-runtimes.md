# 310 — Reference runtimes: third-party columns before tiling

**Status: owner-ratified direction (2026-07-28); the torch AND jax
columns are LANDED.** Before the tiling era opens, the stack grows reference points
it intends to exceed: framework-served runtime columns (torch, then
JAX), a benchmarking discipline over them, and a cuda.tile
reconnaissance probe. 270's arc is unchanged — whole-program analysis
and our own CUDA remain the destination; these are the measuring sticks
carried along, never foundations built upon.

## The two-column law

Every framework contributes TWO artifacts, and conflating them voids
both:

- **The CHECK column** (`conformance/torch_evaluator.py`): a
  graph-level interpreter over dialect regions — the SAME region the
  numpy reference serves, evaluated on the framework's substrate. Its
  assertion is the zoo's own denotation at the interpreter tolerance
  (rtol 1e-9 / atol 1e-12, f64), whole-chain on the device column
  (PR #8's rule). It exists for correctness assurance of compilation
  stages against an independent implementation.
- **The BASELINE column** (`conformance/torch_zoo.py`): what a fluent
  framework author writes today — sdpa, F.layer_norm, index_add —
  independently authored from the entry's math, never a port of the
  interpreter. It is the performance bar the compiled stack must meet,
  and a third authorship angle on the denotation (asserted under its
  own stated tolerance).

## The evaluator's mechanism (why it is small)

The type rules already ran at region construction, so every node's type
carries its result dims — name, start, stop, in presentation order. The
evaluator therefore computes DATA only: each node evaluates to a dense
tensor whose axes are exactly its type dims. Layout ops that move
coordinates but never values (shift, rename, with_charts, ...) are data
no-ops whose entire effect lives in the type. Composite markers and
reducers are marker-DSL trees, so `zoo.gelu` or `zoo.flashsm` evaluate
through the same declarations the reference reads — one primitive table
per substrate is the whole porting surface. Uncovered ops raise
`Untranslatable` naming the op (wgsl_executor's law); the zoo's full
vocabulary (19 tl ops + 4 core) is covered.

ONE core, per-framework hooks (283's emitter philosophy, and the line
discipline's answer): the dims machinery, the take/scatter/fold
algorithms, and the dispatch live once in
`conformance/region_evaluator.py`; a framework column contributes a
`Substrate` — ~16 array hooks plus the marker table (~120 lines each for
torch and jax). Where the frameworks genuinely agree (operators, basic
indexing, advanced-indexing gather) the core uses the shared spelling;
hooks exist only for true divergences (mutation vs `.at[]`, axis-op
namespaces, x64 policy).

## Placement: conformance now, rt Pair at the seam

The columns live under `conformance/` today — the conformance-executor
doctrine's home. When the graphics team's `pdum.rt` skeleton lands
(283), they mount as a Pair behind the selection door, and THAT step is
the recorded, conscious supersession of the translation-only,
conformance-only placement (owner's ruling, 2026-07-28). The Pair must
stay interpreter-open (the PR #9 comment): these generators return
callables over arrays, not source text, and every LaunchContract clause
degenerates honestly — no thread_size, guard moot, bindings are
parameter order, math rows are the framework's libm. `TORCH_FP` already
carries the torch version so the content door keys artifacts correctly
on arrival.

Dependencies are OPT-IN groups (`--group torch`), never default: the
core stays numpy+wgpu and the 21-second gate is untouched (framework
tests importorskip). CI runs the columns anyway — as their OWN job
(`conformance-frameworks` in ci.yml) on CPU-wheel twins of the groups
(`torch-cpu` pins PyTorch's CPU index conditionally; `jax-cpu` is plain
jax), so substrate correctness gates without multi-GB CUDA stacks on
hosted runners. The batteries' cuda halves skip there and light up
unchanged the day a GPU worker exists.

## The benchmark discipline (`benchmarks/`)

Outside the default pytest gate on purpose — benchmarks run
deliberately, never as a suite side effect. Two laws are load-bearing:
**never benchmark a wrong program** (every column asserts against the
entry's denotation before its timed loop) and **never benchmark a
recompile** (timed loops run under `events.forbid` on every cache-miss
event — trivially green while the columns are cacheless interpreters,
doing real work the day they mount as Pairs with artifact caches).

Two measurement levels, matching the campaign's two questions: the
assemblage/fusion level (this rig — e.g. the flash entry's composite
reducer runs ~50x behind fused sdpa on CUDA, which is precisely the
fusion gap made visible) and the tiling level (arrives with the tiling
era; the gemm/heat2d entries are its subjects).

## The dtype axis

Dtype is representation, never semantics (200 §4), so it is a PARAMETER
of the columns, not a variant of an entry: the substrates carry their
float width, `--dtype f32` runs the framework columns at the dtype the
hardware actually competes at (the 4090 runs f64 at 1/64 rate — f64
boards understate the machine). Three laws hold the axis honest:

- **conformance stays tight**: `test_f32_columns.py` asserts the f32
  CHECK columns against the f64 oracle under stated tolerances (first
  subjects heat2d and gemm — pointwise/reduce chains, no tensor-core
  paths), and asserts the output dtype IS f32 so a silent f64
  promotion cannot pass;
- **the rig's f32 verification is scale-aware sanity**: XLA routes f32
  matmuls through TF32 tensor cores whose error rides the output's
  magnitude (torch keeps matmul TF32 off by default) — those defaults
  ARE the benchmarked thing, so the check separates wrong-program from
  tensor-core rounding at 1e-2 of the oracle's scale;
- **discontinuous entries refuse at f32**: moe's top-k routing flips
  when a rounded logit crosses a choice boundary (observed: one token
  of 64 rerouted), so no tolerance separates rounding from a different
  program — the rig refuses the entry VISIBLY (the ledger's law).

## Sequence

1. **torch** — landed: evaluator, 12-entry zoo differential (CPU+CUDA),
   9 idiomatic baselines, the rig.
2. **jax** — landed (`jax[cuda13]` group): same two columns; the whole
   zoo passed the substrate-core port on the first run. The rig's first
   four-column signal: jitted-jax is the strongest baseline on most
   entries, while the EAGER jax translated column pays XLA's per-op
   dispatch tax (up to ~25x slower than eager torch on the same
   region, below the numpy reference on flash/moe) — the measured
   motivation for step 3.
3. **graph-level compile wrap** — `torch.compile` / `jax.jit` around
   the translated callable: the level-1 fusion bar for free.
4. **cuda.tile probe** — reconnaissance for our own tiling design
   (283 §5's bend-point), not a commitment.
