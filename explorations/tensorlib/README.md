# tensorlib

A design exploration for a tensor library about **memory layout and views**,
plus a minimal compute layer over it. See [DESIGN.md](DESIGN.md) for the
layout-layer decisions (D1–D18), [COMPUTE.md](COMPUTE.md) for the compute
vision, [LEAN.md](LEAN.md) for the Lean 4 modeling diary,
[REPRESENTATIONS.md](REPRESENTATIONS.md) for the memory/scheduling levels
above the IR, and [CONCERNS.md](CONCERNS.md) for the open edges.

## The three steps

**Step 1 — the core family** (`layout.py`, `tensor.py`, `buffer.py`): a
layout is an affine map plus a box domain,

```
loc(coords) = offset + sum(stride_d * i_d),    start_d <= i_d < stop_d
```

with named dims, fully expanded byte strides (zero = repeat, negative =
flip), and raw coordinates (slicing shrinks domains, never renumbers). Ops:
`slice`, `select`, `shift`, `rename`, `repeat`, `flip`, `split`
(blocking), `merge`, `diagonal` (n-ary; labeling caller-supplied, never
guessed — `characteristic(rate, ...)` for rate diagonals), `window` (with
`dilation`), `decimate`, `field`; analyses: `footprint`, `injectivity`,
`overlaps`, `check`, `is_contiguous`, `canonical`, and
`alignment(*tensors)` — a DIAGNOSIS of what stops operands from being
elementwise-combinable, with the exact fixing primitive per item; the
library never applies the fixes (aligning is the caller's conscious act).
There is deliberately no permute: dim order carries no meaning, so export
order is a parameter of materialization (`to_numpy(order=...)`) and
order-free identity is a query (`canonical()`).

**Step 2 — the guarded extension** (`guarded.py`): affine map + box +
linear-form guards + fill value,

```
lo <= sum(c_d * i_d) < hi        (else: fill)
```

built by `pad` (guard on one coordinate) and `stencil` (guard on
`x + dilation·x_k`). Every view op rewrites guards algebraically — including
flip, diagonal (padded/banded diagonals work), window, decimate, and merge
(conditionally: guard coefficients must be proportional to the mixed-radix
weights) — and `simplify` discharges them by interval arithmetic, collapsing
back to the core family when the boundary is sliced away.

**Step 3 — exact units and charts** (`units.py`, `chart.py`): a dim may carry
a Chart — `phys(i) = origin + i·step` with exact rational Quantities and
units — making physical indexing (`t.item(x=q("0.75 um"))`), physical
slicing, displacement-labeled kernels, and frame ops (`shift` vs `recenter`)
all exact. The lattice and guard algebra are untouched: plain ints always
mean lattice coordinates (*compiler mode*), and `strip_charts()` drops the
labeling entirely. `value_units` labels the value space (per-field for
structured dtypes) as metadata for the future compute layer. Magnitudes are
stdlib `fractions.Fraction` — floats are rejected everywhere; construct with
`q("0.75 um")`, `3 * u.mm`, `Fraction(3,4) * u.um`, or a PEP 750 t-string
`q(t"{Fraction(3,4)} um")`. A dim may instead carry categorical `labels`
(`with_labels(c=("R","G","B"))`) — the nominal rung of the measurement
ladder: names with no arithmetic, upgradeable to a chart later.

**The compute reference layer** (`compute.py`, vision in
[COMPUTE.md](COMPUTE.md)): three primitives — `pointwise(f, A, B, ...)`
(operands must be 100% aligned; the error quotes the `alignment()` recipes),
`reduce(f, A, dims)`, and `scan(f, A, dim)` (prefix reduce; reverse =
flip∘scan∘flip) — plus `iota`: coordinates as data, built TIGHT as a
`FunctionalBuffer` (no memory; exact rational read function; closed-form
under every view op — layout ops cannot destroy iota-ness). Tensors carry a
`carrier` (bool/int/rat/real/complex): the algebraic object values
approximate — semantics; the dtype is mere representation. `f`s are
declared markers (`pw.*`, `red.*`), not callbacks — and the **marker DSL**
(`mdsl.py`, zero imports from the main pdum package) extends them:
`defmarker("sigmoid", 1, lambda x: 1/(1+exp(-x)))` traces a plain lambda
into an owned expression-tree IR whose partial derivatives are DERIVED by
tree rewriting (new activations differentiate automatically), and
`defreducer` defines structured-state reducers — the SSM recurrence
h_t = a_t·h_{t-1} + b_t runs as an associative pair-combine scan over
multiple aligned inputs — and differentiates: the SSM backward pass is
emitted as IR (the state cotangent is itself a reversed-time linear
recurrence over derived Jacobian trees). Markers carry SIGNATURES
(signatures.py): carrier/unit facts propagate through trees and programs,
`exp` of micrometers refuses, and `grad` infers `target_unit` on its own.
`ops_count` (opcount.py) tallies exact per-primitive operation Counters
("mul"/"add"/"exp"/"copy"), with MAC fusion and cost models as separate,
explicit steps. Frontends are pluggable producers of the Node schema; the
main repo's syntax tooling can target it later without any rewrite.
Beyond scalar-tuple state, `fold` is the TENSOR-state scan — the step is
itself an IR Program (state = named tensors with a fixed-layout carry
contract), covering SSM matrix states (Mamba-2/DeltaNet-style gated
linear attention) and PDE time-stepping (FDTD leapfrog) with one
combinator; its adjoint is derived by differentiating the step program
and folding the VJP backward. LEVELS.md holds the machine-modeling
roadmap (representation ladder × machine tree) these feed into; its first
two rungs are real: the MODEL ZOO (`tensorlib.zoo`: GPT-2, a Llama block
with RoPE/GQA/SwiGLU, sliding-window/gated/QK-norm attention, the
online-softmax flash reducer with a DERIVED backward, 2D heat, and 1D FDTD
on an exactly-charted staggered grid — every entry checked against a pure
numpy denotation) and the L1 PEAK-MEMORY SIMULATOR (`peak_memory`: layout
ops are zero-byte aliases, iota/const/masks are free, the schedule is an
optimizable argument, folds simulate their step recursively).
Deliberately inefficient numpy semantics: repeats and windows materialize —
that is the correctness contract a real backend must match while treating
those views as virtual. Matmul = repeat·mul·reduce; conv = window/stencil
·mul·reduce; softmax = reduce→repeat→pointwise; causal masks via iota —
all pinned in `tests/test_compute.py`.

**The IR and autodiff** (`ir.py`, `autodiff.py`): programs as linear SSA —
(var, op, operands, params) over layout ops + the primitives, plus leaves
(`input`/`const`/`iota`) and `materialize` (identity copy in a chosen dim
order). `run` interprets over the reference layer; `infer` propagates
layouts only. `grad(prog, target, input_layouts, seed=None)` is reverse-mode
AD as a program transformation: one backward pass yields d(target)/d(v) for
every SSA var, the generated gradient is itself IR, and every adjoint rule
is validated against finite differences in `tests/test_autodiff.py`
(repeat†=reduce-sum, slice†=pad, window/stencil†=overlap-add,
decimate†=zero-stuffing, scan(sum)†=reverse scan, ...). Scalar targets seed
with 1; non-scalar targets require an explicit seed (VJPs).

## Quick taste

```python
import numpy as np
from tensorlib import Tensor, q, u

t = Tensor.from_numpy(np.arange(16), ("x",)).with_charts(x=("0 um", "0.25 um"))
t.item(x=q("0.75 um"))                  # exact physical indexing
b = t.split("x", xb=4, xi=4)            # blocks: xb is pos[.. step 1 um],
b.item(xb=q("2 um"), xi=q("0.75 um"))   # xi a displacement within the block
s = t.stencil("x", k=(q("-0.25 um"), q("0.25 um")), fill=0)  # taps at ±step
```

## Walkthrough notebooks

`notebooks/` contains an executed series — each operation shown as a
before/after on the layout (dims, strides, offset, charts, guards):

0. `00_units_and_quantities.ipynb` — the exact unit system: construction,
   arithmetic, t-strings, float rejection, `define`.
1. `01_buffers_layouts_tensors.ipynb` — buffers are bytes; layouts by hand;
   the `get_loc` contract; charts as physical labels; both faces of
   indexing; compiler mode.
2. `02_view_ops.ipynb` — slice / select / shift / recenter / rename / flip /
   repeat with physical coordinates, and why there is no permute.
3. `03_restructuring_ops.ipynb` — split (block position + within-block
   displacement), merge, decimate, alignment diagnosis, diagonal (incl. the
   rate combinator), window, field selection with value units, categorical
   labels.
4. `04_guarded_layouts.ipynb` — pad, stencil with physical taps, guard
   rewrites under select/shift/split, `simplify` collapsing back to the core
   family.
5. `05_autodiff_cheatsheet.ipynb` — the adjoint of every instruction, each
   with a one-line inner-product derivation and finite-difference
   validation; the chain rule mechanically; seeds as VJPs; softmax's
   analytic gradient; training a linear model end to end in the IR.
6. `06_adjoints_from_scratch.ipynb` — the conceptual prequel to 05,
   library-free: sensitivities, path-sums, and THE one rule
   (⟨ȳ, Op dx⟩ = ⟨Op†ȳ, dx⟩ — swap the double sum), with the whole adjoint
   zoo derived by hand and verified by the pairing test alone.
7. `07_the_marker_dsl.ipynb` — the marker DSL in action: tracing lambdas
   into the Node schema, partials derived by tree rewriting (activations
   invented in-cell differentiate on first use), gradient-free positions and
   unit signatures, structured-state reducers (cumsum, the SSM pair
   recurrence), associativity as a declared claim, BPTT emitted as IR (with
   the derived registry and the generated matrix-linrec backward scan),
   training through the scan, and op counts over composite trees.
8. `08_fold_tensor_state.ipynb` — `fold`, the tensor-state scan: the step as
   a first-class IR Program, the scalar case agreeing with 07's composite
   reducer, gated linear attention's matrix state, the carry/`final`/chart
   refusals, the adjoint derived by self-application (the folds the backward
   pass generates), FDTD leapfrog with the space-time trajectory and
   gradients w.r.t. the initial fields, the empty fold, per-step op counts,
   and units surviving the carry fixed point.
9. `09_the_model_zoo.ipynb` — `tensorlib.zoo`: L0 surface programs kept
   honest by an independent numpy denotation. The `Build` name-manager (not a
   frontend); GPT-2 as ordinary IR with heads born as dims (never splits);
   the Llama block's RoPE-without-splits (pairs born in the weights, rotation
   = selects + trig, scores = sum of two u-slot contractions) and GQA as
   repeat-by-declaration; flash attention's online-softmax defreducer, equal
   to naive in forward AND gradients (the backward DERIVED, no hand rule);
   physics — heat2d's Dirichlet ghosts as pad guards, and staggered FDTD with
   exact half-integer charts (the differencing-misalignment refusal, the
   recharting, gradients back on the right grids); the recorded boundaries.
10. `10_peak_memory.ipynb` — the L1 footprint simulator
    (`memory.peak_memory`): the timeline and the peak; views are free but keep
    their root alive; closed forms (iota/const/masks) cost nothing; THE
    SCHEDULE IS AN ARGUMENT (two topological orders, 144 vs 200 bytes); folds
    simulate their step recursively; GPT-2 forward vs forward+backward (the
    grad joint peaks ~2.4x higher — the saved-activations problem that
    motivates DCE/checkpointing); and the model's honest coarseness.

Re-run them with
`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb notebooks/1*.ipynb`.

## Running the tests

```bash
uv run pytest explorations/tensorlib/tests -q
```

(`bfloat16` support activates when `ml_dtypes` is installed; one test skips
without it. No other dependencies — exact rationals are stdlib
`fractions.Fraction`.)

## Deliberately not sketched yet

- The **piecewise family** (roll, reflect/circular padding,
  concat/interleave-as-view): non-affine as single views; each is a union of
  guarded affine pieces. See CONCERNS.md #1.
- Sub-lattice rate diagonals: `characteristic(rate, ...)` covers the case
  where the lattice steps match the rate exactly; commensurable-but-unequal
  steps need a conscious decimate-and-diagonal composition (CONCERNS #5).
- Non-uniform numeric coordinates (city longitudes, irregular sample times)
  — the missing rung between categorical labels and affine charts
  (CONCERNS #14).
- `coalesce` (auto-merging adjacent dims) — needs a naming policy.
- `materialize(tensor, new_layout)` — specified as the boundary with the
  future compute layer; only the naive testing read path (`item`,
  `to_numpy`) exists here.
- Writes through views (would be gated on `injectivity()` per D6).
- Real device buffers / DLPack import-export (Buffer carries `device` but
  only host buffers are readable).
- See CONCERNS.md for the step-3 specific edges.
