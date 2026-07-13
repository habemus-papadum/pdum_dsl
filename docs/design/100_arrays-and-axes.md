# 100 — Arrays & axes: data, loops, and the pedantic indexing surface

**Status:** canon (2026-07-12, step 11 opening). Companions: 090 §5 (the
buffer/interop contract this implements), 080 (backend resolution tiers),
research/R10–R11 (construct surveys). Consumed by step 11; the xarray
exercise is user-directed ("the most pedantic indexing possible, as long as
the machine code is efficient").

## 1. The array type algebra

One base summary, two satellite refinements — all three are frozen
structural types, so *everything* below is cache-key material and nothing
below costs anything at runtime:

- **`Array(dtype, ndim, layout, byteorder, writeable, device)`** (kernel,
  §13's rank-generic default). `device` is the 090 dispatch axis, landing
  now as a field with default `"cpu"` — the ONE kernel line this step
  spends (ledger). With only CPU targets today the resolution wiring stays
  routed/explicit; when a second device exists (step 14), device-in-type
  becomes dispatch tier 1 with zero further type changes.
- **`ShapedArray(Array, shape)`** (satellite) — the §13 dial turned one
  notch: the full shape enters the type. Cost: one specialization per
  shape. Payoff: strides are compile-time constants (const-folded
  indexing, no shape/stride slots in staging). The chapter shows both
  regimes with compile counters; neither is "right" — the dial is the
  point.
- **`NamedArray(Array, dims)`** (satellite) — axis *names* in the type,
  the xarray exercise. Names are strings in a frozen summary: they ride
  fingerprints and lower-time checking and are GONE by codegen — the
  machine code for named and positional indexing is identical. This is
  the thesis working for us: pedantry that would cost a runtime dict
  lookup elsewhere costs nothing here because names live on the
  types-not-values side of the line.

## 2. The indexing decision (the pedantic pick)

Surveyed shapes: NumPy positional `a[i, j]`; xarray's name-keyed
`a.isel(y=i, x=j)` / label-based `a.sel(...)`; einops strings; Dex/Futhark
index-set types (each axis its own index *type* — the theoretical maximum
of pedantry). The pick:

- **Anonymous arrays index positionally**: `a[i, j]`, rank-checked (every
  axis, exactly once — no partial indexing, no views in v1), indices
  strictly `i64`.
- **Named arrays index by name, MANDATORY**: `a.isel(y=i, x=j)` — keyword
  arguments only, every axis named exactly once, unknown or missing names
  are loud lower-time errors listing the type's axes. **Positional
  indexing on a named array is REFUSED**: transposition is precisely the
  bug names exist to kill, so offering the positional back door would buy
  convenience with the entire safety budget. (This is the pedantic choice;
  xarray itself still allows positional. We are deliberately stricter.)
- Keyword syntax rides `isel` *method* form because Python has no
  subscript keywords (PEP 637 was rejected). We adopt xarray's own verb —
  familiar to the ecosystem, and honest: integer selection by axis name.
- **`sel` (label-based lookup) is deferred, with a reason**: coordinate
  labels are *values* (a coords table), and label→position lookup is data-
  dependent work that belongs on the host or in a real gather kernel. A
  future `sel` on a `Literal`-lifted coordinate could const-fold — noted,
  not built.
- Dex-grade index-set types (an index value *typed by its axis*) are the
  logical endpoint; v1 stops at name-checked call sites because our index
  values are ordinary `i64` scalars produced by arithmetic. Revisit when
  transforms mint index values worth tagging.

## 3. Marshaling: how an array crosses the seam

The kernel already left the door: `BufferLeaf` exists, `dest=None` slots
travel the leaves channel, and `legalize_params` refuses leaves-channel
captures with a message pointing at the kind that must supply the binding
op. Arrays are inputs-only in v1 (captures or args); results stay scalar/
tuple DPS.

- **Leaves of `Array`**: one `BufferLeaf` (the payload → leaves channel)
  plus `ScalarLeaf("i64")` per axis for shape and per axis for strides
  *in elements* (→ staging bytes, packed per call — that is what makes the
  rank-generic regime hit the cache across shapes). `ShapeLeaf`/
  `StrideLeaf` remain reserved for backends with dedicated dest
  vocabularies (the WGSL dims-uniform case); using plain i64 scalar slots
  today is zero-kernel-edit and the plan's leaf *paths* preserve which
  slot is which.
- **Leaves of `ShapedArray`**: the `BufferLeaf` alone — shape and strides
  are in the type, so indexing const-folds and staging shrinks.
- **The binding op**: indexing lowers to `array.load(buf, linear)` where
  `buf` is `array.buffer` (attrs: the env/arg path) and `linear` is
  explicit stride arithmetic over `core.env`/`core.param` i64 reads that
  `legalize_params` turns into ordinary `abi.slot`s. Renderers resolve
  `array.buffer`'s path to a leaves-channel position via the plan. No new
  legalize stage, no kernel edits: the "array" namespace is legal because
  registered ops' namespaces already survive to render.
- **Adoption (090 both-directions contract, v1 slice)**: NumPy arrays are
  adopted zero-copy, **C-contiguous only** — a non-contiguous view is a
  loud `TypeError` at the def site ("copy it or wait for views"). The
  holder keeps the array referenced; in-place mutation of *contents* is
  visible to the next call by construction (the pointer travels fresh
  every launch — buffers are data, like uniform values, never identity).
  xarray's `DataArray` adopts through its `.data` NumPy payload, dims →
  `NamedArray.dims`.

## 4. Statements: `if`, `for`, and the single tail return

Base-pack widening (recorded for ch12a):

- **`if`/`elif`/`else` statements** lower to `core.if` yielding the JOIN
  of rebound locals: every name assigned in either branch must have the
  same type on both paths (strict join — no unification, no numba
  fixpoint; this is the §13 "what type inference is here" promise kept).
  A name assigned in only one branch must already exist outside (the
  other path keeps the old value) — else refused.
- **`for i in range(n)` / `range(lo, hi)`** lowers to `core.for` with
  loop-carried values = the pre-existing locals the body rebinds. Bounds
  are strict `i64`. Names *created* inside the body die with it. `range`
  step, `while`, `break`, `continue`: refused loudly (bounded loops only —
  the GPU-honest subset; R11's kernel-DSL survey says every serious
  kernel language draws this line).
- **Single tail return, enforced**: `return` may appear only as the
  function body's final statement. A `return` inside a branch or loop is
  a loud error naming the policy (nearer Taichi's strictness than numba's
  unification — settled at the ch07a matrix review). `core.yield` IS the
  return.

## 5. The C backend (the seam beyond GPUs)

`backends/c.py` — the first *citizen* of the contribution point (the
namespace package stops being infra-only):

- **Render**: legalized Region → C99 via the shared dominator walker
  (`_emit` grows `core.for` emission — loop binders join `core.if`
  branches as the second region construct it knows). Scalars: staging
  reads by offset (`*(double*)(staging + 24)`); buffers: `bufs[k]`
  pointers; math intrinsics spell to `<math.h>` (`sqrt`, `fabs`, `fmin`…).
- **Compile**: source → `cc -O2 -shared -fPIC` (probe `cc`/`clang`;
  `is_available()` gates tests and notebook cells) → `ctypes.CDLL`, cached
  content-addressed like every artifact.
- **ABI**: `double kernel(const char* staging, void** bufs)` — scalar
  returns only in v1 (tuple results refused loudly; they are WGSL's
  fragment territory and DPS-array territory later).
- **Launcher**: peels `Out` (refused — no launch domain on a scalar CPU
  kernel), passes staging bytes + buffer pointers.
- The point is not speed; it is N=3 render targets over ONE IR, and a
  second *runtime* shape (dlopen/ctypes vs exec vs wgpu) under the same
  `launch(staging, leaves)` — evidence for 090 §4 before step 14 abstracts
  it.

## 6. Scope cuts (recorded, not regretted)

Views/slicing, partial `isel` (both produce arrays, not scalars — they
arrive with array *results*), `sel`, array writes in kernel bodies (DPS
out-arrays: needs store legality + aliasing rules — with chaining),
result-side arrays (`rebuild` for Array raises with this pointer),
non-contiguous adoption, `range` step, `while`. Each is one satellite
registration away when its step comes; none blocks the ray-march verdict.
