# tensorlib — design exploration

A library about **memory layout and views of data**. There is no computational
layer here; that comes later, on top. This layer's job is to describe, with
complete algebraic transparency, *where things live* — so that a compiler
(e.g. pdum.dsl) can reason about addresses symbolically.

## 1. Core objects

### Buffer

A rank-1, featureless run of storage:

- a pointer (or handle),
- an address space (host, `cuda:0`, WebGPU device, …),
- a length.

No shape, no strides, no dimension semantics — just a 1-D extent.
DLPack is the natural interchange format for importing/exporting these.

Open point: is a Buffer raw *bytes*, or typed elements? (See decision D4.)

### DType

- numpy dtypes as the base vocabulary, including **structured dtypes with
  padding** between fields;
- extended float types (bfloat16 etc.) via `ml_dtypes`, which plugs into the
  numpy dtype system.

### Layout

A layout is:

- an **offset** into the buffer (possibly zero; whether we need it at all is
  decision D2), plus
- a set of **named dimensions**, each with:
  - a **name** (dimension identity is by name, not position),
  - a **stride** — *fully expanded*, in units of the dtype (D4 discusses
    bytes vs elements). May be **zero** (repeats/broadcast). Nothing in the
    formulation forbids negative strides (flips).
  - a **min index** and **max index** — the coordinate domain. Indices need
    not start at 0 and may be negative. Crucially, a negative index here is a
    *literal coordinate*, not Python's "count from the end".
    (Inclusive vs half-open max is decision D1.)

"Fully expanded" means: strides are explicit per-dimension constants, e.g. for
a conventional dense matrix, the first dimension has stride 1 and the second
has stride `count(first)`, a third would have stride `count(first)·count(second)`,
and so on. Because every stride is explicit, **row-major vs column-major
disappears as a concept** — memory order is entirely carried by the strides,
and dimension *order* carries no addressing semantics at all (D5).

### Tensor

    Tensor = Buffer + Layout + DType

## 2. The layout contract

The semantic ground truth of the whole library:

    layout.get_loc(x=3, y=-1)  ->  location within the buffer

i.e. a layout is a function from a *named coordinate tuple* (each coordinate
within its dimension's [min, max] domain) to a buffer location, respecting the
offset. For the core layout family this function is **affine**:

    loc(i) = offset + Σ_d  stride_d · i_d

(whether the raw coordinate `i_d` or the normalized `i_d − min_d` enters the
sum is decision D3; raw is recommended — see below).

Everything derivable falls out of the (map, domain) pair:

- size of each dimension, total element count;
- **footprint**: the min/max buffer locations the layout can touch
  (→ bounds-check against the buffer's length);
- contiguity / density;
- **injectivity**: does the map send distinct coordinates to distinct
  locations? (False for stride-0 repeats and overlapping windows — fine for
  reads, a hazard for writes.)

### Layouts are white-box

The contract is generic, but the library supports only a **finite, closed
family of layout types whose algebra is 100% understood**. Layouts are never
opaque index functions: a compiler is free to compose them, simplify the
resulting address arithmetic, and plan materializations (temporary buffers
holding the same data under a different layout).

The core family is affine-map + box-domain. One planned extension (§5) adds a
*guard + fill value*: affine inside the domain, a constant outside. That's
still white-box — it's affine arithmetic plus a predicate the compiler can
hoist, split loops around, etc.

## 3. Operations (as specified)

All operations return a **new tensor sharing the same buffer** with a
different layout. Nothing here reads or writes data.

1. **Slice** — `t[x=-1:2, y=3:4]` → same buffer, same strides, shrunken
   coordinate domains. Under the raw-coordinate convention (D3) a slice
   doesn't even touch the offset: it *only* shrinks [min, max]. Coordinates
   keep their identity — element `x=3` is still called `x=3` after slicing.

2. **Blocking / dimension split** — `x -> (x1, x2)`: replace one dimension by
   two (or more), with the *caller controlling the new dimensions' index
   ranges* so coordinates stay natural. Motivating example: view a matrix as
   a matrix of matrices (2-D → 4-D: block coords × within-block coords).
   General contract: an affine bijection from the (x1, x2) box onto x's
   range, `x = a + b1·x1 + b2·x2`; strides follow mechanically
   (`stride_x2 = stride_x · b2`, etc.).

3. **Repeat** — add a new named dimension with **stride 0** and a specified
   [min, max]. All coordinates along it alias the same storage.

## 4. Additional natural operations (proposed)

Same-buffer view ops that fall out of the same (map, domain) algebra:

- **Merge / flatten** — inverse of blocking: `(x1, x2) -> x` when the nesting
  is compatible (`stride_x1 = stride_x2 · size_x2` and the box is full).
  Partial inverse: not every pair of dims merges.
- **Rename** — dimension identity is a name, so renaming is a real
  (if trivial) operation.
- **Permute** — deliberately absent. Order carries no addressing meaning, so
  reordering is not an operation on the tensor: export order is a parameter
  of materialization (`to_numpy(order=...)`), and order-free identity is a
  query (`canonical()`: dims sorted by name — names are semantic, order is
  not).
- **Select / point-index** — `t[x=3]`: drop dimension x, fold
  `stride_x · 3` into the offset. (This is the op that makes the offset
  earn its keep — see D2.)
- **Shift / rebase** — relabel coordinates: translate [min, max] by δ and
  compensate the offset by `−stride·δ` so the same memory is addressed.
  Keeping this separate from slicing is what lets slices stay "natural".
- **Flip** — negate a stride, reflect the domain.
- **Squeeze / unsqueeze** — drop or add extent-1 dimensions.
- **Diagonal** — `(x, y) -> z` with `stride_z = stride_x + stride_y`:
  the classic strided trick, and a second source of non-injectivity's dual
  (an injective map whose image is sparse).
- **Sliding window / unfold** — add `x_k` with `stride_{x_k} = stride_x` and
  shrink x's domain. This *is* the stencil layout in its interior (§5);
  overlapping windows make it non-injective.
- **Pad** — the dual of slice: *extend* a domain beyond the mapped region,
  reads outside produce a fill constant. Requires the guard+fill extension;
  same machinery as stencil boundaries.
- **Field selection** on structured dtypes — `t.field("re")`: bump the offset
  by the field's byte offset, keep all strides, change the dtype. A very cheap
  and very useful view op — and the main argument in D4 for byte-granular
  offsets.

Queries and analyses (no new tensors, but part of this layer's contract):

- `numel`, per-dimension sizes, `footprint`, `fits_in(buffer)`;
- `is_contiguous`, `is_injective` (⇒ safe to write through);
- `overlaps(a, b)` — do two views alias memory? (needed by any future
  compiler for hazard analysis);
- `coalesce` — merge mergeable adjacent dims (canonicalization, useful for
  copy planning);
- `compact_layout(shape_like)` — propose a canonical dense layout for a given
  set of named extents (the "target" half of a materialization plan).

One deliberately *specified but not implemented* op marks the boundary with
the future compute layer: **materialize(tensor, new_layout)** — copy into a
temporary buffer under a different layout. It's the only op that touches
data; this layer only needs to be able to *describe* it (source map, dest
map, iteration domain).

Out of scope even as a view: concatenation (two buffers — not expressible as
one buffer + layout).

## 5. Stencil layouts (advanced / planned extension)

For a dimension x, add a kernel dimension `x_k` giving:

    t_new[x=X, x_k=k] = t[x = X + k]        if X + k in bounds
                      = constant c          otherwise

- The interior is a pure affine view: `stride_{x_k} = stride_x` (exactly the
  sliding-window op above).
- The boundary breaks pure affineness: the layout becomes
  **affine map + domain guard + fill value**. This is the *one* planned
  extension to the core family, and it's shared with `pad`.
- This doesn't map cleanly onto "layout = strides + boxes" alone — which is
  fine, because layouts are a closed set of known types, not arbitrary
  functions. The compiler knows the guard's algebra and can, e.g., peel
  boundary iterations and use the pure affine map in the interior.

## 6. Open design decisions

- **D1 — max inclusive or exclusive?** Half-open `[min, max)` composes better
  (splitting ranges, representing empty domains); inclusive `[min, max]` is
  more natural for symmetric stencil kernels like `[-1, 1]`. Recommendation:
  half-open in the core representation, inclusive sugar in stencil/repeat
  constructors.
- **D2 — keep the offset?** Recommendation: yes. It's the constant term of
  the affine map, and it's forced on us anyway by point-indexing and field
  selection (both fold constants into it). Without it, min-index gymnastics
  would have to smuggle the constant in. Slicing, notably, never touches it.
- **D3 — raw vs normalized coordinates in get_loc?** Does `i_d` enter the sum
  directly, or as `i_d − min_d`? Recommendation: **raw**. Then min/max are
  purely a *domain* (which coordinates are legal) and never affect
  addressing; slicing = domain shrink, rebasing = explicit separate op, and
  coordinates "stay natural" through slicing and blocking — which the
  blocking use case explicitly wants.
- **D4 — units of stride/offset: dtype elements or bytes?** Element units are
  friendlier and match the "stride 1 = one element" intuition. But structured
  dtype field selection and viewing one buffer under several dtypes both want
  byte granularity. Recommendation: element units in the public API, bytes in
  the stored representation (offset especially), with the tensor's dtype
  providing the conversion.
- **D5 — does dimension order mean anything?** Recommendation: no addressing
  semantics ever; retain insertion order only as a default for printing,
  iteration, and materialization planning.
- **D6 — writes through non-injective views.** Stride-0 repeats and
  overlapping windows alias storage. Reads are always fine; the library
  should refuse (or at least flag) writes through a view unless
  `is_injective` holds.

## 7. Prior art worth stealing from

- **numpy** `as_strided` + structured dtypes — the raw mechanics; `ml_dtypes`
  for bfloat16 et al.; **DLPack** for buffer interchange.
- **CuTe (CUTLASS)** — the closest thing to a full "layout algebra":
  composition, `logical_divide` (≈ our blocking), products, complements.
  Validates the white-box-algebra thesis, though CuTe is positional and
  0-based; our names + arbitrary index boxes are the differentiators.
- **tinygrad's ShapeTracker** — a pure view layer whose movement ops
  (reshape / permute / expand / pad / shrink / flip) are nearly exactly the
  inventory above, minus names and index boxes. Evidence the op set is
  complete-ish for a view layer.
- **xarray / named tensors** — names as dimension identity.
- **Halide / TVM / Triton** — boundary conditions as wrappers (≈ guard+fill),
  and the general separation of "what memory means" from "what compute does".

## 8. Objectives (restated)

1. A minimal core — Buffer, DType, Layout, Tensor — where the layout's
   `get_loc` contract is the single source of semantic truth.
2. **View operations only**: every op transforms (map, domain) over the same
   buffer; no data movement, no compute.
3. **Names first**: dimension identity by name; positional order carries no
   meaning; index domains are arbitrary integer boxes, not 0-based.
4. **Fully expanded strides**: no memory-order conventions anywhere.
5. **White-box layout family**: finite set of layout types (affine + box now;
   affine + guard + fill later) whose algebra a compiler can exploit —
   compose, simplify, and plan layout-changing materializations.
6. Rich enough *analysis* surface (footprint, injectivity, overlap,
   contiguity) that a later compute/compiler layer never has to reverse-
   engineer a layout.

## 9. Charts and units (step 3)

The observation that unlocks this step: **the raw integer indexing of step 1
was already a coordinate chart** — `Dim(start, stop)` is the lattice
`n ∈ [0, count)` wearing the implicit chart `coord = start + 1·n`. Step 3
widens the chart family from `(origin ∈ ℤ, step = 1, dimensionless)` to
`(origin ∈ ℚ, step ∈ ℚ, unit)`:

    phys(i) = origin + i · step        (exact Quantities)

Nothing complicated is added, and everything stays exact. Crucially, **the
address and guard algebra is untouched**: `get_loc`, footprints, injectivity,
guard rewrites, and `simplify` all operate on lattice integers exactly as in
steps 1–2. Quantities are normalized to lattice ints at the API boundary;
charts are metadata the ops rewrite mechanically:

- **slice / pad**: bounds may be on-lattice Quantities; the chart itself
  never changes (domains are domains).
- **shift**: relabels the lattice; the chart's origin compensates so physical
  labels stay glued to the data. Storage-side relabeling is not a physics
  change.
- **recenter** (new op): the converse — moves the physical frame
  (origin += Δ) and touches nothing else.
- **flip**: storage reversal; the chart follows the data (origin re-anchors,
  step negates), so each datum keeps its physical label.
- **split**: the outermost part inherits a *position* chart with step scaled
  by its weight (the block pitch); inner parts get *displacement* charts —
  physically, block position + offset-within-block.
- **merge**: requires the parts' charts to nest affinely (or all absent);
  reconstructs the original chart exactly.
- **window / stencil**: the kernel dim gets a displacement chart with x's
  step, so taps carry labels like ±25 nm; k ranges may be given as
  step-multiple Quantities.
- **diagonal**: z inherits the first part's chart (labels along that axis).

### Decisions

- **D7 — charts are a conservative extension.** Optional per-dim; plain ints
  always mean lattice coordinates, with or without charts (*compiler mode*);
  `strip_charts()` drops all labeling. The lattice keeps full citizenship.
- **D8 — exactness is non-negotiable.** Magnitudes are `fractions.Fraction`
  (stdlib — no new dependency); floats are rejected at every boundary
  because `0.1 != 1/10`. Construction: strings `q("0.75 um")`, rationals
  `Fraction(3,4) * u.um`, ints `3 * u.mm`, or PEP 750 t-strings
  `q(t"{Fraction(3,4)} um")` whose interpolations carry ints/Fractions
  exactly. Fixed point is the normalized profile of this (a common
  denominator), useful later for integer-only codegen.
- **D9 — the unit system is owned, minimal, and pint-flavored.** A Unit is an
  exact rational scale onto base units plus a dimension exponent vector;
  `UnitRegistry.define` adds exact aliases (`define("min", "60 s")`) or new
  base dimensions (`define("px", dim="pixel")`). No offset units (°C), no
  floats, no physics — a labeling algebra, not a quantity library.
- **D10 — position vs displacement.** Charts carry a `kind` tag: positions
  have an origin (stage coordinates, timestamps); displacements are
  origin-free differences (kernel taps, within-block offsets — the °C vs ΔT
  distinction). Ops set it correctly; the tag is advisory (the type system
  does not prevent adding two positions — see CONCERNS).
- **D11 — membership is exact-only.** `item`/`slice`/`get_loc` with a
  Quantity require it to be on-lattice; `snap(name, value, floor|ceil|
  nearest)` is the explicit, deliberate rounding op. Coordinates are
  identities, not approximations (the D3 spirit, extended).
- **D12 — steps are canonically positive, but flip may negate them**, since
  charts glue physics to data and flip is a pure storage op. Physical
  slicing on a negative-step chart is refused (slice in lattice space).
- **D13 — value units are metadata.** `Tensor.value_units` labels the VALUE
  space (a Unit, or a field→Unit mapping threaded by `field()`). Values are
  inexact machine numbers, so this is type-level information for the future
  compute layer's dimensional analysis — `item()` returns raw scalars, and
  the exact `Quantity` type remains coordinates-only.

- **D14 — axis identity.** A chart carries an `axis` tag (defaulting to the
  dim's name); split parts and window/stencil kernels share their parent's
  axis. The invariant: *the physical position on axis A is the sum of the
  labels of all dims tagged A*, at most one of which is a position (enforced).
  `select` preserves the invariant by folding a removed dim's label into a
  remaining sibling — the position dim if present, else the widest-step
  displacement, promoted to a position. This makes "labels glued to data" a
  theorem across ops (shift, flip, select, split, decimate) rather than a
  per-op convention: decimation by split+select is labeled correctly, and
  extracting a block yields absolute in-block positions. Axis tags are
  free-standing labels: they do not follow dim renames.
- **D15 — convolution completeness.** `decimate(name, factor, phase)` (the
  one op besides shift that renumbers the lattice; charts keep the physical
  truth), `dilation=` on window/stencil (taps `dilation` steps apart; guard
  form `x + dilation·k`, still linear), and `align(*tensors)` (common-lattice
  views for elementwise compute: equal steps, origins integer steps apart,
  broadcast missing dims). Together with guards rewriting under flip /
  diagonal / window / decimate / conditional merge, the affine + guard
  family covers convolution-shaped workloads short of the piecewise family
  (roll, reflect padding, concat) — deliberately deferred, see CONCERNS.

- **D16 — diagonal never guesses a labeling.** `diagonal` is n-ary
  (`(x_1..x_n) -> z`, strides sum, domains intersect, guard coefficients
  sum). Parts sharing ONE axis get the *forced* label-sum chart (origins and
  steps add — the unique chart satisfying D14; a zero step-sum leaves z
  uncharted). Everything else is UNCHARTED unless the caller supplies
  `chart=` — a Chart, or a combinator `(consumed_dims) -> Chart | None`
  receiving the full Dim objects of the parts (their charts and domains are
  gone after the op). `characteristic(rate, label_along)` is the first
  combinator: it validates the CFL-style commensurability condition
  (`step_along == rate · step_other`, exactly) and labels z along the chosen
  axis — all the physics in a named, testable object.
- **D17 — alignment is diagnosis, never surgery.** There is no `align` op.
  `alignment(*tensors)` reports exactly what stops operands from being
  elementwise-combinable and, per item, the primitive call that fixes it
  (`flip` / `shift` / `slice` / `repeat` / `decimate` / `with_charts`);
  `aligned(*tensors)` is the predicate. The caller applies fixes
  consciously, one cheap view at a time (diagnosis is iterative: flip
  unlocks shift unlocks slice). Same reasoning as no-permute: ops that
  bundle semantic choices teach users not to make them.
- **D18 — categorical labels: the nominal rung.** A dim may carry `labels`
  (a bijection lattice <-> names, e.g. R/G/B) instead of a chart — the
  nominal level of the measurement ladder (nominal -> ordinal -> interval),
  of which the affine Chart is the interval rung. Labels have NO arithmetic:
  no step, no displacements, no sums. Ops that need arithmetic (split,
  merge, window, stencil, pad, diagonal) refuse labeled dims; slice, shift,
  flip, decimate, select thread the labels glued to the data. Attaching a
  chart REPLACES the labels — the nominal -> interval upgrade ("later we
  can replace the categorical data with a coordinate system") — and
  vice versa. Non-uniform numeric coordinates (city longitudes, RGB
  wavelengths) are a third, unimplemented rung: explicit coordinate arrays
  (see CONCERNS). Note the kinship between a labeled dim and a structured
  dtype's fields: a record of same-typed fields IS a categorical dim in
  disguise, which the compute layer may want to unify.

### New analyses

- `Chart.commensurable(other)`: can two lattices interoperate pointwise
  without resampling? True iff same dimensions and the origins differ by an
  integer multiple of the gcd of the steps. This is the layout-level answer
  to "can these two tensors be combined elementwise" — the future compute
  layer's alignment check.
- What it buys, concretely: staggered grids (Yee/FDTD fields at Δx/2
  offsets), texel-center vs corner conventions as explicit chart data,
  finite-difference stencils with taps labeled {-Δx, 0, +Δx}, decimation and
  hopped windows with natural labels ({0, 2, 4, ...} is just a step-2
  chart), and dimensional analysis as a free type check at view-construction
  time.
