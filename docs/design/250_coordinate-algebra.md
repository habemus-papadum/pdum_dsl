# 250 — The coordinate algebra: Frame, Coordinate, Displacement, Slice

**Status: ratified design (owner-ruled 2026-07-27); the P8.6 pivot.**
This document is normative for the indexing surface across all tiers —
host tensors, compute kernels, vertex/fragment shaders. It lands as a
staged pivot (§7) inserted mid-P8, before the vertex-buffer and records
work, because those build on this law. The spec (200) owns the roadmap;
this document owns the algebra.

## 1. Why this layer exists

The layout algebra already obeys an affine discipline it never names.
`Dim.to_lattice` converts positions; `Dim.delta_to_lattice` converts
displacements; window taps divide a physical delta by the chart step;
`Chart.kind` records position-vs-displacement; the axis invariant caps
positions at one per axis. But the discipline lives in verbs, not nouns
— so `Quantity` arithmetic cannot tell a position from a displacement
(the recorded BOUNDARIES gap), and the kernel tier checks subscripts by
structural iota membership rather than by type.

This layer names the structure. A dim's integer lattice is a ℤ-torsor:
points (Coordinates) and differences (Displacements) are distinct
sorts; charts extend the torsor exactly to ℚ·units. `Quantity` stays a
plain field of scalars — the engine room, deliberately unpoliced — and
the affine distinction is policed one layer up, in objects that carry
their dim. Nothing here doubles Quantity's arithmetic surface, and
nothing here adds IR: the objects are the missing nouns for laws the
verbs already obey.

## 2. The four objects

All are host-tier vanilla Python: frozen dataclasses with Layout-grade
citizenship. None ever holds a stride or an offset — they bind to the
**observable frame**, never to the shadow. That is what makes a
coordinate portable across every tensor sharing the frame, which is
what `dst[y, x] = src[y, x]` requires.

- **`Frame`** — one dim's observable identity, reified:
  `(name, start, stop, chart?, labels?, level?)` — exactly the tuple
  `TensorType` has always keyed on (`_dim_key`), now an object.
  `Dim` is Frame + stride. `TensorType.dims` is a tuple of Frames.
- **`Coordinate`** — a point: `(frame, i)` with `start ≤ i < stop`
  **by construction**. There is no out-of-bounds coordinate, only a
  refusal at the constructor. On a charted frame it has an exact
  physical reading `phys = origin + i·step`; on a labeled frame, a
  name.
- **`Displacement`** — a vector: `(frame, k)` lattice steps, or born
  physical as an exact step-multiple Quantity. Displacements are
  unbounded — the domain bounds points, not differences.
- **`Slice`** — an arithmetic progression:
  `(start: Coordinate, stop: Coordinate, step: Displacement)`, all on
  one frame, half-open per D1. Its points are
  `{start + n·step | n ≥ 0} ∩ [start, stop)`. Step is positive; a
  negative step refuses — a Slice is a forward progression, and
  orientation is `flip`'s job, a separate deliberate act.

## 3. The algebra

The torsor law, with refusals as theorems:

| expression | result |
|---|---|
| `Coordinate − Coordinate` (same frame) | `Displacement` |
| `Coordinate ± Displacement` | `Coordinate` (bounds re-checked; out-of-domain refuses) |
| `Displacement ± Displacement`, `int · Displacement`, `−Displacement` | `Displacement` |
| `Coordinate + Coordinate` | **refuses** — adding points is the affine crime |
| cross-frame arithmetic (`y − x`) | **refuses** — geometry, not lattice structure; coerce both |
| arithmetic with a float, or any other numeric use of a Coordinate | **refuses toward explicit coercion** |

Promotion happens at operator boundaries only: an `int` or a
step-multiple `Quantity` promotes to a `Displacement` (exact-only —
off-lattice refuses; `snap` remains the one deliberate rounding door).
A scalar never promotes to a Coordinate: there is no origin-free way to
make a point.

**The explicit-coercion doctrine (owner-ruled).** A Coordinate is not a
number. Using one in numeric arithmetic refuses with a message naming
the door: coerce to a stated value type (`f32(c)`, `i32(c)`) and
compute with values, on the record that the value has left the affine
world. Nothing silently degrades to float. When in-kernel coordinate
arithmetic arrives (§8), it returns Coordinates — scalars and unit
values promote to Displacements, the sum is a point again, chart-aware
and exact — never a bare number.

**The call-boundary law (owner-ruled, the P8.6 veto).** Coordinates
cross call boundaries AS Coordinates — no coercion happens for you at
an application. Both casting sites are legal and the language never
picks: the CALLER casts (`f(f32(i), f32(j))` — f stays a plain scalar
citizen: liftable, spliced, its captures riding arg-rooted slots) or
the CALLEE casts (`f(i, j)` with the coercion inside f's body — f is a
frame-aware function and INLINES through the kernel rules, where the
coordinate law still holds). The observers make the callee style pay
rent: `extent(c)` reads the frame's width — a host int, a build-time
fact (and exactly where the keying ladder's extent-generic license
will later swap in a uniform) — so one Coordinate argument carries
location AND domain. `extent` has ONE rule at every tier
(owner-ruled): the width, a host int — `extent(c)` on a Coordinate,
`extent(t, "y")` on a tensor, and the kernel-lowered form all agree;
the full domain pair is the Frame's job (`.start`/`.stop`). (Before
the ruling, the eager form returned `(start, stop)` while the lowered
form returned the width — the same spelling meant different things
per tier, working only for 0-based dims.)

```python
u = f32(j) / f32(extent(j))
v = f32(i) / f32(extent(j))   # one denominator: aspect preserved
```

**Ints promote explicitly.** `f32`/`i32` accept host ints alongside
Coordinates (`f32(extent(j))`); ints never silently join float math —
the style is `x * 0.5`, never `x / 2`. (Blanket enforcement of the
int/float boundary awaits the carrier machinery of the dtype era;
recorded, not half-built.)

**The index algebra (owner-ruled, P8.6c).** Coordinates admit no
arithmetic — but **layout isomorphisms induce coordinate maps**, and
that is the whole index algebra: the layout algebra acting on points,
one map per op as consumers appear. `rename` is the trivial map. The
merge map is `global_thread_idx(block, thread, grid)` — the coordinate
face of layout `merge`: it consumes the raw (block, thread) pair and
returns Coordinates on the writable's FLAT frames, the affine
evaluation `b·T + t` performed inside the algebra with `T` read from
the grid layout — correct by construction. The inverse direction
(global→raw) stays banned: div/mod is piecewise, and the algebra is
deliberately fine→coarse. `global_idx("y", "x")` is the STANDARD door
— an ambient intrinsic *defined* as the merge over the raws (the raws
remain the primitives; a backend may bind the name to
`thread_position_in_grid`): my position in the writable lattice, one
uniform meaning under every geometry, degenerating to the thread pair
structurally under one block (the grid IS the flat lattice — the old
coincidence as a theorem, not a spelling). `thread_idx` is thereby
re-scoped, not deprecated: it is the primitive for genuinely
block-local work (split-aligned tensors, shared memory, the tile
tier) and the definitional substrate of `global_idx`. Store legality
is TYPED: a store index whose frame is the writable's flat dim IS the
declared global — only the merge map mints those — so the merge-back
needs no registry. Derivative slopes ride the maps (`f32` slope 1,
`rename` slope 1, merge slope 1 in the thread direction and `T` in
the block direction), which is what makes `with_respect_to` a
frame-targeted operator when the AA work returns to it — d/d(unit
step of the writable frame), well-defined under any geometry; with
charts, d/d(phys) exactly.

## 4. Factories — where coordinates come from

Each tier has one factory, and they rhyme:

- **Kernel and shader tiers**: `thread_idx(...)` — the one ambient
  function. Its meaning is kind-dependent: in a compute kernel it
  yields Coordinates over the writable lattice's frames; in a vertex
  shader, over the draw domain (`vertex_id`, `instance_id`). Binding
  names are free; frames are the identity.
- **Host tier**: the frame itself. `y, x = t.frames("y", "x")` — the
  same tuple-unpack gesture as `thread_idx`. The frame handle is the
  point factory, with the same tri-acceptance as `to_lattice`:

```python
y, x = img.frames("y", "x")

c = y[128]           # Coordinate; refuses if 128 ∉ [start, stop)
p = y[q("1.5 um")]   # chart door, exact-only; snap() to round deliberately
r = chan["red"]      # label door
d = y[256] - y[128]  # Displacement: 128 steps ≡ physical via the chart
```

The frame handle's `__getitem__` is **point-only**. Slices are built
from coordinate endpoints (§5), not inside the handle — one door each.

## 5. The slice spelling

Slices are spelled with Coordinate endpoints. In subscript position the
native colon display is the canonical form — Python hands the display
to `__getitem__`, and the objects carry the meaning:

```python
crop = img[y[128] : y[256], x[0] : x[128]]     # order-free
band = img[y[q("1.5 um")] : y[q("2.5 um")]]    # physical endpoints
even = img[y[0] : y[512] : 2]                  # step promotes to Displacement
tail = img[y[128] :]                           # to the frame's stop
win  = img[c : c + d]                          # endpoints are expressions
```

This spelling is tier-uniform: kernel bodies have no frame handles —
they have Coordinates from `thread_idx` — so coordinate-endpoint slices
are the only form that can ever appear in a kernel, and the host uses
the same one. The step slot takes anything that promotes to a
Displacement. One-sided forms take the missing endpoint from the known
endpoint's frame; a bare `[:]` has no frame and refuses (and is never
needed — unmentioned dims pass through). Mixed-frame endpoints refuse.
Standalone, outside brackets, the dataclass constructor
`Slice(c0, c1, step)` is the spelling.

One consequence the in-bounds law forces (stage-1 ruling): the
frame-end **exclusive** endpoint has no point — `y[512]` on a
`[0, 512)` frame refuses like any other out-of-bounds coordinate — so
a slice to the end of the domain is spelled by omission
(`img[y[128]:]`, `Slice(c0)` with `stop=None`). The one-sided form is
not sugar; it is the only spelling of that endpoint.

## 6. The subscript law — one law, all tiers

`tensor[index, ...]` where each index is a Coordinate or a Slice:

1. **Named, order-free.** Each index binds to the tensor dim carrying
   its frame's name. Order is semantically void:
   `dst[y, x] = src[y, x]` ≡ `dst[x, y] = src[y, x]`. Unmentioned dims
   pass through — no `:` placeholders, ever.
2. **Typed: strict identity, containment extent (owner-ruled).** The
   frames must agree on everything they know — name equal; chart equal
   when both present, both absent otherwise; labels agreeing at the
   indexed points, never labeled-meets-unlabeled; level equal when
   present. The one relaxation is extent: the point(s) must lie in the
   target dim's domain — containment, not equality. (P9's
   gather/scatter step depends on this relaxation.) A frame mismatch
   refuses; `rename` is the adapter.
3. **A Coordinate drops the dim; a Slice keeps it.** `img[y[128]]` is
   the x-row view; `img[y[128] : y[129]]` keeps a size-1 y.
4. **Never promote.** Indexing never leaves tensor-land:
   `img[y[128], x[7]]` is a rank-0 tensor (`Layout(dims=())`,
   numel 1), not a Python scalar. `.item()` stays the one explicit
   scalar exit — parallel to `snap` as the one explicit rounding exit.
5. **The one promotion, store-side.** Assigning a bits-compatible
   scalar through an indexed writable view promotes it to a memoryless
   const broadcast over that view — pointwise's law, the same rule
   `tl.store` already applies in kernels; the single-element assign is
   its smallest instance. A tensor-valued right-hand side meets the
   assigned view under the ordinary alignment law.

The kernel tier is the same law with larger extents: `thread_idx`
coordinates range over the whole ambient lattice, a host Coordinate
over one point, a Slice in between. Subscripting yields fields/views at
every extent; nothing is ever a bare scalar.

## 7. Integration: the staged pivot

Every stage lands green, committed, and separately reversible; the
Layout kwargs methods (`select`/`slice`/`decimate`/`flip`) never change
— they are the engine-room normal form the new surface lowers onto.

1. **Objects.** `Frame`/`Coordinate`/`Displacement`/`Slice` in their
   own module with exhaustive algebra tests: torsor laws, every
   refusal, chart/label doors, exact-only conversion, boundary
   promotion. Nothing consumes them. Risk: zero.
2. **Identity reification.** `_dim_key` → `Frame`; `TensorType.dims`
   becomes tuple-of-Frames. The canary stage: identity semantics are
   unchanged, so the whole suite — kernel caches, artifact keys,
   conformance — must pass **with no test edits**.
3. **Host indexing.** `t.frames(...)`, `__getitem__`/`__setitem__`
   lowering onto the Layout normal form; rank-0; never-promote; the
   broadcast store law. Indexing becomes the canonical Tensor surface.
   (`shift`, `rename`, `repeat`, `recenter`, `split`, `merge` are not
   index sets — they stay methods.) Stage-3 finding: `select`/`slice`/
   `decimate` are ALSO traced spellings the step tier lowers by
   inspection (`_METHODS`), so their retirement moves to the respell
   stages — the methods die when the traced tiers lower the subscript
   spelling, not before.
4. **Kernel law.** `thread_idx` returns Coordinates (the `CoordType`
   leaf, iota-backed); the typed order-free subscript replaces the
   structural iota-membership checks; the explicit-coercion refusal
   plus the `f32`/`i32` vocabulary; `rename` as the first adapter.
   The committed respells live here (the S.3 example, quad, box3,
   cylinder, the conformance battery), each mechanical, each commit
   green, with one hard invariant: **conformance golden outputs are
   numerically identical before and after respelling**.
5. **Vertex tier + buffers.** `thread_idx("vertex_id", "instance_id")`
   respell, then vertex pulling and records built on the coordinate
   law rather than retrofitted.

## 8. What the IR does NOT get

No new ops. A subscript normalizes into the existing view algebra
before any load — Coordinate → `select`, Slice → `slice` (+
`decimate` for step > 1) — and the loads that follow are the identity
and iota reads that already exist. Displacements are host-static data,
exactly what window/stencil taps already are. The one new IR citizen is
the `CoordType` leaf at the kernel tier. The complexity budget is spent
on the type discipline — frame agreement and refusal wording — not on
the dialect.

The stratification is load-bearing: coordinate arithmetic with
**static** Displacements stays affine — interval arithmetic on domains,
bounds provable at lowering time, the stencil/window family surfacing
as subscript syntax. **Dynamic** (value-computed) indices are a
different door — P9's gather/scatter family, value-typed, with the
reference refusing out-of-bounds. Neither bleeds into the other.

## 9. Recorded future work

- **In-kernel coordinate arithmetic** (`src[v + dv]`, SDFs): returns
  Coordinates per §3; static-displacement bounds proved by interval
  arithmetic; arrives with its first consumer. Until then the
  explicit-coercion refusal holds the line.
- **Chart-aware arithmetic yielding Quantities**: `c2 − c1` on a
  charted frame reads as an exact physical Quantity; user charts with
  units make kernel geometry unit-checked.
- **Fancier index sets** (unions, reflections, rolls) belong to the
  piecewise-guarded-affine family, adopted as a family decision when a
  concrete need appears (BOUNDARIES).
