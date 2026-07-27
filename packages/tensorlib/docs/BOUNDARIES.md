# BOUNDARIES — the deliberate limits

Every known coarseness, convention, and refusal, written next to the code
that has it. Each limit is a choice; where one bites, the code refuses
loudly rather than guessing. Forward-looking items (codegen
normalization, buffer reuse, materialize elision, the indexing family)
live in the workspace spec, `docs/design/200_the-spec.md` §8.

## Layout algebra

- **The piecewise family is absent.** `roll`, reflect/circular padding,
  and concat-as-view are each a union of guarded affine pieces; adopting
  piecewise layouts is a family decision to make consciously when a
  concrete need appears. Constant-fill pad covers convolutional practice.
- **`Quantity` arithmetic cannot tell a position from a displacement** —
  `q("1 um") + q("2 um")` is unpoliced, deliberately: Quantity is the
  engine room. The affine distinction is policed one layer up by the
  coordinate algebra (Frame/Coordinate/Displacement/Slice — design
  ratified, `docs/design/250_coordinate-algebra.md`, landing as the
  P8.6 pivot) without doubling Quantity's arithmetic surface.
- **Select's promotion rule is a convention, not a theorem.** When an
  axis's position dim is selected away, the removed label folds into the
  widest-step displacement sibling (tie: name order), which becomes the
  position. `recenter` + `with_charts` overrides it.
- **Physical slicing on a negative-step chart refuses** — after a flip,
  slice in lattice space.
- **Sub-lattice rate diagonals** (steps commensurable with, but unequal
  to, the rate) need a decimate-and-diagonal composition the library does
  not automate: a conscious construction.
- **Merge through guards is conditional.** A guard survives merging only
  when its coefficients are proportional to the mixed-radix weights (true
  for split-born guards); parts must be all-charted or all-uncharted.
- **Decimate renumbers the lattice** (j = (i − phase)/factor); the chart
  keeps the physical truth, so physical indexing is unaffected.
- **Charts and labels on repeat (broadcast) dims are unpoliced.**
- **The measurement ladder has missing rungs.** Non-uniform numeric
  coordinates (irregular sample times, RGB center wavelengths) and
  ordinal categories are neither labels nor affine charts — a distinct
  labeling family, adopted when a concrete need appears.
- **Guard errors and reprs speak lattice integers, not physics.**
- **The unit registry is deliberately small.** No SI-prefix
  auto-generation, no offset units; `define()` is the extension point;
  compound parsing covers `m/s`, `um**2`, `s**-1`, not parentheses.
- **Fractions are unbounded rationals.** Exactness is guaranteed; long
  chart chains can grow denominators (per-axis normalization is an L2
  item, 200 §8).
- **Per-label value units do not exist.** A categorical dim of
  measurement channels and a structured dtype with a value-units mapping
  express the same thing; one canonical spelling should win.
- **FunctionalBuffer reads refuse byte locations off the functional
  layout's scale**; layout ops preserve alignment by construction, so
  only hand-built offsets can misalign — loudly, never silently.

## Cost models and transforms

- **opcount counts names, not flops.** Exact per-instruction tallies;
  cost weights are a machine property supplied separately; a MAC is a
  recognized fusion, never a primitive; guarded operands count over the
  guard box.
- **The peak-memory model is deliberately coarse.** Uniform 8-byte
  itemsize, numpy temporaries ignored, inputs resident unless freed —
  but every layout op is exactly a zero-byte alias, and
  iota/const/guards occupy nothing, which is the part planners guess at.
- **`materialize` is the IR's one copying op.** Split/decimate adjoints
  insert it to guarantee merge's stride nesting; eliding it when nesting
  already holds is bufferization's job (200 §8).
- **fold is sequential; its reference adjoint stores everything by
  default.** `fold_segments=K` applies uniform checkpointing and
  `fold_slots=S` runs binomial revolve over the same certified pieces —
  mutually exclusive knobs, global per grad call, gradients
  bit-identical across store-all/uniform/revolve. A declared associative
  combine is a license a compiler may exploit; it never changes the
  sequential denotation.
- **Checkpointing's min cut optimizes the fwd/bwd boundary, not the peak
  directly**; the ban set is a policy, not a cost model. Recompute
  duplicates break name = value until a value-numbering pass exists
  (200 §8).
- **The traffic model is v1-coarse, loudly.** Ring formulas only; no
  overlap, topology, or resharding; backward collectives are unfused;
  lattice surgery on bound dims refuses. Cross-placement operands refuse
  toward an explicit collective — applying it is the caller's conscious
  act.
- **The zoo's numeric hygiene is toy-scale.** `-1e9` masking, float64,
  tiny widths — denotation tests and cost-model corpora, not a numerics
  benchmark (adversarial input families arrive with the L4 flagships,
  200 §8).

## Graphics (v1)

- **Texture sampling is deliberately narrow.** One format
  (rgba8unorm-srgb), 2D, `lod=0` only, the R channel, filter
  nearest|linear, address clamp|repeat. Mip chains (explicit-LOD then
  analytic auto-LOD from the wrt-ambient gradient), texture arrays,
  cube faces, the format registry, and `sample` inside compute kernels
  or oracle-class device functions are future work. The conformance
  goldens state their tolerances: hardware sRGB decode may deviate
  from the exact IEC curve by ~0.5/255 in linear light, and bilinear
  weights carry ~8-bit fractional precision.
- **The reference rasterizer has no depth buffer** — triangles compose
  in draw order (painter's), and vertex buffers do not yet reach the
  device render path (the vid-only subset renders there).
