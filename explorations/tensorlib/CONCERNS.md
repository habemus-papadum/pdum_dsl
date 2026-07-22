# Concerns / open edges (steps 3–4: charts, units, axis identity, conv primitives)

Things implemented with a pragmatic choice, deferred, or deserving judgment
before the compute layer lands on top.

1. **The piecewise family is deliberately skipped.** `roll` (circular
   shift), reflect/circular padding, and concat/interleave-as-view are all
   non-affine as single views; each is expressible as a union of guarded
   affine pieces (tinygrad's multi-view ShapeTracker is the precedent).
   Constant-fill pad covers most convolutional practice without it. Adopting
   piecewise layouts is the one real *family* extension decision left; make
   it consciously when a concrete need appears, not by accretion.

2. **Position + position is still not prevented by the type system.** Axis
   tags now give the model real affine structure (the per-axis label-sum
   invariant, select compensation, one-position-per-axis enforcement), but
   `Quantity` arithmetic itself cannot tell a position from a displacement —
   `q("1 um") + q("2 um")` is unpoliced. A dedicated affine `Position` type
   would close this at the cost of doubling the arithmetic surface. Revisit
   if the compute layer starts doing coordinate arithmetic on user values.

3. **Select's promotion rule is a convention.** When the *position* dim of
   an axis is selected away and several displacement siblings remain, the
   removed label is folded into the widest-step displacement (tie: name
   order), which is promoted to a position. This is deterministic and
   order-free (per D5), but it is a choice, not a theorem — e.g. after
   selecting the block of a 3-deep split, the mid-level dim becomes the
   position. If a use case wants a different target, an explicit
   `recenter`+`with_charts` fixes it up.

4. **Physical slicing on negative-step charts is refused.** After `flip`, a
   chart's step is negative (charts glue to data). A physical interval
   [a, b) on a descending axis inverts at the lattice with fiddly open/closed
   ends; rather than pick a semantics silently, `slice` raises and tells you
   to slice in lattice space.

5. **Diagonal labeling is caller-supplied (D16); sub-lattice rate diagonals
   remain future work.** Same-axis parts get the forced label-sum chart;
   everything else is uncharted unless a Chart or combinator is passed, and
   `characteristic(rate, label_along)` covers the exact-rate case
   (step_along == rate·step_other). What is still missing: the rate diagonal
   whose steps do NOT match the rate but are commensurable with it — the
   true characteristic then lives on a coarser sub-lattice, which needs a
   decimate-and-diagonal composition the library does not yet automate
   (deliberately: that is a conscious construction, per D17's spirit).

6. **Merge through guards is conditional by design.** A guard survives
   merging only when its coefficients are proportional to the mixed-radix
   weights (true for any guard that came from a split); otherwise the form
   would need div/mod and merge raises. Similarly, merge requires parts
   all-charted (one shared axis, affine step nesting) or all-uncharted —
   compiler-mode code that strips charts must strip consistently.

7. **Alignment diagnosis is iterative and frame-first.** `alignment`
   reports frame issues (flip/shift/chart mismatches) before domain issues,
   so some recipes only appear after earlier ones are applied — apply and
   re-run until clean. Recipes are strings (human-facing); if the compute
   layer wants machine-applicable plans, promote them to structured op
   descriptions then. Resampling is never suggested as automatic: unequal
   commensurable steps are reported as a problem with a `decimate` *hint* —
   a decimating alignment is expressible with existing ops but must be a
   conscious act. Labeled (categorical) dims align only when their label
   tuples match exactly. `Chart.commensurable` remains the "could we" query.

8. **Decimate renumbers the lattice.** It is the one op besides `shift`
   that relabels lattice coordinates (j = (i − phase)/factor — forced,
   since domains are dense boxes). The chart keeps the physical truth, so
   physical indexing is unaffected; pure-lattice users must expect the
   renumbering. Phase is normalized mod factor.

9. **Value units are unchecked metadata.** `value_units` is threaded by
   `field()` and displayed, but nothing validates it and `item()` returns
   raw machine numbers. The compute layer should introduce an *inexact*
   quantity concept (float magnitude + unit) and use `value_units` for
   dimensional checking of kernels — including the sampled-function vs
   discrete-filter question for convolution (whether a kernel sum carries a
   ·Δx measure factor; the chart's exact step is available either way).

10. **Guard errors and reprs speak lattice, not physics.** A guard violation
    on a charted dim reports lattice integers. Presentation-layer
    translation would help debugging but touches error paths everywhere;
    deferred.

11. **The unit registry is deliberately small.** No SI-prefix
    auto-generation, no offset units (°C); `define()` is the extension
    point. Compound-unit parsing covers `m/s`, `um**2`, `s**-1` but not
    parentheses. Derived symbols from arithmetic are literal (`"m/s"`) and
    can get ugly through long chains.

12. **Fractions are unbounded rationals, not fixed point.** Exactness is
    guaranteed; a long chain of chart arithmetic can grow denominators. For
    codegen, normalize a chart family to a common denominator per axis
    (true fixed point) — the representation supports it; the normalization
    pass is future work.

13. **Charts and labels on `repeat` dims are unpoliced.** A broadcast dim
    can carry any chart or label set; expressive but nothing checks that it
    is sensible (alignment recipes suggest exactly this for missing dims).

14. **The measurement ladder has a missing rung.** Nominal (labels) and
    interval (affine charts) exist; *non-uniform numeric coordinates* —
    city longitudes, RGB center wavelengths, irregular sample times — are
    neither: they need explicit per-point coordinate arrays (xarray-style),
    a distinct labeling family with its own exactness story. Ordinal
    (ordered categories without distances) is likewise absent; today order
    is implied by lattice order. Decide these when a concrete need appears.

15. **Per-label value units overlap with structured dtypes.** A categorical
    dim of measurement channels ({v, i} in volts and amps) duplicates what a
    structured dtype + `value_units` mapping already expresses; a labeled
    dim currently has no per-label value units. The compute layer should
    pick ONE canonical spelling (a record of same-typed fields is a
    categorical dim in disguise) rather than supporting both forever.

16. **Carriers are inferred-and-unchecked.** The carrier field (bool/int/
    rat/real/complex) is threaded metadata; marker carrier signatures
    (`exp : real → real`, per-carrier `div`, comparisons → bool) and the
    coercion chain are declared in COMPUTE.md but not enforced anywhere.
    The compute layer should harden this when marker signatures land —
    until then `exp(iota)` silently "works" in the reference layer.

17. **Scan is single-dim and ufunc-only in the reference layer.** Multi-dim
    scan is deliberately absent (order ambiguity); pair-state associative
    scans (SSM recurrences h_t = a_t·h_{t-1} + b_t) need the marker DSL —
    the reference `ufunc.accumulate` cannot express them. Reverse scan is
    spelled flip∘scan∘flip rather than a parameter; revisit if it grates.

18. **FunctionalBuffer reads assume layout-respecting locs.** A functional
    read raises on byte locations that are not multiples of its scale;
    layout ops preserve alignment by construction, but `field()` on a
    functional tensor (or hand-built offsets) could misalign — the error is
    loud, not silent. `pointwise` outputs materialize; only pristine
    view-chains of iota stay tight, which is exactly what a compiler wants
    to detect.

19. **Cotangents live on the lattice — charts are stripped in backward
    programs.** The AD transform strips charts/labels from the seed and
    from every forward value it references, so gradient tensors carry no
    physical labeling. Whether a gradient *should* inherit its primal's
    chart (it is a cotangent — arguably it lives on the dual axis, and its
    value-unit is 1/unit(primal)·unit(loss)) is a real design question for
    the compute layer's unit checking. Deferred deliberately.

20. **The differentiable subset is explicit and partial.** Markers:
    add/sub/neg/mul/div/exp/log/maximum/minimum/where (comparisons and iota
    are gradient-free by design). Reducers: sum/mean/max/min (max/min with
    the tie caveat); prod deferred. scan(sum) only. decimate's adjoint
    needs a factor-divisible domain (pad the source first); n-ary diagonal
    adjoints deferred. Each gap raises loudly.

21. **`materialize` is the IR's one copying op.** Adjoints of split (and
    decimate) must merge, and merge needs real stride nesting that an
    arbitrary cotangent chain does not guarantee — so the transform inserts
    an explicit materialize (identity, chosen dim order). This is honest
    (export order is a materialization property, D5) but it is also the
    first place the *correctness* layer forces a copy; the memory layer
    (REPRESENTATIONS.md) will want to elide it when nesting already holds.
