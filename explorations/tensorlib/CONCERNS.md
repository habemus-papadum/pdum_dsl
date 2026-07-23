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

16. **RESOLVED: markers have carrier/unit signatures** (signatures.py).
    One pass propagates (carrier, unit) facts through trees, primitives,
    reducers (fixed-point over structured state), and whole programs —
    `exp` of micrometers, `x_volts + t_seconds`, and cross-dimension
    comparisons now REFUSE, statically (`infer_signatures`) and at run time
    (`pointwise` enforces `marker_signature`). Policy: None = unknown
    unifies with anything (unlabeled programs pass untouched); nonzero
    constants are dimensionless, zero is unit-polymorphic; carriers join
    bool<int<rat<real<complex. Payoff: `grad` infers `target_unit` from
    the target's signature — units-in-gradients needs no annotation when
    inputs declare units. Honest gaps: `prod` of a dimensioned quantity
    (needs static extent), pad fills, structured-dtype unit maps.

17. **PARTLY RESOLVED: pair-state scans exist via the marker DSL.**
    `defreducer` defines structured-state reducers (lift / associative
    combine / init / project) and `scan`/`reduce` evaluate them over
    multiple aligned element tensors — the SSM recurrence h_t = a_t·h_{t-1}
    + b_t is `linrec` (pair combine, associativity property-tested; a Lean
    lemma later). Composite-reducer ADJOINTS now DERIVE too
    (autodiff.composite_scan_adjoint): the state cotangent of a
    structured-state scan is itself a linear recurrence in reversed time,
    emitted as a generated matrix-linrec composite scan over derived
    Jacobian trees; reduce† = embed-at-last then scan†. The boundary needs
    no special case BECAUSE init is the monoid identity (C(init, r) = r ⇒
    ∂C/∂right = I there) — an obligation to state in Lean. Still open:
    multi-dim scan stays deliberately absent; reverse scan stays
    flip∘scan∘flip. The reference sweep is a sequential Python loop — the
    O(log n) parallel evaluation licensed by declared associativity is the
    compiler's job, and the adjoint re-scans the forward per state
    component (k extra scans) rather than caching the trajectory.

18. **FunctionalBuffer reads assume layout-respecting locs.** A functional
    read raises on byte locations that are not multiples of its scale;
    layout ops preserve alignment by construction, but `field()` on a
    functional tensor (or hand-built offsets) could misalign — the error is
    loud, not silent. `pointwise` outputs materialize; only pristine
    view-chains of iota stay tight, which is exactly what a compiler wants
    to detect.

19. **RESOLVED (2026-07-22): gradients carry their primal's labeling.**
    Implemented: every cotangent contribution is restamped with its
    primal's coordinate charts and labels at the moment it is recorded
    (with_charts/with_labels metadata ops), which also absorbs select's
    axis-compensation; composite adjoints (decimate, diagonal) work on the
    bare lattice internally and are restamped on the way out. Value units:
    `grad(..., target_unit=u_L)` annotates gradients of unit-bearing inputs
    with u_L/u_v (u_L alone when the input is unlabeled); a runtime seed
    should carry u_L/u_target. Remaining honest gap: u_L is
    caller-declared, because forward unit propagation through markers
    (CONCERNS #16) does not exist yet — when marker unit signatures land,
    target_unit becomes inferable and seed units checkable.

20. **The differentiable subset is explicit and partial — but no longer
    closed.** Primitives: add/sub/neg/mul/div/exp/log/tanh/maximum/minimum/
    where (comparisons and iota gradient-free by design), and every
    COMPOSITE marker differentiates automatically via derived partials
    (mdsl.diff) — the hand table stops growing. Reducers: sum/mean/max/min
    (tie caveat) and every composite reducer (BPTT as generated IR, #17);
    prod deferred; plain scan(sum) plus all composite scans. decimate's
    adjoint needs a factor-divisible domain; n-ary diagonal adjoints
    deferred. Each remaining gap raises loudly.

21. **`materialize` is the IR's one copying op.** Adjoints of split (and
    decimate) must merge, and merge needs real stride nesting that an
    arbitrary cotangent chain does not guarantee — so the transform inserts
    an explicit materialize (identity, chosen dim order). This is honest
    (export order is a materialization property, D5) but it is also the
    first place the *correctness* layer forces a copy; the memory layer
    (REPRESENTATIONS.md) will want to elide it when nesting already holds.

22. **Marker-DSL registries are process-level — and naming is a live design
    tension.** Composite markers/reducers resolve by name, like PW/RED —
    programs stay pure data, but deserializing in a fresh process requires
    re-registering composites first; derived machinery (`name.d0`,
    `name.C0`, `name.adj0`) registers on demand. Mitigation landed:
    `defmarker(None, ...)` derives a CONTENT-ADDRESSED name (`m_<digest>`
    of the tree) and re-registration of an equal body is a no-op — the
    registry behaves as a cache keyed by structure, which is the main
    repo's build-in-a-loop philosophy rather than a namespace one. Open
    (deliberately unresolved until pdum.dsl integration): registries as
    LIBRARIES — multiple registries with dependencies, so precompiled
    marker packages can ship; and whether reducers get content addressing
    too. Also: traced bodies are trees, not DAGs — reuse duplicates
    subtrees (correct; CSE is the compiler's job), and derivative trees
    grow multiplicatively for deep compositions.

23. **The ops-count model counts names, not flops** (opcount.py). Exact
    per-instruction Counters keyed by primitive name; cost is a separate
    weights dict (`ProgramOps.weighted`), because what an exp or a div
    costs is a machine property, not a program property. MACs are a
    recognized FUSION (mul consumed solely by reduce-sum → "mac", adds
    absorbed), not a primitive — matmul counts m·n·k macs. Guarded
    operands count over the guard BOX (reference semantics evaluates
    fills); λ-proportional counts and memory-traffic modeling
    (REPRESENTATIONS.md Level 1) remain future work.

24. **`fold` (the tensor-state scan) is sequential, and its reference
    adjoint stores everything.** The step is an IR Program (the one
    structured combinator; still no branching); the carry must keep the
    state's exact layout (checked, D17-style); the scan dim must be
    chartless in the reference (strip first, glue charts back after).
    The adjoint is DERIVED by self-application — grad of a scalarized
    wrapper around the step yields the VJP program, folded in reversed
    time — which means: forward state trajectories are re-emitted per
    state component and held whole (store-everything BPTT; the L1
    checkpointing work will trade this off properly), each element/init
    cotangent re-runs the reverse fold (one fold per output, like the
    state_scanner pattern), and out=("final", v) is restricted to carry
    vars. An associative tensor COMBINE (the Mamba-2 chunk-parallel
    license) is a declaration the compiler may exploit later; it does not
    change the sequential denotation. Second-order through fold is
    untested.

25. **The peak-memory model (memory.py) is deliberately coarse.** Uniform
    8-byte itemsize (the shadow convention); numpy's internal temporaries
    ignored; fold transient = new-carry + one recursive step simulation
    (step inputs alias outer storage but are counted — an upper bound);
    guarded shadows count their box; dead values free immediately; inputs
    resident unless free_inputs. What it gets EXACTLY right is the part
    heuristic planners guess at: every layout op is a zero-byte alias
    (root-tracking through view chains), and iota/const/masks-as-guards
    occupy nothing. Dtype-exact sizes, buffer reuse (L2), and
    rematerialization live in later passes.

26. **The zoo's numeric hygiene is toy-scale.** -1e9 masking (not -inf),
    tiny widths, float64 everywhere, no dropout/training-time noise, and
    the flash reducer's -1e30 init relies on exp underflow for its
    identity. Adequate for denotation tests and cost-model corpora; not a
    numerics benchmark. Also: reduce(max) tie-splitting (the #-caveat) is
    reachable through the flash combine's maximum partials if two scores
    tie exactly — tests use continuous random scores.

27. **Checkpointing (transforms.py) optimizes the BOUNDARY, not the peak
    directly.** The min cut minimizes saved bytes across the fwd/bwd
    boundary; the peak usually follows (GPT-2: 76% of joint) because
    recompute is placed just-in-time, but no theorem connects the two —
    schedule search / pebbling is the direct-peak attack (later). The ban
    set is a POLICY (default reduce/scan/fold ≈ don't redo contractions),
    not a cost model: a tiny reduce is banned while a huge pointwise
    recomputes. Recompute duplicates break name=value (v and v.rc denote
    one semantic value — fine for run/measure; a value-numbering pass
    would be needed to see through it). Fold interiors: grad's
    `fold_segments=K` applies Chen-style uniform checkpointing inside fold
    adjoints (boundary states + per-segment just-in-time recompute; FDTD
    T=12 peaks 2680→1816 B with the minimum at K≈√T, ops rising
    monotonically — the curve, measured), and `fold_slots=S` now runs
    BINOMIAL revolve (Griewank & Walther) over the SAME certified pieces: a
    recursive schedule that stores a checkpoint at each split point, follows
    the optimal C(S+r,S) offline frontier, keeps ~O(S·state) live
    checkpoints, and needs NO divisibility on T (FDTD T=24, out=final:
    revolve holds the peak to 856 B at S=1 — below uniform's 1384 B √T floor
    — while paying recompute; gradients bit-identical across store-all,
    uniform, and revolve; measured). The two knobs are mutually exclusive
    (raise if both given). Still open: K/S is global per grad call rather
    than per-fold; and only `fold_segments` carries the divide-T constraint
    (revolve lifts it).

28. **The traffic model (placement.py) is v1-coarse, loudly.** Ring
    all-reduce and all-gather formulas only; distribution and shard-views
    cost zero (weights assumed pre-placed — loading is unmodeled); no
    overlap or topology (alpha-beta over a single per-level link); lattice
    surgery on bound dims, and scan/fold ALONG a bound dim, refuse rather
    than guess. Gradients now CARRY bindings (restamp binds; backward
    repeats rebind), so traffic covers training steps — but the backward's
    collectives are UNFUSED (Megatron joint: 6 all-reduces where fused f/g
    operators give 4) and no resharding/ZeRO exists. Alignment now
    refuses cross-placement operands (D17 at L3); the fix recipe it quotes
    is a collective, but applying it is still the caller's conscious act.
