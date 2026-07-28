# 211 — The stratification handoff (graphics campaign → L4)

**For the L4 team, on claiming the tier-stratification task
(owner-cleared, PR #7 thread).** This is the handoff you asked for:
the owner rulings your design must implement, the raw
translatability/admission data from our spikes, the runtime-side
admission implications you asked about, and our roadmap so neither
team surprises the other. Line references are against
main@4f7aa19 unless a spike file is named; every spike claim is
rerunnable from `explorations/graphics/` on the
`worktree-graphics-design` branch (now on origin), with a FINDINGS.md
per spike as the durable record — the Metal spikes cannot rerun on
your Linux/RTX box, so those files are the evidence of record.

## 1. Owner rulings your stratification implements (from 282, 2026-07-28)

1. **The kernel rulebook (ratified).** A kernel body admits exactly:
   reading a tensor at an index, writing a tensor at an index, scalar
   math on those values, captured scalars as uniforms, and records.
   Everything tensor-shaped — the 18 layout ops, reductions, fold,
   the indexing family — is a host act applied at the call site, with
   the view passed in. All the accidental cases (below) get ONE
   refusal class; the model voice is the existing reduction refusal
   ("'red' is a host citizen here — kernels use it at a call site,
   not as a value").
2. **Control flow (ratified with an owner amendment).** The mask law
   stands (spike-measured free for branches: 1.000× vs real `if`,
   both divergence regimes). Expression `if` (`a if c else b`,
   `and`/`or` on predicates) becomes legal in ALL kernel bodies —
   compute included — exactly as the vertex tier already spells it.
   **Amendment: statement `if` in kernel bodies is admitted when its
   arms are STORE-FREE**, and only an `if` containing a store
   refuses (the law: values may branch, effects may not). This is
   statically adjudicable at lowering — a store is syntactically an
   assignment to a buffer subscript, so "does this if contain a
   store" is a trivial AST check, and the admitted form lowers by
   select-joining each variable the arms assign (the same join shape
   the value tier's `If` rule already implements; a variable assigned
   in only one arm with no prior definition refuses, as there). Loops
   remain refused at this tier (see §2, bounded loop).
3. **The coverage ledger (ratified).** Two different gap kinds get
   two different treatments: never-legal constructs REFUSE at the
   tier (your stratification); legal-but-not-yet-translatable
   constructs stay allowed on the reference but live in a **visible,
   versioned ledger** — a machine-checked table whose drift from
   reality fails a test, so device-coverage regressions can't hide in
   green skip-runs. Suggestion (ours, not a ruling): make the ledger
   the stratification's own artifact — one table of
   tier × construct → {admitted, refused, ledgered}, adjudicated by
   your mechanism, consumed by the conformance batteries to decide
   skip-vs-fail. We'll wire conformance to whatever shape you pick.
4. **The bounded loop is YOURS, whole (owner ruled option "a").** A
   fixed-max-count loop with a declared early-exit, reference
   semantics = run to the bound and mask after exit (straight-line
   for AD and analyses), backend-licensed to lower to a real `break`.
   Consumer requirements from our side, evidence-backed
   (`spike_controlflow/FINDINGS.md`): the flat form costs 2.2–4.0× on
   a 64-step raymarcher (ratio tracks 64/mean-steps; an 85%-saturated
   control scene drops to 1.09×, pinning it as skipped work); and
   today a `for` in a `@jit` body doesn't even reach flattening — it
   silently drops to the per-element oracle path
   (`_liftable`, kernel.py:597, rejects nested regions except
   `core.if`), which no device translates. Whatever you design, make
   that drop LOUD in the stratification. Our demos avoid raymarching
   until this lands; no timeline pressure from us.

## 2. The raw data you asked for

### 2a. What translates today (the device-admissible set, both targets)

Compute path (`conformance/wgsl_executor.py:123-192`; MSL twin in
`spike_metal/msl_backend.py` — identical row set): `core.param`
buffer reads, `tl.iota` (→ thread id), `core.const`/`tl.const`,
`abi.slot` uniforms, `tl.pointwise` with markers
{add sub mul div neg, the six comparisons, where, maximum minimum,
sqrt exp log tanh abs floor sin cos}, the `core.*` scalar twins
(`add sub mul div neg cmp select`), `pw.*` scalar ops (same set),
`tl.read` (computed-index gather, unchecked — reference-certified),
`tl.store` to params. Fragment adds `tl.sample` (lod 0). Geometry:
rank ≤ 2 launch lattices; buffer dims must lie in the launch lattice;
reads/stores on non-parameter tensors refuse.

### 2b. What raises `Untranslatable` (per target: BOTH, identically — the row sets match)

- Core ops with no row: `core.mod`, `core.pow` (emitted for `%`/`**`,
  value.py:33-34), `core.cast` (emitted by `float()/int()/bool()`),
  `core.tuple`/`extract`/`vec`/`field`, `core.load`/`core.store`, and
  ALL THREE region-carrying ops (`core.if`, `core.for`, `core.call`)
  — no translator descends into regions at all.
- tl ops with no row: `tl.reduce`, `tl.scan` (dialect.py:523),
  `tl.fold`, `tl.repeat_like`, `tl.random`,
  `tl.take`/`scatter_add`/`argtopk`/`argsort` (dialect.py:537-551),
  `tl.materialize`/`round_to`/`with_value_units`, and all 18 layout
  ops. Two layout ops matter most because the KERNEL EMITS THEM:
  `tl.split` (kernel.py:396, the grid bracket) and `tl.merge`
  (kernel.py:899, global-index stores) — so any multi-block/tiled
  launch is untranslatable even with a trivial body. This is the
  single biggest coverage cliff for your tiling-era work.
- Markers: `stop_gradient`; every `defmarker` composite (resolved by
  name, misses the tables); every oracle marker
  (`kernel.fn.<sha256[:10]>`, kernel.py:815).
- Autodiff: the value-tier tangent path DOES translate (it splices at
  lower time into ordinary rows — pinned by both conformance
  batteries). The region-VJP path CANNOT: `_VJP_SUPPORTED` is
  {pointwise, reduce, repeat_like, const, param, yield}
  (dialect.py:975) and `reduce`/`repeat_like` have no device row —
  the two supported sets are nearly disjoint, so `derive_vjp` output
  is untranslatable BY CONSTRUCTION. Fold steps likewise
  (`_FOLD_STEP_SUPPORTED`, dialect.py:863).
- Dtypes: devices emit f32 and bool, nothing else. Three SILENT
  narrowing sites (all should become declared or refused under your
  admission design): buffers (`ascontiguousarray(f32)` up,
  `astype(f64)` back), uniform slots (i64 `<q` slots →
  `np.float32` array — wgsl_executor.py:267/269 and the Metal twin
  identically), int consts (`tl.const` dtype int64 renders as a float
  literal).

### 2c. Near-universal vs target-bent rows (`spike_metal/rowdiff.py`)

100.0% of emitted WGSL statements (123/123) convert to MSL under
three lexical rules: cast spelling (`f32(`→`float(`), float-literal
`f` suffix, C declaration form. NO operator/builtin row differs —
`select` argument order and semantics identical, all ten math
builtins spelled the same, bool-widening identical. The target-bent
remainder is structural, not per-op: buffer declarations
(module-scope `@group/@binding` + address-space access modes vs
entry-point `[[buffer(i)]]` params + `const device`), entry-point
naming (`main` reserved in MSL), and thread-size placement (the
supersession ruling in 210). Practical consequence for you: ONE
shared expression emitter with per-target leaf/declaration hooks is
the right shape — the marker tables currently exist three times
(twice inside wgsl_executor.py alone) and should exist once.
Incidental datum: hand-written select pairs beat Metal's `clamp()`
builtin by ~1.2% — no urgency to pattern-match min/max idioms.

### 2d. What the kernel tier ADMITS by accident today (the probe)

`explorations/graphics/probe_kernel_bleed.py`, rerunnable: `flip` in
a `@compute` body LOWERS AND RUNS on reference (then would be
`Untranslatable` on device — the silent reference/device gap the
ledger ruling exists to close); `shift`/`pad`/`select` fail but on
incidental grounds (containment/alignment errors from downstream
machinery — not a tier refusal); `red.sum` refuses properly (the
model voice); all statement control flow refuses properly. Mechanism,
as you noted on the thread: `KERNEL_RULES = {**TL_RULES, ...}` —
inheritance of the whole tensor surface, illegality only surfacing
downstream or never.

### 2e. Structural facts any adjudication/emission pass must know

(Each bit us during the spikes; details in the FINDINGS files.)

- **`args` is not dataflow.** `tl.iota` carries the lattice tensor as
  args[0] and never reads it — a naive `.args` walk drags the output
  buffer into the DAG (our one silent spike bug: numerically correct,
  wrong code). The op table, not the arg list, defines dataflow.
- **`walk_region` does not descend into `n.regions`**
  (dialect.py:866-878) — moot while the splicer flattens `core.if`
  pre-artifact; bites the moment any region-carrying construct
  becomes admissible (your bounded loop will be the first).
- The `_freeze_params`/`_thaw_params` dict round-trip and
  attrs-sorting trap you already carry in the excavation notes apply
  to any new admission metadata.
- The conformance battery has NO `where`/`select` subject (those rows
  are live-but-untested) and its inputs are too small to reach math
  edges (Metal's tanh NaN at |x|≥44.36 was invisible at 5×7). If your
  stratification adds battery subjects, wide-range inputs and a
  select subject close both holes; otherwise we'll add them in our
  cleanup — say which you'd prefer.

## 3. Runtime-side admission implications (your question b)

- **Unified-memory adoption raises the stakes on aliasing, not on
  admission.** Under `newBufferWithBytesNoCopy:` adoption, two
  tensors overlapping in host memory are device-aliased too — the
  store seam's overlap refusal (`shares_memory` over leaves) must
  stay authoritative against HOST memory truth, and it already is.
  No new op admission follows; the existing refusal becomes more
  load-bearing.
- **Do not require contiguity of store/read targets at the tier.**
  Our residency experiment (`spike_runner/residency.py`) proved a
  kernel can read/write field views of a record buffer THROUGH the
  view's own affine layout (offset/stride indexing in the shader),
  bit-identical to the repack path — one resident buffer serving
  compute field views and the render record simultaneously. Admission
  should demand an affine layout (guaranteed by construction), never
  contiguity — contiguity is today's executor limitation, a ledger
  row, not a law.
- **OOB: never admit an op whose safety depends on a device-side
  net.** WebGPU implementations clamp storage accesses per spec;
  Metal raw `device` pointers are genuine UB; CUDA sits with Metal.
  The keying-ladder discipline (reference refuses OOB at run time;
  device runs only reference-certified cases) is the invariant your
  admission rules should preserve.
- **i64 captures are admitted today and silently narrowed to f32 on
  every device path** — under your admission design this should
  become a declared narrowing or a refusal for values outside f32's
  exact-integer range (the 210 numeric policy already names the
  principle: narrowing is declared, never silent).
- **No op's SEMANTICS may depend on residency** (transfers stay
  explicit acts; the committed-future residency contract in the
  syntax tour is unchanged by anything we found). What residency
  changes is COST — and on that, note the measurement warning in 210:
  host repack is 88–100% of launch cost today, so any admission or
  scheduling argument of the form "X is fine because the device is
  fast" is unfounded until the repack path dies.

## 4. Our roadmap (so you can plan around us)

Sequencing agreed with the owner: **you merge the stratification; we
rebase the campaign on top of it and continue.** We are NOT
implementing tier admission ourselves — we wait for yours. What we
build after the rebase, in order:

1. **A new workspace package for runtimes + backend emitters**
   (owner-ratified; name proposed in its own PR before code moves).
   Contents: the device registry (feature requests at creation — our
   spikes had to monkeypatch the singleton to reach timestamp
   queries), the WebGPU and Metal runtimes, the shared expression
   emitter with per-target hooks (killing the three table copies),
   and device compilation routed THROUGH the existing
   `(region.key, executor fp)` content door — your PR-thread
   suggestion, adopted as stated. `conformance/` shrinks back to
   tests only, importing translators from the package.
2. **Backend/runtime selection spelling** (owner-ruled): a generator
   dataclass and a runtime dataclass (both empty for now), a pair
   dataclass holding one of each; user-facing names (`metal`,
   `webgpu`) are pre-assembled pair instances; a public
   `kernel.artifact(...)` door so a compiled artifact can exist
   without launching.
3. **The encodable + frame runner** (shape proven in
   `spike_runner/FINDINGS.md`): compile-once
   `compile_pso(pso, inputs, args, target, residency)`; per frame
   `bind(**slots)` (queue-touching) strictly separate from
   `encode(pass)` (encoder-touching, host-owned pass); target format
   keys the artifact, resolution provably must not. **Depth buffering
   enters graphics v2 here** (owner-ruled): `position` grows z, the
   reference rasterizer gains a z-buffer, the pass declares a depth
   attachment.
4. **Residency direction only** (full design later, with the
   encoding-descriptor work): `Buffer.device` becomes a reference
   into the runtime/device registry, residency becomes a set. The L2
   epoch/ownership handshake remains the eventual owner of this; we
   build only what the frame loop needs.
5. **Demos.** Procedural/analytic-AA and mesh demos first;
   raymarching waits for your bounded loop.

**What we know we might do but haven't committed to:** an async/
persistent-surface readback path (210 names it; nothing needs it
until a demo does); texture residency (the current `upload()` makes a
texture per call — we'll hit it when a textured demo lands);
if-reconstruction as a backend peephole (built, proven, shelved —
`spike_controlflow/ifrecon.py`, ~135 LOC when a compute-bound
consumer appears).

**What we genuinely don't know yet (de-risk by not building against
it):** the final package name/layout; whether the encodable API
survives MRT + depth + textures unchanged (we'll prototype against
the spike code before freezing anything); who fixes `Tensor.to_numpy`
— the per-element materializer is reference-tier code adjacent to
your excavation, and its fast path (a strided numpy view over the
buffer) may be cheapest done from your side; we'd take it in our
cleanup otherwise. Flag which you prefer — it's the single
highest-leverage small fix in the tree (88–100% of every device
launch).

## 5. De-risking notes, both directions

- Your stratification touches frozen refusals; so does the
  thread-sizing supersession (the geometry refusal repin, ruled on
  this PR's thread). If you're repinning the battery anyway,
  bundling both repins in one deliberate change avoids two API-break
  events.
- We will not touch `pdum/tl/kernel.py`, `dialect.py`, or the rule
  packs until your merge lands — the campaign branch adds only new
  files (docs + explorations) precisely so the rebase is trivial.
- If the ledger becomes your artifact (§1.3 suggestion), give it a
  stable machine-readable form early — our conformance wiring and
  your admission checks then can't drift apart.
- Anything you want verified on Apple hardware (Metal behavior,
  wgpu-on-Metal quirks), send it our way — that's the one thing your
  box can't do, and our spikes left a working PyObjC harness
  (`spike_metal/metal_runtime.py`) ready to point at new questions.
