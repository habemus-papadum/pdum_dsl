# 195 — Indexed computation: take, routing, taps, and the unrolled trainer

**Status: proposal against the 190 spec, pre-P0.** Three threads, one
document: (1) `take` re-founded as a first-class *computation* in the
tensor tier, with the factoring that makes the whole indexing family
nearly free; (2) taps — the scope's output side, which dissolves the
"functional programs hide their internals" worry; (3) the unrolled
autoregressive trainer as the forcing exercise, worked end to end.
Adoptions amend 190 where §6 says so. One small notation fix rides
along (§1).

---

## 1. The pipe requires marked units (notation fix)

Plain Python functions do not overload `|`. The assemblage pipe
composes **Unit objects**: a unit is created by the `@unit` decorator
(or `unit(fn)` inline), remains an ordinary callable, and carries the
`|` overload — the same move the kernel-tier fuse pipe makes with its
stage objects. Makers return decorated units; combinators (`seq`,
`pipe`) accept and return Units. Library tensor-functions need no
marking — they are called, never piped. 190's samples read as if bare
closures compose; the normative form is:

```python
def make_attn(s, cfg):
    ...
    @unit
    def attn(h):
        ...
    return attn
```

## 2. `take` is a computation, not a view

**The re-founding principle.** The layout algebra is affine and
data-independent — that is what makes alignment decidable, adjoints
derivable, and the whole white-box story possible. Gather looked like
it threatened that: a "view" whose addresses depend on runtime data
would wreck the layout schema. The resolution is to refuse the premise:
**`take` is not a view. It is a computation** — in the loose sense that
it either moves memory or computes — and like every computation it
**materializes a fresh, plainly-laid-out tensor**. The layout algebra
never learns about data-dependent addressing; the output of `take` is
as boring as the output of `pointwise`. Zero-cost data-dependent views
never enter the system.

**Semantics.** `take(table, idx, dim="v")`: `table` over dims
`(v, d)`, `idx` an integer-carrier tensor over dims `(t,)` →
output over `(t, d)`, `out[t, d] = table[idx[t], d]` — the indexed dim
is replaced by the index tensor's dims. The reference executor
**refuses out-of-range indices loudly**; device-tier behavior
(clamp/UB) is a descent-license matter, decided there, never silently.

**The adjoint pair.** `take† = scatter_add`:
`d_table = scatter_add(zeros_like(table), idx, d_out)` — duplicates sum
(the same token appearing twice in a batch accumulates both
contributions, which is exactly the embedding gradient), and addition
makes the result order-independent, so the adjoint is deterministic by
construction. `d_idx = None` (integer, gradient-free). `scatter_add` is
itself user-facing (routing needs it) and its adjoint is `take` — a
self-dual pair, like `repeat† = reduce`.

**The factoring that makes the family nearly free.** Every other
indexing operation decomposes into a **gradient-free index producer**
plus `take`:

- `argtopk(x, k, dim)` → integer indices; gradient-free, **no adjoint
  rule needed**. Then `topk values = take(x, argtopk(x, k), dim)` — and
  the values' gradient comes out *via take's scatter-add*, correct by
  composition (ties resolve first-wins, inheriting 190's partition
  law). No multi-output instructions needed — confirmed against the IR:
  instructions stay `(var, op, operands, params)`, single-output.
- `argmax = argtopk(k=1)`. `argsort` likewise: indices gradient-free;
  any differentiable reordering is `take` by those indices.
- `concat` (needed below) requires **no new op at all** today:
  `concat(a, b, dim) = a.pad(after) + b.pad(before)` — two guarded pads
  and an add, materializing exactly once. A dedicated materializing
  `concat` is a later cost optimization, not semantics.

So the entire indexing family costs the tensor tier exactly **two
differentiable primitives** (`take`, `scatter_add` — adjoints of each
other) **plus a family of gradient-free index producers** (`argtopk`,
`argsort`, …) that never touch the derivative table. This is the
derive-don't-enumerate discipline applied to indexing, and it is why
the original omission is cheap to repair.

**Cost semantics.** `take` counts one read+write per output element
(opcount gains a "gather" counter); it allocates its output
(peak-memory: a real node, never a free view); under placement, a
`take` whose table and indices live on different machine levels is a
data-dependent exchange — the traffic pass refuses cross-placement
`take` in v1 (D17: quote the fix — colocate or all-gather the table),
with modeled all-to-all as later work.

## 3. MoE top-k routing, in the subset

With the family above, capacity-factor MoE (Switch-style) is
expressible **with static shapes** — no dynamic-shape boundary crossed:

1. `logits_e = contract(h, w_router)`; `sel = argtopk(logits_e, k,
   dim="e")` (gradient-free); `gates = softmax(take(logits_e, sel))`
   (differentiable through take).
2. Position assignment by **cumsum over a one-hot mask** (`scan` —
   already a primitive): each token's slot within its expert's
   fixed-capacity buffer `(e, cap, d)`; overflow beyond `cap` is
   masked out (`where`) — the standard capacity-factor semantics,
   dropped tokens and all, stated honestly.
3. `scatter_add` tokens into the `(e, cap, d)` buffer (slots are unique
   by construction — injective scatter); run every expert dense over
   its buffer (a `seq` over experts, or one batched contract with `e`
   as a named dim); `take` results back to token order; weight by
   `gates`; combine.

Fully dynamic MoE (true variable-size dispatch) remains a recorded
boundary — it is a dynamic-shapes problem, not an indexing problem.
The capacity-factor form becomes a zoo entry (§6).

## 4. Taps: the scope's output side

**The worry:** a functional model is a black box — write GPT-2 once,
and you cannot later get its internals (activations for visualization,
KV for reuse) without rewriting it. **The dissolution:** internals are
*named potential outputs*, and the scope — which already owns names on
the input side (parameters) — gains the symmetric output side.

`tap(x, s / "k")` marks a value as a potential output under its site
path and returns `x` unchanged. Two consumers, one mechanism:

- **Within one build** (the unrolled trainer's case): tapping writes
  the value *reference* into the scope's build-time registry — the same
  class of build-time collection as `s.param`. The enclosing builder
  reads `s.taps["h.3.attn.k"]` after applying a unit and wires it
  anywhere — **ordinary dataflow**, no export machinery, no state, no
  mutation. Within a single Program, KV reuse is just SSA.
- **Across the run boundary** (visualization, probing, serving):
  requested taps become **additional named Program outputs** —
  `model.outputs(["logits", "h.*.attn.k"])` selects by name pattern,
  and `run` returns them in the name-keyed result. Unrequested taps
  are **pruned by DCE and cost nothing** — this is the
  requested-gradients machinery generalized to requested-*outputs*:
  taps are the output-side dual of the keep-set, running through the
  identical mechanism.

Purity is intact (a tap is an identity with a name; selection is
post-hoc, like freezing); the naming law carries it (tap sites are
contract names); mode interplay is free (a tap under `mode="eval"`
regions works like any site). Visualization of internals stops being a
framework feature and becomes a *query*: name patterns over tapped
sites, exactly like freezing is a query over parameter names.

**Selection, identity, and the derivation request.** The training step
is derived by one request carrying three name-pattern selection sets,
all identity-bearing:

```python
step = compile(loss_fn, wrt=trainable, taps=["h.*.attn.k", "h.*.wq.grad"])
```

— the keep-set (`wrt`), the output set (`taps`), and freeze overrides.
A different tap set is a different derived Program, cached like any
other; bufferization sees exactly which values escape, so buffer reuse
is planned against the true output set. The derived step is **one
Program** — forward, loss, backward, optimizer — with updated
parameters among its outputs; in-place update is L2 buffer donation,
never syntax. **Sites vs activation:** sites live in code (declared at
scope paths, possibly inside policy regions — a site under a
`mode="eval"` region exists in that build); activation is the
derivation-time pattern set; policies may contribute *default*
patterns, but the request is the truth. Inside a `fold`, an activated
tap along the scan dim is the fold's `emit` — per-step stacking already
has its semantics.

**Gradient names are derived names.** The naming law's derived-suffix
set (`name.d{i}`, `.rc`) gains `.grad`: the cotangent of any named
value `x` is `x.grad` — parameter gradients (`h.3.attn.wq.grad`) and
activation gradients (`h.3.attn.k.grad`) are selectable by the same
patterns through the same DCE that prunes unrequested gradients today.
**Names and the compiler:** contract names — inputs, outputs, tap
sites, and their derived forms — are **ABI: linker symbols** that
legitimately survive compilation, exactly as symbol names survive a
linker; interior names are debugging metadata excluded from
content-addressing; compiler passes operate on structure with names
riding as annotations. The scope needs no redesign for this — it is
what the contract/internal division was for.

## 5. The unrolled autoregressive trainer, worked

The exercise: unroll the sampling step K=2–3 times inside training —
chunk 1 is the normal forward over the sequence (loss over all
positions); between chunks, sample the next token by top-k with
temperature; chunks 2, 3 predict one step further, **reusing chunk 1's
KV** (elision, not mutation); the total loss sums all chunks — making
sampling parameters (temperature τ) trainable. (Prior-art honesty:
close relatives exist — scheduled sampling, professor forcing,
self-critical training, ST-Gumbel — the specific
temperature-as-trained-parameter arrangement may or may not be novel;
this document's question is only expressibility.)

**Backprop through sampling needs a declared estimator.** A discrete
sample has no derivative; the spec-honest options: (a)
**straight-through Gumbel** — sample hard forward, gradient of the
relaxed distribution backward; (b) full Gumbel-softmax relaxation; (c)
score-function/REINFORCE (needs stochastic-node machinery — out of
scope, recorded). Everything (a)/(b) need already exists:

- Gumbel noise is a **derived closed-form field**:
  `g = -log(-log(uniform(stream, layout)))` — the §1.7 randomness
  doctrine, no new mechanics; reproducible, recompute-exact,
  device-lowered as Philox + `log`.
- Temperature enters as `softmax((logits + g) / τ)` with
  `τ = s.param("tau")` — an ordinary trainable leaf.
- Top-k restriction = `argtopk` + mask (`where` over a
  scatter-of-ones), gradient-free where it must be.
- Straight-through = `hard + (soft - stop_gradient(soft))` —
  `stop_gradient` is already in the spec. The hard token then feeds
  `take(wte, token)` for the next chunk's embedding.

**KV reuse without mutation.** The whole unrolled trainer is **one
Program built by one builder**, and that changes everything: weight
sharing across chunks is automatic (the same `wq`/`wte` objects
captured in every chunk = one leaf each, gradients summed across the
unroll — the tying rule doing macro-scale work for free), and KV
"reuse" is ordinary dataflow via build-time taps. The decode-form
chunk is the same *library* core with different wiring:

```python
def make_gpt2_unrolled(s, cfg, K=3):
    gpt   = make_gpt2_parts(s, cfg)          # same makers; units tap k, v per layer

    def trainer(ids):
        h, logits1 = gpt.trunk(gpt.embed(ids))          # chunk 1: full forward
        kv    = s.taps.collect("h.*.attn.{k,v}")        # build-time references — just SSA
        loss  = xent(logits1, shift(ids))
        tok   = ids
        for j in range(1, K):                            # host-level unroll
            nxt    = st_gumbel_topk(last(logits1), k=50,
                                    tau=s.param("tau"),
                                    stream=s / ("sample", j))
            e_new  = take(gpt.wte, nxt, dim="v") + gpt.wpe_at(len_plus(j))
            h_new, logits_j, kv = gpt.decode_step(e_new, kv)   # concat'd attention:
                                                               # new position only
            loss   = loss + xent(logits_j, target_at(j))       # chunks 2..K
        return loss
    return assemblage(trainer)
```

`decode_step` is the honest cost of "write it once": the *library*
layer (norms, projections, `attention(q, k, v)`) is 100% shared; the
*maker* layer gains one decode-form wiring (~5 lines per block: compute
q/k/v for the new position, `concat` with the tapped kv — pad+add
today — attend). Deriving the incremental program automatically from
the full one is program incrementalization — real research, honestly
out of scope, recorded as a far-future note, not promised. What the
design does deliver: GPT-2's trunk is written once, *without knowing*
about unrolling; the unrolled trainer is a different **builder** over
the same makers and taps; nothing in the model file changes except two
`tap` lines per attention block, which cost nothing when unrequested.

Memory honesty: K-step unrolling multiplies saved activations by ~K;
this is precisely what min-cut checkpointing and revolve exist for, and
the recompute theorem (§190 1.7) means the Gumbel draws replay exactly
under recompute — the unrolled trainer checkpoints *correctly by
construction*.

## 6. Control flow: branching is a host right

190's standing invariant — control flow confined to value-language
kernels and the host — has a positive form, and it is a doctrine, not a
workaround: **model-level branching is host-level branching over cached
linear segments, and the cache polarity is what makes it cheap.**

**The shape.** A *prolog* Program runs (linear, as always); its
predicate outputs — a boolean, five values, whatever the decision needs
— are read back to host; host Python decides which *branch* segment
runs next. Each branch is an ordinary Program: built on first visit,
compiled once, content-cached forever. Revisiting a branch is a warm
hit with zero recompiles — the live-knob thesis extended from values to
*paths*. Nothing branch-shaped ever enters the IR: no phi nodes, no
divergence, no conditional adjoints; every invariant of the
straight-line representation survives untouched, because control flow
was never in the representation to begin with.

**Three tools, complete.** Every control-flow need has exactly one
designated tool, and each preserves straight-line semantics:

1. **Element-wise selection** → `where` in-program: both sides
   computed, masked — the SIMT-honest form.
2. **Per-example routing within a batch** → the §3 regrouping idiom:
   argtopk/scatter/take reshape the batch so each branch runs dense
   over its members (MoE is this).
3. **Program-level decisions** → host branching over cached segments —
   this section.

**Gradients across the joint need no new machinery.** The backward of
the taken path is VJP chaining over named segments: the branch's
backward produces cotangents with respect to its inputs — which are the
prolog's *named* outputs — and those seed the prolog's VJP (explicit
seeds are already the non-scalar-target contract). Cross-joint
activation reuse is taps (§4). For small branch counts the joint
program (prolog + branch fused) is equally valid and cheaper at run
time; both are cached derivations. The unrolled trainer (§5) is
precisely this pattern with the decision replaced by sampling — the
mechanics are identical.

**JIT-on-demand over unbounded branch spaces.** The branch space may be
combinatorially large or infinite — architectures that recurse over
per-example structure (tree-shaped models), adaptive depth/early exit,
data-statistic-dependent topologies. Compilation cost scales with the
number of *distinct structures visited*, not with steps: each
encountered structure compiles once and is warm thereafter. This is
where the doctrine earns "transformative": structure-dependent
architectures that mainstream compilers handle with guard soup or
retracing are, here, just host programs over a content-addressed
program cache.

**Honest costs.** (i) Each joint is a host synchronization — a
readback; this is the *right* cost at program-level granularity (the
host must know to decide) and the wrong tool below it (use tools 1–2);
device-era pipelining across joints is an L5 concern. (ii) High branch
cardinality with low revisit rates pays build cost per visit — the
mitigation is the same bucketing idiom as sequence lengths (collapse
the cardinality, mask the difference). (iii) If the branch *choice*
itself must receive gradients, that is the discrete-choice estimator
question of §5 (straight-through/relaxation), declared at the site —
routing without learned choice has no such question.

## 7. Incremental compilation is a cache phenomenon

A training loop derives many programs from one model: the training step
(forward; loss; backward; optimizer), the validation loss (forward;
loss), the eval forward — plus every branch and unroll chunk of §5–§6.
The doctrine: **incremental compilation is not a compiler mode; it is
what the existing caches already do**, and no special mechanism may be
built for it. "Compilation" here is four memoized tiers, each sharing
work by content:

1. **Build** — Programs are content-addressed (tier 2); identical
   subprograms are identical entries.
2. **Derivation** — grad/DCE/checkpoint transforms are
   derivation-under-cache: cache entries computed from cache entries;
   the training step composes the *cached* forward rather than
   rebuilding it.
3. **Descent** — the certified-lowerings registry is chunk-granular,
   keyed by (chunk fingerprint, boundary contract, licenses, …): the
   training step's attention chunk and the eval forward's attention
   chunk hit the *same entry* whenever key and contract agree.
4. **Kernel codegen** — per-artifact, content-addressed as always.

**Warm start without constraint.** Reuse is an *outcome* of content
addressing, never an input to the compiler. Whole-program optimization
of the training step may legitimately choose different work for the
same source — the canonical case is already in the spec: training-flash
must export the logsumexp (the saved-set demand, which is *in the
registry key*), eval-flash needn't — so those are two entries and the
eval-compiled chunk simply does not hit. No staleness, no invalidation
heuristics: where the optimizer's choices coincide, artifacts are free;
where they diverge, the miss is the correct answer.

**Why the sharing rate is high in practice.** Within one codebase, the
eval forward and the forward inside the training step are built by the
*same makers*, so their subgraphs are syntactically identical — sharing
works before the Program-normalization pass exists; normalization later
*broadens* sharing across differently-spelled sources, it does not
enable it. One honest split: `mode="eval"` changes the built Program
(dropout gone), so train- and eval-forward differ at the top and share
at chunk granularity beneath (the contractions and dropout-free cores
coincide). The branching doctrine (§6) multiplies the payoff: every
host-orchestrated segment draws from the same memoized substrate.

**The one new commitment.** The L4 descent *search* may consume a
sibling program's plan as a **non-binding hint** — seeding the
training step's partition search from the eval forward's chosen
boundaries, cost-model-checked, freely discarded. Correctness never
depends on hints; this line carries into the L4 brief.

## 8. Amendments to 190

1. **Units are marked** (§1): the assemblage pipe composes `@unit`
   objects; makers return them; samples updated accordingly. Lands
   with the scope (P5).
2. **The indexing family enters the primitive set** (§2): `take` +
   `scatter_add` (one adjoint pair) + gradient-free index producers
   (`argtopk`, `argsort`) + reference OOB refusal + cost entries +
   the cross-placement refusal. `concat` = pad+add idiom now,
   materializing op later. Lands at P9 as already scheduled — the
   factoring keeps it small — with two new zoo entries: the embedding
   sample and **capacity-factor MoE** (§3).
3. **Taps join the scope** (§4): `tap(x, site)` + build-time
   collection + requested-outputs selection running through the DCE
   machinery (the output-side keep-set). Lands with the scope (P5);
   the requested-outputs DCE generalization lands with grad wiring.
4. **Sampling idioms are library, not mechanics** (§5): Gumbel field,
   temperature-softmax, top-k masking, straight-through — all derived
   over existing primitives; recorded as zoo/library code with the
   estimator named explicitly at the call site. Score-function
   estimators are a recorded boundary.
5. **The unrolled trainer becomes a zoo entry** once P9's indexing
   family lands — it exercises take, taps, ST-Gumbel, trainable τ,
   weight-sharing-by-capture, and checkpointing-with-replay in one
   program, which makes it the natural end-to-end gate for this whole
   document.
6. **Selection and derived names** (§4): the three-set derivation
   request (`wrt`/`taps`/freeze, identity-bearing); 190 §1.5's naming
   law gains the derived-suffix clause (`.grad` joins `name.d{i}`/`.rc`)
   and the linker-symbol sentence; 190 §8's L2 blocker list gains the
   requested-output sets as a bufferization input; P5's gate gains a
   tap-set identity pin (different tap sets never share a derived
   Program).
7. **The branching doctrine** (§6) enters canon as the positive form of
   190 §1.2's control-flow invariant: three tools (where / routing /
   host branch), VJP chaining across named segments, JIT-on-demand over
   branch spaces. No mechanics required — it is a consequence of the
   cache polarity; an adaptive-depth or tree-structured zoo sample is a
   natural later addition.
8. **Incremental compilation as doctrine** (§7): no mechanism is ever
   built for it — the four memoized tiers are the mechanism; reuse is
   an outcome of content addressing, never a compiler input; the
   descent-search plan-as-hint line (non-binding) joins the L4 brief.
