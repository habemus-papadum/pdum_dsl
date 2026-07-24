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

## 6. Amendments to 190

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
