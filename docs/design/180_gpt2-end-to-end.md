# 180 — GPT-2 end to end: the user experience and the compiler's view

**Status: red-team exercise against the 170 spec, run before P0.** Two
questions. First, the user level: one GPT-2 definition — weights in
closures, sub-pipelines composed with `|` — must serve checkpoint loading,
controlled random initialization, zero-allocation analysis, and a training
loop with selective freezing, with no wasted allocation or copying
anywhere. Second, the compiler level: is the representation at the right
granularity for the optimizations we intend (distribution, tiling,
fusion, checkpointing), or does the primitive language hide what the
optimizer needs to see? Findings and the amendments they force on 170 are
in §6. Headline: **no structural blocker found; two genuine gaps
(parameter provisioning, gather) and one ruling (`|` at the assemblage
tier) must be adopted; the RNG/dropout stance is specified here and
GPT-2's three dropout sites exercise it.**

---

## 1. The model, as a user writes it

GPT-2 per the zoo's canon: heads are born as dims (`wq: (d, nh, hk)` —
never split into existence), the causal mask is an iota comparison (a
closed form, not memory), and token embedding is treated in §5f. All code
below is the 170 surface (S.1/S.2 syntax); everything it needs already
exists or is committed in P0–P9 except where §6 says otherwise.

**The two-layer discipline.** Model code has exactly two layers, and only
one of them knows about parameters:

- **The standard library is parameter-blind.** `layernorm`, attention
  cores, dense blocks — functions from tensors to tensors, with
  parameters passed as *ordinary arguments*. No namespace, no names, no
  knowledge that provisioning exists. Architecture variants (Llama =
  rmsnorm + RoPE + GQA) reuse these wholesale.
- **The binding layer owns names.** Concrete model code (`make_attn`,
  `make_gpt2`) holds the parameter namespace, declares leaves under
  chosen names, and hands them to library functions as arguments. Name
  assignment happens in exactly one visible place — never inside library
  code. This is deliberate: the naming law is load-bearing (loading,
  init, freezing, and RNG streams all join on names), so names are never
  produced by module-internal magic, only by binding-layer declaration.

### 1.1 Device functions and blocks

```python
GELU_C = 0.7978845608028654

def gelu(x):                                   # value-language body; partials derive
    return 0.5 * x * (1 + tanh(GELU_C * (x + 0.044715 * x*x*x)))

def layernorm(x, g, b, *, feat, eps):          # assemblage helper: ordinary Python
    mu = x.mean(feat)
    xc = x - mu.repeat(feat, x.extent(feat))
    sd = ((xc * xc).mean(feat) + eps).sqrt()
    return xc / sd.repeat(feat, x.extent(feat)) * g.repeat_like(x, but=feat) \
           + b.repeat_like(x, but=feat)

def causal_softmax(sc, *, q="t", k="s"):
    mask = iota_of(sc, k) <= iota_of(sc, q)    # closed form; costs nothing
    sm   = where(mask, sc, const_like(sc, -1e9))
    e    = exp(sm - sm.max(k).repeat_like(sm, but=None, dim=k))
    return e / e.sum(k).repeat_like(e, but=None, dim=k)
```

### 1.2 The blocks, weights in closures, dropout in place

`p` is a parameter namespace (§2). Blocks **declare their leaves at first
use** — `p.param(name, **dims)` with dim names as keyword keys and extents
as values — and receive back tensors: virtual, loaded, or initialized, the
block code cannot tell. Declaration is idempotent (re-declaring the same
name with the same shape is the same leaf; a conflict refuses). `key` is a
**program input** (an ordinary value); blocks close over the key *symbol*
and derive per-site streams by contract name. `training` is host-level:
two cached Programs (train/eval), no runtime mode flag.

```python
def make_attn(p, cfg, key, *, training):
    D, H, K = cfg.d, cfg.nh, cfg.hk
    ln1g, ln1b = p.param("ln1g", d=D), p.param("ln1b", d=D)
    wq = p.param("wq", d=D, nh=H, hk=K)
    wk = p.param("wk", d=D, nh=H, hk=K)
    wv = p.param("wv", d=D, nh=H, hk=K)
    wo = p.param("wo", nh=H, hk=K, d=D)

    def attn(h):
        a  = layernorm(h, ln1g, ln1b, feat="d", eps=cfg.eps)
        q  = contract(a, wq)                           # unique shared axis: "d"
        k  = contract(a.rename(t="s"), wk)
        v  = contract(a.rename(t="s"), wv)
        sc = contract(q * cfg.scale, k, axis="hk")     # "nh" rides; axis named to
                                                       # break the genuine ambiguity
        pr = causal_softmax(sc)
        if training:
            pr = dropout(pr, cfg.p_attn, key / "attn") # site-named stream
        cx = contract(pr, v, axis="s")
        o  = contract(cx, wo, axis=("nh", "hk"))
        if training:
            o = dropout(o, cfg.p_resid, key / "attn.out")
        return h + o
    return attn

def make_mlp(p, cfg, key, *, training):
    D, M = cfg.d, cfg.m
    ln2g, ln2b = p.param("ln2g", d=D), p.param("ln2b", d=D)
    w1, b1 = p.param("w1", d=D, m=M), p.param("b1", m=M)
    w2     = p.param("w2", m=M, d=D)

    def mlp(h):
        a = layernorm(h, ln2g, ln2b, feat="d", eps=cfg.eps)
        m = gelu(contract(a, w1) + b1.repeat_like(a, but="m"))
        o = contract(m, w2)
        if training:
            o = dropout(o, cfg.p_resid, key / "mlp.out")
        return h + o
    return mlp
```

Note the shape of the discipline: `attn`/`mlp` bodies call only
parameter-blind library functions with the declared tensors as plain
arguments; every naming decision is a `p.param(...)` line at the top of
the binding function. An architecture edit — swapping layernorm for
rmsnorm, adding a gate weight — touches exactly these lines and nothing
else in the file.

### 1.3 Assembly: sub-pipelines with `|`

```python
def make_gpt2(P, cfg, key, *, training):
    wte = P.param("wte", v=cfg.v, d=cfg.d)             # declared ONCE — tied below
    wpe = P.param("wpe", t=cfg.t_max, d=cfg.d)
    lnfg, lnfb = P.param("lnfg", d=cfg.d), P.param("lnfb", d=cfg.d)

    def embed(ids):
        tok = wte.take(ids, dim="v")                   # gather — §5f/§6
        e   = tok + wpe.slice(t=(0, ids.extent("t")))
        return dropout(e, cfg.p_embd, key / "embd") if training else e

    blocks = pipe(
        make_attn(P / f"L{i}", cfg, key / f"L{i}", training=training)
        | make_mlp(P / f"L{i}", cfg, key / f"L{i}", training=training)
        for i in range(cfg.layers)
    )
    def head(h):
        hf = layernorm(h, lnfg, lnfb, feat="d", eps=cfg.eps)
        return contract(hf, wte, axis="d")             # TIED: the same object

    model = embed | blocks | head
    return assemblage(model)                           # Handle: capture + 2-tier cache
```

**The `|` ruling this forces (adopted into 170):** at the assemblage tier,
`f | g` over tensor-functions is **build-time function composition
threading one value** — the same fuse semantics `|` already means, realized
as program-fragment composition into one Program. It is legal precisely
because it is *not* dispatch sequencing: nothing here launches. The unary
constraint holds (one threaded value, `h`); `key` and the parameters ride
as closed-over program symbols, not threaded values — which is what makes
the user's `attn | mlp` sketch work without widening the pipe. `pipe(...)`
over a generator is the n-fold spelling of the same operator.

**Weight tying is the identity rule:** `wte` is declared once and the
same object is captured by both `embed` and `head` → **one input leaf**
(capture identity, not name-string, decides). Its gradient is
automatically the sum of both uses' contributions because there is only
one leaf to seed. Pinned by a test (§6.4). GPT-2 makes this unavoidable;
the rule was implicit in 170 and is now explicit — and declare-at-use
makes it visually obvious: tying looks like what it is, one declaration
used twice.

---

## 2. Provisioning: one definition, three materializations

**The resting state of a model is virtual, and the builder is the single
source of truth.** A fresh namespace holds nothing; running the builder
against it *collects* the spec — every `p.param(...)` declaration
registers a leaf (name, dims, extents, carrier; **no buffer**), and the
blocks capture exactly the tensors they declared. They cannot tell
virtual from real, because `typeof` is identical (layout + carrier; no
buffer in the type). There is no separately-maintained parameter table to
keep in sync with the code:

```python
P     = namespace()
model = make_gpt2(P, cfg, key, training=True)   # builds AND collects
P.spec()          # the full table, derived: "L0.wq": (d:768, nh:12, hk:64), ...
```

A schema-first door remains for those who want it — `namespace(schema=…)`
validates each declaration against a written table, and provisioning
validates against a checkpoint's manifest either way — but the code is
authoritative.

**What `P` is (and is not).** A string-keyed namespace whose **flat name
space is primary** — `P / "L3"` is a prefix *view*, not a container
boundary — and which exists **only at build time**: after the build, the
Program has named inputs, and runtime state (weights, grads, moments) is
plain name-keyed dicts. It is deliberately not a pytree subsystem: pytree
machinery (treedefs, container registration, positional flattening)
exists to turn arbitrary containers into positional argument lists, and
nothing here consumes positions — Programs, `grad` maps, provisioning,
and optimizers all **join on names**, so "zip params with grads with
moments" is a dict join. The whole object is on the order of a hundred
lines.

**Scenario 3 — analysis, zero allocation.** Virtual is not a mode; it is
the unprovisioned state, and it is also the collector. The build above
already produced the full Program (layouts are known);
`ops_count(model.program)`, `peak_memory(model.program, schedule)`,
traffic and placement analysis all read layouts and never values. Nothing
allocates. Only `ir.run`/`item()` refuse, quoting the fix ("provision the
parameters"). **Cache dividend, pinned as a test:** the virtual build and
the provisioned build have identical types, therefore identical
fingerprints — analyze first, provision later, and the Program cache hits
warm.

**Scenario 1 — load.** `weights = provision(P, source=safetensors("gpt2.st"))`.
safetensors is an mmap format: each entry becomes a boundary descriptor
over the **mmap'd region directly** — Buffer (DLPack shim over the map) +
Layout (from shape metadata) + Encoding (the file's dtype: f32, or bf16 as
a *fact*). Zero copies on the host; exact decode per 170 §4. Names join on
the contract names — the checkpoint's `"L0.wq"` is our `"L0.wq"` (a
translation table handles foreign checkpoints' naming schemes; it is data,
not code). Tied weights stored once in the file arrive as one Buffer → one
descriptor → one leaf, preserving §1.3's rule. Device transfer, when
backends exist, is one explicit copy per buffer at provisioning — never
per dispatch, never allocate-then-overwrite.

**Scenario 2 — init.** Strategies are keyed by name pattern; randomness is
the §4 machinery, so initialization is *reproducible and per-value
controlled by construction*:

```python
weights = provision(P, source=init(
    key / "init",
    default   = normal(std=0.02),
    overrides = {
        "*.ln?g": ones,   "*.ln?b": zeros,
        "*.b?":   zeros,
        "*.wo":   normal(std=0.02 / sqrt(2 * cfg.layers)),   # GPT-2's scaled resid init
    },
))
```

Each leaf's values are the closed-form random field
`normal(fold_in(init_key, leaf_name), leaf_layout)` — materialized
**directly into the leaf's one allocation** (or, later, generated
on-device by the same field lowered to Philox threads: no host array, no
transfer). Same key → same init, forever, on any device. There is no
allocate-uninitialized-then-fill-then-maybe-copy sequence anywhere in any
scenario: virtual allocates nothing; load allocates nothing on host
(mmap) and copies once to device; init allocates once and writes once.

**Name stability under refactoring, stated honestly.** Leaf-level and
block-internal architecture edits never churn names: declare-at-use means
the edit and the name live on the same line, and RNG streams — being
name-derived, not position-derived — ride along unchanged. The one
instability is **index-derived layer names**: inserting a layer shifts
`L{i}` for everything after it, and a loaded checkpoint stops joining.
This is universal (PyTorch `state_dict` keys, Flax scopes — all
index-derived, all break identically), and the mitigation is the same
name-translation table that foreign checkpoints already require; it is
data, not code.

---

## 3. Training: trainable by default, frozen by name

All captured parameter leaves are trainable by default — "the program is
the parameter container." Freezing is name selection; the naming law is
the whole mechanism:

```python
trainable, frozen = P.partition(freeze=["L0.*", "wpe"])   # glob on contract names

step  = grad(loss_fn, wrt=trainable)      # keep-set → requested-gradients DCE:
                                          # frozen weights' backward work is PRUNED,
                                          # not just discarded
opt   = adam(trainable)                   # moment dicts keyed by the same names

for t, batch in enumerate(data):
    lr    = sched(t)                                      # live knob: value, never keys
    grads = step(weights, batch, key / ("step", t))       # per-step streams, warm cache
    weights, opt = opt.update(weights, grads, lr)
```

The tied `wte` receives one gradient (embedding and head contributions
summed — one leaf, §1.3). `key / ("step", t)` is `fold_in` twice — a new
value each step, identity unchanged, zero recompiles: the live-knob
thesis applied to randomness. Freezing changes the *keep-set*, which is a
different derived Program (different DCE result) — cached like any other;
flipping a layer between frozen and trainable alternates between two warm
entries.

---

## 4. RNG and dropout — the stance, specified

Forced by principles 170 already holds (purity, content-addressed caching,
recompute-based checkpointing), and now part of the spec:

1. **Randomness is a counter-based, coordinate-indexed, closed-form
   field.** `uniform(key, layout)` is a pure function of (key, lattice
   coordinates) — Philox-class bits, element *i* computed directly, no
   sequential state. It is a `FunctionalBuffer`-class citizen exactly like
   `iota`: zero memory, exact under view ops, free in the cost models,
   materialized only at a boundary that demands it. Bits are exact
   (`u32/2³²` is a rational) — carrier-consistent.
2. **Keys are ordinary values** — program inputs or captures, never hidden
   state; type-keyed, so new keys are warm hits. Streams derive by
   **`fold_in` on contract names and step indices** (`key / "L3.attn"`,
   `key / ("step", t)`) — insertion-stable and refactor-stable where
   positional splitting is not. `split` exists underneath; users rarely
   touch it.
3. **Dropout is an idiom, not an op**:
   `where(uniform(key, x.layout) < p, 0, x / (1-p))`. Its AD falls out of
   existing rules (comparisons gradient-free; mask acts as a constant
   field). Train/eval are **build-time variants** — host `if training:`,
   two cached Programs.
4. **The recompute theorem, pinned:** checkpointing and revolve recompute
   forward segments; the mask field regenerates bit-identically (same key,
   same coordinates), so gradients under recompute are exact *by
   construction* — no RNG-state stashing exists because none is needed.
   One test pins it: revolve-checkpointed training step ≡ store-all, with
   dropout on.
5. **Device lowering is the same story**: Philox is pure integer
   arithmetic — a value-language device function (or vendor intrinsic),
   lowered like iota→thread_idx. Oracle and device produce bit-identical
   masks, so differential testing survives dropout.

Recorded stance: all randomness is named and keyed — reproducible by
default, always; there is no "fast nondeterministic stream." Deferred:
generator ceremony beyond Philox4x32, distributions beyond
uniform/normal (composites derive), stochastic-rounding interplay,
rejection sampling (variable consumption is not straight-line — outside
the subset, refused with the boundary stated).

---

## 5. The compiler's view: what the optimizer needs to see

The granularity question, worked per optimization against this exact
model.

**(a) Distribution — GREEN.** Named dims + `bind` are the mechanism;
Megatron's tensor-parallel block already proved collectives are *read off
the algebra* (two all-reduces, discovered not written) and the placed
backward carries bindings. GPT-2 distributes by binding `nh` (head
parallel) or `d`/`m` (Megatron-style) — nothing in this model's Program
hides from the traffic pass. The batch dim, when it arrives, is one more
named dim riding every declaration.

**(b) Tiling and mma selection — YELLOW, known and bounded.** `contract`
is the `repeat·mul·reduce` normal form; L4 selects tensor cores by
*recognizing* mul-solely-consumed-by-reduce. The known miss: training
saves activations, and if an optimizer chose to save the raw product
`a*w` for backward, the "solely consumed" pattern breaks. Worked against
GPT-2: the standard saved set (inputs and normalized activations, per the
min-cut with contractions recompute-banned) does **not** save the
product, so recognition holds on this model. The residual risk is real
but already owned: it is the open mma half of the L4 brief
(pattern-match vs stated annotation), and the naive→flash move as a
registered rewrite shows the escape hatch (a recognized pattern can be
*promoted to* an annotation by a named rewrite). No new gap.

**(c) Fusion, including flash + dropout — GREEN, better than baseline.**
Flash derives from the declared online-softmax combine (its backward
derives too — no hand rule), and because the attention-dropout mask is a
closed-form field over coordinates, **fusing dropout into the flash
kernel requires materializing nothing**: the kernel computes mask bits
from (key, coords) in-register — exactly what hand-written flash
implementations do with in-kernel Philox, arrived at here by
construction. The mask's zero-cost status in the memory model is truthful
for the fused form.

**(d) Checkpointing/revolve — GREEN.** The §4.4 theorem: recompute
regenerates masks exactly. The min-cut's exact-byte capacities now read
descriptor-fed sizes (170 §4). Dropout adds `where` + comparison nodes —
pointwise, recompute-cheap, precisely what the cheap-chain heuristic
wants to recompute rather than save.

**(e) Module boundaries — GREEN with one observation.** After `|`
composition, "this was the attention block" survives in exactly one
place: the **name prefixes** (`L3.attn.*`) — which are machine-readable
and stable under the naming law. Partitioning operates on dataflow plus
names; the kernel boundary is an annotation anyway. If L4 partition
search ever wants explicit block scopes, a scope annotation is an
erasure-preserving addition in the existing style — a nicety, not a
missing representation.

**(f) The genuine gap: gather.** Token embedding (`P.wte.take(ids,
dim="v")`) is data-dependent indexing — tensorlib's recorded boundary,
which is why the zoo starts from hidden states. A user-facing GPT-2
cannot: embedding lookup is unavoidable, and its adjoint (scatter-add
into the embedding gradient) is exactly how the tied `wte` trains.
One-hot-contract is a semantically correct spelling but violates the
no-waste constraint (T·V work for T lookups). **Finding: `take`/gather
joins the primitive set** — op + layout semantics + scatter-add adjoint +
cost entries — scheduled per §6.5. Same family, explicitly deferred
beyond it: top-k/MoE routing; sampling (argmax/top-k/categorical over
logits) stays **host-side** in the inference loop for now — logits out,
host samples, next token in.

**(g) Sequence length — stated honestly.** Extents are structural
(Literal-typed) at the assemblage tier: a new `t` is a new Program.
Building is cheap and cached per length; the practical idiom is length
bucketing (pad to the bucket, mask via the same closed-form comparisons).
This is a cost to know about, not a blocker; kernel-tier staging keeps
per-shape artifact reuse underneath. Decode-time KV caching composes as
the ring/window boundary sample from the L2 runway.

**(h) What the representation genuinely cannot express — unchanged and
intended.** Data-dependent control flow inside programs (rejection
sampling, early exit), dynamic shapes *within* one Program, mutation
outside the store seam. All are stated subset boundaries with refusals;
GPT-2 needed none of them.

---

## 6. Verdict and amendments to 170

**No structural blocker.** The composition story, the caching thesis, the
naming law, the precision doctrine, and the cost/transform machinery all
carried GPT-2 without strain — and twice (flash+dropout fusion,
recompute-with-dropout) the design produces for free what mainstream
frameworks implement as special machinery. The exercise forces six
amendments, adopted into 170:

1. **`|` at the assemblage tier** (§1.3): build-time function composition
   threading one value — the same fuse semantics; `pipe(...)` as the
   n-fold form. Never dispatch sequencing.
2. **Provisioning** (§2): the namespace + `provision(source=...)` with
   virtual as the resting state and **declare-at-use as primary** —
   `p.param(name, **dims)` at the binding layer; the spec is *collected*
   by the virtual build (the builder is the single source of truth), with
   `namespace(schema=…)` as the optional schema-first door and checkpoint
   manifests validated at provisioning. Safetensors = mmap'd descriptors,
   zero-copy; init strategies keyed by name pattern over the §4 fields;
   the virtual↔provisioned warm-cache pin. Lands with `@assemblage` (P5),
   with the descriptor pieces in P6.
3. **The RNG stance** (§4) enters the spec: fields in P4 (expression
   syntax + derivative table entries), Philox device function and the
   recompute-theorem test in P7.
4. **The tying rule** (§1.3): capture identity, not name strings, decides
   leaf identity; one leaf per object; pinned by a tied-gradient test in
   the zoo gate.
5. **`take`/gather scheduled** (§5f): op + scatter-add adjoint + cost
   model entries, landing between P7 and P9 so the runway's training-loop
   story is honest; top-k/MoE and in-program sampling remain recorded
   boundaries.
6. **The two-layer discipline and the namespace semantics** (§1, §2): the
   standard library is parameter-blind (tensors in, tensors out,
   parameters as ordinary arguments); only binding-layer code touches the
   namespace, which is build-time-only, flat-name-primary (the tree is a
   prefix view), and deliberately not a pytree subsystem — names replace
   positions as the join key everywhere. The index-derived layer-name
   instability is recorded with its translation-table mitigation.

With these adopted, the 170 plan proceeds unchanged in shape: P0–P9 with
amendments 2–6 slotted as noted. The exercise's residual watch item is
(b): the mma-recognition question stays open in the L4 brief, now with a
GPT-2-specific worked case showing where recognition holds and what
breaks it.
