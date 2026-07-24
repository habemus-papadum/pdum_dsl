# 180 — GPT-2 end to end: the user experience and the compiler's view

**Status: red-team exercise against the 170 spec, run before P0.** Two
questions. First, the user level: one GPT-2 definition — weights in
closures, blocks composed with `|` — must serve checkpoint loading,
controlled random initialization, zero-allocation analysis, and a training
loop with selective freezing, with no wasted allocation or copying
anywhere. Second, the compiler level: is the representation at the right
granularity for the optimizations we intend (distribution, tiling, fusion,
checkpointing), or does the primitive language hide what the optimizer
needs to see? Findings and the amendments they force on 170 are in §6.
Headline: **no structural blocker; the exercise produced the scope (one
threaded context unifying parameters, randomness, and policies — and
closing an identity hazard), the maker convention over four pinned
substrate contracts, a provisioning design, the RNG/dropout stance, and
one genuine primitive gap (gather).**

---

## 1. The model, as a user writes it

GPT-2 per the zoo's canon: heads are born as dims (`wq: (d, nh, hk)` —
never split into existence), the causal mask is an iota comparison (a
closed form, not memory), and token embedding is treated in §5f. All code
below is the 170 surface (S.1/S.2 syntax); everything it needs already
exists or is committed in P0–P9 except where §6 says otherwise.

### 1.0 What threads through a model: the scope, and nothing else

A model factory takes exactly two things: **the scope `s`** and **your
data `cfg`**. The criterion for what goes where is crisp: *the scope
carries what must agree with the naming law — path-addressed facets; cfg
carries what doesn't — your values* (widths, rates, eps: ordinary host
data, closed over like any value, structured however you like).

The scope is an explicit, immutable value — your position in the model's
one address space — carrying that path's facets:

- **Path.** `s / "attn"` derives a child scope; the path IS the
  structural address the naming law reifies. There is no second
  path-shaped thing: parameters, randomness, and policies are all
  addressed by it, so they can never drift apart (passing `P/"L3"` next
  to `key/"L4"` is a bug class the scope makes unwritable).
- **Parameters.** `s.param(name, **dims)` — dim names as keyword keys,
  extents as values — declares a leaf at `path.name` and returns its
  tensor (virtual, loaded, or initialized; the code cannot tell).
  Declaration is idempotent; a conflict refuses.
- **Randomness.** The scope carries the randomness root (a program
  input); streams derive from **site paths** — `dropout(x, p, s /
  "attn_drop")` gets `fold_in(root, path)`. No key is threaded anywhere.
- **Policies.** An open, string-keyed set of aspects scoped to the
  subtree: `s.with_(mode="eval")`, `s.with_(trainable=False)`, later
  `init=`, precision regions, modes not yet invented. The scope
  *interprets none of them*; library idioms read them by convention
  (dropout reads `mode`; grad reads `trainable`). `training` is not a
  blessed boolean in any signature — it is one policy among many.

**Policies are identity-bearing — this closes a real hazard.** A mode
selects which Program gets built, but a plain captured bool has the same
*type* either way, so type-keyed identity would let a train build and an
eval build collide in the cache and silently serve the wrong Program.
The scope folds its policy map into build identity the way `Literal`
folds values into types — the mode-as-loose-bool mistake becomes
unwritable.

**Context managers are sugar, never a global.** `with s.with_(mode=
"eval") as se:` is fine — lexical sugar over deriving an explicit value,
equally writable without `with`. There is no module-level "current
scope" stack, ever; that is where framework naming magic and
irreproducibility come from, and the functional core refuses it. The
only mutation anywhere is the leaf registry's build-time collection.

**The two-layer discipline.** Only one layer of model code touches the
scope:

- **The standard library is parameter-blind.** `layernorm`, attention
  cores, dense blocks — functions from tensors to tensors, parameters
  passed as *ordinary arguments*. No scope, no names, no knowledge that
  provisioning exists. Architecture variants (Llama = rmsnorm + RoPE +
  GQA) reuse these wholesale.
- **The binding layer owns names.** Makers (`make_attn`, `make_gpt2`)
  hold the scope, declare leaves, and hand tensors to library functions.
  Name assignment happens in exactly one visible place — never inside
  library code.

### 1.1 Device functions and library blocks

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

### 1.2 The makers: declare-at-use, dropout without a mode branch

A **maker** is a plain function `(s, cfg) -> unit`. Leaves are declared
at the top; the unit body calls only parameter-blind library functions.
`dropout` consults the scope's `mode` policy itself — it is the identity
under `mode="eval"` — so no `if training:` appears in user code; the
branch lives in the library idiom, and train/eval are two cached
Programs distinguished by identity-bearing policy.

```python
def make_attn(s, cfg):
    D, H, K = cfg.d, cfg.nh, cfg.hk
    ln1g, ln1b = s.param("ln1g", d=D), s.param("ln1b", d=D)
    wq = s.param("wq", d=D, nh=H, hk=K)
    wk = s.param("wk", d=D, nh=H, hk=K)
    wv = s.param("wv", d=D, nh=H, hk=K)
    wo = s.param("wo", nh=H, hk=K, d=D)

    def attn(h):
        a  = layernorm(h, ln1g, ln1b, feat="d", eps=cfg.eps)
        q  = contract(a, wq)                           # unique shared axis: "d"
        k  = contract(a.rename(t="s"), wk)
        v  = contract(a.rename(t="s"), wv)
        sc = contract(q * cfg.scale, k, axis="hk")     # "nh" rides; axis named to
                                                       # break the genuine ambiguity
        pr = dropout(causal_softmax(sc), cfg.p_attn, s / "attn_drop")
        cx = contract(pr, v, axis="s")
        o  = contract(cx, wo, axis=("nh", "hk"))
        return h + dropout(o, cfg.p_resid, s / "resid_drop")
    return attn

def make_mlp(s, cfg):
    D, M = cfg.d, cfg.m
    ln2g, ln2b = s.param("ln2g", d=D), s.param("ln2b", d=D)
    w1, b1 = s.param("w1", d=D, m=M), s.param("b1", m=M)
    w2     = s.param("w2", m=M, d=D)

    def mlp(h):
        a = layernorm(h, ln2g, ln2b, feat="d", eps=cfg.eps)
        m = gelu(contract(a, w1) + b1.repeat_like(a, but="m"))
        return h + dropout(contract(m, w2), cfg.p_resid, s / "resid_drop")
    return mlp
```

An architecture edit — swapping layernorm for rmsnorm, adding a gate
weight — touches the `s.param` lines and the body that uses them, in one
file, and nothing else.

### 1.3 Assembly: level-first names, `seq`, and the tie

Composition points name their sub-scopes explicitly, giving the
**level-first hierarchy** checkpoints use (`h.3.attn.wq`, `h.3.mlp.w1`).
Two blocks declared under one path collide loudly — a designed refusal
naming the fix — never an auto-suffix (suffixing would reintroduce
positional instability).

```python
def make_block(s, cfg):
    return make_attn(s / "attn", cfg) | make_mlp(s / "mlp", cfg)

def make_gpt2(s, cfg):
    wte = s.param("wte", v=cfg.v, d=cfg.d)             # declared ONCE — tied below
    wpe = s.param("wpe", t=cfg.t_max, d=cfg.d)
    lnfg, lnfb = s.param("lnfg", d=cfg.d), s.param("lnfb", d=cfg.d)

    def embed(ids):
        tok = wte.take(ids, dim="v")                   # gather — §5f/§6
        e   = tok + wpe.slice(t=(0, ids.extent("t")))
        return dropout(e, cfg.p_embd, s / "embd_drop")

    trunk = s.seq("h", make_block, cfg, n=cfg.layers)  # h.0.attn.wq, h.1.mlp.w1, ...

    def head(h):
        hf = layernorm(h, lnfg, lnfb, feat="d", eps=cfg.eps)
        return contract(hf, wte, axis="d")             # TIED: the same object

    return assemblage(embed | trunk | head)

train_model = make_gpt2(root.with_(mode="train"), cfg)   # two Programs, both cached —
eval_model  = make_gpt2(root.with_(mode="eval"),  cfg)   # policies are identity-bearing
```

`seq` is deliberately **thin enough to print** — it is the explicit host
loop, named; the loop form remains legal and identical in meaning, and
this transparency is how the code stays visibly host-level:

```python
def seq(s, name, maker, cfg, n):           # cfg may be a value or a fn of i
    units = [maker(s / name / str(i), cfg(i) if callable(cfg) else cfg)
             for i in range(n)]
    return pipe(units)                     # n-fold unit composition
```

**The `|` ruling this forces (adopted into 170):** at the assemblage
tier, `f | g` over tensor-functions is **build-time function composition
threading one value** — the same fuse semantics `|` already means,
realized as program-fragment composition into one Program. It is legal
precisely because it is *not* dispatch sequencing: nothing here launches.
The scope and `cfg` ride as closed-over symbols, never threaded values.
`|` composes **units only** — never makers: a maker-level pipe would be a
third composition semantics punned onto the operator, which 170 forbids.

**A considered and rejected alternative:** a double-curried maker
convention (`make_foo(p)(cfg)(inputs)` with maker-level `|` and
`p.seq(block, repeats)(cfg)`). Rejected on three grounds: maker-`|` puns
a third semantics onto the pipe; currying standardizes ceremony rather
than power (three nested defs per block, late confusing errors, a
convention to learn) — closures already stage bindings; and cfg applied
once at the end forecloses per-layer variation (GPT-2's own scaled init
wants the layer index; hierarchical models change widths per stage). The
plain-maker form achieves the same goals — composition before scope
binding, machinery-assigned level-first paths, a named sequencing
operation — with one convention instead of three.

**Weight tying is the identity rule:** `wte` is declared once and the
same object is captured by both `embed` and `head` → **one input leaf**
(capture identity, not name-string, decides). Its gradient is
automatically the sum of both uses' contributions because there is only
one leaf to seed. Pinned by a test (§6.4). Declare-at-use makes it
visually obvious: tying looks like what it is — one declaration used
twice.

---

## 2. Provisioning: one definition, three materializations

**The resting state of a model is virtual, and the builder is the single
source of truth.** A fresh scope holds nothing; running the builder
against it *collects* the spec — every `s.param(...)` registers a leaf
(name, dims, extents, carrier; **no buffer**), and the makers capture
exactly the tensors they declared. They cannot tell virtual from real,
because `typeof` is identical (layout + carrier; no buffer in the type).
There is no separately-maintained parameter table to keep in sync with
the code:

```python
root  = scope()                                  # rng root = a program input
model = make_gpt2(root.with_(mode="train"), cfg) # builds AND collects
root.spec()      # the full table, derived: "h.0.attn.wq": (d:768, nh:12, hk:64), ...
```

A schema-first door remains — `scope(schema=…)` validates each
declaration against a written table, and provisioning validates against
a checkpoint's manifest either way — but the code is authoritative.

**What the scope is (and is not).** A string-keyed address space whose
**flat name space is primary** — `s / "h" / "3"` is a prefix *view*, not
a container boundary — and which exists **only at build time**: after the
build, the Program has named inputs, and runtime state (weights, grads,
moments) is plain name-keyed dicts. It is deliberately not a pytree
subsystem: pytree machinery (treedefs, container registration,
positional flattening) exists to turn arbitrary containers into
positional argument lists, and nothing here consumes positions —
Programs, `grad` maps, provisioning, and optimizers all **join on
names**, so "zip params with grads with moments" is a dict join. The
whole object is on the order of a hundred lines, and it stays thin by
law: path, registry, randomness root, small policy map — interpreting
nothing.

**Scenario 3 — analysis, zero allocation.** Virtual is not a mode; it is
the unprovisioned state, and it is also the collector. The build above
already produced the full Program (layouts are known);
`ops_count(model.program)`, `peak_memory(model.program, schedule)`,
traffic and placement analysis all read layouts and never values.
Nothing allocates. Only `ir.run`/`item()` refuse, quoting the fix
("provision the parameters"). **Cache dividend, pinned as a test:** the
virtual build and the provisioned build have identical types, therefore
identical fingerprints — analyze first, provision later, and the Program
cache hits warm.

**Scenario 1 — load.** `weights = provision(root, source=safetensors("gpt2.st"))`.
safetensors is an mmap format: each entry becomes a boundary descriptor
over the **mmap'd region directly** — Buffer (DLPack shim over the map) +
Layout (from shape metadata) + Encoding (the file's dtype: f32, or bf16
as a *fact*). Zero copies on the host; exact decode per 170 §4. Names
join on the contract names — the checkpoint's `"h.0.attn.wq"` is ours (a
translation table handles foreign naming schemes; it is data, not code).
Tied weights stored once in the file arrive as one Buffer → one
descriptor → one leaf, preserving §1.3's rule. Device transfer, when
backends exist, is one explicit copy per buffer at provisioning — never
per dispatch, never allocate-then-overwrite.

**Scenario 2 — init.** Strategies are keyed by name pattern; randomness
is the §4 machinery, so initialization is *reproducible and per-value
controlled by construction*:

```python
weights = provision(root, source=init(
    root_key / "init",
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
allocate-uninitialized-then-fill-then-maybe-copy sequence anywhere in
any scenario: virtual allocates nothing; load allocates nothing on host
(mmap) and copies once to device; init allocates once and writes once.
(Init strategies may also hang on the scope as an `init=` policy at
declaration sites — region-scoped defaults with the glob door as the
post-hoc override; both join on names underneath.)

**Name stability under refactoring, stated honestly.** Leaf-level and
block-internal architecture edits never churn names: declare-at-use
means the edit and the name live on the same line, and RNG streams —
being name-derived, not position-derived — ride along unchanged. The one
instability is **index-derived layer names**: inserting a layer shifts
`h.{i}` for everything after it, and a loaded checkpoint stops joining.
This is universal (PyTorch `state_dict` keys, Flax scopes — all
index-derived, all break identically), and the mitigation is the same
name-translation table that foreign checkpoints already require; it is
data, not code.

**Beyond ML.** None of this is an ML concept. In scientific computing
the same three scenarios are: load = observational data entering as
boundary facts; init = synthesized initial conditions from closed-form
fields; virtual = costing a solver before buying the cluster. Scope
declarations carry units and charts as naturally as extents
(`s.param("dt", unit=u.s)` — more declaration facts through the same
door), and policies cover precision regions or boundary-condition
variants exactly as they cover train/eval. The scope stays domain-neutral
because it interprets nothing.

---

## 3. Training: trainable by default, frozen by name or by region

All parameter leaves are trainable by default — "the program is the
parameter container." Freezing has two doors, both keep-set arithmetic
underneath:

- **In the model** (region-scoped, a policy): declaring under
  `s.with_(trainable=False)` — e.g. a frozen pretrained encoder, usually
  paired with `mode="eval"` for its dropout — sets the *default*
  keep-set at the declaration site.
- **Post hoc** (call-time, by name): `partition(freeze=["h.0.*", "wpe"])`
  overrides the default. Declaration is the default; invocation is the
  override.

```python
trainable, frozen = root.partition(freeze=["h.0.*", "wpe"])

step  = grad(loss_fn, wrt=trainable)      # keep-set → requested-gradients DCE:
                                          # frozen weights' backward work is PRUNED
opt   = adam(trainable)                   # moment dicts keyed by the same names

for t, batch in enumerate(data):
    lr    = sched(t)                                      # live knob: value, never keys
    grads = step(weights, batch, root_key / ("step", t))  # per-step streams, warm cache
    weights, opt = opt.update(weights, grads, lr)
```

The tied `wte` receives one gradient (embedding and head contributions
summed — one leaf, §1.3). `key / ("step", t)` is `fold_in` twice — a new
value each step, identity unchanged, zero recompiles: the live-knob
thesis applied to randomness. A different freeze-set is a different
derived Program (different DCE result) — cached like any other. For
gradient control on *activations* rather than parameters,
`stop_gradient(x)` is a plain IR op (identity forward, zero backward) —
dataflow, not scope.

---

## 4. RNG and dropout — the stance, specified

Forced by principles 170 already holds (purity, content-addressed
caching, recompute-based checkpointing), and now part of the spec:

1. **Randomness is a counter-based, coordinate-indexed, closed-form
   field.** `uniform(key, layout)` is a pure function of (key, lattice
   coordinates) — Philox-class bits, element *i* computed directly, no
   sequential state. It is a `FunctionalBuffer`-class citizen exactly
   like `iota`: zero memory, exact under view ops, free in the cost
   models, materialized only at a boundary that demands it. Bits are
   exact (`u32/2³²` is a rational) — carrier-consistent.
2. **Keys are ordinary values; the scope carries the root; streams
   derive from site paths.** The randomness root is a program input
   (type-keyed — new keys are warm hits); per-site streams are
   `fold_in(root, site_path)` via `s / "attn_drop"`, and per-step
   streams `fold_in(root, t)` — insertion-stable and refactor-stable
   where positional splitting is not. No key is ever threaded through
   model code; `split` exists underneath and is rarely touched.
3. **Dropout is an idiom, not an op**:
   `where(uniform(stream, x.layout) < p, 0, x / (1-p))` — and it is
   **mode-aware**: it reads the scope's `mode` policy and is the
   identity under eval, so mode branches live in the idiom, not in user
   code. Train/eval are build-time variants — two cached Programs,
   distinguished by identity-bearing policy. AD falls out of existing
   rules (comparisons gradient-free; the mask acts as a constant field).
4. **The recompute theorem, pinned:** checkpointing and revolve recompute
   forward segments; the mask field regenerates bit-identically (same
   key, same coordinates), so gradients under recompute are exact *by
   construction* — no RNG-state stashing exists because none is needed.
   One test pins it: revolve-checkpointed training step ≡ store-all,
   with dropout on.
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
Megatron's tensor-parallel block already proved collectives are *read
off the algebra* (two all-reduces, discovered not written) and the
placed backward carries bindings. GPT-2 distributes by binding `nh`
(head parallel) or `d`/`m` (Megatron-style) — nothing in this model's
Program hides from the traffic pass. The batch dim, when it arrives, is
one more named dim riding every declaration.

**(b) Tiling and mma selection — YELLOW, known and bounded.** `contract`
is the `repeat·mul·reduce` normal form; L4 selects tensor cores by
*recognizing* mul-solely-consumed-by-reduce. The known miss: training
saves activations, and if an optimizer chose to save the raw product
`a*w` for backward, the "solely consumed" pattern breaks. Worked against
GPT-2: the standard saved set (inputs and normalized activations, per
the min-cut with contractions recompute-banned) does **not** save the
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
construction. The mask's zero-cost status in the memory model is
truthful for the fused form.

**(d) Checkpointing/revolve — GREEN.** The §4.4 theorem: recompute
regenerates masks exactly. The min-cut's exact-byte capacities now read
descriptor-fed sizes (170 §4). Dropout adds `where` + comparison nodes —
pointwise, recompute-cheap, precisely what the cheap-chain heuristic
wants to recompute rather than save.

**(e) Module boundaries — GREEN with one observation.** After
composition, "this was the attention block" survives in exactly one
place: the **level-first name prefixes** (`h.3.attn.*`) — which are
machine-readable and stable under the naming law. Partitioning operates
on dataflow plus names; the kernel boundary is an annotation anyway. If
L4 partition search ever wants explicit block scopes, a scope annotation
is an erasure-preserving addition in the existing style — a nicety, not
a missing representation.

**(f) The genuine gap: gather.** Token embedding (`wte.take(ids,
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
bucketing (pad to the bucket, mask via the same closed-form
comparisons). This is a cost to know about, not a blocker; kernel-tier
staging keeps per-shape artifact reuse underneath. Decode-time KV
caching composes as the ring/window boundary sample from the L2 runway.

**(h) What the representation genuinely cannot express — unchanged and
intended.** Data-dependent control flow inside programs (rejection
sampling, early exit), dynamic shapes *within* one Program, mutation
outside the store seam. All are stated subset boundaries with refusals;
GPT-2 needed none of them.

---

## 6. Verdict and amendments to 170

**No structural blocker.** The composition story, the caching thesis,
the naming law, the precision doctrine, and the cost/transform machinery
all carried GPT-2 without strain — and twice (flash+dropout fusion,
recompute-with-dropout) the design produces for free what mainstream
frameworks implement as special machinery.

**The substrate contracts, and the fashion test.** Everything
user-facing in §1–§3 is host-level *convention* over four pinned
substrate contracts: (i) leaves are declared at paths — the naming law;
(ii) units are tensor→tensor functions; (iii) unit composition is
build-time function composition; (iv) identity follows the scope rules —
policies identity-bearing, structural values Literal-typed. The de-risk
criterion for any future idiom is the **fashion test**: Flax-style
modules, PyTorch-style modules, a curried point-free style, and the
plain-maker style must all be expressible as satellites over these four
contracts — and they are, which is why no maker fashion can paint the
substrate into a corner. The idiom layer is deliberately left
conventional; only the contracts are spec.

The exercise forces seven amendments, adopted into 170:

1. **`|` at the assemblage tier** (§1.3): build-time function
   composition threading one value — the same fuse semantics; composes
   **units only**, never makers. Never dispatch sequencing.
2. **Provisioning** (§2): virtual as the resting state with
   **declare-at-use primary** — `s.param(name, **dims)` at the binding
   layer; the spec *collected* by the virtual build (the builder is the
   single source of truth), `scope(schema=…)` as the optional
   schema-first door, checkpoint manifests validated at provisioning.
   Safetensors = mmap'd descriptors, zero-copy; init strategies by name
   pattern (or `init=` region policy) over the §4 fields; the
   virtual↔provisioned warm-cache pin. Lands with `@assemblage` (P5),
   descriptor pieces in P6.
3. **The RNG stance** (§4): fields in P4 (expression syntax + derivative
   table entries), Philox device function and the recompute-theorem test
   in P7; streams derive from site paths via the scope.
4. **The tying rule** (§1.3): capture identity, not name strings,
   decides leaf identity; one leaf per object; pinned by a tied-gradient
   test in the zoo gate.
5. **`take`/gather scheduled** (§5f): op + scatter-add adjoint + cost
   model entries, landing between P7 and P9 so the runway's
   training-loop story is honest; top-k/MoE and in-program sampling
   remain recorded boundaries.
6. **The scope** (§1.0): one threaded context — path + parameters +
   randomness + policies — replacing separate `P`/`key`/mode arguments;
   policies are an **open set** and **identity-bearing** (closing the
   mode-as-captured-bool cache-collision hazard); context managers are
   sugar over explicit derivation, and a global scope stack is refused;
   `cfg` stays user data by the stated criterion. The two-layer
   discipline: library code is parameter-blind; only makers touch the
   scope.
7. **The maker convention over the substrate contracts** (§1.3, §6):
   makers are plain `(s, cfg) → unit` functions; composition points name
   sub-scopes explicitly (level-first hierarchy); collisions refuse;
   `seq` exists and is thin enough to print; the curried maker-pipe
   alternative is recorded as considered-and-rejected. The four
   substrate contracts are pinned as spec; idioms above them are
   convention, gated by the fashion test.

With these adopted, the 170 plan proceeds unchanged in shape: P0–P9 with
amendments slotted as noted (the scope rides P5 with `@assemblage`; its
policy-identity rule is part of the P5 gate). The exercise's residual
watch item is (b): the mma-recognition question stays open in the L4
brief, now with a GPT-2-specific worked case showing where recognition
holds and what breaks it.
