# Deep-learning library design notes

**Status:** satellite design notes (2026-07-11, extended during step 2). How a
JAX-style deep-learning library would be built *on top of* pdum.dsl, and which
structural properties of the kernel solve problems PyTorch and JAX users
accept as facts of life. Companion to `docs/desiderata.md` §2.1; implies **no
kernel changes** — every mechanism named here is one of the five surfaces or
an already-planned milestone. Prerequisites before building: M4 (grad) and a
tensor-capable backend (MLX-class). Evidence citations refer to
`design/research/R4-jax.md` and `R6-torch.md`.

---

## 1. The shape of the library

Modules are **closure factories**; the network is built bottom-up by ordinary
Python, passing jitted closures into jitted closures:

```python
def dense(w, b):
    @jit()
    def layer(x): return relu(w @ x + b)
    return layer

def block(p):
    attn, ff = attention(p.wq, p.wk, p.wv, p.wo), mlp(p.w1, p.w2)
    @jit()
    def blk(x): return ff(attn(x))
    return blk

model = block(p0)
for p in params[1:]:
    model = compose(model, block(p))      # or a `>>` sequencing operator
loss  = make_loss(model, batch)
```

**The program is the parameter container.** There is no separate module tree,
no `params` pytree threaded through `apply`: a capture that is itself a
`Handle` contributes its `FnType` to the parent's identity, so the network is
a tree of Handles whose *identity* is value-free and whose *leaves* are the
weights. Everything else in these notes follows from that one move.

## 2. Structural properties vs. the incumbents' pain

| Pain (incumbent) | Structural answer here |
|---|---|
| **JAX: closures retrace.** Captures bake into the trace as constants; jit keys on function `id()`; a closure rebuilt per step retraces perpetually (documented ~240× pathology; `JAX_EXPLAIN_CACHE_MISSES` exists because of it) — R4 | Identity is `(code value-equality, env *types*)`. Rebuilding closures with fresh weights every step **is the designed hot path**: phase A is reflection with memoized fingerprints; the thesis cache hits. |
| **JAX: parameter plumbing.** Flax/Haiku/Equinox exist to reconcile "models hold state" with "functions must take params explicitly" | Weights are captures. The flat labeled parameter tree the optimizer needs is **derived** by the marshaling layer (LeafPaths through nested envs), not declared and threaded by the user. |
| **JAX: `static_argnums` footguns; PyTorch: `model.train()` mutable global** | `Literal` lift per capture: `mode=Literal("train")` enters the key explicitly → two artifacts, dropout branch folded out, no runtime flag, no positional indices. |
| **JAX: PRNG key threading** | Counter-based RNG (Philox-style) as an op; seed/step-counter are ordinary runtime captures — uniforms, in shader terms. |
| **PyTorch: stateful modules & aliasing bugs; torch.compile: opaque guard recompiles with an 8-entry limit** — R6 | Pure rebuild semantics; per-tier miss counters *name the key component that changed*; `no_compile` assertion mode makes "this loop must not recompile" a testable claim. |
| **Both: "what triggers recompilation" is folklore** | The two-tier law is a contract: type/code/Literal changes miss the artifact tier; unit/byte-level knobs miss only the pack memo; weight values miss nothing. |

## 3. Identity economics (the part measured in ch02)

- **O(1) incorporation.** `Handle.fp` is a precomputed subtree digest; a
  parent fingerprints a child by reading it. Building an N-module network is
  O(N) total phase-A work — ch02 measures a mocked 100-block transformer at
  linear scaling, microseconds per rebuild step.
- **Unchanged subtrees are free.** Because identity is compositional and
  value-free, a frozen backbone can be built *once outside the loop* and
  captured by the per-step head: phase A then costs only the changed spine,
  and the root identity is bit-identical to a full rebuild (ch02 asserts
  this). Fine-tuning, LoRA-style adapters, and staged unfreezing fall out of
  the programming model instead of requiring framework features
  (`requires_grad_`/`stop_gradient` bookkeeping).
- **Escalation, pre-shaped:** Python re-hashes nested fp tuples on dict
  probes (structure traversal is avoided; raw hashing is O(subtree)). If the
  microbench gate ever flags deep spines, `Handle.fp` becomes a flat memoized
  digest (the `Node.key` technique) with no contract change.

## 4. The training-loop mechanics, mapped to kernel features

- **Labels.** Mostly derived from structure: factory qualnames + capture
  names + position (`net.block[3].attn.wq`), the grown-up version of M0's
  `weave2_k` merged-uniform naming. Explicit annotation (`label="encoder.q"`)
  is static metadata for the places structure isn't enough.
- **Gradients.** `grad(loss, wrt="encoder.*")` mints a `Derived` template
  identity (tag + base + static label pattern) that flows through the
  unchanged thesis cache — `grad(f)` rebuilt every step is a cache hit. The
  optimizer receives `(LeafPath → gradient)` mappings.
- **Optimizers.** optax-shaped: composable pure transformations over the
  derived gradient map (`chain(clip_by_global_norm(1.0), adam(3e-4))`);
  optimizer state is itself a labeled tree keyed by the same LeafPaths.
- **Gradient clipping** — three homes, choose per need: optimizer-side value
  transformation (default, zero compiler involvement); fused into the
  backward program as a post-transpose IR pass (global-norm needs a
  cross-gradient reduction — a whole-program rewrite, which passes are);
  per-module `custom_vjp` (straight-through estimators and friends).
- **Dropout / mode-dependent layers.** `mode` is a `Literal`; dropout lowers
  to counter-RNG + mask under `Literal("train")` and folds to identity under
  `Literal("eval")`.
- **Normalization.** LayerNorm/RMSNorm (stateless) are ordinary batteries.
  BatchNorm's *running statistics* are the honest wart: mutable train-mode
  state in a functional system — options are threaded state (extra
  input/output leaves, Flax-style) or boundary stores; modern
  architectures' drift toward stateless norms shrinks the problem.
- **Checkpointing (weights).** LeafPaths give every parameter a stable
  structural address derived from the program — serialization is
  `(path → array)`, robust to code refactors that preserve structure, with
  the same explicitness about when it isn't.

## 5. Execution reality (honest scope)

- **Tensor performance is delegated.** The tensor dialect (matmul, reduce,
  broadcast, softmax…) lowers through a Surface-D backend to a mature tensor
  runtime — MLX first (custom kernels + graphs on Metal), CuPy/CUDA later. A
  `Runtime.compile` that builds a backend graph is a legitimate backend.
  Our layer owns identity, caching, marshaling, labels, and AD orchestration;
  matmul speed belongs to the delegate. We do not build fusion/tiling/memory
  planning (the tinygrad/XLA/Inductor middle) — settled in
  `proposed-architecture.md` §14.
- **AD depth is real work.** Correct transpose rules for tensor ops are
  registrations (M4); *production* NN autodiff also needs numerically stable
  custom VJPs (logsumexp/softmax accumulate in `custom_vjp` packs) and
  **rematerialization** for activation memory — a serious post-M4 milestone,
  not a rule pack.
- **Scope.** Wins on workflow for interactive/medium-scale differentiable
  programs — the desiderata's design-optimization domain wearing an NN
  costume. Does not contest LLM-scale training: XLA/TPU/sharding and the
  cuDNN/flash-attention ecosystem are moats ergonomics don't cross.

## 6. Open questions (for when this is pulled forward)

1. **BatchNorm-style state**: threaded leaves vs boundary stores — decide on
   the first real user, not before.
2. **Mixed precision**: is a compute-dtype policy a backend `type_map`
   concern, a `Quantity`-like tag in the type, or a transform? (Leaning:
   backend policy + explicit cast ops; keep the honest-types rule.)
3. **Weight initialization**: pure Python outside the kernel (it runs once) —
   probably needs nothing from us; confirm.
4. **Whole-step compilation** (forward+backward+update as one artifact) vs
   optimizer-in-Python: start with the latter (per-step overhead is µs
   against ms-scale steps); revisit only with evidence.
5. **The `>>`/compose sugar**: a tiny combinator library (Surface B
   batteries) — where does it live and what's its minimal op set?
6. **Distributed**: out of scope; recorded so nobody wonders whether it was
   forgotten.
