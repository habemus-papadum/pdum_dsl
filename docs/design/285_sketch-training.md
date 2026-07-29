# 285 — The training sketch, redone in the zoo idiom (DRAFT)

**Status: DRAFT FOR OWNER MARKUP.** This supersedes 284's Sketch 2 and
Sketch 3. The markup on 284 said two things: the training program must
be written in the model-zoo shape — parameter-blind library functions,
makers that declare parameters at use, gradients over the *closed-over*
leaves via the naming law, never a list of parameter names at the grad
site — and the compiler pipeline should compose with `|`, not `.then`.
Both are done here. Everything is defined in this file; the divergence
ledger at the end says exactly where the sketch departs from today's
tree so markup can rule per line.

## What this sketch treats as law (it already is)

These are not proposals — they are the shipped idiom, and the sketch
is built on them unchanged:

- **The library is parameter-blind** (`zoo/zoo_common.py`): plain
  functions from tensors to tensors. No scope, no names, no build.
- **Makers own names** (`zoo/gpt2.py`): a maker is `(s, cfg) -> unit`;
  it declares leaves at use with `s.param("w1", d=D, m=M)` and returns
  a `@unit` function that *closes over* them. Units compose with `|`;
  `s.seq("blk", make_block, cfg, n=2)` stacks them under `blk.0.…`,
  `blk.1.…`. The flat contract name (`blk.0.w1`) is the naming law's
  assignment, pinned by literal test expectation.
- **Dropout is the mode-aware idiom** (`scope.py`, `assemblage.py`):
  `dropout(u, rate, s / "drop")` in a unit body. Under
  `s.with_(mode="eval")` it lowers to identity; policies are
  identity-bearing, so a train build and an eval build can never
  collide in the cache.
- **The grad site takes no name list** (`autodiff.py`,
  `test_naming_law.py`): `grad(region, out, names=a.names)` returns
  gradients addressable by contract name — every closed-over leaf,
  joined on the naming law. One leaf captured twice gets ONE summed
  gradient. `wrt=` exists for narrowing; the training loop never
  writes it.

284's Sketch 2 violated the first, second, and fourth of these
(explicit weight arguments, `value_and_grad(loss_fn, wrt=(...))`).
Retired.

---

## The program: `train_mlp.py`

A two-block residual MLP classifier: layer normalization, GELU,
dropout, skip connections. The library contributes five pure
transforms; the makers wire an actual architecture; the training loop
computes gradients over whatever the makers declared — it never names
a weight.

### 1. The library — five pure transforms, no names anywhere

Three exist in `zoo_common.py` today (`layernorm`, `gelu`, `softmax`);
`lse_of` is promoted from a trainer-local helper; `dense` is new but
trivial. All are tensors-to-tensors; none touches a scope.

```python
from pdum.tl.compute import contract, pointwise, red, reduce, repeat_like
from pdum.tl.markers import exp, log, sqrt
from pdum.tl.zoo.zoo_common import gelu, layernorm   # as shipped


def dense(h, w, b, *, axis):
    """Affine map along one named dim."""
    u = contract(h, w, axis=axis)
    return u + repeat_like(b, u)


def lse_of(logits, dim):
    """log-sum-exp along `dim` (numerically safe)."""
    mx = reduce(red.max, logits, dim)
    e = pointwise(exp, logits - repeat_like(mx, logits))
    return pointwise(log, reduce(red.sum, e, dim)) + mx
```

### 2. The makers — the architecture, declared at use

Dim names: `n` batch, `f` input features, `d` model width, `m` hidden
width, `k` classes.

```python
from dataclasses import dataclass

from pdum.tl.assemblage import assemblage, unit
from pdum.tl.indexing import take
from pdum.tl.scope import dropout, scope


@dataclass(frozen=True)
class MLPConfig:
    n: int = 128     # batch
    f: int = 16      # input features
    d: int = 32      # model width
    m: int = 64      # block hidden width
    k: int = 10      # classes
    blocks: int = 2
    pdrop: float = 0.1
    eps: float = 1e-5


def make_embed(s, cfg):
    w = s.param("w", f=cfg.f, d=cfg.d)
    b = s.param("b", d=cfg.d)

    @unit
    def embed(x):
        return dense(x, w, b, axis="f")

    return embed


def make_block(s, cfg):
    """Pre-norm residual block: norm -> dense -> gelu -> dropout -> dense,
    added back onto the skip path."""
    lng, lnb = s.param("lng", d=cfg.d), s.param("lnb", d=cfg.d)
    w1, b1 = s.param("w1", d=cfg.d, m=cfg.m), s.param("b1", m=cfg.m)
    w2, b2 = s.param("w2", m=cfg.m, d=cfg.d), s.param("b2", d=cfg.d)

    @unit
    def block(h):
        a = layernorm(h, lng, lnb, feat="d", eps=cfg.eps)
        u = pointwise(gelu, dense(a, w1, b1, axis="d"))
        u = dropout(u, cfg.pdrop, s / "drop")     # mode-aware; site-keyed
        return h + dense(u, w2, b2, axis="m")     # the skip

    return block


def make_head(s, cfg):
    g, b = s.param("g", d=cfg.d), s.param("b", d=cfg.d)
    wo = s.param("wo", d=cfg.d, k=cfg.k)

    @unit
    def head(h):
        return contract(layernorm(h, g, b, feat="d", eps=cfg.eps), wo, axis="d")

    return head


def make_xent(s, cfg):
    """Mean cross-entropy from integer labels. The labels are a declared
    DATA FIELD: named like a param, fed like a batch, never trained.
    (Divergence 1 — `s.field`.)"""
    tgt = s.field("tgt", n=cfg.n)

    @unit
    def xent(logits):
        nll = lse_of(logits, "k") - take(logits, tgt, dim="k")   # the aligned take
        return reduce(red.mean, nll, "n")

    return xent


def make_step(s, cfg):
    return (
        make_embed(s / "in", cfg)
        | s.seq("blk", make_block, cfg, n=cfg.blocks)
        | make_head(s / "out", cfg)
        | make_xent(s, cfg)
    )
```

The parameter table falls out of the declarations — nobody wrote it:

```python
>>> root = scope(root_key=41)
>>> model = assemblage(make_step(root.with_(mode="train"), cfg),
...                    scope=root.with_(mode="train"),
...                    x=tl.layout(n=cfg.n, f=cfg.f))          # divergence 7
>>> list(model.params)
['in.w', 'in.b',
 'blk.0.lng', 'blk.0.lnb', 'blk.0.w1', 'blk.0.b1', 'blk.0.w2', 'blk.0.b2',
 'blk.1.lng', 'blk.1.lnb', 'blk.1.w1', 'blk.1.b1', 'blk.1.w2', 'blk.1.b2',
 'out.g', 'out.b', 'out.wo']
```

`tgt` is not in that list — a field is data, not a leaf to train.

### 3. Gradients and the optimizer — the step is params-to-params

The grad call names nothing. It differentiates the scalar loss with
respect to every closed-over leaf, and the result is addressable by
the same contract names the makers declared:

```python
from pdum.tl.autodiff import grad
from pdum.tl.optim import sgd     # divergence 3

rg = grad(model.region, model.output, names=model.names)
# rg.grads["blk.0.w1"] is the gradient's name. No wrt list, ever.
```

The optimizer is a pure rule, `(p, g) -> p'`, applied per leaf. It
extends the joint region with one update row per trainable leaf, so
the whole training step becomes ONE program:

```python
step = sgd(lr=1e-3)(rg, over=model.params)
# step takes  {x, tgt, tick} | params      (named inputs)
# and yields  {loss} | updated params      (named outputs)
# step.updates maps each leaf to its update's name: {"blk.0.w1": "blk.0.w1'", ...}
```

Why this shape holds up: swap `sgd(lr)` for `momentum(lr, beta=0.9)`
and the velocity state simply *joins the store* under the naming law
(`opt.v.blk.0.w1`, …), appears in `step.updates` like any other row,
and the loop below does not change by one character. The optimizer is
just more pure functions mapping tensor parameters to tensor
parameters; the naming law is what lets its state ride along
anonymously.

### 4. The loop — resident params, one readback, warm by construction

```python
import numpy as np

from pdum.rt import metal, on

rng = np.random.default_rng(0)
params = {
    name: Tensor.from_numpy(
        0.1 * rng.standard_normal(tuple(e for _, e in p.dims)),
        tuple(n for n, _ in p.dims),
    ).to(metal)                                   # resident; divergence from nothing — .to was 284's
    for name, p in model.params.items()
}

run = step[on(metal)]                             # the launchable; nothing compiles yet

for t, (xb, yb) in enumerate(batches()):
    env = run({"x": xb, "tgt": yb, "tick": t} | params)      # dict union — plain Python |
    params = {k: env[v] for k, v in step.updates.items()}    # device handles; no readback
    if t % 100 == 0:
        print(t, float(env["loss"]))              # THE one explicit readback
```

What each piece is doing:

- `step[on(metal)]` — the selection bracket, exactly as it works on
  `@compute` kernels today, applied to a whole training step
  (divergence 4 — the tl-entry integration; was 284's divergence 2).
- The call face is a named environment in, a named environment out —
  the device twin of `run_named(region, inputs, names)` (divergence 5).
- `"tick": t` feeds the dropout streams. Sites still derive their
  streams from their paths (`fold_in`), but the key becomes a program
  *input* mixed with the tick, so every step draws fresh masks and any
  step is replayable by number. `scope.py` already records this
  future: "the program-input form arrives with @compute"
  (divergence 2).
- Steady state does no compiling and no host round-trips except
  `float(env["loss"])`. Enforceable, not hoped for:

```python
from pdum.dsl import events

with events.forbid("assemblage.miss"), events.forbid("rt.compile"):
    env = run({"x": xb, "tgt": yb, "tick": t} | params)      # step 2 onward
```

### 5. Evaluation — same makers, different policy, disjoint cache row

```python
ev = root.with_(mode="eval")
val = assemblage(make_step(ev, cfg), scope=ev, x=tl.layout(n=cfg.n, f=cfg.f))
vrun = val[on(metal)]                             # forward only — no grad, no updates

vloss = float(vrun({"x": xv, "tgt": yv} | params)["loss"])
```

Same collection, same leaves, same resident tensors. The eval build
has no random ops at all (the dropout lowered to identity), and
policies are identity-bearing, so it can never collide with the train
build in any cache.

---

## The compiler pipeline — `|`, not `.then`

The three-role model from 284 stands: the runtime is the compiler
frontend (`on(metal)` is the only thing you invoke); the compiler is a
pluggable, fingerprinted pipeline of denotation-preserving passes
ending in source emission; every runtime carries a default; cache rows
key on (program, compiler fingerprint, device, threads) and never die.
Only the composition spelling changes.

```python
from pdum.rt import compilers

naive = compilers.default(metal)          # today's translate-and-emit
fused = naive | compilers.fuse_pointwise  # one pass appended
tight = fused | compilers.reuse_buffers   # order is meaning; fp is ordered
```

**The shape that keeps `|` from doing anything silly:** a compiler is
(passes…, emit) with emission *structurally terminal* — it is not a
pass in the list you can append past. `compiler | pass` returns a new
compiler with the pass appended to the passes half; emission stays
last by construction, so there is no way to pipe yourself into
"optimize after emitting". Finer placement keeps the surgical form:
`tight.insert_before("fuse_pointwise", my_split)`.

Refusals, mirroring the unit pipe's existing one (`Unit.__or__`:
"`|` composes UNITS only"):

| expression | outcome |
|---|---|
| `compiler \| pass` | new compiler, pass before emission |
| `pass_a \| pass_b` | a pass chain; `naive \| chain` appends both in order |
| `compiler \| compiler` | refuses — two emitters; a compiler is terminal |
| `compiler \| (lambda r: ...)` | refuses — no fingerprint; "wrap it: `@compilers.rewrite`" |
| `unit \| pass`, `pass \| unit` | refuses — units compose models, passes compose compilers |

(Operator mechanics are safe: both operands are always our types, and
Python's `|` binds tighter than comparisons, so `a | b | c` chains
left-to-right with no surprises. The only other `|` in this file are
the unit pipe and dict union — three types, three `__or__`s, no
overlap.)

A custom pass declares its identity, which is what the cache keys on:

```python
@compilers.rewrite
def my_split(program):
    """Split wide pointwise chains at reduction boundaries."""
    ...
    return program            # denotation-preserving, checked by conformance

experimental = naive | my_split | compilers.fuse_pointwise
```

**The research loop stays warm through a compiler swap** — this is
the whole point of fingerprint-keyed immortal rows. Params are device
handles; they do not care which compiler produced the executor:

```python
run = step[on(metal)]                       # steps 0..899:   launches/step 9
run = step[on(metal, compile=fused)]        # one compile;    launches/step 3
run = step[on(metal, compile=tight)]        # one compile;    launches/step 3, fewer bytes
run = step[on(metal)]                       # ZERO compiles — the old row is still warm
```

And the bake-off is one batch through each candidate, bitwise-compared
because passes preserve denotation:

```python
for c in (naive, fused, tight, experimental):
    env = step[on(metal, compile=c)]({"x": xb, "tgt": yb, "tick": 0} | params)
    report(c, float(env["loss"]), events.last_span())   # same loss, different cost
```

---

## Divergence ledger (vs today's tree — rule per line)

1. **`s.field(name, **dims)`** — a declared, named, non-trainable data
   leaf (labels here; the Gumbel draws of `zoo/trainer.py` are the
   precedent: "randomness enters as NAMED, KEYED INPUT FIELDS").
   Today targets only enter via `lift_step`'s explicit arguments.
2. **Program-input randomness key** — `"tick"` mixed into site-derived
   streams so dropout redraws per step and replays by number. Today
   `tl.random` keys are build-time constants; `scope.py` explicitly
   records the program-input form as a planned arrival.
3. **`pdum.tl.optim`** — optimizer rules as region-to-region
   constructors: append `p' = rule(p, g)` rows over `model.params`,
   updates named by priming, optimizer state joining the store under
   the naming law. No such module exists today.
4. **The bracket on a step** — `step[on(metal, compile=…)]` where
   `step` wraps a grad-joint region, and `val[on(metal)]` on an
   Assemblage. Today the bracket lives on `ComputeKernel` only. This
   is the tl-entry integration and the largest commitment in the file.
5. **The dict-env call face** — named tensors in, named env out, on
   device; `run_named` is its host reference. Today `run_named` is the
   only face.
6. **The compiler `|` law** — pass-append before a structurally
   terminal emission, the refusal table, ordered fingerprints.
   Replaces 284 Sketch 3's `.then`; everything else from that sketch
   (default per runtime, per-invocation override, immortal rows,
   `@compilers.rewrite`, `insert_before`) carries over unchanged.
7. **`tl.layout(n=…, f=…)`** — a public layout constructor for input
   declarations. Today this is spelled `_dense_like((Dim(...), ...))`,
   underscore-private even in the zoo.

Not in the ledger, because they are today's shipped idiom used
verbatim: makers, `s.param` declare-at-use, `@unit`, the unit `|`,
`s.seq`, policies as identity, the dropout idiom, capture-identity
ties, and `grad(..., names=...)` joining on the naming law.
