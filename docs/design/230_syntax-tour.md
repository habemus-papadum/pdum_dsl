# 230 — The syntax tour

**Status: living document.** Every snippet below is running code lifted
from green tests or the zoo — not sketches — except the final section,
which is clearly marked as committed-but-arriving. The tour pops around
rather than building up; read it in any order, though the order given
has a certain logic to it.

## A shader that differentiates its argument

```python
def circle(cy, cx, r):
    @jit()
    def go(y, x):
        d = sqrt((y - cy) * (y - cy) + (x - cx) * (x - cx))
        return d - r                       # signed distance
    return go

@compute
def aa_shader(f, img):
    y, x = thread_idx("y", "x")
    g = value_and_grad(f, wrt=("y", "x"))  # staged: f's identity is compile-time
    v, (dy, dx) = g(y, x)                  # the pattern declares the structure
    w = sqrt(dy * dy + dx * dx)            # fwidth — analytic, no 2×2 quad
    img[y, x] = clamp(v / w + 0.5, 0.0, 1.0)

aa_shader(circle(7.5, 7.5, 5.0), img)      # black inside, white outside,
                                           # a one-pixel analytic AA band between
```

The kernel takes a *function* and differentiates it. `f` is a
compile-time citizen (a handle, not a tensor), so `value_and_grad`
stages at lowering; the base parameter rides the rebind channel, so
calling again with `circle(7.5, 7.5, 2.0)` is a warm cache hit that
renders the *new* circle — no recompile, no stale capture. `clamp` is
an ordinary library function that inlines into the kernel. Nothing here
was registered, subclassed, or configured.

## Derivatives are symbolic, local, and honest

```python
@jit()
def go(y, x):
    d = sqrt(y * y + x * x)
    return with_respect_to(d, y)     # d(local)/d(upstream local), forward-seeded

reference(go)(3.0, 4.0)              # 0.6 — exact
reference(go)(0.0, 0.0)              # nan — IEEE non-trapping, like a device
```

At a kink the derivative is a *contract*, not an accident — a tie at
`maximum` sends everything left and nothing right (the partition law):

```python
with_respect_to(maximum(y, x), y)    # 1.0 at y == x
with_respect_to(maximum(y, x), x)    # 0.0 at y == x
```

And where there is no rule, there is a refusal, never a guess:
differentiating through a branch says *branch derivatives are not yet
vetted*; differentiating `y ** 2.0` says *no entry in the derivative
table*. The table is one page; it grows only when a primitive joins the
core.

## One vocabulary object, every tier

```python
from pdum.dsl.markers import sqrt, maximum, where

sqrt(4.0)                  # 2.0 — a marker is ordinary math on host scalars
pointwise(sqrt, t)         # the SAME object, applied over a tensor
sqrt([1.0, 2.0])           # TypeError: tensor-tier application is spelled pointwise(...)
```

numpy is the naming authority — `maximum`/`minimum` are the binary
pointwise ops and `max`/`min` the reductions, which is exactly the
pointwise/reduce line this system draws. A marker carries its numpy
function (the executable semantics), its derivative row, and its
per-backend spellings, declared once. Adding vocabulary is a two-door
decision: compose it from primitives (a battery like `clamp` — no op,
no table row, derivative free through inlining) or declare a primitive
(one marker + one table row, or an explicit gradient-free row).

## Write it unbatched; the batch is a layout fact

```python
def layernorm(x, g, b, *, feat, eps):
    mu = reduce(red.mean, x, feat)
    xc = x - repeat_like(mu, x)
    sd = pointwise(sqrt, reduce(red.mean, xc * xc, feat) + eps)
    return xc / repeat_like(sd, x) * repeat_like(g, x) + repeat_like(b, x)
```

This layernorm never mentions a batch dimension, and never will —
`repeat_like(a, like)` derives the missing axes from its like-operand's
*layout*. Feed it `("t", "d")` and it normalizes a sequence; feed it
`("b", "t", "d")` and it broadcasts correctly, same five lines. Axes
are *names*, not positions: `reduce(red.mean, x, "d")` cannot silently
reduce the wrong dimension. Contraction axes are mandatory and spelled
by the author (`contract(q, k, axis="hk")`) — batch dims ride along,
named dims never collide by accident.

## A transformer block is a closure over parameters

```python
def make_attn(s, cfg):
    D, H, K = cfg.d, cfg.nh, cfg.hk
    wq = s.param("wq", d=D, nh=H, hk=K)
    wk = s.param("wk", d=D, nh=H, hk=K)
    wv = s.param("wv", d=D, nh=H, hk=K)
    wo = s.param("wo", nh=H, hk=K, d=D)
    scale = 1.0 / math.sqrt(K)

    @unit
    def attn(h):
        a = layernorm_t(h, ln1g, ln1b, feat="d", eps=cfg.eps)
        q = contract(a, wq, axis="d")
        k = contract(a.rename(t="s"), wk, axis="d")
        v = contract(a.rename(t="s"), wv, axis="d")
        sc = contract(q * scale, k, axis="hk")
        pr = causal_softmax_t(sc)
        cx = contract(pr, v, axis="s")
        return h + contract(cx, wo, axis=("nh", "hk"))

    return attn
```

No module class, no parameter registry: the program *is* the parameter
container. A maker takes a scope and returns a unit; parameters are
declared with named dims and get hierarchical names for free
(`h.0.attn.wq` — checkpoint-stable because the naming law derives them
from structure). The body is the same parameter-blind library
(`contract`, `layernorm`) that runs eagerly on numpy-backed tensors —
one definition, run eagerly or lowered by inspection, differentially
tested against each other.

## Flash attention's backward is derived, not written

```python
@dataclass(frozen=True)
class SoftmaxState:          # running max, denominator, weighted sum
    m: object
    den: object
    o: object

def _flash_combine(L, R):    # the online-softmax lemma, as a combine
    m = maximum(L.m, R.m)
    sl, sr = exp(L.m - m), exp(R.m - m)
    return SoftmaxState(m, L.den * sl + R.den * sr, L.o * sl + R.o * sr)

flashsm = defreducer("zoo.flashsm", state=SoftmaxState, element=2,
                     lift=_flash_lift, combine=_flash_combine,
                     init=SoftmaxState(-1e30, 0.0, 0.0), project=_flash_project)

# inside attention:
return reduce(flashsm, (se, ve), "s")
```

A reduction with *record* state. Because the combine is a small named
body over primitives, the AD machinery differentiates it by inspection
— flash's backward exists because the case analysis is finite, not
because anyone wrote it.

## Dropout costs zero bytes, and recomputation is bit-identical

```python
mask = uniform(fold_in(key, "train.drop"), lattice.layout)   # a FIELD, not an array
assert mask.buffer.data is None                              # zero bytes, exact under views
```

Randomness is a counter-based closed-form field: reading coordinates
regenerates the same bits by construction. The consequence is the
recompute theorem, pinned as a test: a revolve-checkpointed training
step and the store-all schedule produce **bit-identical** gradients
with dropout on — no mask is ever stored anywhere.

## Debugging is a name, not an API

```python
@compute
def shade(img):
    y, x = thread_idx("y", "x")
    dist = (y - 1.0) * (y - 1.0) + x * 0.0    # a named binding IS a tap site
    img[y, x] = dist * 2.0

shade(img)                                     # plain call
shade[config(taps={"dist": buf})](img)         # bind the site; buf gets the full lattice
shade.taps(img)                                # every binding, validity, dims, reasons
```

No `tap()` call exists — the naming law is the claiming mechanism. And
it is honest: inline a helper twice and its binding goes non-unique, so
the site is reported INVALID with the reason, never silently renamed.

## Recompilation is something you can forbid

```python
shade(twill(0.5, 0.5) | zoom(2.0), img)        # compile once...
with events.forbid("kernel.miss"):
    for i in range(1, 50):                     # ...then 49 fresh pipelines,
        shade(twill(float(i), 0.1 * i) | zoom(1.0 / i), img)   # zero compiles — proved
```

Identity is *types, never values*: new captured values are a warm hit
riding the uniform channel; a new shape or a swapped stage is honestly
a new artifact. `forbid` turns "this loop is hot" from a hope into an
assertion, and guards catch a rebound capture loudly instead of serving
bytes from a world that no longer exists.

## The oracle is readable, and it fails like a device

```python
reference(f)(3.0, 4.0)       # oracle execution is always SPELLED — never silent fallback
```

The reference evaluator renders each program to readable Python (the
artifact carries its own listing) and computes floats on numpy scalars:
0/0 flows as nan and `sqrt(-1)` is nan, exactly as a GPU would behave —
the oracle every device backend will be differential-tested against,
including its edge cases.

## Memory is a boundary fact you declare

```python
enc = QuantGroupEncoding(int4, group=32)       # int4 weights + per-group scales
x.round_to(enc)                                # precision loss is an IR op with a gradient story
```

Encodings (float formats, structured records, quant groups, srgb) live
in descriptors at the boundary; interiors are exact. Cost oracles read
descriptors, so op counts and peak memory are exact numbers computed
from layouts alone — before anything runs.

## Views are free, and physics keeps its units

```python
def fdtd_step(E, H, n: Literal[int]):
    dE = (E.shift(x=-1).slice(x=(0, n-1)) - E.slice(x=(0, n-1))).with_charts(x=h_chart)
    H1 = H + c * dE
    dH = (H1.slice(x=(1, n-1)) - H1.shift(x=1).slice(x=(1, n-1))).with_charts(x=e_chart)
    return E + c * dH.pad(x=(0, n), fill=0.0), H1
```

`slice`/`shift`/`window`/`decimate`/`split`/`merge` are zero-cost views
in an affine layout algebra (deliberately not piecewise — every
adjoint of a layout op is again a layout op). Charts carry physical
coordinates through the views, so a staggered-grid FDTD step is five
lines and the alignment is *checked*, not hoped.

## Committed, arriving (specified as skipped tests)

These spellings are ratified and pinned in `test_kernel_spec.py`; each
un-skips when its machinery lands:

```python
by, bx = block_idx("y", "x")                       # the RAW ambient pair
g = grid_layout()                                  # the launch grid AS A LAYOUT
gy, gx = global_thread_idx((by, bx), (ty, tx), g)  # layout evaluation, not an intrinsic

@vertex
def quad():                                        # no vertex buffers
    vid = vertex_index()
    u = 1.0 if (vid == 1 or vid == 3 or vid == 4) else 0.0
    v = 1.0 if (vid == 2 or vid == 4 or vid == 5) else 0.0   # claimed varyings
    return position(u * 2.0 - 1.0, v * 2.0 - 1.0)

@fragment
def shade(f, varying):                             # requirements INFERRED from use;
    return f(varying.v * 23.0, varying.u * 39.0)   # any superset producer pairs

pso = pair(quad, shade)                            # PSO composition — never |
render(pso[config(taps={"dist": gbuf})], f, target=img)   # MRT: a tap bound to a buffer

tex = upload(src)                                  # a wgpu runtime object — NOT a tensor
sample(tex, sampler(filter="linear"), (u, v), lod=0)
```

The through-line, never stated in the stops above but present in all of
them: one body language, one derivative table, names as contract, and
identity by type — the same four decisions, visible from every angle.
