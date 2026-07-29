# 284 — Syntax sketches: three programs, written to be marked up (DRAFT)

> **MARKED UP (2026-07-29).** Sketch 2 and Sketch 3 are superseded by
> `285_sketch-training.md` — the training program redone in the model-zoo
> idiom (makers, closed-over params, gradients via the naming law) and
> the compiler pipeline moved from `.then` to `|`. Sketch 1 (graphics)
> is ruled substantially off; its replacement is the next step and will
> be built by transforming a real render loop toward the target syntax.

**Status: DRAFT FOR OWNER MARKUP.** These are NOT transcriptions of the
current design. Per the vetting process: assume the built design is
wrong where it disagrees with these; this is the syntax I believe the
owner's model implies, written out whole — every name defined in this
file, no dangling references. The divergence ledger at the end lists
exactly where each sketch departs from what exists, so markup can say
"keep", "bend", or "kill" per line.

## The model these sketches assume (one paragraph, three roles)

A **program** is what you wrote — tl/dsl code; its meaning is fixed by
the reference and never changes below this line. A **runtime** is the
compiler FRONTEND: the thing you hand a program to (`on(metal)`); it
owns devices, memory, launch, presentation — and it DRIVES compilation
but does not define it. The **compiler** is pluggable: a fingerprinted
pipeline of denotation-preserving passes ending in source emission;
every runtime carries a DEFAULT compiler (today's naive
translate-and-run), and any invocation may hand the runtime a
different one. The executor cache keys on (program content, compiler
fingerprint, device, thread sizing) — so swapping compilers mid-loop
creates a new row and NEVER invalidates the old one: flipping back is
warm, and A/B-ing two compilers on a live loop is free. Uniform values
never appear in any of this: they ride the captured-scalar channel,
rebound by ordinary Python assignment.

---

## Sketch 1 — the anti-aliased circle, mouse-driven, with live controls

```python
# aa_circle.py — a windowed WebGPU app: an analytically antialiased
# circle follows the mouse; two immediate-mode sliders tune it.

from pdum.dsl import jit, value_and_grad
from pdum.dsl.intrinsics import clamp
from pdum.dsl.markers import sqrt
from pdum.rt import on, webgpu
from pdum.rt.present import window
from pdum.tl import i32, thread_idx
from pdum.tl.graphics import fragment, pair, position, vertex

# --- the captured state -----------------------------------------------------
# Plain module globals. A shader body that reads one CAPTURES it — that
# read IS the binding. No bind call, no uniform declaration, no name
# ever repeated at a call site. Rebinding = assignment.

_CX = 0.0   # circle center x  <- the mouse (clip space)
_CY = 0.0   # circle center y  <- the mouse
_R = 0.25   # radius           <- a slider
_SOFT = 1.0  # edge-width scale <- a slider


# --- the program ------------------------------------------------------------

@jit()
def sd(y, x):
    """The circle's signed distance. _CX/_CY/_R are captures: four
    bytes each in the slot channel, refreshed per frame, never named
    again below."""
    return sqrt((x - _CX) * (x - _CX) + (y - _CY) * (y - _CY)) - _R


F = value_and_grad(sd, wrt=("y", "x"))  # the analytic edge: no fwidth, no 2x2 quad


@vertex
def quad():
    """The screen quad from the draw ambient — no vertex buffers. u and
    v are claimed varyings (assignment claims; the tagless law)."""
    (vid,) = thread_idx("vertex_id")
    i = i32(vid)
    u = 1.0 if (i == 1 or i == 3 or i == 4) else 0.0  # noqa: F841
    v = 1.0 if (i == 2 or i == 4 or i == 5) else 0.0  # noqa: F841
    return position(u * 2.0 - 1.0, v * 2.0 - 1.0)


@fragment
def shade(f, varying):
    """f is the paired (value, gradient) field — the fragment's ONE
    fn-argument. Coverage from signed distance over one screen
    footprint; the spotlight-free minimal case."""
    d, (dy, dx) = f(varying.v * 2.0 - 1.0, varying.u * 2.0 - 1.0)
    w = sqrt(dx * dx + dy * dy) * _SOFT + 1e-6
    return clamp(0.5 - d / w, 0.0, 1.0)


CIRCLE = pair(quad, shade)  # the PSO: constant for the app's lifetime


# --- the app ----------------------------------------------------------------
# The window is the runtime's presenter: it owns the surface, the swap
# chain, the frame clock, and an immediate-mode UI layer. The program
# and its fn-arguments are handed over ONCE.

app = window(CIRCLE, args=(F,), on=on(webgpu), title="aa circle", size=(900, 600))


@app.frame
def tick(ui):
    """Once per frame, BEFORE encode. Everything here is ordinary
    Python mutating the captured globals; the machinery re-extracts
    captures and rewrites ~16 slot bytes. ui is immediate-mode: the
    slider's label is a UI label, not a binding name — the STATE lives
    in our globals (current value in, new value out)."""
    global _CX, _CY, _R, _SOFT
    _CX, _CY = ui.mouse.clip          # (-1..1, -1..1), y up
    _R = ui.slider("radius", lo=0.02, hi=0.60, value=_R)
    _SOFT = ui.slider("edge width", lo=0.5, hi=3.0, value=_SOFT)


app.expect(lowerings=1, pipelines_after_first_frame=0)  # warmth, PINNED (events seam)
app.run()  # the host owns the loop: tick -> update captures -> encode -> present
```

What to notice: the program is built once, `_R` appears in exactly one
shader line and one slider line, and there is no object in the file
whose job is "binding". The `expect` line is the warmth doctrine made
enforceable — if a numpy float sneaks into a capture and rekeys the
artifact, this app FAILS instead of quietly recompiling per frame.




-------------- 
owner's comments:
This is too much like a traditional shader program. I want something that's more functional. 
Have a look at https://github.com/pygfx/wgpu-py/blob/main/examples/imgui_backend_sea.py -- this is not what I want either. But it does show a bunch of actual details of the web GPU render loop that are important to understand. I want the example to include that detail and not have some magical app. I like to control the loop myself. Which, again, I don't think this application actually does. 



---

## Sketch 2 — a tensorlib training loop: MLP, SGD, printed loss

```python
# train_mlp.py — two-layer MLP on streamed batches, plain SGD,
# loss printed every 100 steps. The MODEL spelling is schematic
# (the zoo's conventions); the LOOP is the thing being vetted.

import numpy as np

from pdum.dsl import value_and_grad
from pdum.dsl.markers import maximum
from pdum.rt import metal, on
from pdum.tl import Tensor, red

LR = 1e-2  # a capture of the update program, like any other scalar


# --- the model (schematic) --------------------------------------------------

def mlp(x, w1, b1, w2, b2):
    """x: ("batch","d_in"); w1: ("d_in","hidden"); w2: ("hidden",).
    Contraction by NAME (the naming law); relu via maximum."""
    h = maximum(red.sum(x * w1, "d_in") + b1, 0.0)
    return red.sum(h * w2, "hidden") + b2


def loss_fn(x, y, w1, b1, w2, b2):
    p = mlp(x, w1, b1, w2, b2)
    return red.mean((p - y) * (p - y), "batch")


GRAD = value_and_grad(loss_fn, wrt=("w1", "b1", "w2", "b2"))


def sgd(w, g):
    return w - LR * g  # functional: a NEW tensor. In-place is a compiler's


def step(params, x, y):
    """One training step, whole: loss, grads, update. This function IS
    the program the compiler sees — how it becomes device kernels
    (how many launches, which buffers get reused) is the COMPILER's
    business, not this file's."""
    loss, grads = GRAD(x, y, *params)
    return loss, tuple(sgd(w, g) for w, g in zip(params, grads))


# --- the loop ---------------------------------------------------------------

def init_params(rng):
    d_in, hidden = 32, 128
    return (
        Tensor.from_numpy(rng.normal(0, 0.1, (d_in, hidden)), ("d_in", "hidden")),
        Tensor.from_numpy(np.zeros(hidden), ("hidden",)),
        Tensor.from_numpy(rng.normal(0, 0.1, hidden), ("hidden",)),
        Tensor.from_numpy(np.zeros(()), ()),
    )


def batches(rng):
    while True:
        x = rng.normal(0, 1, (64, 32))
        yield (
            Tensor.from_numpy(x, ("batch", "d_in")),
            Tensor.from_numpy(np.sin(x).sum(axis=1), ("batch",)),
        )


rng = np.random.default_rng(0)
params = tuple(p.to(metal) for p in init_params(rng))  # ONE conscious transfer each
train = step[on(metal)]                                # the frontend, default compiler

for i, (x, y) in enumerate(batches(rng)):
    loss, params = train(params, x.to(metal), y.to(metal))
    if i % 100 == 0:
        print(i, float(loss))  # float() IS the readback — the ONE explicit
        #                        device->host act, paid only when you print
        #                        (a sync readback is ~1.4 ms of protocol; the
        #                        loop's other 99 steps never leave the device)
    if i == 1000:
        break
```

What to notice: `params` are device-resident for the whole loop and
the update writes stay on the device; the ONLY host↔device traffic in
steady state is the batch upload and the occasional `float(loss)`.
There is no optimizer object, no `.backward()`, no parameter registry
— the step is a function, the optimizer is a two-line program, and
what fuses into which kernels is deliberately not visible here.

---

## Sketch 3 — the research case: swapping the compiler under a warm loop

```python
# compilers.py-flavored continuation of train_mlp.py — the part 270
# calls "transform, run, measure": the PROGRAM never changes; the
# COMPILER is the experiment.

from pdum.rt import compilers, events, metal, on

# A compiler is a fingerprinted pass pipeline ending in emission.
# `on(metal)` implies the default one; naming it makes it a value:

naive = compilers.default(metal)                 # translate one-op-per-launch
fused = naive.then(compilers.fuse_pointwise)     # elementwise chains -> one kernel
tight = fused.then(compilers.reuse_buffers)      # intermediates share allocations

# The same step, three ways. Three cache rows, all warm once built —
# these are HANDLES, and switching between them costs nothing after
# the first touch of each:

t_naive = step[on(metal)]                        # == on(metal, compile=naive)
t_fused = step[on(metal, compile=fused)]
t_tight = step[on(metal, compile=tight)]

# (a) the A/B: same live loop, compiler swapped mid-stream, effect
# visible immediately in the events the runtime already emits

rec = events.recorder()
for i, (x, y) in enumerate(batches(rng)):
    train = t_naive if i < 300 else t_fused if i < 600 else t_tight
    with rec.span(f"step[{i}]"):
        loss, params = train(params, x.to(metal), y.to(metal))
# rec now shows: launches/step 9 -> 3 -> 3, allocations/step N -> N -> 0,
# and exactly TWO compile events in 900 steps (one per new row).
# Meaning never moved: any step can be replayed on the reference.

# (b) the measured bake-off — one batch, every candidate, one table

for c in (naive, fused, tight):
    with rec.span(c.name):
        step[on(metal, compile=c)](params, x.to(metal), y.to(metal))
print(rec.table())  # gpu-time per candidate, from timestamp queries

# (c) a candidate of your own: passes are program -> program over
# regions, DECLARED denotation-preserving; the certification is a
# differential against the reference, not trust

@compilers.rewrite
def my_split(program):
    """A hand-rolled experiment: split the batch dim so the update
    kernels tile. Regions in, regions out; strings never appear."""
    return compilers.split(program, dim="batch", by=16)


mine = naive.insert_before("emit", my_split)     # pipeline surgery; new fingerprint
loss, params = step[on(metal, compile=mine)](params, x.to(metal), y.to(metal))
```

What to notice: the runtime is the only thing you ever *invoke*; a
compiler is a value you hand it. Compilers never touch devices,
runtimes never transform programs, and the cache is what makes the
research loop livable — every candidate you've ever built stays warm
behind its fingerprint, so "try the January pipeline again" is a cache
hit, not an archaeology dig.

---

## Divergence ledger — where these sketches depart from what is built

1. **The generator disappears from the user surface.** 282 §7 ruled a
   (Generator, Runtime) pair; these sketches imply Runtime + pluggable
   Compiler, with source emission as the default compiler's LAST PASS.
   `on(metal)` = the metal runtime with its default compiler;
   `on(metal, compile=c)` overrides per invocation. The rt package's
   `select.Pair` machinery survives internally (emission still needs a
   dialect per target) but stops being the user-facing selection.
   NEEDS AN OWNER RULING amending 282 §7.
2. **`on(...)` on non-kernel programs.** The bracket exists on
   `@compute` kernels today; sketch 2 applies it to a whole training
   step (assemblage + autodiff + updates). That is the real
   integration: the tl entry tier must hand rt a program, not a
   kernel. Large, known, and the point.
3. **`compilers.*` is a new namespace** — default/then/insert_before/
   rewrite, fingerprinted pipelines, pass certification by
   differential. The pass MACHINERY (fusion, buffer reuse, split) is
   L4's; the SPELLING and the cache law (compiler fp keys the row,
   rows never die) are what this sketch commits us to.
4. **`present.window` + `@app.frame` + immediate-mode `ui`** — the
   presenters increment, now with the corrected surface: no bind
   calls; UI values flow into captured globals by assignment;
   `app.expect(...)` pins warmth through the events seam.
5. **`x.to(metal)` / resident params** — the L2 residency contract
   (the tour's committed-future cell), assumed here with runtime
   objects instead of strings. The demo proved the mechanism
   (one resident buffer, strided binding); the tensor-tier surface is
   still future work.
6. **`float(loss)` as THE readback spelling** — `.item()` exists; the
   sketch commits to "leaving the device is explicit, and printing is
   the act that pays it".
7. **`events.recorder()/span/table/expect`** — the dsl events seam
   exists and `expect()` budgets exist; the rt-facing surface
   (launch/compile/allocation events, gpu-time spans) is new.
```
