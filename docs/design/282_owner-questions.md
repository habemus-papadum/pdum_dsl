# 282 — Owner questions (working file)

**Status: WORKING FILE for the graphics/runtime campaign.** Twelve
decisions, written in plain language, sorted into three buckets:
six real decisions, three where a one-line direction is enough, and
three proposed postponements. Each item has an empty **Comments**
block — write rulings, edits, or pushback there and I'll fold them
into 280 and start the cleanup against them.

Jargon used throughout: a *kernel* is any `@compute`/`@vertex`/
`@fragment` function; the *reference* is the numpy CPU implementation
that defines correct answers; the *mask form* means "compute both
options, then pick per element with the condition" (what the code
calls `where`/`select`); an *artifact* is the compiled-and-cached form
of a kernel.

---

## Bucket 1 — real decisions

### 1. The kernel rulebook: what are you allowed to write inside a kernel body?

**Background.** Right now the answer is accidental. The probe
(`explorations/graphics/probe_kernel_bleed.py`) showed that
`x.flip("i")` — a tensor-layout operation that belongs at the host
level — *works* inside a `@compute` body, while `shift` and `pad`
fail there, but only because they trip over unrelated errors, not
because anyone ruled them out. Only reductions refuse with a real
message ("`red` is a host citizen here"). So the boundary exists, but
nobody drew it — it's whatever the machinery happens to accept.

**The proposal** (the owner's seed position): a kernel body may
contain exactly — reading a tensor at an index, writing a tensor at
an index, scalar math on those values, captured variables as
uniforms, and records. Everything tensor-shaped (flips, pads,
reductions, reshapes) happens *outside* the kernel; you apply it at
the call site and pass the resulting view in. All the accidental
cases get one consistent, well-worded refusal.

**Cost/risk.** Small implementation (one refusal class in the
lowering rules); nothing in the repo relies on the accidental
behavior.

**Leaning: ratify as stated.**

**Comments:**
Claimed by L4 team -- see their comments on the PR.  

---

### 2. Control flow: one consistent rule for `if`

**Background.** Today there are three different behaviors. (a) Inside
a `@jit` scalar function you can write real `if` statements — they
get converted to the mask form when the function is inlined into a
kernel. (b) Inside `@compute` bodies, *all* `if` forms are refused.
(c) Inside `@vertex` bodies, the expression form (`a if c else b`) is
allowed and becomes the mask form, but the statement form isn't. The
spike proved the mask form costs literally nothing for branches on
the GPU (measured 1.000× vs. a real `if`), so the underlying law is
good — the inconsistency is just in what you're allowed to *write*
where.

There's also a principled reason kernels refuse `if` statements
that's worth making explicit: kernel bodies contain *stores* (writes
to buffers). "Compute both sides and pick" works for values but is
simply wrong for effects — you can't half-execute a write. So
branching around a store can never be allowed under this law.

**The proposal.** Three parts. (i) Ratify the mask-form law itself —
the spike says it's free. (ii) Make the *expression* form
(`a if c else b`, and `and`/`or` on comparisons) legal in **all**
kernel bodies, compute included, exactly as the vertex tier already
does — it's pure by construction, so it's safe. (iii) Keep
*statement* `if` refused in kernel bodies (because bodies contain
stores) and keep it allowed in `@jit` scalar functions (which are
pure). The rule becomes teachable in one sentence: *"values may
branch, effects may not."*

**Leaning: ratify all three parts.**

**Comments:**
Is it possible to adjudicate if a kernel if statement has no stores?
Could we just have it be that only those if statements are not allowed in the kernel? 

---

### 3. Depth buffer: in or out for graphics v2?

**Background.** Our rasterizer has no depth test — triangles just
paint over each other in submission order. You can see the
consequence in the spike's hero image
(`explorations/graphics/spike_runner/frames/hero_256x384.png`): the
cylinder's far wall bleeds through the near wall. This is fine for 2D
and procedural work, but any real 3D mesh demo will look broken.
Adding depth is not just a GPU feature toggle: it changes the
*reference* semantics (the numpy rasterizer needs a z-buffer), the
vertex shader's return type grows a z coordinate (`position(x, y)` →
`position(x, y, z)`), and the render pass needs a depth attachment
declaration.

**Options.** (a) Rule it in scope now, build it together with the
encodable/runner work so the device path is only built once. (b) Keep
v1 flat and pick demos that don't need depth. (c) Defer until after
the cleanup.

**Leaning: (a) — rule it in now, implement with the encodable.** The
part-3 demos (3D meshes) are not credible without it, and doing it
during the encodable build is the cheap moment.

**Comments:**
agreed -- do a
---

### 4. Where does the new code live?

**Background.** The spikes proved we need real homes for two kinds of
code: *runtimes* (device management, buffers, launch, timing,
presentation — the Metal spike's runtime imports nothing from pdum at
all) and *backend emitters* (IR → WGSL/MSL text — these import the
IR). Today device code squats in `graphics.py` (which is supposed to
be pure reference semantics) and the translators live in
`conformance/` (which is supposed to be tests only). One constraint
to know about: the spec ruled the conformance executor must never be
called a "backend" and never live in `backends/` — but that ruling
was about not dignifying a dumb translator, not about banning the
directory name forever. Second constraint: the spec's title is
literally "one workspace, two packages" — adding a third package is a
spec amendment only the owner can make.

**Options.** (a) A third workspace package — say `pdum.rt` — holding
runtimes and per-target emitters, with `conformance/` keeping only
the differential tests and importing the real translators from it
(this kills the three duplicate copies of the translation tables).
(b) A subpackage inside tensorlib, `pdum/tl/rt/`, promoted to a
package later if it earns it. (c) Keep everything under
`conformance/` until the design is more settled.

**Leaning: (a), the third package.** The Metal spike showed runtime
code has *zero* pdum dependencies — it's genuinely a separate layer,
and both the dsl tier and tensorlib will want it. (b) is the safe
middle if the spec shouldn't be amended yet; take it over (c).

**Comments:**
Agree a
---

### 5. The bounded loop: who designs it, and when?

**Background.** This is the spike's biggest finding. For *branches*,
our mask form is free. For *loops that stop early* (a raymarcher
marching until it hits a surface), the mask form costs 2.2–4.0× —
every thread pays for the slowest thread. And currently the situation
is worse than slow: a `for` loop in a `@jit` function doesn't even
reach the GPU — it silently falls back to a per-element Python path
that no device can run. So the canonical procedural-graphics demo
(raymarching) is unreachable today. What's needed is a new construct:
a loop with a fixed maximum count and a *declared* early-exit
condition, whose reference semantics stay simple (run all iterations,
ignore results after the exit condition triggers — so autodiff and
analyses still see straight-line code) but which a backend is
licensed to compile to a real `break`.

**The decision is about process, not design:** kernel-language
constructs are the L4 team's territory (their queued K-A…K-G design
conversation), but *our* demo campaign is the first consumer, and the
spike is the evidence.

**Options.** (a) Hand the whole question to L4 with the spike as an
exhibit; our demos avoid raymarching until it lands. (b) We design
the *surface syntax and reference semantics* in 280 (as the first
consumer), send it to L4 as a second communiqué, and they own the
IR/lowering design. (c) We design *and* prototype the lowering
ourselves in the campaign.

**Leaning: (b).** Consumer-driven design is the house doctrine
("doors open when a consumer arrives"), and we're the consumer — but
building kernel-IR machinery is exactly what the L4 brief says not to
fragment. If their timeline leaves our demos blocked, revisit (c)
with owner say-so.

**Comments:**
option a
---

### 6. Policy for "works on CPU but can't reach the GPU"

**Background.** Today, when a kernel uses something the WGSL
translator doesn't handle, the conformance test *skips* — the test
suite stays green, and the gap is invisible unless someone reads skip
messages. Separately, there are two very different kinds of gaps:
things that will *never* be legal in a kernel (a layout op like
`flip` — decision 1 handles those with refusals), and things that are
legal but *not yet implemented* on the device (3-D thread grids,
reductions). The question is what to do about the second kind.

**Options.** (a) Hard line: a kernel tier refuses anything the device
can't currently run (this would break currently-working
reference-tier code, e.g. rank-3 kernels). (b) Soft line:
legal-but-not-yet-translatable stays allowed on the reference, but
the gap list becomes a *visible, versioned ledger* (a table in the
docs / a test that fails when the ledger drifts from reality), so
coverage regressions can't hide in green runs.
Intentionally-reference-only features (the per-element oracle path)
get explicitly marked as such.

**Leaning: (b).** (a) punishes users for the backend's immaturity;
(b) fixes the actual problem, which is silence.

**Comments:**
accepted -- do b
---

## Bucket 2 — a one-line direction is enough

### 7. What does a name like `metal` mean in `kernel[on(metal)]`?

The Metal spike showed the code generator (makes MSL text) and the
runtime (runs it) are genuinely separable — the same MSL could be
launched by a different harness, the same runtime could launch
precompiled libraries. But users shouldn't have to know that.
**Proposed direction: user-facing names like `metal` and `webgpu` are
pre-assembled generator+runtime *pairs*; internally the two halves
stay separate objects, so an unusual pairing can be constructed
explicitly when someone needs one.** Nothing else to decide until a
second pairing exists.

**Comments:**
Have user specify both generator and runtime -- make them dataclasses (nothing in them for now), and create dataclass to hold the pair. Then the user-facing names are just pre-assembled instances of that dataclass.
---

### 8. What replaces the `Buffer.device` string?

Today every buffer carries a text label like `"host"` saying where it
lives. Unified memory breaks the concept: after adoption, one
allocation *is simultaneously* host memory and Metal device memory,
and with two runtimes "device" must identify a specific live device
object, not a word. **Proposed direction: device becomes a reference
to a registered runtime/device object (not a string), and a buffer's
residency is a *set* (it can be resident in more than one place).
Full design lands with the residency/encoding-descriptor work already
queued — not now.**

**Comments:**
Agreed
---

### 9. Test tolerance for frame-sweep conformance

The GPU-vs-CPU image test passes at the single camera angle it was
written for; sweeping angles, two pixels out of six thousand land at
0.005 error (tolerance is 0.002) — both GPU paths agree with each
other perfectly, and the offending pixels sit exactly on the steepest
part of the anti-aliasing gradient, where float32-vs-float64
differences concentrate. **Proposed direction: the frame-sweep
battery checks two things separately — pixel-exact agreement on
*which* pixels are covered, and a value tolerance that excludes
pixels where the reference's local gradient is steeper than a stated
threshold. Owner reviews the wording when it exists.**

**Comments:**
Agree
---

## Bucket 3 — postpone, and why

### 10. If-reconstruction shipping policy

We built a pass that can turn mask-form code back into real GPU `if`
statements — proven correct, but measured worthless on every kernel
shape we can currently produce (they're memory-bound, not
compute-bound). We already agreed not to ship it. The open policy
question ("should it branch only on masks that came from a user's
`if`, or on any mask with exclusive arms?") only matters when it
ships. **Postpone until a compute-bound consumer exists; the question
is recorded in 280.**

**Comments:**
agree postpone
---

### 11. The per-target math-function registry

Metal's `tanh` returns NaN for large inputs; the fix is a free clamp.
The *category* — per-target math quirks, their fix rows, and proofs
the fixes are harmless — eventually wants a declared registry. But we
have exactly one data point. **Postpone the mechanism until CUDA
arrives with its own list (it will); stopgap now: the clamp row goes
into the emitters during cleanup, and the test battery gains
wide-range inputs so this class of bug can't hide.**

**Comments:**
agree postpone

---

### 12. The NaN-gradient trap

There's a classic autodiff bug in systems that compute both sides of
a mask: the gradient of `where(c, sqrt(x), 0)` can become NaN at
points where the *unused* side is NaN, because 0 × NaN = NaN. Whether
our system has the bug is unknown. **No decision needed: the test
gets written during cleanup. If it passes, done. If it fails, the
standard fix touches the derivative table — which is frozen behavior
— and *then* it becomes an owner decision.**

**Comments:**
agree postpone
