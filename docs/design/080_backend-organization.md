# 080 — Backend organization: families, targets, cells

*Settled at the ch10 walkthrough (2026-07-12, user-driven). Extends 070
(which chose WHICH backends and HOW to invoke them) with WHERE the concepts
live in code and HOW backends get chosen at dispatch. Companion sections:
070 §1 (the taxonomy this doc gives a code layout), §3 (invocation).*

## 1. Two declarations, one sparse matrix

**Kind ≠ backend.** They are different axes, declared in different places,
connected by a sparse matrix of thin *cells*:

- **A kind is declared by its FAMILY (dialect) package.** A family owns the
  semantic contract: the role registration and composition rules (what `|`
  means for it), the family's ops and lowering conventions, the parameter
  contract (compute v1: params ARE thread coordinates), the launch *shape*
  (grid dispatch vs draw call vs audio render callback), and the output
  discipline (DPS plans, out-shape rules). The ch04 rule — role vocabularies
  ship with their owning package — extended a level up. A family says
  nothing about where code runs.
- **A target is declared by its backend package.** A target owns the
  execution substrate: device management, the compile pipeline (NVRTC, MSL,
  naga), memory and streams, and its spelling tables (`code_for_op`,
  `type_map`). A target says nothing about what a kernel means.
- **A cell = family × target** is what actually registers into routing: the
  composition of one family's contract with one target's substrate. Cells
  are THIN — the evidence is both prior art (tinygrad's renderers are
  string tables over one IR; Warp is one namespace over CUDA + CPU) and our
  own demo: `COMPUTE` and `FRAGMENT` are two Backend records sharing one
  wgsl runtime module, differing exactly in the family-varying columns
  (`plan`, `param_types`, launcher shape).

```
families:   compute      fragment       audio_node (hypothetical)
targets:
  cpu       cell         cell           cell   (render for the audio thread)
  cuda      cell         — (no raster)  cell   (block DSP on GPU)
  metal     cell         cell           cell
  wgsl      cell         cell           —
```

The matrix is **sparse by design and holes are loud**: CUDA×fragment does
not exist because CUDA has no raster stage; an unrouted kind raises
`NoBackend`, never approximates.

**Shared logic has two homes, by axis.** Cross-target-per-family
(launch-domain math, DPS planning, an audio graph scheduler) lives in the
family package. Cross-family-per-target (the CUDA context/stream/compile
machinery that compute and audio cells both need) lives in the target
package — most of each 130–220-line runtime, written once per target.
Cross-everything (the dominator-placed emitter) lives at the
`backends/_emit.py` level. Cells are the remainder: binding one to the
other, plus a dotted name.

## 2. Backend resolution: three tiers

Kernels declare **kind only** (`@jit(kind="compute")`) — kind is the
contract the kernel is written against; the backend is resolved at
dispatch. Ordered by priority:

1. **Data-driven** (the workhorse once arrays land, ch12): compute where the
   data lives. DLPack gives every array a device (kDLCUDA, kDLMetal, …);
   the device is a structural property of the value, so it belongs in the
   `Array` Type that `typeof` produces — and then backend selection **is**
   type dispatch, the operation this framework is built on. No new
   mechanism; the backend fp is already a specialization-key component, so
   a device change is a *named* key miss (two-tier law holds for free).
2. **Explicit override** (per call / per configured stage): rare for users,
   REQUIRED for us — the differential harness and ch15's
   four-targets-one-IR chapter force specific backends on the same kernel.
3. **Routed default** (`Registry.routes: kind → cell name`, falling back to
   the registry default): activation-lite. This is the matplotlib move but
   tamer — explicit registry state, one line to change, and a `Registry`
   instance is itself the scoping mechanism. Deliberately NO activation API
   (no context managers, no `use_backend()`): routes-as-data covers today,
   and tier 1 shrinks the problem before an API would earn its surface.

Honest note recorded so the demo is not mistaken for doctrine: today's
kernels are all-scalar, so they carry NO device signal — tier 3 does all
the work, which is why it currently *feels* like "kind picks backend."
Degenerate case, not the design.

## 3. The code layout

```
pdum/dsl/
  stdlib/                    # base-language dialect; declares "device"
  families/                  # BORN AT STEP 14 with the second compute target:
                             #   compute.py, fragment.py (roles, contracts,
                             #   family ops, shared planning). Not created
                             #   before it has two consumers — the demo would
                             #   be its only tenant today.
  backends/                  # NAMESPACE PACKAGE (no __init__.py): the
    _emit.py                 #   contribution point. First-party shared infra
    cuda/     (step 14)      #   may live here; target packages are its
    metal/    (step 14)      #   citizens: runtime.py + one thin cell module
    wgsl/     (grown-up)     #   per served family (cuda/compute.py, …).
  demo/                      # THE SPECIAL CASE (this doc's occasion):
    simple_shader/           #   the ch09/ch10 vertical-slice pair, family+
      python.py  wgsl.py     #   target FUSED in one module each — on purpose,
                             #   they are the book's teaching artifacts.
                             #   Cell names: demo.simple_shader.python,
                             #   demo.simple_shader.wgsl.{compute,fragment}.
```

Why `dsl.demo` and not `backends/demo/`: backends/ citizens are expected to
be target packages serving families through cells; the demos deliberately
violate that shape (fused), and placing them inside the contribution point
would present them as the pattern to copy. Their fused form is right for
chapters and wrong for real backends — so they live where their nature is
explicit.

Cells live with their TARGET (they depend on target internals and import
family machinery). Cell registration names are dotted paths
(`cuda.compute`, `demo.simple_shader.python`) — the naming scheme exists so
implementation names can never collide with or be mistaken for role names.

One scheduled migration, recorded now: the demo wgsl module currently
declares the `compute`/`fragment` ROLES itself — acceptable only because it
is family-and-target fused. When the second compute-family target lands
(step 14), the family declarations MUST move to `families/` or the two
targets would fight over role ownership. That is the moment `families/` is
born, with a real consumer.

## 4. The contribution contract (specified now, implemented at step 10)

`backends/` is a PEP 420 implicit namespace package: no `__init__.py`, so
separately-installed distributions can place target packages inside it
(the `sphinxcontrib.*` pattern). Registration must therefore not depend on
package-import side effects. The contract:

- Every backend package exposes `install(registry) -> None` (the seam the
  demo and stdlib packages already honor).
- Discovery for third parties: an entry-point group **`pdum.dsl.backends`**
  (`importlib.metadata.entry_points`) whose entries name `install`
  functions; the Registry (surface E, completed at step 10) gains
  `load_entry_points()` — explicit, lazy, no import-order dependence.
  External packages need not live inside our namespace to be discovered;
  the shared namespace is coherence, the entry point is the mechanism.
- First-party batteries remain explicit imports in `pdum.dsl.__init__`
  (today: `from . import demo`), never implicit discovery — batteries must
  be readable in one place.

## 5. What changed in code at this settlement

- `backends/{python,wgsl}.py` → `demo/simple_shader/{python,wgsl}.py`;
  registration names and fps became the dotted cell names
  (`demo.simple_shader.python`, `demo.simple_shader.wgsl.compute`,
  `demo.simple_shader.wgsl.fragment`). Free today: fps live only in
  in-process cache keys; the moment artifacts persist to disk this rename
  would have stopped being free — which is why it happened now.
- `backends/__init__.py` deleted (namespace-ready); `_emit.py` remains as
  first-party shared infrastructure inside the namespace package.
- Budget buckets split: `sat:backends` (infra, 150) and `sat:demo` (600).
- KIND names follow the same hygiene as cell names (settled at the same
  walkthrough): the demo registers `simple_shader.compute` /
  `simple_shader.fragment`, RESERVING the plain `compute`/`fragment` for
  the real family packages (step 14) — a demo must never squat on a name a
  future package will use with richer semantics. `device` is NOT demo-owned
  and keeps its name: it is stdlib's base-language kind (`@jit`'s default,
  user-affirmed at ch04), its semantics — the neutral composable kernel —
  are already its final semantics, and it reaches the Python twin only via
  the registry DEFAULT fallback, never via a route.
