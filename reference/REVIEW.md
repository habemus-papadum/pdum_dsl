# Code Review Guide

A guided tour of the `pdum.dsl` implementation, organized for the review path you asked
for: start from the **most specific use case** (WebGPU), confirm it gets to the right
place, then step out to the **framework / DSL-design** level, then go deep into the
**internals & rationale**. Each section lists files in reading order, what to scrutinize,
how the pieces connect, which tests prove what, and the open design decisions to weigh in
on.

Prose docs for each layer live under `docs/` (run `uv run mkdocs serve`):
[WebGPU guide](../docs/m0/guide/webgpu-shaders.md), [DSL-design guide](../docs/m0/guide/designing-a-dsl.md),
[Theory](../docs/m0/theory/overview.md). This file is about reviewing the *code*.

## How to run things first

```bash
uv run pytest reference/tests -q                              # 22 tests (unit + GPU integration)
uv run ruff check src tests                   # lint
uv run python reference/demos/disk.py --frames 120 # GPU window demo, prints compiles=1
uv run mkdocs serve                           # the docs site

# See the generated WGSL for the inlined higher-order example:
uv run python - <<'PY'
from pdum.dsl_reference import builtins, jit
from pdum.dsl_reference.backends.wgsl import compile_fragment
@jit(kind="fragment")
def shader(f):
    i, j = builtins.FragCoord.xy
    return (f(i, j) / 800.0, 0.3, 0.6)
def make_img(k):
    @jit(kind="device")
    def weave(x, y): return x + y + k
    @jit(kind="device")
    def img(x, y): return weave(x, y) + 10
    return img
print(compile_fragment(shader(make_img(7))).wgsl)
PY
```

Total implementation is ~9 small modules under `src/pdum/dsl_reference/` plus tests. It is meant to
be readable end-to-end in a sitting.

---

## Level 1 — The WebGPU use case (does it get to the right place?)

**Goal of this pass:** decide whether the user-facing API and runtime behavior are what
you want, independent of how they're built.

Read in this order:

1. **`reference/demos/disk.py`** — the API as a user sees it: `@jit(kind="fragment")`,
   `builtins.FragCoord.xy`, captured `cx/cy/radius`, `Context` → `window_drawer` →
   `update`/`show` → `run`. *Does this read the way you want the framework to feel?*
2. **`reference/tests/test_m03_thesis.py`** — the thesis as an executable assertion: a render loop
   that changes capture values shows `compile_count == 1`, `uniform_writes == N`, and a
   pixel check that it actually renders. *Is this the behavior you expected?*
3. **`reference/tests/test_m04_inline.py`** — the demo.py shape: higher-order `shader(img)` where
   `img` calls `weave` (capturing `k`), all inlined, still one compile across changing `k`.
4. **`src/pdum/dsl_reference/webgpu/runtime.py`** — the runtime. Focus on `Drawer.update` (the
   `flatten → cache.get_or_compile → pack → write_buffer` sequence) and `_build` (the
   cached expensive step). `Context`, `OffscreenTarget`/`WindowTarget`, `GpuProgram`.
5. Run the WGSL-printing snippet above and read the emitted `fs_main`.

**What to scrutinize / decide:**

- Is `@jit(kind="fragment" | "device")` the surface you want? (Note `kind` is a string;
  `vertex`/`compute` are accepted but not emitted.)
- The `update(program)` / `show()` split, vs. the original `demo.py` sketch's
  `drawer.update(shader(img)); drawer.show()`. It matches — confirm the ergonomics.
- Captures-as-uniforms: today **scalars only** (`int/float/bool`). The disk demo captures
  `cx, cy` as separate floats, not `center=(cx, cy)`. Is the scalar-only limitation
  acceptable for now? (Vector uniforms are the top near-term gap — see Level 3.)
- The no-recompile guarantee is per-`Drawer` (each `Drawer` owns a `SpecCache`). Is a
  per-drawer cache right, or do you want a shared/global one?

**Gap check (confirm these are flagged, not hidden):** the functional pipeline style
(`twill | widen(2) | ...`, the `🌺` operator, `weave`/`zoom`/color tables) is **not built**
— only the higher-order machinery that would make it inline. See the "Planned" callouts in
[the WebGPU guide](../docs/m0/guide/webgpu-shaders.md).

---

## Level 2 — The framework / DSL-design level (is the core reusable?)

**Goal of this pass:** decide whether the backend-independent core is a clean foundation
for *other* DSLs/backends, not just WebGPU.

Read in this order:

1. **`src/pdum/dsl_reference/jit.py`** — capture. `make_handle` (phase A: code object + closure
   cells → `env_types` → `FnType`), `Handle`, `Program` (calling a `Handle` builds a
   deferred application). *Is this genuinely backend-agnostic? (It is — no WGSL here.)*
2. **`src/pdum/dsl_reference/cache.py`** — `SpecCache.get_or_compile(fntype, arg_types, compile_fn)`,
   generic over the artifact type; the per-key future; `generation`. *Is the cache contract
   the right seam for a new backend?*
3. **`src/pdum/dsl_reference/passes/inline.py`** — `flatten(program)`: inline device fns, merge
   uniforms, resolve higher-order args. Returns `Flattened(fn, names, types, values)`. This
   is reused unchanged by any backend. *Is the `Flattened` contract clear?*
4. **`src/pdum/dsl_reference/types.py`** — the lattice + `typeof`. *Is the extension path (add a
   `Type` subclass + a `typeof` case + a backend layout/emit rule) clear?*
5. Skim **`docs/m0/guide/designing-a-dsl.md`** — it includes an illustrative (non-repo) CPU
   backend sketch showing the `flatten → get_or_compile → feed values → execute` contract.

**What to scrutinize / decide:**

- The **coupling** to flag: `frontend/ast_lower.py` imports `INTRINSIC_NAMES` and
  `passes/infer.py` imports `BUILTIN_CALLS`/`INTRINSIC_WGSL` from `backends/wgsl/`. That is
  a temporary core→backend dependency. The intended fix is a backend-supplied "dialect"
  object injected into the frontend. *Is that the right factoring, and should it happen
  before a 2nd backend or now?*
- The cache key is `(FnType, arg_types, generation)`. A backend whose output depends on
  extra parameters (WGSL: target texture format) must add them to the key — today the WGSL
  format is **not** in the key (one `Drawer` per format). *Acceptable, or fix now?*
- `@jit` returns a `Handle`, not a callable that runs Python. Calling it builds a
  `Program`. *Is this Julia-like "calling builds an application graph" model what you want,
  or surprising?*

---

## Level 3 — Internals & rationale (why, and what the limits are)

**Goal of this pass:** evaluate the compiler internals and confirm the limitations are
understood and acceptable.

Read in this order:

1. **`src/pdum/dsl_reference/ir.py`** — the node set. Small expression/statement tree (not SSA).
2. **`src/pdum/dsl_reference/frontend/ast_lower.py`** — the accepted Python subset and name
   classification (`uniform`/`arg`/`local`), the `builtins.*` intrinsic recognition, the
   tuple-unpack-via-temp lowering, and that any `Call` lowers uniformly (builtin vs device
   resolved later).
3. **`src/pdum/dsl_reference/passes/infer.py`** — bottom-up typing; the promotion rule; note it runs
   against **narrowed** uniform types (the "two type levels" in
   [Type System](../docs/m0/theory/type-system.md)).
4. **`src/pdum/dsl_reference/backends/wgsl/layout.py`** — the WGSL alignment rules (the `vec3`
   size-12/align-16 footgun), `build_layout`, `pack`, `narrow_type`. Cross-check against
   `reference/tests/test_m02_wgsl.py`.
5. **`src/pdum/dsl_reference/backends/wgsl/emit.py`** — `IR -> str`; the full-screen triangle; the
   `f32(...)` numeric coercion (`_emit_f`/`_is_floatish`); `select(false, true, cond)`
   ordering; color coercion (`_emit_color`).
6. **`src/pdum/dsl_reference/backends/wgsl/compile.py`** — ties flatten → layout → infer → emit.
7. **`src/pdum/dsl_reference/lang.py`** — the `builtins` sentinel (iterable so `x, y = ...xy` is
   statically valid; never executed).
8. **`reference/tests/test_m01_core.py`**, **`reference/tests/test_m02_wgsl.py`** — unit coverage of the
   `closure(5)/closure(6)` → one compile property and golden WGSL/layout.

**Decisions made that are worth a second opinion:**

| Decision | Where | Alternative |
|---|---|---|
| Full **monomorphization** (no polymorphic call sites; everything inlines) | `passes/inline.py` | a polymorphic-inline-cache for `Callable` union types (different design) |
| **AST** (source-based) frontend | `frontend/ast_lower.py` | CPython bytecode lowering (more general, heavier) |
| WGSL emitted as **text** | `backends/wgsl/emit.py` | a structured WGSL AST |
| **Per-frame `flatten`** to re-collect uniform values | `webgpu/runtime.py` `update` | cache "extraction paths"; collect values without re-lowering |
| Device fns must be **single-return** | `passes/inline.py` `_inline` | multi-statement with let-hoisting |
| Deterministic uniform naming `prefix+counter` (`weave2_k`) | `passes/inline.py` | path-based names; user-controlled names |
| **Per-`Drawer`** `SpecCache` | `webgpu/runtime.py` | a shared/global cache |
| `generation` as a global **sledgehammer** | `cache.py` | dependency-graph world age |

**Known gaps (confirm the docs flag each):**

- numeric coercion is minimal (int→`f32` only in a float context)
- captured **vector/tuple/struct** uniforms unsupported (`typeof`→`TupleType`, which
  `narrow_type` rejects); scalars only
- core imports the WGSL dialect tables (coupling, above)
- target format not in the cache key
- no cache eviction, no on-disk cache, no dependency-graph invalidation
- no `vertex`/`compute`, varyings, textures, storage buffers, multiple bind groups
- no `if`/`for` statements, lists/indexing, `len()`, user structs, `Literal`, pipe operator
- `resolution`/`time`/`mouse` sentinels exist but aren't wired into emission

All of these are listed as "Planned" in the docs and in the project memory; the goal of
this section is for you to confirm nothing is silently missing.

---

## Suggested review outcome

As you go, it helps to separate three kinds of feedback:

1. **API/behavior** (Level 1) — what the framework should *feel* like; cheapest to change
   now, before more is built on top.
2. **Boundaries/contracts** (Level 2) — the core/backend seam and the cache contract; these
   shape how every future feature and backend plugs in.
3. **Internals** (Level 3) — correctness, coercion, and which limitations to lift first
   (vector uniforms and the dialect-decoupling are the highest-leverage next steps).

The design docs this was built from are in
[`design/`](../design/dsl_caching_layer.md); the approved plan and roadmap are in the project
memory and the plan file.
