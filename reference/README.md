# The Milestone-0 reference asset

The original end-to-end proof of concept of `pdum.dsl` — a type-keyed JIT
compiling Python closures to WebGPU/WGSL fragment shaders — preserved **frozen
and runnable** while the framework is redesigned from scratch (see
[`docs/desiderata.md`](../docs/desiderata.md)).

**Do not extend or refactor this code.** It exists to be consulted: as a worked
example of the caching thesis, as a source of test cases, and as a baseline the
new implementation can be compared against. New work happens in `src/pdum/dsl/`.

## Layout

| Piece | Location |
|---|---|
| The implementation (importable, installed) | `src/pdum/dsl_reference/` |
| Its test suite (not collected by default) | `reference/tests/` |
| The code-review tour | `reference/REVIEW.md` |
| The window demo | `docs/demos/disk.py` |
| Prose docs (guides, theory, API reference) | `docs/` (mkdocs site) |
| Original design notes | `design/` |

The package was moved verbatim from `src/pdum/dsl/`; only the import name
changed (`pdum.dsl` → `pdum.dsl_reference`). All internal imports are relative,
so the code is untouched apart from the package docstring.

## Running it

```bash
# The M0 test suite (22 tests; needs a GPU/Metal — excluded from `uv run pytest`)
uv run pytest reference/tests -q

# The window demo
uv run python docs/demos/disk.py --frames 120

# Print the WGSL emitted for the inlined higher-order example
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

The main test suite (`uv run pytest`) no longer includes these tests; only the
new implementation's tests under `tests/` are collected by default.
