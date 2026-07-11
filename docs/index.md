# pdum.dsl

A Python **DSL compiler framework**: a numba-like `@jit` decorator workflow
with **Julia-like type-keyed caching**. A closure is *(code identity, typed
environment, environment values)*; compiled artifacts are cached on the
**types** of captures and arguments — never their values — so a tight Python
loop that rebuilds closures with fresh parameter values every iteration never
recompiles. First proven use case: Python functions compiled to WebGPU/WGSL
fragment shaders whose captured values become uniforms.

## Project status

The project is in a **ground-up redesign**. The proof of concept (Milestone 0)
is complete, frozen, and preserved; the redesigned kernel is being built
step by step, with each step ending in an executable book chapter.

| What | Where |
|---|---|
| The redesign brief (wants, influences, open questions) | [Desiderata](desiderata.md) |
| The synthesized architecture (primitives, hooks, structure) | [`design/proposed-architecture.md`](https://github.com/habemus-papadum/pdum_dsl/blob/main/design/proposed-architecture.md) |
| The step-by-step implementation plan | [`design/implementation-plan.md`](https://github.com/habemus-papadum/pdum_dsl/blob/main/design/implementation-plan.md) |
| The research corpus behind the architecture (R/V/P/J) | [`design/research/`](https://github.com/habemus-papadum/pdum_dsl/tree/main/design/research) |
| The book (bottom-up chapters, one per implementation step) | `docs/book/` — *forthcoming* |
| The frozen M0 proof of concept | [M0 reference asset](m0/index.md) · [`reference/README.md`](https://github.com/habemus-papadum/pdum_dsl/blob/main/reference/README.md) |

!!! note "Reading the M0 section"
    Everything under **M0 reference asset** documents the frozen proof of
    concept (`pdum.dsl_reference`), kept runnable as a worked example and
    baseline. It is historical: it does **not** describe the redesigned
    kernel, whose documentation is the book.
