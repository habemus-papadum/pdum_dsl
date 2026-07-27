# PROVENANCE — inspirations and acknowledgments

PHILOSOPHY.md names the convictions; this file names where they came
from. Roughly: what we took, from whom. Errors of interpretation are ours.

## Program representation & transformation

- **MLIR / linalg bufferization** — the value-semantics-first,
  bufferize-late discipline; the destination-passing lessons we plan to
  meet properly at L2. Our "transform where it is safe; commit late" is
  their hard-won lesson, adopted wholesale.
- **JAX** — AD as program transformation over a pure IR; `scan`/carry
  idioms that shaped `fold`; the general stance that transformations
  compose when programs are data.
- **Halide** (Ragan-Kelley et al.), **Exo**, **TVM** — the
  algorithm/schedule split. Our human-as-compiler workflow with certified
  rewrites is that split with a proof story attached.
- **Classic AD literature** (Griewank & Walther, *Evaluating
  Derivatives*) — the adjoint calculus, the tape-is-the-program view, and
  **revolve**: provably optimal binomial checkpointing, which
  `fold_slots` implements via the C(S+r, S) frontier.
- **Chen et al. 2016** (√n checkpointing), **Checkmate**,
  **Rockmate/Rotor** — the memory/recompute design space our uniform
  segmenting and min-cut planner walk.
- **AOTAutograd's min-cut partitioner** (PyTorch) — the saved-set
  selection as a max-flow problem, which `transforms.checkpoint` adapts
  with exact byte capacities and closed-form-free tensors.
- **Pebbling theory** — Sethi's register pebble game; **Hong & Kung's**
  red-blue pebbling for I/O lower bounds: the one formalism spanning our
  L1 (checkpointing) and the coming L4 (kernel traffic).

## Distribution & placement

- **GSPMD / XLA** (Xu et al.) — the global-view single-program stance and
  per-dim sharding; our `Dim.level` is a sharding annotation living where
  our metadata lives.
- **Megatron-LM** (Shoeybi et al.) — tensor parallelism's column/row
  split and the two-all-reduce block, which the traffic pass re-derives;
  their f/g conjugate operators are why we could *recognize* our
  backward's collectives as correct-but-unfused.
- **PyTorch DTensor**, **Alpa** — named sharding specs; automated
  inter/intra-op parallelism as the search problem we deliberately left
  for later.

## Layouts, kernels, numerics

- **CuTe / CUTLASS** — the closest kin to our layout algebra: nested
  integer strides as composable functions. Their swizzles mark the one
  known extension beyond our affine world (recorded for L6).
- **FlashAttention** (Dao et al.) and **online softmax** (Milakov &
  Gimelshein) — the associative rescaling accumulator that became our
  flagship composite reducer, and the fusion story L4 will measure.
- **Blelloch** — parallel scans; the associativity-buys-parallelism
  license that composite reducers declare and the compiler will spend.
- **Mamba-2 / SSD** (Dao & Gu), **DeltaNet**, and the SSM line — the
  matrix-state recurrences that (together with PDE time-stepping) forced
  `fold` into existence.
- **Yee 1966** — the staggered grid; our exact half-integer charts exist
  so his half-steps could be written down honestly.

## Interfaces & idioms

- **xarray / named tensor notation** (Chiang, Rush, Barak; PyTorch named
  tensors) — names-first dims. We pushed it further (order-free
  everywhere, no permute) but the seed is theirs.
- **pint** — unit-registry ergonomics; ours is exact-rational and
  t-string-based, but the UX debt is real.
- **Unison / Nix / git** — content-addressed definitions; `defmarker`'s
  digest-named registry is that idea in miniature.
- **Sebastian Raschka's LLM architecture gallery** — the survey that
  seeded the zoo's spanning set.
- **Lean 4 / Mathlib** — the eventual home of the proofs, and already the
  discipline: every design note asks "what is the denotation, what is the
  theorem" (LEAN.md) before code believes itself correct.

## And

The numpy reference layer rests, as everything does, on **NumPy**; the
notebooks on **Jupyter**; the exactness doctrine on Python's
`fractions.Fraction` quietly doing heroic work throughout.
