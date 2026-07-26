# tensorlib

`pdum.tl` — the assemblage tensor language's library layer: the exact
layout algebra (affine map + box + guards + charts + units + placement),
the compute primitives (pointwise/reduce/scan/fold), the Program/Instr
IR with derived adjoints, the transforms (DCE, checkpointing, revolve),
the cost semantics (opcount, peak memory, traffic), and the model zoo
with numpy-pinned denotations.

The LANGUAGE — kernels, steps, assemblages, and the dialect machinery
they lower through — is specified at the workspace level:
`docs/design/200_the-spec.md` is the spec,
`docs/design/220_principles.md` the principles, and
`docs/design/230_syntax-tour.md` the living syntax tour. A user guide
for the compute surface arrives with the P8/P9 API freeze; until then
the syntax tour is the door.

## The documents

- [DESIGN.md](DESIGN.md) — the layout layer's decision record (D1–D18):
  buffers, layouts, views, guards, charts, labels, units.
- [BOUNDARIES.md](BOUNDARIES.md) — the deliberate limits: every known
  coarseness, convention, and refusal, next to the code that has it.
- [PLACEMENT.md](PLACEMENT.md) — machine-bound dims (L3-lite): binding
  lives on `Dim`, collectives are conscious acts, the traffic model.
- [LEVELS.md](LEVELS.md) — the representation ladder × the machine
  tree, down to lanes; [REPRESENTATIONS.md](REPRESENTATIONS.md) holds
  the detailed memory-level notes it delegates to.
- [PHILOSOPHY.md](PHILOSOPHY.md) — the convictions;
  [PROVENANCE.md](PROVENANCE.md) — where they came from.
- [LEAN.md](LEAN.md) — the Lean 4 modeling diary (dated sketches).

## The notebooks

Executable teaching notebooks over the eager layer (`../notebooks`):

- **00–06, the core semantics**: units and quantities; buffers, layouts,
  tensors; view ops; restructuring ops; guarded layouts; the autodiff
  cheatsheet; adjoints from scratch.
- **07–13, the derived machinery and cost models**: the marker DSL; fold
  and tensor state; peak memory; memory transformations; binomial
  revolve; placement.

Re-run them with

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  packages/tensorlib/notebooks/*.ipynb
```

## Running the tests

```bash
uv run pytest packages/tensorlib/tests -q
```

(`bfloat16` support activates when `ml_dtypes` is installed; one test
skips without it.)
