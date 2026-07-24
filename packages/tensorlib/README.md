# pdum.tl

The assemblage tensor language: the exact layout algebra (affine map + box
+ guards + charts + units + placement), carriers, the compute primitives
(pointwise/reduce/scan/fold and the indexing family), the Program/Instr
IR, reverse-mode AD with derived adjoints, the transforms (DCE,
checkpointing, revolve), the cost semantics (opcount, peak memory,
traffic), placement, signatures, and the model zoo with its numpy-pinned
denotations.

Part of the [pdum_dsl](https://github.com/habemus-papadum/pdum_dsl)
workspace; published in lockstep with `habemus-papadum-dsl`. See
`docs/design/200_the-spec.md` for the specification.

**Status: migration P0 skeleton — contents arrive at P2.**
