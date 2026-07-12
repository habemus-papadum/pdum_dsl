"""The pdum.dsl kernel: twelve frozen data structures and two engines.

Everything in this package is governed by ``docs/design/010_proposed-architecture.md``
and the CI line budget (``scripts/loc_budget.py``). Each module is introduced
by a chapter of the book (``docs/book/``); read the chapters to understand not
just the API but the reasoning.

Modules land step by step (``docs/design/020_implementation-plan.md``):

- ``types``     — the structural type lattice + template identity   (step 1)
- ``valuekind`` — value -> Type summaries and fingerprints           (step 1, completed step 7)
- ``capture``   — phase A: make_handle, Handle, SourceSnapshot       (step 2)
- ``api``       — @jit (capture only until the step-8 call path)     (step 2, completed step 8)
- ``cache``     — the two-tier cache: specialization + artifact       (step 3)
- ``ir``        — Node/Region, content key, Builder, verify            (step 4)
- ``ops``       — OpDef + the core dialect table                       (step 4)
- ``printer``   — the MLIR-flavored textual form                       (step 4)
- ``rewrite``   — Pat/RuleSet, the one driver, Stage legality          (step 5)
- ``lower``     — the fused typing+lowering driver; fates; inlining    (step 6)
- ``pack``      — leaves, plans, the generic packer, the ABI stages    (step 7)
"""

from . import pack  # noqa: F401  — registers the marshaling aspects on BUILTINS

# Importing the package registers them, so ANY entry point (`from
# pdum.dsl.kernel.valuekind import BUILTINS` included) gets a table that can
# plan. Without this, `extend()` — which snapshots the aspect registry — could
# mint a child table that is permanently unable to marshal.
