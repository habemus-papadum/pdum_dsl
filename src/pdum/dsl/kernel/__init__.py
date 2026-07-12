"""The pdum.dsl kernel: twelve frozen data structures and two engines.

Everything in this package is governed by ``design/proposed-architecture.md``
and the CI line budget (``scripts/loc_budget.py``). Each module is introduced
by a chapter of the book (``docs/book/``); read the chapters to understand not
just the API but the reasoning.

Modules land step by step (``design/implementation-plan.md``):

- ``types``     — the structural type lattice + template identity   (step 1)
- ``valuekind`` — value -> Type summaries and fingerprints           (step 1, completed step 7)
- ``capture``   — phase A: make_handle, Handle, SourceSnapshot       (step 2)
- ``api``       — @jit (capture only until the step-8 call path)     (step 2, completed step 8)
- ``cache``     — the two-tier cache: specialization + artifact       (step 3)
- ``ir``        — Node/Region, content key, Builder, verify            (step 4)
- ``ops``       — OpDef + the core dialect table                       (step 4)
- ``printer``   — the MLIR-flavored textual form                       (step 4)
"""
