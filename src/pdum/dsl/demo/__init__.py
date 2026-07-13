"""Demo satellites: pedagogical implementations, deliberately special-cased.

``demo.simple_shader`` is the vertical-slice pair from chapters 9–10 — the
Python reference twin and the WGSL compute/fragment programs. They are
FUSED family+target implementations (the family contract and the target
runtime live in one module each), which is exactly why they live here and
not in ``pdum.dsl.backends``: backends/ is the contribution point for real
target packages (`080_backend-organization.md`), and its citizens are
expected to be target packages serving families through thin cells. The
demos predate that split on purpose — they are the book's teaching
artifacts, frozen to their chapters' shape.

Importing this package wires the demo backends into DEFAULT (batteries —
the stdlib's intrinsic spellings need the demo backends to exist first).
``demo.graphics`` (the toy Color record + 2D helpers) is NOT auto-imported:
consuming it is one explicit import, which is the lesson — domain vocabulary
arrives like any ecosystem package would (090's stdlib-minimalism policy).
"""

from . import simple_shader  # noqa: F401
