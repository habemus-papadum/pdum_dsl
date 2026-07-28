"""Put THIS worktree's packages (and conformance/) on sys.path.

The repo venv lives in the main checkout and has no editable install of
pdum, so a spike importing `pdum.tl` gets nothing unless the worktree's
source dirs go on the path first. Import this module before any pdum
import.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]

for _p in ("packages/dsl/src", "packages/tensorlib/src", "conformance"):
    _s = str(ROOT / _p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
