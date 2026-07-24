#!/usr/bin/env python3
"""The line-budget gate ("budgets are architecture"; tripwire policy: crossing
any cap means a conscious, recorded decision — never silent growth).

Counts *tokenized* lines — lines carrying at least one token that is not a
comment or a docstring — so documentation is free and code is not. Every file
in pdum.dsl, recursively, must have an explicit cap: a new file without a cap
is an error by design (budgeting is a conscious act, not an audit).

Redrawn at migration P1 (design 200 §7): one package, one bucket. The old
kernel/satellite split died with the purge — pdum.dsl now carries the engine,
the value language, the pipe, the recorder, and the reference oracle under a
single total. pdum.tl joins the budget when it converts onto the core (P3).

Usage:  python scripts/loc_budget.py [--json]
Exit 1 on any breach. Run in CI via packages/dsl/tests/test_budgets.py.
"""

from __future__ import annotations

import ast
import io
import json
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "packages" / "dsl" / "src" / "pdum" / "dsl"

KERNEL_TOTAL_CAP = 2600  # P1 redraw: measured 2253 at the move + headroom for the
# P4-P7 installments (random fields, scope, stores); each future raise is a
# conscious act with the reason recorded here.

# Per-file caps. Engine files carry their pre-move caps (same code, same
# discipline); the P1 arrivals were capped at their measured size + slack.
FILE_CAPS = {
    "__init__.py": 40,  # batteries-included install + the version anchor
    "types.py": 100,
    "valuekind.py": 95,
    "capture.py": 85,
    "api.py": 50,
    "cache.py": 175,
    "ir.py": 150,
    "ops.py": 110,
    "printer.py": 80,
    "rewrite.py": 150,
    "derived.py": 45,
    "lower.py": 170,
    "pack.py": 175,
    "registry.py": 180,  # P1: + the kind vocabulary, the spelled-oracle door (F33)
    "naming.py": 55,  # P3: claim/derive — the naming law's core mechanism seed
    "events.py": 60,  # the seam (emit/span/forbid)
    "recorder.py": 170,  # the observability satellite, now in-package
    "value.py": 330,  # the value language: statements, joins, loops, refusals
    "surfaces.py": 80,  # the five registration surfaces' helpers
    "intrinsics.py": 60,  # scalar intrinsics + DSL batteries
    "pipe.py": 175,  # the fuse pipe: stages, vocabulary checks, build rule
    "reference.py": 140,  # the oracle: renderer + the spelled door
    "render.py": 90,  # the shared dominator-walking emitter
}

_SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}


def _docstring_starts(tree: ast.AST) -> set[tuple[int, int]]:
    """(line, col) of every docstring constant, so its *token* can be skipped
    while other code sharing its lines still counts."""
    starts: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    starts.add((body[0].value.lineno, body[0].value.col_offset))
    return starts


def counted_lines(path: Path) -> int:
    src = path.read_text()
    doc = _docstring_starts(ast.parse(src))  # SyntaxError propagates; report() turns it into a breach
    lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in _SKIP or (tok.type == tokenize.STRING and tok.start in doc):
            continue
        lines.update(range(tok.start[0], tok.end[0] + 1))
    return len(lines)


def report(kernel_dir: Path = KERNEL, caps: dict[str, int] | None = None) -> tuple[dict, list[str]]:
    caps = FILE_CAPS if caps is None else caps
    errors: list[str] = []
    files: dict[str, dict] = {}
    for path in sorted(kernel_dir.rglob("*.py")):
        name = path.relative_to(kernel_dir).as_posix()
        try:
            n = counted_lines(path)
        except SyntaxError as exc:
            errors.append(f"{name}: does not parse ({exc}) — cannot be budgeted")
            continue
        cap = caps.get(name)
        files[name] = {"lines": n, "cap": cap}
        if cap is None:
            errors.append(f"{name}: no cap declared — add it to FILE_CAPS consciously")
        elif n > cap:
            errors.append(f"{name}: {n} counted lines exceeds its cap of {cap}")
    total = sum(f["lines"] for f in files.values())
    if total > KERNEL_TOTAL_CAP:
        errors.append(f"pdum.dsl total {total} exceeds the hard cap of {KERNEL_TOTAL_CAP}")
    return {"kernel_total": total, "kernel_cap": KERNEL_TOTAL_CAP, "files": files}, errors


def main() -> int:
    data, errors = report()
    if "--json" in sys.argv:
        print(json.dumps(data, indent=2))
    else:
        for name, f in data["files"].items():
            cap = f["cap"] if f["cap"] is not None else "MISSING"
            print(f"  {name:<20} {f['lines']:>5} / {cap}")
        print(f"  {'PDUM.DSL TOTAL':<20} {data['kernel_total']:>5} / {data['kernel_cap']}")
    for e in errors:
        print(f"BUDGET BREACH: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
