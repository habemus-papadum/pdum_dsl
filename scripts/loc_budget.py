#!/usr/bin/env python3
"""The line-budget gate (architecture §6: "budgets are architecture").

Counts *tokenized* lines — lines carrying at least one token that is not a
comment or a docstring — so documentation is free and code is not (including
code that shares a physical line with a docstring). Every kernel file,
recursively, must have an explicit cap here: a new kernel file without a cap
is an error by design (budgeting is a conscious act, not an audit).

Usage:  python scripts/loc_budget.py [--json]
Exit 1 on any breach. Run in CI via tests/test_budgets.py.
"""

from __future__ import annotations

import ast
import io
import json
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "src" / "pdum" / "dsl" / "kernel"

KERNEL_TOTAL_CAP = 1500  # tripwire, not a wall (010 §6 policy, 2026-07-13): crossing any cap
# means a LEDGER ENTRY stating what was bought — never silent growth. Raised 1150→1500 when
# three lines of headroom started producing monkeypatches instead of seams (design 120).

# Per-file caps (paths relative to kernel/). types.py runs ~25% over its §5
# estimate (65) after the LiteralType identity fix — noted, consciously.
FILE_CAPS = {
    "__init__.py": 10,
    "types.py": 100,
    "valuekind.py": 95,
    "capture.py": 85,
    "api.py": 50,
    "cache.py": 175,  # §5 estimate 105 + retirement/explain + probe() + the 120 event hooks, consciously
    "ir.py": 150,
    "ops.py": 110,
    "printer.py": 80,
    "rewrite.py": 150,
    "lower.py": 170,  # the fused driver; rule PACKS live in the satellite bucket (V1 calibration)
    # §5 estimated 80 for the input half alone. The real file also carries the
    # OUTPUT half (ResultPlan/unflatten — 040 §3b made marshaling bidirectional
    # from the start), the compiled per-slot extractor (§4.3.10), and the two
    # ABI stages. Raised consciously at the step-7 review, not by drift.
    "pack.py": 175,
    "registry.py": 125,  # surface E v1 + the traced-dispatch twin and miss-path spans (design 120)
    "events.py": 55,  # the 120 seam: emit/span/forbid; measured 41, headroom for sink-protocol growth
}

SATELLITE_CAPS = {  # separately-counted buckets (the honesty clause): src/pdum/dsl/<name>
    "combinators.py": 250,
    "stdlib": 1500,
    "viz.py": 450,
    "bench.py": 350,  # step 10b: adaptive sampling, phase instrument, timelines (shrinks with 120 step 6)
    "backends": 500,  # _emit infra + the C target (step 11 pulled the first citizen forward); step 14 raises again
    "demo": 600,  # the fused ch09/ch10 simple-shader pair (080: special-cased OUT of backends/)
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
        errors.append(f"kernel total {total} exceeds the hard cap of {KERNEL_TOTAL_CAP}")
    sats = {}
    base = KERNEL.parent  # satellites live beside the real kernel, not beside a test's tmp dir
    for name, cap in SATELLITE_CAPS.items():
        path = base / name
        if not path.exists():  # a renamed/typo'd satellite must not report a silent 0/cap pass
            errors.append(f"satellite {name}: declared in SATELLITE_CAPS but not found at {path}")
            continue
        targets = [path] if path.is_file() else sorted(path.rglob("*.py"))
        try:
            n = sum(counted_lines(f) for f in targets)
        except SyntaxError as exc:  # same fate as an unparseable kernel file: a breach, not a crash
            errors.append(f"satellite {name}: does not parse ({exc}) — cannot be budgeted")
            continue
        sats[name] = {"lines": n, "cap": cap}
        if n > cap:
            errors.append(f"satellite {name}: {n} counted lines exceeds its cap of {cap}")
    uncapped = sorted(  # a new satellite module must be budgeted consciously, like a kernel file
        p.name for p in base.glob("*.py") if p.name not in SATELLITE_CAPS and p.name != "__init__.py"
    )
    errors += [f"satellite {n}: no cap declared — add it to SATELLITE_CAPS consciously" for n in uncapped]
    return {"kernel_total": total, "kernel_cap": KERNEL_TOTAL_CAP, "files": files, "satellites": sats}, errors


def main() -> int:
    data, errors = report()
    if "--json" in sys.argv:
        print(json.dumps(data, indent=2))
    else:
        for name, f in data["files"].items():
            cap = f["cap"] if f["cap"] is not None else "MISSING"
            print(f"  {name:<24} {f['lines']:>5} / {cap}")
        print(f"  {'KERNEL TOTAL':<24} {data['kernel_total']:>5} / {data['kernel_cap']}")
        for name, f in data.get("satellites", {}).items():
            print(f"  sat:{name:<20} {f['lines']:>5} / {f['cap']}")
    for e in errors:
        print(f"BUDGET BREACH: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
