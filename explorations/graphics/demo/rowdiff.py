"""Does the render stage port under the same lexical rules as compute?

spike_metal measured 123/123 emitted COMPUTE statements converting
WGSL->MSL under three lexical rules (cast spelling, `f` literal suffix,
C declaration form). This module asks the same question of the render
stage, mechanically: take the generated WGSL, apply the rules as text
substitutions, and compare against the generated MSL line by line.

It is a text experiment on purpose. ``program.py`` already generates both
languages from ONE walker, so of course they agree -- that proves our
factoring is sound, not that the languages are close. Re-deriving one
source from the other with a regex is the independent check: if a row
needs anything a substitution cannot express, it shows up here as a
residue line.

The answer, for this program: the rows still port, but the render stage
adds a FOURTH lexical rule the compute stage never needed, because
compute is scalar-only and rasterization is not:

  R4. VECTOR TYPE SPELLING. ``vec4<f32>`` is ``float4``. The clip-space
      position is the only vector value in the whole program, and it is
      unavoidable -- every raster pipeline has one.

Everything that does NOT port is shell, and the shell differences are
structural rather than lexical: see msl_glue's S1-S4.
"""

from __future__ import annotations

import re

# R1 type and cast spelling (R4 is the vector case of the same rule)
_R1 = [(re.compile(r"\bvec4<f32>"), "float4"), (re.compile(r"\bf32\b"), "float"), (re.compile(r"\bi32\b"), "int")]
# R2 float literals carry an `f` suffix (MSL has no double)
_R2 = re.compile(r"(?<![\w.])(\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])")
# R3 C declaration form
_R3_LET = re.compile(r"^(\s*)let (\w+): (\w+) = (.*);$")
_R3_VAR = re.compile(r"^(\s*)var (\w+): (\w+);$")


def to_msl(line: str) -> tuple[str, set[str]]:
    """Apply the rules; report which ones fired."""
    used: set[str] = set()
    s = line
    m = _R3_LET.match(s)
    if m:
        s, used = f"{m[1]}{m[3]} {m[2]} = {m[4]};", used | {"R3"}
    else:
        m = _R3_VAR.match(s)
        if m:
            s, used = f"{m[1]}{m[3]} {m[2]};", used | {"R3"}
    for pat, rep in _R1:
        if pat.search(s):
            used.add("R4" if "vec4" in pat.pattern else "R1")
            s = pat.sub(rep, s)
    if _R2.search(s):
        used.add("R2")
        s = _R2.sub(r"\1f", s)
    return s, used


def rows(source: str) -> list[str]:
    """The emitted STATEMENTS: entry-point body lines, which both shells
    indent by two spaces. Entry-point parameter lines are indented four,
    so they land in the shell where they belong -- and so, explicitly, do
    the members of the varying struct, which LOOK like statements and are
    not: they are the vertex-to-fragment interface declaration, and the
    two languages carry its information differently (S2). Leaving them in
    was the first version of this measurement and it reported a 6-line
    residue that was really one structural fact."""
    out, in_struct = [], False
    for ln in source.splitlines():
        s = ln.strip()
        if s.startswith("struct "):
            in_struct = True
            continue
        if in_struct:
            in_struct = not (s == "}" or s == "};")
            continue
        if ln.startswith("  ") and not ln.startswith("   "):
            out.append(ln)
    return out


def report(name: str, wgsl: str, msl: str) -> dict:
    w, m = rows(wgsl), rows(msl)
    n = min(len(w), len(m))
    ok, residue, fired = 0, [], {}
    for a, b in zip(w[:n], m[:n]):
        conv, used = to_msl(a)
        if conv == b:
            ok += 1
            for r in used:
                fired[r] = fired.get(r, 0) + 1
        else:
            residue.append((a, conv, b))
    shell_w = len(wgsl.splitlines()) - len(w)
    shell_m = len(msl.splitlines()) - len(m)
    return {
        "stage": name,
        "rows": len(w),
        "rows_msl": len(m),
        "converted": ok,
        "residue": residue,
        "rules": fired,
        "shell_wgsl": shell_w,
        "shell_msl": shell_m,
    }


def print_report(reports) -> None:
    print("\nWGSL -> MSL by lexical rule alone (R1 type/cast, R2 literal suffix,")
    print("R3 C declaration form, R4 vector type spelling):")
    tot_rows = tot_ok = 0
    for r in reports:
        tot_rows += r["rows"]
        tot_ok += r["converted"]
        pct = 100.0 * r["converted"] / max(r["rows"], 1)
        print(
            f"  {r['stage']:20s} rows {r['converted']:3d}/{r['rows']:<3d} ({pct:5.1f}%)   "
            f"rules fired {dict(sorted(r['rules'].items()))}   shell lines wgsl/msl "
            f"{r['shell_wgsl']}/{r['shell_msl']}"
        )
        for a, conv, b in r["residue"]:
            print(f"      RESIDUE wgsl:   {a.strip()}")
            print(f"              ->msl:  {conv.strip()}")
            print(f"              actual: {b.strip()}")
    print(f"  {'TOTAL':20s} rows {tot_ok}/{tot_rows} ({100.0 * tot_ok / max(tot_rows, 1):.1f}%)")
