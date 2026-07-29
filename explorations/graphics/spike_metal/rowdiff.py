"""WGSL vs MSL, measured two ways -- the duplication evidence.

The lead asked for "exactly which rows differed between WGSL and MSL and
which were identical". Guessing from a read-through is worthless, so
this module answers it mechanically, and answers it twice.

MEASUREMENT A -- the emitted code. For every subject, translate the SAME
artifact through both backends, then try to turn the WGSL body into the
MSL body using only these three lexical rules:

    1. `f32(` -> `float(`,  `i32(` -> `int(`
    2. float literals take an `f` suffix
    3. `let v: f32 = e;` -> `float v = e;` (C declaration, not a binding)

Every emitted statement that survives that transformation is a row where
the two languages are the SAME LANGUAGE for our purposes. The count of
survivors vs total is the headline. This is a falsifiable claim, not a
narrative: if any operator row genuinely differed -- a different builtin
name, a different `select` argument order, a different bool convention
-- the transformation would fail on it and this script would print it.

MEASUREMENT B -- the translator source. `difflib` over
`wgsl_executor._translate` and `msl_backend._translate` (a deliberate
structural clone), reporting how many lines are byte-identical. That is
the cost of the third copy, in lines, as a maintenance number.

The two measurements answer different questions and both belong in the
report: A says how much of the TARGET LANGUAGE is shared, B says how
much of the TRANSLATOR is duplicated. A is a fact about WGSL and MSL; B
is a fact about our factoring, and only B is ours to fix.
"""

from __future__ import annotations

import difflib
import inspect
import re

import _paths  # noqa: F401
from subjects import SUBJECTS, artifact

_FLOAT = re.compile(r"\d*\.\d+(?:[eE][+-]?\d+)?|\d+\.\d*(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+")
_DECL = re.compile(r"^(\s*)let (v\d+): (f32|bool) = (.*);$")


def wgsl_row_to_msl(line: str) -> str:
    """Apply ONLY the three declared dialect rules. Anything else that
    differs will show up as a mismatch, which is the point."""
    m = _DECL.match(line)
    if m:
        indent, var, ty, rhs = m.groups()
        line = f"{indent}{'float' if ty == 'f32' else 'bool'} {var} = {rhs};"
    line = line.replace("f32(", "float(").replace("i32(", "int(")
    return _FLOAT.sub(lambda mo: mo.group(0) + "f", line)


def _body(src: str, lang: str) -> list[str]:
    """The statement lines -- everything after the entry-point brace, minus
    the closing brace. The preamble (bindings/entry signature) is compared
    separately: it is structural, not row-level."""
    lines = src.splitlines()
    if lang == "wgsl":
        i = next(k for k, ln in enumerate(lines) if ln.startswith("fn main("))
    else:
        i = next(k for k, ln in enumerate(lines) if ln == "{")
    return [ln for ln in lines[i + 1 : -1] if ln.strip()]


def measure_a():
    from msl_backend import _translate as msl_translate

    from wgsl_executor import Untranslatable as WU
    from wgsl_executor import _translate as wgsl_translate

    total = same = 0
    mismatches, per_subject = [], []
    for name, (kernel, mk) in SUBJECTS.items():
        art = artifact(kernel, mk())
        try:
            w_src, _ = wgsl_translate(art)
        except WU as exc:
            per_subject.append((name, None, None, str(exc)))
            continue
        m_src, _ = msl_translate(art)
        wb, mb = _body(w_src, "wgsl"), _body(m_src, "msl")
        if len(wb) != len(mb):
            per_subject.append((name, len(wb), None, f"row COUNT differs: {len(wb)} vs {len(mb)}"))
            continue
        n_same = 0
        for w, m in zip(wb, mb):
            total += 1
            if wgsl_row_to_msl(w) == m:
                same += 1
                n_same += 1
            else:
                mismatches.append((name, w, m, wgsl_row_to_msl(w)))
        per_subject.append((name, len(wb), n_same, None))
    return total, same, mismatches, per_subject


def _code_only(lines: list[str]) -> list[str]:
    """Strip blanks, comment lines and the docstring, so the duplication
    number is about CODE and not about this spike's longer prose."""
    out, in_doc = [], False
    for ln in lines:
        s = ln.strip()
        if in_doc:
            if s.endswith('"""'):
                in_doc = False
            continue
        if s.startswith('"""'):
            in_doc = not (s.endswith('"""') and len(s) > 3)
            continue
        if not s or s.startswith("#"):
            continue
        out.append(ln.split("  # ")[0].rstrip())  # drop trailing comments
    return out


def measure_b():
    from msl_backend import _translate as msl_translate

    from wgsl_executor import _translate as wgsl_translate

    w = inspect.getsource(wgsl_translate).splitlines()
    m = inspect.getsource(msl_translate).splitlines()
    wc, mc = _code_only(w), _code_only(m)
    sm = difflib.SequenceMatcher(None, w, m, autojunk=False)
    smc = difflib.SequenceMatcher(None, wc, mc, autojunk=False)
    return {
        "raw": (len(w), len(m), sum(b.size for b in sm.get_matching_blocks()), sm.ratio()),
        "code": (len(wc), len(mc), sum(b.size for b in smc.get_matching_blocks()), smc.ratio()),
    }


def preamble_diff():
    """The structural (non-row) difference, shown once on one subject."""
    from msl_backend import _translate as msl_translate

    from wgsl_executor import _translate as wgsl_translate

    art = artifact(*(SUBJECTS["uniform"][0], SUBJECTS["uniform"][1]()))
    w, _ = wgsl_translate(art)
    m, _ = msl_translate(art)
    wl, ml = w.splitlines(), m.splitlines()
    wi = next(k for k, ln in enumerate(wl) if ln.startswith("fn main("))
    mi = next(k for k, ln in enumerate(ml) if ln == "{")
    return wl[: wi + 1], ml[: mi + 1]


def main():
    total, same, mism, per = measure_a()
    print("=== MEASUREMENT A: emitted rows, WGSL -> MSL under 3 lexical rules ===")
    print(f"{'subject':<14}{'rows':>6}{'identical':>11}")
    for name, n, s, err in per:
        if err:
            print(f"{name:<14}{'--':>6}{'--':>11}   {err}")
        else:
            print(f"{name:<14}{n:>6}{s:>11}")
    pct = 100.0 * same / total if total else 0.0
    print(f"\nTOTAL: {same}/{total} emitted statements identical under the 3 rules ({pct:.1f}%)")
    if mism:
        print(f"\n{len(mism)} row(s) NOT explained by the 3 rules:")
        for name, w, m, got in mism[:20]:
            print(f"  [{name}]\n    wgsl : {w.strip()}\n    msl  : {m.strip()}\n    xform: {got.strip()}")
    else:
        print("NO operator row differs beyond the 3 lexical rules.")

    print("\n=== the structural (preamble) difference, on subject 'uniform' ===")
    wp, mp = preamble_diff()
    print("--- WGSL ---")
    for ln in wp:
        print("  " + ln)
    print("--- MSL ---")
    for ln in mp:
        print("  " + ln)

    b = measure_b()
    print("\n=== MEASUREMENT B: translator SOURCE duplication ===")
    for label, (nw, nm, ident, ratio) in b.items():
        tag = "with comments" if label == "raw" else "CODE ONLY (no comments/docstring)"
        print(f"  {tag}:")
        print(f"    wgsl_executor._translate {nw} lines | msl_backend._translate {nm} lines")
        print(f"    byte-identical {ident} | changed {nm - ident} | difflib ratio {ratio:.3f}")


if __name__ == "__main__":
    main()
