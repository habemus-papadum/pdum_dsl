"""Architecture §6: the line budget is a CI gate, not an audit."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("loc_budget", ROOT / "scripts" / "loc_budget.py")
loc_budget = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loc_budget)


def test_kernel_line_budget():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "loc_budget.py"), "--json"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"budget breach:\n{proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["kernel_total"] <= data["kernel_cap"]


def test_gate_sees_subdirectories(tmp_path):
    # A kernel file hidden in a subpackage must trip the no-cap error, not escape.
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "sneaky.py").write_text("x = 1\n")
    _, errors = loc_budget.report(kernel_dir=tmp_path)
    assert any("sub/sneaky.py" in e and "no cap" in e for e in errors)


def test_code_sharing_a_docstring_line_still_counts(tmp_path):
    f = tmp_path / "f.py"
    f.write_text('def f():\n    """doc"""; x = 1\n    return x\n')
    assert loc_budget.counted_lines(f) == 3  # def, the x=1 line, return — doc token free


def test_unparseable_file_is_a_breach_not_a_crash(tmp_path):
    (tmp_path / "bad.py").write_text("def f(:\n")
    _, errors = loc_budget.report(kernel_dir=tmp_path)
    assert any("does not parse" in e for e in errors)
