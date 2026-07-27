"""The teaching notebooks EXECUTE under the default pytest run — the same
rot-guard CI applies via nbconvert, so a stale notebook fails here first
(skip-execution-tagged cells are honored by nbclient's default)."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NB_DIR = Path(__file__).parents[1] / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("[0-9]*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_executes(path):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()


def test_the_teaching_set_is_present():
    assert len(NOTEBOOKS) >= 13  # 00..13 — deletions are conscious decisions
