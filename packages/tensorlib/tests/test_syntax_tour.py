"""The syntax tour is EXECUTABLE (owner-ruled): the tour notebook runs
under pytest and in CI, so the tour can never rot. Cells tagged
``skip-execution`` are committed futures or device cells — displayed,
not executed (nbclient's default skip tag)."""

from pathlib import Path

TOUR = Path(__file__).parent.parent / "notebooks" / "20_syntax_tour.ipynb"


def test_the_syntax_tour_executes():
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(TOUR, as_version=4)
    client = NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": TOUR.parent}})
    client.execute()  # skip-execution cells are honored by default
    skipped = [c for c in nb.cells if "skip-execution" in c.metadata.get("tags", ())]
    assert skipped, "the tour carries committed-future cells; losing them all is a regression"
