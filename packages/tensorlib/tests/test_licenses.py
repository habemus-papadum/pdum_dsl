"""The license schema stub (200 §4/§8): declarations, closed taxonomy."""

import pytest

from pdum.tl.licenses import GEMM_F16_TILES, KINDS, Descent, License


def test_the_taxonomy_is_closed():
    assert KINDS == ("none", "reassociation", "precision-demotion")
    with pytest.raises(ValueError, match="taxonomy is closed"):
        License("x", "fastmath", "no", 0, 0, "anywhere")


def test_the_worked_gemm_declaration():
    """The spec's worked check (200 §4): f16 tiles + f32 accumulators —
    reassociation implied by tiling, demotion stated with tolerance and
    domain; the license set joins the registry key SORTED."""
    kinds = {lic.kind for lic in GEMM_F16_TILES}
    assert kinds == {"reassociation", "precision-demotion"}
    for lic in GEMM_F16_TILES:
        assert lic.equivalence and lic.input_domain  # claims are stated, never blank
    d = Descent("tiled_matmul@toy", GEMM_F16_TILES)
    assert d.key() == ("tiled_matmul@toy", ("gemm.f16tile.f32acc", "gemm.k-reassoc"))
