"""The equivalence harness over the flagships (320 §6): each certificate
tier exercised by the flagship that earns it."""

import numpy as np
import pytest

from pdum.tl.certify import certify, normalize
from pdum.tl.licenses import FLASH_ONLINE_SOFTMAX, GEMM_F16_TILES
from pdum.tl.tensor import Tensor
from pdum.tl.transforms import erase_stages
from pdum.tl.zoo.tiles import flash_tile, gemm_tile, stencil_tile

_REASSOC = tuple(lic for lic in GEMM_F16_TILES if lic.kind == "reassociation")


def _t(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def test_stencil_certifies_exact():
    """No licensed deviation exists, and none is needed: exact rewrites
    alone (stage erasure) reach the twin's content key — the region hash
    IS the proof, zero numerics."""
    f = stencil_tile()
    cert = certify(f.region, f.naive)
    assert cert.verdict == "proved-exact"
    assert cert.key_reached and cert.licenses == ()


def test_gemm_certifies_by_licensed_chain():
    """The crown case: fold-of-accumulation inlines to a reduce, splits
    commute out of the pointwise forest, the reduce over (ko, ki)
    collapses to k — and the chain REACHES the naive key. The certificate
    names the reassociation license it consumed; no numerics ran."""
    f = gemm_tile()
    cert = certify(f.region, f.naive, licenses=_REASSOC)
    assert cert.verdict == "proved-licensed"
    assert cert.key_reached
    assert cert.licenses == ("gemm.k-reassoc",)
    # the same fact, stated on the raw machinery
    norm_t, used = normalize(erase_stages(f.region), frozenset({"reassociation"}))
    norm_n, _ = normalize(f.naive, frozenset({"reassociation"}))
    assert norm_t.key == norm_n.key and used == {"reassociation"}


def test_gemm_without_a_license_refuses():
    """The teeth: the same deviation with no declared license is not
    certified — normalization is not allowed to consume what nobody
    licensed, and the differential tier has nothing to quote."""
    f = gemm_tile()
    with pytest.raises(ValueError, match="unlicensed deviation"):
        certify(f.region, f.naive)


def _flash_families(T=6, E=3, OD=2):
    """The adversarial families for flash (260's law: never random draws
    alone), all inside the declared license domain — finite masked scores,
    causal masking guaranteeing the diagonal."""

    def draw(scale_q=1.0, scale_k=1.0, seed=0, dominant=False):
        def factory():
            rng = np.random.default_rng(seed)
            q = scale_q * rng.standard_normal((T, E))
            k = scale_k * rng.standard_normal((T, E))
            if dominant:  # one huge key per row: softmax collapses one-hot,
                k[0] *= 40.0  # the running-max rescale at its extreme
            v = rng.standard_normal((T, OD))
            return {"q": _t(q, ("t", "e")), "k": _t(k, ("s", "e")), "v": _t(v, ("s", "o"))}

        return factory

    return (
        ("gaussian", draw(seed=1)),
        ("wide-scores", draw(scale_q=8.0, scale_k=8.0, seed=2)),  # exp near under/overflow, max-shifted
        ("dominant-key", draw(seed=3, dominant=True)),
    )


def test_flash_certifies_by_licensed_differential():
    """Normalization stops short by design — the online-softmax lemma is
    algebra, not syntax — so the declared license gates the adversarial
    differential and the certificate records the families that ran."""
    f = flash_tile()
    cert = certify(f.region, f.naive, licenses=FLASH_ONLINE_SOFTMAX, families=_flash_families())
    assert cert.verdict == "licensed-differential"
    assert cert.licenses == ("flash.online-softmax",)
    assert cert.families == ("gaussian", "wide-scores", "dominant-key")
    assert not cert.key_reached


def test_differential_tier_demands_families():
    """A licensed differential with no adversarial families is not evidence
    (260: random draws alone never gate a flagship — and NO draws even
    less so)."""
    f = flash_tile()
    with pytest.raises(ValueError, match="adversarial families"):
        certify(f.region, f.naive, licenses=FLASH_ONLINE_SOFTMAX, families=())
