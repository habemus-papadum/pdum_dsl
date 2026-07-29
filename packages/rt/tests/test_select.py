"""Selection: pairs are identity-bearing, empty by ruling, and `on` refuses non-pairs."""

import pytest

from pdum.rt import Pair, cuda, metal, on, webgpu


def test_prefab_pairs_are_distinct_identities():
    fps = {webgpu.fp(), metal.fp(), cuda.fp()}
    assert len(fps) == 3  # three columns, three keys — FAIL-5's lesson


def test_pairs_are_frozen_and_comparable():
    assert webgpu == Pair(webgpu.generator, webgpu.runtime)
    with pytest.raises(Exception):
        webgpu.generator = None  # frozen


def test_on_returns_the_pair_and_refuses_non_pairs():
    assert on(metal) is metal
    with pytest.raises(TypeError, match="takes a Pair"):
        on("metal")


def test_explicit_pairing_is_constructible():
    # Separable halves (283 §3): an unusual pairing is an explicit act.
    odd = Pair(metal.generator, webgpu.runtime)
    assert odd.fp() == ("MslGenerator", "WgpuRuntime")
