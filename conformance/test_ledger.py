"""The coverage ledger's drift tests (290 §2, owner-ratified).

The ledger (pdum/tl/ledger.toml) claims a partition: every op in the
compute tier's vocabulary is either translated by the device executor
or carries a ledger row — and a ledgered op that has GAINED a device
row fails here too, so coverage progress is recorded, never silent.
Facet rows (finer than an op) each name a guard test; the guards live
in this file, and a row whose guard does not exist is itself a failure.
"""

import tomllib
from pathlib import Path

import numpy as np
import pytest
import wgsl_executor as wx
from pdum.tl import Tensor, compute, f32, global_idx, i32, thread_idx  # noqa: F401 — bodies' globals
from pdum.tl.dialect import _FAMILIES, _TIER_EXTRA, TIERS
from pdum.tl.kernel import _compile
from pdum.tl.zoo.zoo_common import gelu  # noqa: F401 — a composite marker: no device row
from wgsl_executor import Untranslatable, _translate

from pdum.dsl import events, jit

LEDGER = tomllib.loads((Path(wx.__file__).parent.parent / "packages/tensorlib/src/pdum/tl/ledger.toml").read_text())

# the device row set, from the executor's own tables + its structural handling
TRANSLATED = (
    set(wx._CORE_INFIX)
    | {"core.neg", "core.cmp", "core.select", "core.param", "core.const", "core.yield"}
    | {"tl.const", "tl.iota", "tl.pointwise", "tl.read", "tl.store", "tl.token", "abi.slot"}
)


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def test_ledger_partitions_the_compute_vocabulary():
    """Both drift directions: no vocabulary op without a home; no stale row."""
    ledgered = {r["op"] for r in LEDGER["ops"] if r["tier"] == "compute"}
    vocab = set().union(*(_FAMILIES[f] for f in TIERS["compute"])) | set(_TIER_EXTRA["compute"])
    stale = TRANSLATED & ledgered
    assert not stale, f"stale ledger rows — these now translate, remove them: {sorted(stale)}"
    missing = vocab - TRANSLATED - ledgered
    assert not missing, f"vocabulary ops with neither a device row nor a ledger row: {sorted(missing)}"
    stray = ledgered - vocab
    assert not stray, f"ledger rows outside the compute vocabulary: {sorted(stray)}"


def test_ledger_facet_guards_exist():
    text = Path(__file__).read_text()
    for f in LEDGER["facets"]:
        assert f"def {f['guard']}(" in text, f"facet {f['tier']}/{f['key']} names a missing guard: {f['guard']}"
    assert isinstance(LEDGER["version"], int)


# --- the guards ---------------------------------------------------------------


def test_rank3_lattice_is_ledgered():
    """compute/lattice-rank-3: legal at the tier, refused by the translator."""

    def k(vol):
        z, y, x = thread_idx("z", "y", "x")
        vol[z, y, x] = f32(z) + f32(y) + f32(x)

    art = _compile(k, (T(np.zeros((2, 3, 4)), ("z", "y", "x")),))
    art.launch((T(np.zeros((2, 3, 4)), ("z", "y", "x")),))  # the reference runs it
    with pytest.raises(Untranslatable, match="rank-3"):
        _translate(art)


def test_multi_block_geometry_is_ledgered():
    """compute/multi-block-geometry: the tl.split/tl.merge machinery
    emissions (the grid bracket) refuse at the translator — 211 §2b's
    coverage cliff, kept visible."""

    def k(img):
        i, j = global_idx("y", "x")
        img[i, j] = f32(i) + f32(j)

    art = _compile(k, (T(np.zeros((4, 4)), ("y", "x")),), (), ((2, 2), (2, 2)))
    ops = {n.op for r in [art.region] for n in wx.walk_region(r)}
    assert {"tl.split", "tl.merge"} & ops, "the grid bracket should emit split/merge"
    with pytest.raises(Untranslatable):
        _translate(art)


def test_oracle_fallback_is_loud():
    """compute/oracle-fn-arg: an unliftable fn-arg body (bounded control
    flow) drops to per-element oracle dispatch — and SAYS so (211 §1.4)."""

    @jit()
    def loopy(a):
        acc = a
        for _ in range(3):
            acc = acc * 0.5
        return acc

    def k(f, img):
        y, x = thread_idx("y", "x")
        img[y, x] = f(f32(y) + f32(x))

    heard = []
    events.SINKS.append(lambda name, key, dur, depth, detail: heard.append(name))
    try:
        img = T(np.zeros((2, 3)), ("y", "x"))
        art = _compile(k, (loopy, img))
        art.launch((loopy, img))
        np.testing.assert_allclose(img.to_numpy(), np.add.outer(np.arange(2.0), np.arange(3.0)) * 0.125)
    finally:
        events.SINKS.pop()
    assert "kernel.oracle_fallback" in heard, "the oracle drop must be loud"


def test_marker_beyond_rows_is_ledgered():
    """compute/scalar-markers-beyond-rows: a composite marker lowers and
    runs on the reference; the translator refuses NAMING it."""

    def k(img):
        y, x = thread_idx("y", "x")
        img[y, x] = gelu(f32(y) * 0.25 - f32(x) * 0.125)

    img = T(np.zeros((2, 3)), ("y", "x"))
    art = _compile(k, (img,))
    art.launch((img,))
    with pytest.raises(Untranslatable, match="gelu"):
        _translate(art)


def test_i64_narrowing_is_pinned():
    """compute/i64-narrowing: int consts render as float literals (site 3
    of three — buffers and uniform slots are the runtime twins), and f32
    cannot carry ints past 2**24. Pinned so the declared-narrowing fix
    flips THIS test instead of landing silently."""
    big = 2**24 + 1
    assert wx._lit_of(big) == f"{float(big)!r}"  # an int const becomes a float literal
    assert int(np.float32(big)) != big  # and f32 cannot hold it


def test_sample_lod_refusal_is_pinned():
    """fragment/sample-lod: lod=0 only today, refused with the recorded
    wording at fragment lowering."""
    pytest.importorskip("wgpu")
    from pdum.tl.graphics import _device, fragment, pair, position, render, sample, sampler, upload, vertex

    try:
        _device()
    except Exception as exc:
        pytest.skip(f"no WebGPU device available: {exc}")
    tex = upload(T(np.zeros((4, 4)), ("y", "x")))
    smp = sampler(filter="nearest", address="clamp")

    @vertex
    def quad():
        (vid,) = thread_idx("vertex_id")
        i = i32(vid)
        u = 1.0 if (i == 1 or i == 3 or i == 4) else 0.0
        v = 1.0 if (i == 2 or i == 4 or i == 5) else 0.0
        return position(u * 2.0 - 1.0, v * 2.0 - 1.0)

    @fragment
    def shade(varying):
        return sample(tex, smp, (varying.v, varying.u), lod=1)

    with pytest.raises(ValueError, match="lod=0 only"):
        render(pair(quad, shade), target=T(np.zeros((4, 4)), ("y", "x")))
