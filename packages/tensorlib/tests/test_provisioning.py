"""Provisioning (200 §1.7/6.4): the virtual resting state analyzes for
free and refuses execution quoting the fix; init strategies key by name
glob over closed-form fields; safetensors joins on contract names over the
mmap'd file; virtual and provisioned builds share one fingerprint."""

import json
import struct

import numpy as np
import pytest
from pdum.dsl import events
from pdum.tl import Tensor
from pdum.tl.assemblage import assemblage, unit
from pdum.tl.ir import run
from pdum.tl.lifting import contract
from pdum.tl.opcount import ops_count
from pdum.tl.provisioning import init, normal, ones, provision, safetensors, zeros
from pdum.tl.scope import scope


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def make_dense(s, cfg):
    w = s.param("w", d=cfg["d"], m=cfg["m"])
    g = (s / "out").param("g", m=cfg["m"])

    @unit
    def dense(h):
        return contract(h, w) * g.repeat_like(h, but="m", dim="t")

    return dense


CFG = {"d": 3, "m": 2}


def _model(root):
    return assemblage(make_dense(root, CFG), scope=root, h=T(np.zeros((2, 3)), ("t", "d")).layout)


def test_virtual_analyzes_for_free_and_execution_refuses_quoting_the_fix():
    root = scope()
    a = _model(root)
    assert ops_count(a.program, a.layouts).total  # layouts only — never values
    with pytest.raises(KeyError, match=r"virtual leaves analyze for free.*provision\(root, source=init"):
        run(a.program, {"h": T(np.zeros((2, 3)), ("t", "d"))})


def test_init_strategies_match_by_glob_and_regenerate_exactly():
    root = scope()
    _model(root)
    src = init(41, default=normal(std=0.02), overrides={"out.g": ones, "*.nothing": zeros})
    w1 = provision(root, source=src)
    w2 = provision(root, source=src)
    assert sorted(w1) == ["out.g", "w"]
    np.testing.assert_array_equal(w1["w"].to_numpy(), w2["w"].to_numpy())  # same key, same init, forever
    np.testing.assert_array_equal(w1["out.g"].to_numpy(), np.ones(2))
    assert abs(float(w1["w"].to_numpy().std()) - 0.02) < 0.02
    w3 = provision(root, source=init(42, default=normal(std=0.02)))
    assert not np.array_equal(w1["w"].to_numpy(), w3["w"].to_numpy())  # a different key differs


def test_the_virtual_provisioned_cache_dividend():
    """GATE 10: identical fingerprints across provisioning — analyze first,
    provision later, hit warm."""
    root = scope()
    a = _model(root)  # virtual build; analyses ran above
    weights = provision(root, source=init(7, default=normal(std=1.0)))
    with events.forbid("assemblage.miss"):  # provisioning never touches identity
        again = _model(root)
    assert again is a
    env = run(a.program, {"h": T(np.ones((2, 3)), ("t", "d")), **weights})
    want = (np.ones((2, 3)) @ weights["w"].to_numpy()) * weights["out.g"].to_numpy()
    np.testing.assert_allclose(env[a.output].to_numpy(order=("t", "m")), want, rtol=1e-12)


def _write_safetensors(path, arrays):
    header, blobs, off = {}, [], 0
    for name, arr in arrays.items():
        raw = arr.astype("<f8").tobytes()
        header[name] = {"dtype": "F64", "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hdr = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for b in blobs:
            f.write(b)


def test_safetensors_joins_on_names_over_the_mmapped_file(tmp_path):
    root = scope()
    a = _model(root)
    rng = np.random.default_rng(1)
    w, g = rng.standard_normal((3, 2)), rng.standard_normal(2)
    path = str(tmp_path / "ck.st")
    _write_safetensors(path, {"w": w, "out.g": g})
    weights = provision(root, source=safetensors(path))
    np.testing.assert_array_equal(weights["w"].to_numpy(), w)
    env = run(a.program, {"h": T(np.ones((2, 3)), ("t", "d")), **weights})
    np.testing.assert_allclose(env[a.output].to_numpy(order=("t", "m")), (np.ones((2, 3)) @ w) * g)


def test_safetensors_translation_tables_and_refusals(tmp_path):
    root = scope()
    _model(root)
    path = str(tmp_path / "ck.st")
    _write_safetensors(path, {"foreign.w": np.zeros((3, 2)), "out.g": np.zeros(2)})
    weights = provision(root, source=safetensors(path, translate={"w": "foreign.w"}))
    assert weights["w"].to_numpy().shape == (3, 2)
    with pytest.raises(KeyError, match="checkpoint has no entry 'w'"):
        provision(root, source=safetensors(path))
    _write_safetensors(path, {"w": np.zeros((9, 9)), "out.g": np.zeros(2)})
    with pytest.raises(ValueError, match="declares extents .* joins on contract names AND shapes"):
        provision(root, source=safetensors(path))
