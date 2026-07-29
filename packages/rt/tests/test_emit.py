"""One walker, two languages — the four lexical rules, mechanically.

The campaign's headline claim (210) is falsifiable, so it is tested as
such: take the statements the ONE generator emits for a real kernel in
each dialect and try to turn the WGSL rows into the MSL rows using only
the declared rules. A fifth difference — a renamed builtin, a swapped
``select`` argument order, a per-dialect branch smuggled into the op
table — fails here and names the row.
"""

import re

import numpy as np
import pytest

from pdum.rt import emit, msl, wgsl
from pdum.tl import Tensor, compute, f32, global_idx, i32  # noqa: F401 — bare in kernel bodies
from pdum.tl.markers import maximum, tanh  # noqa: F401 — bare in kernel bodies

_GAIN = 3.5


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


@compute
def ripple(src, dst):
    i, j = global_idx("y", "x")
    v = src[i, j] * _GAIN + f32(i) * 0.5
    dst[i, j] = tanh(v) + maximum(v, 0.0)


@compute
def diag(tex, img):
    (i,) = global_idx("y")
    img[i] = tex[i32(i), i32(i)] * 2.0


@compute
def ignores_one(unused, dst):
    i, j = global_idx("y", "x")
    dst[i, j] = f32(i) + f32(j)


def _art(kernel, args):
    return kernel.artifact(*args)  # the PUBLIC door (H2, landed)


@pytest.fixture
def art():
    return _art(ripple, (T(np.zeros((3, 4)), ("y", "x")), T(np.zeros((3, 4)), ("y", "x"))))


# --- the four rules, as a transformation ------------------------------------

_FLOAT = re.compile(r"\d*\.\d+(?:[eE][+-]?\d+)?|\d+\.\d*(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+")
_DECL = re.compile(r"^(\s*)let (e\d+): (f32|bool) = (.*);$")


def wgsl_row_to_msl(line: str) -> str:
    """Rules 1-3 applied to one emitted statement, and nothing else.
    (Rule 4, vector spelling, has no compute site — see the test.)"""
    m = _DECL.match(line)
    if m:  # R3: a C declaration, not a `let` binding
        indent, var, ty, rhs = m.groups()
        line = f"{indent}{'float' if ty == 'f32' else 'bool'} {var} = {rhs};"
    line = line.replace("f32(", "float(").replace("i32(", "int(")  # R1: cast spelling
    return _FLOAT.sub(lambda mo: mo.group(0) + "f", line)  # R2: the literal suffix


def _rows(art, d):
    r = emit.compute_rows(art, d)
    return r.lines + r.stores


def test_every_emitted_row_converts_under_the_declared_rules(art):
    w, m = _rows(art, emit.WGSL), _rows(art, emit.MSL)
    assert len(w) == len(m), "the two dialects emitted different ROW COUNTS — that is a fifth difference"
    bad = [(a, b, wgsl_row_to_msl(a)) for a, b in zip(w, m) if wgsl_row_to_msl(a) != b]
    assert not bad, f"rows differ beyond the declared rules: {bad}"
    assert len(w) >= 10, "a subject this small proves little"


def test_rule_four_is_declared_and_has_no_compute_site(art):
    """Vector spelling is the fourth rule and it fires exactly once per
    RENDER program (the clip-space position); compute is scalar-only, so
    it fires zero times here and lives only in the shell's builtin."""
    assert (emit.WGSL.vec(4), emit.MSL.vec(4)) == ("vec4<f32>", "float4")
    assert emit.WGSL.vec(3, "u32") == "vec3<u32>" and emit.MSL.vec(3, "uint") == "uint3"
    assert not [r for r in _rows(art, emit.WGSL) if "vec" in r]


def test_the_tanh_row_applies_in_both_and_is_recorded(art):
    """The target numeric contract at the marker-function site: the row
    substitutes for EVERY dialect (it is free where the library is
    already correct), and the artifact says what it substituted."""
    w_src, w_contract = wgsl.generate(art)
    m_src, m_contract = msl.generate(art)
    assert "tanh(clamp(e6, -20.0, 20.0))" in w_src
    assert "tanh(clamp(e6, -20.0f, 20.0f))" in m_src  # R2 reaches row text like any literal
    assert w_contract.math == ("tanh",) == m_contract.math


def test_the_contract_carries_what_source_cannot(art):
    """Thread size is IN the WGSL text and nowhere in the MSL text; the
    guard is live code on one target and absent on the other. Both facts
    are contract clauses, which is why a generator returns two things."""
    w_src, w = wgsl.generate(art)
    m_src, m = msl.generate(art)
    assert w.thread_size == m.thread_size == (8, 8, 1)  # the 2D policy
    assert "@workgroup_size(8, 8, 1)" in w_src and "8" not in m_src.splitlines()[2]
    assert (w.guard, m.guard) == ("emitted", "exact")
    assert "if (gid.y >= 3u || gid.x >= 4u) { return; }" in w_src
    assert "return;" not in m_src
    assert w.key() == ("threads", (8, 8, 1))


def test_the_binding_table_is_data_and_both_shells_read_it(art):
    w_src, w = wgsl.generate(art)
    m_src, m = msl.generate(art)
    assert len(w.bindings) == len(m.bindings) == 1  # compute is one stage
    assert w.bindings == m.bindings  # ... and one index space, so far
    names = [(b.name, b.index, b.writable) for b in w.bindings[0]]
    assert names == [("buf0", 0, False), ("buf1", 1, True), ("U", 2, False)]
    assert "@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;" in w_src
    assert "    device float* buf1 [[buffer(1)]]," in m_src
    assert "    const device float* buf0 [[buffer(0)]]," in m_src


def test_the_slot_plan_is_the_declared_device_map(art):
    _src, w = wgsl.generate(art)
    assert w.slots == (("_GAIN", 0, "f32"),)  # name, device index, carrier


def test_the_one_dimensional_policy_and_the_computed_index_read():
    art = _art(diag, (T(np.arange(9.0).reshape(3, 3), ("y", "x")), T(np.zeros(3), ("y",))))
    w_src, w = wgsl.generate(art)
    m_src, m = msl.generate(art)
    assert w.thread_size == m.thread_size == (64, 1, 1)
    # the read's own index arithmetic — and the diagonal's two indices are ONE
    # node, so CSE-by-node-id emits e0 twice rather than computing it twice
    assert "buf0[i32(e0) * 3 + i32(e0) * 1]" in w_src
    assert "buf0[int(e0) * 3 + int(e0) * 1]" in m_src
    assert w.slots == () and "U" not in w_src  # no captures: no slot buffer declared


def test_a_parameter_no_row_names_gets_no_binding_slot():
    """WebGPU's ``layout="auto"`` prunes an unreferenced resource from
    the pipeline layout, so declaring one makes the bind group
    unbuildable — the generator anticipates the runtime rule (210), once,
    for every column."""
    art = _art(ignores_one, (T(np.ones((3, 4)), ("y", "x")), T(np.zeros((3, 4)), ("y", "x"))))
    rows = emit.compute_rows(art, emit.WGSL)
    assert rows.bound == (1,)  # param 0 is named by no row
    src, contract = wgsl.generate(art)
    assert [(b.name, b.index, b.writable) for b in contract.bindings[0]] == [("buf0", 0, True)]
    assert src.count("var<storage") == 1
    m_src, m_contract = msl.generate(art)
    assert m_contract.bindings == contract.bindings and m_src.count("[[buffer(") == 1


def test_an_explicit_thread_size_reaches_the_contract_and_the_source(art):
    src, contract = wgsl.generate(art, thread_size=(16, 4, 1))
    assert contract.thread_size == (16, 4, 1) and "@workgroup_size(16, 4, 1)" in src
    _msrc, mcontract = msl.generate(art, thread_size=(16, 4, 1))
    assert mcontract.thread_size == (16, 4, 1)  # SPECIALIZES wherever it lands


def test_an_untranslatable_op_names_itself():
    class _Node:
        op, args, regions, attrs, type = "tl.scatter_add", (), (), (), None

    gen = emit.Gen(emit.WGSL, lambda n, g: None)
    with pytest.raises(emit.Untranslatable, match="tl.scatter_add"):
        gen.expr(_Node())
