"""The naming-law pins (zoo gate 7) and the tied-gradient pin (gate 9).

Contract names are pinned by LITERAL expectation — never by comparison
with dead code. A rebuilt closure maps the same capture to the same name;
the grad map joins on those names; one leaf captured twice receives ONE
summed gradient."""

import numpy as np
from pdum.tl.assemblage import assemblage, unit
from pdum.tl.autodiff import grad, numeric_grad
from pdum.tl.ir import Instr, Program, run
from pdum.tl.scope import scope
from pdum.tl.zoo.gpt2 import GPT2Config, gpt2, make_gpt2

CFG = GPT2Config()


def test_the_naming_law_literal_pins():
    """Gate 7: level-first names, hardcoded. h.{i} from seq, attn/mlp from
    make_block, leaf names from declare-at-use — the checkpoint contract."""
    m = gpt2()
    names = set(m.inputs) - {"x"}
    expected = {"wpe", "lnfg", "lnfb", "wlm"}
    for i in range(CFG.layers):
        expected |= {f"h.{i}.attn.{n}" for n in ("ln1g", "ln1b", "wq", "wk", "wv", "wo")}
        expected |= {f"h.{i}.mlp.{n}" for n in ("ln2g", "ln2b", "w1", "b1", "w2", "b2")}
    assert names == expected
    assert "h.0.attn.wq" in names and "h.1.mlp.w1" in names  # the spec's own examples


def test_a_rebuilt_closure_maps_the_same_capture_to_the_same_name():
    root1, root2 = scope(), scope()
    from pdum.tl.ir import _dense_like
    from pdum.tl.layout import Dim

    lay = _dense_like((Dim("t", 0, 0, CFG.t), Dim("d", 0, 0, CFG.d)))
    a1 = assemblage(make_gpt2(root1, CFG), scope=root1, x=lay)
    a2 = assemblage(make_gpt2(root2, CFG), scope=root2, x=lay)
    assert list(a1.params) == list(a2.params)  # same captures, same names, same order
    assert a1 is a2  # and in fact the same cached build (identical identity)


def test_grad_map_joins_on_contract_names():
    m = gpt2()
    prog = Program(m.program.instrs + (Instr("zloss", "reduce", (m.out,), {"f": "sum", "dims": ("t", "v")}),))
    _, grads = grad(prog, "zloss", m.inputs)
    assert "h.0.attn.wq" in grads and grads["h.0.attn.wq"] is not None
    assert "h.1.mlp.b2" in grads  # every leaf is addressable by its contract name


def test_the_tied_gradient_pin():
    """Gate 9: one leaf declared once and captured twice receives ONE
    gradient — the summed contributions — checked against finite
    differences on the single shared leaf."""
    root = scope()
    w = root.param("w", d=3)

    @unit
    def first(h):
        return h * w.repeat_like(h, dim="t")  # capture 1

    @unit
    def second(h):
        return h + (w * w).repeat_like(h, dim="t")  # capture 2, nonlinearly

    from pdum.tl.ir import _dense_like
    from pdum.tl.layout import Dim
    from pdum.tl.tensor import Tensor

    lay = _dense_like((Dim("t", 0, 0, 2), Dim("d", 0, 0, 3)))
    a = assemblage(first | second, scope=root, h=lay)
    assert list(a.params) == ["w"]  # ONE leaf despite two captures
    rng = np.random.default_rng(3)
    inputs = {
        "h": Tensor.from_numpy(rng.standard_normal((2, 3)), ("t", "d")),
        "w": Tensor.from_numpy(rng.standard_normal(3), ("d",)),
    }
    prog = Program(a.program.instrs + (Instr("zloss", "reduce", (a.output,), {"f": "sum", "dims": ("t", "d")}),))
    jp, grads = grad(prog, "zloss", inputs)
    env = run(jp, inputs)
    got = env[grads["w"]].to_numpy(order=("d",))
    fd = numeric_grad(prog, "zloss", "w", inputs)
    np.testing.assert_allclose(got, fd, rtol=1e-5, atol=1e-8)  # summed, once


def test_derived_marker_names_carry_the_d_suffix():
    """The derived-name law's marker face: partials register as name.d{i}."""
    from pdum.tl.registry import MARKERS
    from pdum.tl.zoo.zoo_common import gelu

    d0 = gelu.partial(0)
    assert d0.name == "zoo.gelu.d0" and d0.name in MARKERS
