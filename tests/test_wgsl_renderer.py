"""The WGSL renderer, CPU-only: golden-shape assertions that need no adapter.
A GPU-less CI keeps full coverage of the text this backend emits."""

import pytest

import pdum.dsl  # noqa: F401
from pdum.dsl.demo.simple_shader.wgsl import COMPUTE, render
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.ir import VerifyError
from pdum.dsl.kernel.lower import lower_handle
from pdum.dsl.kernel.ops import CORE_OPS
from pdum.dsl.kernel.pack import ABI_OPS, NORMALIZE_ENV, legalize_params
from pdum.dsl.kernel.registry import DEFAULT
from pdum.dsl.kernel.rewrite import run_stage
from pdum.dsl.kernel.valuekind import BUILTINS

ALL_OPS = {**CORE_OPS, **ABI_OPS}


def rendered(handle, nargs=2):
    arg_types = (T.f64,) * nargs
    region = lower_handle(handle, DEFAULT.lower_rules, ALL_OPS, arg_types=arg_types)
    plan = COMPUTE.plan(handle.env_types, arg_types, BUILTINS)
    region = run_stage(region, NORMALIZE_ENV, ALL_OPS)
    region = run_stage(region, legalize_params(plan), ALL_OPS)
    return render(region, plan)


def test_golden_shape_signature_env_and_lazy_if():
    def make(cx, n, flag):
        @jit(kind="simple_shader.compute")
        def k(i, j):
            d = i - cx
            return (d / j) if flag else float(n)

        return k

    text = rendered(make(0.5, 3, True))
    assert "fn kernel_body(p0: f32, p1: f32) -> f32 {" in text
    assert "struct Env {" in text
    # member types follow the SLOT FORMAT: f64->f32, i64->i32, bool->u32
    assert "m0: f32" in text and "i32" in text and "u32" in text
    assert "(env.m" in text and "!= 0u)" in text  # bool reads compare against 0u
    assert "var v" in text and "if (" in text and "} else {" in text  # lazy statement-if


def test_select_argument_order_is_false_true_cond():
    """WGSL's select() is (false_value, true_value, condition) — reversed from
    the IR's (cond, a, b). A swap here is silent wrong answers on every ternary."""
    from pdum.dsl.kernel.ir import Builder, Region

    b = Builder(ALL_OPS)
    c = b.emit("core.env", type=T.boolean, slot=(0,))
    x = b.emit("core.env", type=T.f64, slot=(1,))
    y = b.emit("core.env", type=T.f64, slot=(2,))
    region = Region(body=(b.emit("core.yield", b.emit("core.select", c, x, y)),))
    plan = COMPUTE.plan((T.boolean, T.f64, T.f64), (), BUILTINS)
    region = run_stage(region, legalize_params(plan), ALL_OPS)
    text = render(region, plan)
    cond = next(ln for ln in text.splitlines() if "!= 0u" in ln).split(":")[0].replace("let", "").strip()
    args = next(ln for ln in text.splitlines() if "select(" in ln).split("select(")[1].rstrip(";").rstrip(")")
    assert args.split(", ")[2] == cond  # condition LAST — anything else is the swap bug


def test_nonfinite_and_bigint_consts_are_refused():
    def make_inf():
        @jit(kind="simple_shader.compute")
        def k(i, j):
            return i * 1e999

        return k

    with pytest.raises(VerifyError, match="inf/nan"):
        rendered(make_inf())

    def make_big(n):
        @jit(kind="simple_shader.compute")
        def k(i, j):
            m = n * 2654435761
            return float(m) + i

        return k

    with pytest.raises(VerifyError, match="does not fit i32"):
        rendered(make_big(3))


def test_env_member_names_avoid_reserved_words():
    """Offset 16 produced member `f16` — a reserved WGSL keyword (found live
    in ch11's six-capture kernel). Six captures force offsets through 16."""

    def make(a, b, c, d, e, f):
        @jit(kind="simple_shader.compute")
        def k(i, j):
            return i * a + b + c + d + e + f

        return k

    text = rendered(make(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    assert "m16: f32" in text and "f16" not in text
