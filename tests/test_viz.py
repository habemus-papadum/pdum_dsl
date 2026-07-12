"""The viz satellite: composable static HTML, both notebook hosts, no kernel edits."""

from pdum.dsl import viz
from pdum.dsl.combinators import op, register_composition, register_role
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.cache import SpecializationCache
from pdum.dsl.kernel.ir import Builder, Loc, Region
from pdum.dsl.kernel.ops import CORE_OPS

register_role("device")
register_composition("pipe", "device", "device", "fuse")

b = Builder(CORE_OPS)


def make(k):
    @jit()
    def go(x):
        return x * k

    return go


def small_region():
    e = b.emit("core.env", type=T.f64, slot=(0,), loc=Loc("art.py", 7))
    return Region(body=(b.emit("core.yield", b.emit("core.mul", e, e)),))


def test_fragments_compose_and_style_emits_once():
    @op
    def stage(k):
        @jit()
        def go(x):
            return x + k

        return go

    p = stage(1.0) | stage(2.0)
    page = viz.render(p)._repr_html_()
    assert page.count("<style>") == 1  # containers embed fragments, style once at root
    assert page.count("pv-chip") >= 2  # both stage chips embedded


def test_handle_card_has_pills_source_and_escaping():
    html = viz.render(make(3.0))._repr_html_()
    assert "pv-card" in html and "pv-pill" in html and "<details>" in html
    assert "<script" not in html.lower()


def test_region_widget_has_tooltips_with_type_and_loc():
    html = viz.render(small_region())._repr_html_()
    assert 'data-tip="f64 · art.py:7"' in html  # hover = type + provenance
    assert "core.mul" in html and "key " in html


def test_cache_and_fallback():
    c = SpecializationCache()
    assert "generation" in viz.render(c)._repr_html_()
    assert "<pre>" in viz.render(object())._repr_html_()  # unknown types degrade politely


def test_install_attaches_repr_html_without_kernel_edits():
    viz.install()
    assert "pdum-viz" in make(1.0)._repr_html_()
    assert "pdum-viz" in small_region()._repr_html_()


def test_save_writes_standalone_page(tmp_path):
    path = viz.save(small_region(), str(tmp_path / "r.html"))
    text = open(path).read()
    assert text.startswith("<!doctype html>") and "pdum-viz" in text
