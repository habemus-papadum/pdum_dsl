"""M0.2 — AST → IR → WGSL. Golden-ish emission, the value/type-change invariant
that makes the cache work, and uniform-buffer layout (incl. the vec3 padding rule)."""

import struct

from pdum.dsl_reference import builtins, jit
from pdum.dsl_reference.backends.wgsl import compile_fragment
from pdum.dsl_reference.backends.wgsl.layout import build_layout
from pdum.dsl_reference.types import VecType, f32


def make_disk(cx, cy, radius):
    @jit(kind="fragment")
    def shader():
        x, y = builtins.FragCoord.xy
        dx = x - cx
        dy = y - cy
        d2 = dx * dx + dy * dy
        return (1.0, 0.5, 0.0) if d2 < radius * radius else (0.05, 0.05, 0.1)

    return shader


# --- emission ---------------------------------------------------------------


def test_emits_expected_wgsl_pieces():
    mod = compile_fragment(make_disk(100.0, 100.0, 50.0))
    w = mod.wgsl
    assert "struct Uniforms {" in w
    assert "cx: f32," in w and "cy: f32," in w and "radius: f32," in w
    assert "var<uniform> u: Uniforms;" in w
    assert "@vertex" in w and "fn vs_main" in w
    assert "@fragment" in w and "fn fs_main(@builtin(position) fragcoord: vec4<f32>)" in w
    assert "fragcoord.xy" in w
    assert "u.cx" in w and "u.radius" in w
    assert "select(" in w  # the ternary
    # vec3 color return padded to vec4
    assert "vec4<f32>(select(" in w


def test_value_change_is_byte_identical_wgsl():
    # Same types, different captured values -> identical WGSL (values are uniforms).
    a = compile_fragment(make_disk(10.0, 20.0, 30.0)).wgsl
    b = compile_fragment(make_disk(99.0, 88.0, 77.0)).wgsl
    assert a == b


def test_type_change_changes_wgsl():
    floats = compile_fragment(make_disk(10.0, 20.0, 30.0)).wgsl
    mixed = compile_fragment(make_disk(10.0, 20.0, 30)).wgsl  # radius is int now
    assert floats != mixed
    assert "radius: i32," in mixed


# --- uniform layout: WGSL alignment rules -----------------------------------


def test_layout_scalar_packing_roundtrips():
    layout = build_layout(("cx", "cy", "radius"), (f32, f32, f32))
    assert [f.offset for f in layout.fields] == [0, 4, 8]
    assert layout.size == 16  # rounded up to a multiple of 16
    raw = layout.pack({"cx": 1.5, "cy": 2.5, "radius": 3.5})
    assert len(raw) == 16
    assert struct.unpack("<3f", raw[:12]) == (1.5, 2.5, 3.5)


def test_layout_vec3_is_aligned_to_16():
    # scalar then vec3: the vec3 must start at offset 16 (align 16), not 4.
    layout = build_layout(("a", "color"), (f32, VecType(f32, 3)))
    by = layout.by_name()
    assert by["a"].offset == 0
    assert by["color"].offset == 16
    assert layout.size == 32


def test_layout_matches_spec_example():
    # docs example: resolution: vec2, time: f32, color: vec3  -> offsets 0, 8, 16; size 32
    layout = build_layout(
        ("resolution", "time", "color"),
        (VecType(f32, 2), f32, VecType(f32, 3)),
    )
    assert [f.offset for f in layout.fields] == [0, 8, 16]
    assert layout.size == 32
