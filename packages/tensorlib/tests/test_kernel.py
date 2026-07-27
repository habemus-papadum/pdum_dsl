"""@compute on the reference evaluator — the P7 gate (200 §S.3, §7).

The S.3 example runs; the iota-unification differential; the two-consumers
differential (S.2); the key-discipline pins (shape miss / value hit /
launch never keys / fn-swap miss); the compile-once thesis for
function-valued arguments; the day-one overlap refusal; the
struct-element round-trip through a structured encoding."""

import numpy as np
import pytest
from pdum.dsl import events, jit, op
from pdum.dsl.reference import reference
from pdum.tl import Tensor
from pdum.tl.compute import iota, pointwise
from pdum.tl.kernel import KERNELS, compute, config, thread_idx
from pdum.tl.zoo.zoo_common import GELU_C, np_gelu
from pdum.tl.zoo.zoo_common import gelu as gelu_marker


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


@op
def twill(a, b):
    @jit()
    def go(x):
        return x * a + b

    return go


@op
def zoom(scale):
    @jit()
    def go(x):
        return x * scale

    return go


@compute
def shader(f, img):
    y, x = thread_idx("y", "x")
    img[y, x] = f(y + x)


def _expected(f, shape):
    Y, X = np.meshgrid(np.arange(float(shape[0])), np.arange(float(shape[1])), indexing="ij")
    ref = reference(f)
    return np.vectorize(lambda a, b: ref(a + b))(Y, X)


# --- the S.3 example + the iota-unification differential --------------------


def test_the_s3_example_runs_on_the_reference_evaluator():
    f = twill(4.0, 3.0) | zoom(0.5)
    img = T(np.zeros((3, 4)), ("y", "x"))
    shader[config(blocks=(1, 1), threads=(16, 16))](f, img)
    np.testing.assert_allclose(img.to_numpy(), _expected(f, (3, 4)), rtol=1e-12)


def test_the_iota_unification_differential():
    """The same kernel three ways: @compute, hand pointwise-over-iotas, and
    the per-element spelled-oracle loop — all agree (S.3)."""
    f = twill(2.0, 1.0) | zoom(1.0)
    img = T(np.zeros((3, 4)), ("y", "x"))
    shader(f, img)
    ops = _region_ops(f, img)  # the unification is LITERAL now: coords ARE tl.iota
    assert "tl.iota" in ops and "tl.store" in ops
    # hand assemblage: pointwise over coordinate iotas
    base = T(np.zeros((3, 4)), ("y", "x"))
    yv, xv = iota(base, "y"), iota(base, "x")
    from pdum.tl.markers import add

    s = pointwise(add, yv, xv)
    ref = reference(f)
    hand = np.vectorize(lambda c: ref(float(c)))(s.to_numpy(order=("y", "x")))
    np.testing.assert_allclose(img.to_numpy(), hand, rtol=1e-12)
    # per-element loop through the spelled oracle
    loop = np.array([[ref(float(y + x)) for x in range(4)] for y in range(3)])
    np.testing.assert_allclose(img.to_numpy(), loop, rtol=1e-12)


def _region_ops(f, img):
    from pdum.tl.dialect import walk_region
    from pdum.tl.kernel import _arg_fp, _code_fp, _env_fp

    key = (_code_fp(shader.fn), _env_fp(shader.fn), (_arg_fp(f), _arg_fp(img)), ())
    return [n.op for n in walk_region(KERNELS.peek(key).region)]


# --- the two-consumers differential (S.2) -----------------------------------


from pdum.tl.mdsl import tanh  # noqa: E402


def _gelu(v):
    return 0.5 * v * (1 + tanh(GELU_C * (v + 0.044715 * v * v * v)))


@compute
def gelu_kernel(img):
    (y,) = thread_idx("y")
    img[y] = _gelu(y * 0.1)


def test_the_two_consumers_differential():
    """One definition, two consumers: gelu INLINED as a device function in a
    kernel body ≡ gelu as a registered pointwise marker under ir.run."""
    img = T(np.zeros(7), ("y",))
    gelu_kernel(img)
    ys = np.arange(7.0) * 0.1
    via_marker = pointwise(gelu_marker, T(ys, ("y",))).to_numpy()
    np.testing.assert_allclose(img.to_numpy(), via_marker, rtol=1e-12)
    np.testing.assert_allclose(img.to_numpy(), np_gelu(ys), rtol=1e-12)


# --- key discipline ----------------------------------------------------------


def test_key_discipline_shape_miss_value_hit_launch_never_keys_fn_swap_miss():
    img = T(np.zeros((3, 4)), ("y", "x"))
    shader(twill(1.0, 0.0) | zoom(1.0), img)  # warm the entry
    with events.forbid("kernel.miss"):
        # VALUE HIT: new captured values, same pipeline shape
        shader(twill(9.0, -2.0) | zoom(3.0), img)
        # TAP TENSORS NEVER KEY: config carries invocation data, same entry
        shader[config()](twill(1.0, 0.0) | zoom(1.0), img)
    # GEOMETRY IS VALIDATED LAUNCHER DATA (owner-ruled): it never keys and
    # never re-renders — this entry was built under the default geometry, so
    # presenting an explicit one REFUSES rather than silently meaning nothing
    with pytest.raises(ValueError, match="validated launcher data"):
        shader[config(blocks=(9, 9), threads=(2, 2))](twill(1.0, 0.0) | zoom(1.0), img)
    with pytest.raises(events.EventForbidden):
        with events.forbid("kernel.miss"):  # SHAPE MISS: a new lattice is a new artifact
            shader(twill(1.0, 0.0) | zoom(1.0), T(np.zeros((5, 5)), ("y", "x")))
    with pytest.raises(events.EventForbidden):
        with events.forbid("kernel.miss"):  # FN-SWAP MISS: a new pipeline shape
            shader(twill(1.0, 0.0) | zoom(1.0) | zoom(1.0), img)


def test_a_new_frame_is_a_new_artifact():
    """C4 tail: the labeling frame joined type identity, and the arg
    fingerprint IS that identity — same dims with a new frame is an honest
    MISS (a warm artifact's types would carry the wrong frame); the
    recompiled kernel then serves the charted lattice."""
    src, dst = T(np.ones((2, 3)), ("y", "x")), T(np.zeros((2, 3)), ("y", "x"))
    copy_kernel(src, dst)
    with events.forbid("kernel.miss"):  # same frame: warm
        copy_kernel(src, dst)
    cs = T(np.ones((2, 3)), ("y", "x")).with_charts(x=("0 um", "1 um"))
    cd = T(np.zeros((2, 3)), ("y", "x")).with_charts(x=("0 um", "1 um"))
    with pytest.raises(events.EventForbidden):
        with events.forbid("kernel.miss"):  # FRAME MISS: a new frame is a new lattice
            copy_kernel(cs, cd)
    copy_kernel(cs, cd)  # the recompile, outside the tripwire
    np.testing.assert_array_equal(cd.to_numpy(), 2.0 * np.ones((2, 3)))


def test_compile_once_thesis_for_function_valued_arguments():
    """The thesis at the kernel tier: 50 fresh pipelines with fresh values,
    one compile — values ride the rebind channel."""
    img = T(np.zeros((2, 3)), ("y", "x"))
    shader(twill(0.5, 0.5) | zoom(2.0), img)
    with events.forbid("kernel.miss"):
        for i in range(1, 50):
            shader(twill(float(i), 0.1 * i) | zoom(1.0 / i), img)
    f = twill(7.0, 0.25) | zoom(2.0)
    shader(f, img)
    np.testing.assert_allclose(img.to_numpy(), _expected(f, (2, 3)), rtol=1e-12)


# --- the day-one overlap refusal --------------------------------------------


@compute
def copy_kernel(src, dst):
    y, x = thread_idx("y", "x")
    dst[y, x] = src[y, x] * 2.0


def test_writable_overlapping_readable_refuses_ping_pong():
    buf = T(np.arange(12.0).reshape(3, 4), ("y", "x"))
    with pytest.raises(ValueError, match=r"overlaps readable.*ping-pong between two buffers"):
        copy_kernel(buf, buf)  # the same buffer readable AND writable
    src = T(np.arange(12.0).reshape(3, 4), ("y", "x"))
    dst = T(np.zeros((3, 4)), ("y", "x"))
    copy_kernel(src, dst)  # disjoint buffers: fine
    np.testing.assert_allclose(dst.to_numpy(), src.to_numpy() * 2.0)


# --- struct elements round-trip through a structured encoding ----------------


@compute
def complex_kernel(re_out, im_out):
    (y,) = thread_idx("y")
    re_out[y] = y * 0.5
    im_out[y] = 1.0 - y * 0.25


def test_struct_element_kernel_round_trips_through_structured_encoding():
    """The kernel writes the FIELDS; the memory shape is the descriptor's
    structured encoding (§4) — decode recovers the records exactly."""
    from pdum.tl.encoding import NumpyEncoding, adopt

    dt = np.dtype([("re", "<f8"), ("im", "<f8")])
    raw = np.zeros(5, dtype=dt)
    struct = Tensor.from_numpy(raw, ("y",))
    re_view, im_view = struct.field("re"), struct.field("im")
    complex_kernel(re_view, im_view)
    decoded = NumpyEncoding(dt).decode(raw)
    np.testing.assert_allclose(decoded["re"], np.arange(5.0) * 0.5)
    np.testing.assert_allclose(decoded["im"], 1.0 - np.arange(5.0) * 0.25)
    t = adopt(np.asarray(raw["re"]), NumpyEncoding(np.float64), ("y",))
    np.testing.assert_allclose(t.to_numpy(), np.arange(5.0) * 0.5)


# --- refusals ----------------------------------------------------------------


def test_kernels_return_nothing_and_data_dependent_indexing_refuses():
    @compute
    def bad_return(img):
        (y,) = thread_idx("y")
        return img

    with pytest.raises(ValueError, match="kernels return nothing"):
        bad_return(T(np.zeros(3), ("y",)))

    @compute
    def bad_index(img):
        (y,) = thread_idx("y")
        img[y * 2] = 1.0

    with pytest.raises(ValueError, match=r"exactly the thread coordinates.*arriving P9"):
        bad_index(T(np.zeros(3), ("y",)))


@jit()
def _looped_device_fn(cr, ci):
    zr = 0.0
    zi = 0.0
    n = 0.0
    for i in range(8):
        zr2 = zr * zr - zi * zi + cr
        zi = 2.0 * zr * zi + ci
        zr = zr2
        if zr * zr + zi * zi < 4.0:
            n = n + 1.0
    return n / 8.0


@compute
def _escape_kernel(f, img):
    y, x = thread_idx("y", "x")
    img[y, x] = f(y * 0.5 - 1.0, x * 0.5 - 1.0)


def test_liftable_fn_args_inline_with_zero_oracle_dispatch():
    """240 C5.3: the LIFTABLE class inlines — the pipeline's own scalar
    region splices into the kernel region as pointwise rows over
    arg-rooted abi slots, with NO launch-rebind markers and NO
    per-element oracle dispatch; capture values ride the
    extract->pack->launch channel fresh each launch."""
    from pdum.tl.dialect import walk_region
    from pdum.tl.kernel import _arg_fp, _code_fp, _env_fp

    img = T(np.zeros((3, 4)), ("y", "x"))
    f = twill(4.0, 3.0) | zoom(0.5)
    shader(f, img)
    key = (_code_fp(shader.fn), _env_fp(shader.fn), (_arg_fp(f), _arg_fp(img)), ())
    art = KERNELS.peek(key)
    assert art.fn_markers == {} and [b[0] for b in art.arg_slots] == ["f"]
    slots = [dict(n.attrs) for n in walk_region(art.region) if n.op == "abi.slot"]
    assert [(s["src"], s["offset"]) for s in slots] == [
        (("arg", "f", 0, 0), 0),  # twill's a
        (("arg", "f", 0, 1), 8),  # twill's b
        (("arg", "f", 1, 0), 16),  # zoom's s
    ]
    rows = [dict(n.attrs)["f"] for n in walk_region(art.region) if n.op == "tl.pointwise"]
    assert not any(str(r).startswith("kernel.fn.") for r in rows)


def test_all_scalar_subtrees_stay_value_dialect():
    """A spliced fn whose interior op never meets a tensor (a*b) keeps
    that op in the VALUE dialect inside the kernel region — run_region
    has its rows (the markers compute on host scalars by law) — and the
    values still ride per launch."""

    def gain(a, b):
        @jit()
        def g(x):
            return x * (a * b)

        return g

    @compute
    def apply_gain(f, img):
        (y,) = thread_idx("y")
        img[y] = f(y)

    img = T(np.zeros(3), ("y",))
    apply_gain(gain(2.0, 3.0), img)
    np.testing.assert_allclose(img.to_numpy(), np.arange(3.0) * 6.0)
    with events.forbid("kernel.miss"):  # fresh values, warm artifact
        apply_gain(gain(5.0, -1.0), img)
    np.testing.assert_allclose(img.to_numpy(), np.arange(3.0) * -5.0)


def test_the_reference_refuses_out_of_bounds_reads():
    """P8 buffer reads + the keying-ladder ruling: an out-of-bounds
    computed read REFUSES at the reference — an oracle has no undefined
    behavior, so running on reference IS the UB detector; device
    backends need not check."""

    @compute
    def shift_read(tex, img):
        (y,) = thread_idx("y")
        img[y] = tex[y + 3.0]

    tex, img = T(np.arange(5.0), ("y",)), T(np.zeros(3), ("y",))
    with pytest.raises(ValueError, match="out of bounds.*oracle has no undefined behavior"):
        shift_read(tex, img)


def test_fn_arg_with_loops_and_module_global_name():
    """Two pins: (a) per-pixel LOOPS/BRANCHES live in @jit device functions
    (the value language) — the kernel body stays straight-line plumbing;
    (b) REGRESSION: an argument handle also visible as a module global under
    its own name must bind through the PARAMETER slot (the lookup once found
    the global first and the launch rebind broke)."""
    img = T(np.zeros((4, 4)), ("y", "x"))
    _escape_kernel(_looped_device_fn, img)
    ref = reference(_looped_device_fn)
    want = np.array([[ref(y * 0.5 - 1.0, x * 0.5 - 1.0) for x in range(4)] for y in range(4)])
    np.testing.assert_allclose(img.to_numpy(), want, rtol=1e-12)


# --- the bracket config + taps (owner-ruled syntax, TAGLESS) -----------------
# The naming law is the claiming mechanism (S.4 amendment): there is no
# tap() call — every uniquely-named binding IS a site.


from pdum.tl.kernel import shared  # noqa: E402


@compute
def tapped_kernel(img):
    y, x = thread_idx("y", "x")
    dist = (y - 1.0) * (y - 1.0) + x * 0.0
    img[y, x] = dist * 2.0


def test_config_bracket_taps_write_into_caller_tensors():
    img = T(np.zeros((3, 4)), ("y", "x"))
    dt = T(np.zeros((3, 4)), ("y", "x"))
    tapped_kernel[config(taps={"dist": dt})](img)
    Y = np.arange(3.0)[:, None] * np.ones((1, 4))
    np.testing.assert_allclose(dt.to_numpy(), (Y - 1.0) ** 2)  # the tap, full lattice
    np.testing.assert_allclose(img.to_numpy(), (Y - 1.0) ** 2 * 2.0)
    tapped_kernel(img)  # the plain call still works; no tap written


def test_tap_name_set_specializes_tensors_do_not():
    """The config contract: the tap NAME SET is identity-bearing (a
    different set is a different artifact); the tap TENSORS are pure
    invocation data; GEOMETRY is validated launcher data — it never keys,
    and an incoherent geometry refuses instead of silently meaning
    nothing."""
    img = T(np.zeros((2, 2)), ("y", "x"))
    t1, t2 = T(np.zeros((2, 2)), ("y", "x")), T(np.zeros((2, 2)), ("y", "x"))
    tapped_kernel[config(taps={"dist": t1})](img)
    with events.forbid("kernel.miss"):
        tapped_kernel[config(taps={"dist": t2})](img)  # new TENSOR: warm hit
    with pytest.raises(ValueError, match="validated launcher data"):
        tapped_kernel[config(taps={"dist": t1}, blocks=(9, 9), threads=(2, 2))](img)
    with pytest.raises(events.EventForbidden):
        with events.forbid("kernel.miss"):
            tapped_kernel(img2 := T(np.zeros((5, 5)), ("y", "x")))  # noqa: F841 — shape miss still misses


def test_tap_introspection_lists_every_named_binding():
    """Tagless: EVERY uniquely-named binding is a site — the thread
    coordinates included (tapping a coordinate dumps its iota field)."""
    img = T(np.zeros((2, 3)), ("y", "x"))
    sites = tapped_kernel.taps(img)
    assert sites["dist"] == {"valid": True, "dims": ("y", "x"), "reason": None}
    assert sites["y"]["valid"] and sites["x"]["valid"]  # bindings, so sites


def _tapped_helper(v):
    inner = v * 3.0
    return inner * 2.0


@compute
def colliding_kernel(img):
    y, x = thread_idx("y", "x")
    a = _tapped_helper(y + 0.0)
    b = _tapped_helper(x + 0.0)  # the SAME binding inlined twice: non-unique
    img[y, x] = a + b


def test_colliding_tap_sites_invalidate_honestly():
    """The naming law never auto-suffixes: a binding inlined into
    non-uniqueness is reported INVALID, and requesting it refuses."""
    img = T(np.zeros((2, 2)), ("y", "x"))
    sites = colliding_kernel.taps(img)
    assert sites["inner"]["valid"] is False
    assert "more than one site" in sites["inner"]["reason"]
    with pytest.raises(ValueError, match="tap 'inner' is INVALID.*more than one site"):
        colliding_kernel[config(taps={"inner": img})](img)


def test_unknown_tap_refuses_listing_sites():
    img = T(np.zeros((2, 2)), ("y", "x"))
    with pytest.raises(ValueError, match=r"no tap site 'nope' — sites: .*dist"):
        tapped_kernel[config(taps={"nope": img})](img)


def test_shared_mem_slot_is_reserved():
    img = T(np.zeros((2, 2)), ("y", "x"))
    with pytest.raises(NotImplementedError, match=r"tile tier \(L4\).*slot is reserved"):
        tapped_kernel[config(shared_mem=shared(t1=("ty", 16)))](img)


# --- 240 C1: the explicit staging door + the closed kernel key ---------------


def test_undeclared_function_returning_host_call_refuses():
    """Door 4 is explicit now: a host call may return structural data, but
    a FUNCTION CITIZEN crossing the door must come from @staged."""

    def sneaky_factory():  # NOT @staged
        @jit()
        def go(y):
            return y * 2.0

        return go

    @compute
    def k(img):
        (y,) = thread_idx("y")
        f = sneaky_factory()  # noqa: F841
        img[y] = y * 1.0

    with pytest.raises(ValueError, match=r"@staged.*or build the value outside"):
        k(T(np.zeros(3), ("y",)))


def test_staged_transforms_compose_and_restage():
    """Recipes chain (functional, composable — owner ruling): a staged
    transform built on another staged transform replays through BOTH on a
    warm hit, with the current parameter."""
    from pdum.dsl import staged, value_and_grad

    @staged
    def slope_only(f):  # a smaller transform composed from a smaller one
        return value_and_grad(f, wrt=("y",))

    def line(a):
        @jit()
        def go(y):
            return y * a

        return go

    @compute
    def k(f, img):
        (y,) = thread_idx("y")
        g = slope_only(f)
        v, (dy,) = g(y)
        img[y] = dy  # the slope field: constant a

    img = T(np.zeros(4), ("y",))
    k(line(3.0), img)
    np.testing.assert_allclose(img.to_numpy(), 3.0)
    with events.forbid("kernel.miss"):  # warm hit, new capture...
        k(line(5.0), img)
    np.testing.assert_allclose(img.to_numpy(), 5.0)  # ...restaged, not stale


_C1_SCALE = 2.0


def test_unmarked_captured_scalar_is_data_warm_hit_fresh_value():
    """THE LITERAL DOCTRINE (240 C4.2b, owner-ruled): an unmarked captured
    scalar is DATA — a per-launch uniform slot. Rebinding it is a WARM HIT
    with the fresh value: no recompile, never stale."""

    @compute
    def k(img):
        (y,) = thread_idx("y")
        img[y] = y * _C1_SCALE

    img = T(np.zeros(3), ("y",))
    k(img)
    np.testing.assert_allclose(img.to_numpy(), np.arange(3.0) * 2.0)
    globals()["_C1_SCALE"] = 7.0
    try:
        with events.forbid("kernel.miss"):  # data changed, the PROGRAM did not
            k(img)
        np.testing.assert_allclose(img.to_numpy(), np.arange(3.0) * 7.0)
    finally:
        globals()["_C1_SCALE"] = 2.0


def test_uniform_slots_ride_the_dsl_marshaling_dialect():
    """240 C5 (do-not-forget B): tl.uniform is GONE — an unmarked captured
    scalar is an ``abi.slot`` node (src/offset/fmt, the dsl's marshaling
    dialect), extracted and byte-packed fresh at every launch."""
    from pdum.tl.dialect import walk_region
    from pdum.tl.kernel import _arg_fp, _code_fp, _env_fp

    @compute
    def k(img):
        (y,) = thread_idx("y")
        img[y] = y * _C1_SCALE

    img = T(np.zeros(3), ("y",))
    k(img)
    key = (_code_fp(k.fn), _env_fp(k.fn), (_arg_fp(img),), ())
    ops = list(walk_region(KERNELS.peek(key).region))
    (slot,) = [n for n in ops if n.op == "abi.slot"]
    assert dict(slot.attrs) == {"src": ("env", "_C1_SCALE"), "offset": 0, "fmt": "<d"}
    assert not any(n.op == "tl.uniform" for n in ops)


def test_block_idx_default_geometry_and_split_geometry_warmth():
    """P8 ambient raws: under the DEFAULT geometry one block spans the
    lattice — block_idx is the zero field and thread_idx IS the global
    coordinate; under explicit geometry the raws are iotas of the split
    target's (block, thread) dim pairs, and the same geometry stays
    WARM."""
    from pdum.tl.kernel import block_idx  # noqa: F401 — resolved from the body's globals

    @compute
    def k_default(img):
        (by,) = block_idx("y")
        (ty,) = thread_idx("y")
        img[ty] = by + ty  # by == 0 everywhere

    img = T(np.zeros(4), ("y",))
    k_default(img)
    np.testing.assert_allclose(img.to_numpy(), np.arange(4.0))
    with events.forbid("kernel.miss"):  # same (default) geometry: warm
        k_default(img)
        k_default[config(blocks=(1,), threads=(8,))](img)  # all-ones blocks IS the default: same entry
    with pytest.raises(ValueError, match="does not cover"):
        k_default[config(threads=(2,))](img)  # an under-provisioned one-block launch refuses

    @compute
    def k_split(tiled):
        (by,) = block_idx("y")
        (ty,) = thread_idx("y")
        tiled[by, ty] = by * 4.0 + ty

    base = T(np.zeros(8), ("y",))
    k_split[config(blocks=(2,), threads=(4,))](base.split("y", by=2, ty=4))
    with events.forbid("kernel.miss"):  # same geometry, same lattice: warm
        k_split[config(blocks=(2,), threads=(4,))](base.split("y", by=2, ty=4))
    np.testing.assert_allclose(base.to_numpy(), np.arange(8.0))
    with pytest.raises(ValueError, match="validated launcher data"):
        k_split[config(blocks=(4,), threads=(2,))](base.split("y", by=2, ty=4))


def test_identical_ir_shares_one_executor_across_kernels():
    """240 C5.2: the content tier. Two kernels, two templates, ONE
    artifact — the second lowers (a spec miss: new code fp) but its
    executor comes from the content cache (artifact HIT on
    (region.key, executor fp)): identical IR never builds twice."""
    from pdum.tl.kernel import _arg_fp, _code_fp, _env_fp

    @compute
    def k_left(img):
        (y,) = thread_idx("y")
        img[y] = y * 3.0

    @compute
    def k_right(img):
        (y,) = thread_idx("y")
        img[y] = y * 3.0

    assert _code_fp(k_left.fn) != _code_fp(k_right.fn)  # the pin's premise: two templates
    img = T(np.zeros(3), ("y",))
    k_left(img)  # warms BOTH tiers
    with events.forbid("artifact.miss"):  # tier 1 misses (new template); tier 2 must NOT
        k_right(img)
    np.testing.assert_allclose(img.to_numpy(), np.arange(3.0) * 3.0)
    a = KERNELS.peek((_code_fp(k_left.fn), _env_fp(k_left.fn), (_arg_fp(img),), ()))
    b = KERNELS.peek((_code_fp(k_right.fn), _env_fp(k_right.fn), (_arg_fp(img),), ()))
    assert a.region.key == b.region.key and a.executor is b.executor


from pdum.dsl import literal  # noqa: E402

_C4_GAIN = literal(3.0)


def test_literal_wrapped_capture_bakes_and_recompiles_by_choice():
    """The declared door: literal(...) bakes the value into the program and
    keys identity — changing it RECOMPILES, because the user chose that."""

    @compute
    def k(img):
        (y,) = thread_idx("y")
        img[y] = y * _C4_GAIN

    img = T(np.zeros(3), ("y",))
    k(img)
    np.testing.assert_allclose(img.to_numpy(), np.arange(3.0) * 3.0)
    globals()["_C4_GAIN"] = literal(5.0)
    try:
        with pytest.raises(events.EventForbidden):
            with events.forbid("kernel.miss"):  # a different literal IS a different program
                k(img)
        k(img)  # and the recompiled program computes with the new literal
        np.testing.assert_allclose(img.to_numpy(), np.arange(3.0) * 5.0)
    finally:
        globals()["_C4_GAIN"] = literal(3.0)


def _c1_helper(v):
    return v * 10.0


def test_edited_captured_helper_misses_never_stale():
    """Same law for captured helper CODE: redefining the helper is a miss."""

    @compute
    def k(img):
        (y,) = thread_idx("y")
        img[y] = _c1_helper(y)

    img = T(np.zeros(3), ("y",))
    k(img)
    np.testing.assert_allclose(img.to_numpy(), np.arange(3.0) * 10.0)
    original = globals()["_c1_helper"]

    def _c1_helper_v2(v):
        return v * 100.0

    globals()["_c1_helper"] = _c1_helper_v2
    try:
        k(img)
        np.testing.assert_allclose(img.to_numpy(), np.arange(3.0) * 100.0)
    finally:
        globals()["_c1_helper"] = original


def test_staged_transform_must_return_a_function_citizen():
    """Declaration-first, checked both ways (240 C2): a DECLARED staged
    transform that returns structural data is a broken declaration."""
    from pdum.dsl import staged

    @staged
    def not_a_transform(f):
        return 42.0  # structural, not a citizen

    @compute
    def k(f, img):
        (y,) = thread_idx("y")
        g = not_a_transform(f)  # noqa: F841
        img[y] = y * 1.0

    with pytest.raises(ValueError, match=r"not a\s+function citizen"):
        k(twill(1.0, 0.0) | zoom(1.0), T(np.zeros(3), ("y",)))
