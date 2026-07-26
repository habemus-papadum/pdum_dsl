"""THE COMMITTED KERNEL SYNTAX — specified now, implemented later.

Every test here is SKIPPED and is the executable contract for a gated
milestone: it un-skips when its machinery lands, and its body is the
ratified spelling (owner-ruled). The governing law (200 §S.3 amendment,
220 §14):

    There is ONE body language: the value language. A kernel body is the
    value language plus exactly three dialect extensions — the thread
    AMBIENT, token-threaded STORES, and buffer READS. The scalar marker
    subset is its straight-line, effect-free core. One derivative engine
    (forward seeding over the one derivative table) serves every
    per-element tier.

Names used below (block_idx, grid_layout, global_thread_idx,
with_respect_to, value_and_grad, shared_alloc, shared_bind, barrier,
sample) arrive with their implementations; this file records the
spellings. The ambient's primitives are the RAW block/thread pair plus
the launch grid reified as a LAYOUT; global_thread_idx is a stdlib
device function over the full triple (block, thread, grid) — layout
evaluation, not an intrinsic.
"""

import numpy as np
import pytest
from pdum.dsl.intrinsics import clamp
from pdum.dsl.markers import sqrt
from pdum.tl import (  # noqa: F401 — ambient vocabulary resolved from bodies' globals
    Tensor,
    block_idx,
    compute,
    global_thread_idx,
    grid_layout,
    thread_idx,
)
from pdum.tl.kernel import config

P8 = pytest.mark.skip(reason="specified now; lands at P8 (graphics + the value-tier derivative engine)")
P9 = pytest.mark.skip(reason="specified now; lands at P9 (the indexing family)")
L4 = pytest.mark.skip(reason="specified now; lands with the tile tier (L4)")


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


# --- P8: the ambient is the RAW lattice; global is a device function ---------


def test_raw_block_thread_ambient_and_derived_global():
    """The ambient PRIMITIVES: the raw lattice pair — block_idx and
    thread_idx (within block; under the default one-block-spans-the-
    lattice geometry the two coincide, which is why every existing
    kernel reads unchanged) — plus grid_layout(), the launch geometry
    reified as a LAYOUT with block and thread levels per axis
    (config(blocks, threads) is its constructor sugar; the same
    split+bind object as the tile tier and placement's machine-bound
    dims). global_thread_idx is NOT an intrinsic: it is a stdlib
    DEVICE FUNCTION taking the full triple — (block, thread, grid) —
    and it IS layout evaluation: the grid's affine map applied at the
    raw pair. Nothing ambient hides inside it. A backend may bind the
    NAME to a hardware built-in when g is the launch grid (Metal's
    thread_position_in_grid) or take the computed floor (CUDA) —
    declarations over recognition; named axes make the vendors'
    index-component conventions a binding detail."""

    @compute
    def k(img):
        g = grid_layout()  # ambient: the launch grid AS A LAYOUT
        by, bx = block_idx("y", "x")  # raw
        ty, tx = thread_idx("y", "x")  # raw: within block under explicit geometry
        gy, gx = global_thread_idx((by, bx), (ty, tx), g)  # the triple
        img[gy, gx] = (gy - (by * 8.0 + ty)) + (gx - (bx * 8.0 + tx))  # identity → 0

    img = T(np.ones((16, 16)), ("y", "x"))
    k[config(blocks=(2, 2), threads=(8, 8))](img)
    np.testing.assert_allclose(img.to_numpy(), 0.0)


def test_split_aligned_tensors_index_by_raw_coordinates():
    """A tensor already split to the (block, thread) lattice is indexed
    by the RAW pair directly — respecting the split, no global round
    trip. The direction is FORCED by the affine-only layout algebra:
    raw→global is affine (by·T + ty); global→raw is div/mod — piecewise,
    banned. So the raws are the primitives and global is sugar."""

    @compute
    def k(tiled):
        (by,) = block_idx("y")
        (ty,) = thread_idx("y")
        tiled[by, ty] = by * 8.0 + ty  # the global coordinate, via raws alone

    base = T(np.zeros(16), ("y",))
    k[config(blocks=(2,), threads=(8,))](base.split("y", by=2, ty=8))
    np.testing.assert_allclose(base.to_numpy(), np.arange(16.0))


def test_global_thread_idx_survives_coordinate_transforms():
    """Thread identity is AMBIENT everywhere: a device function sees the
    global thread id even after its coordinates were scaled/translated —
    transforms change what you SAMPLE, never who you are."""
    from pdum.dsl import jit

    def probe():
        @jit()
        def go(y, x):
            g = grid_layout()  # the ambient reaches device functions too
            gy, gx = global_thread_idx(block_idx("y", "x"), thread_idx("y", "x"), g)
            return (y - gy) + (x - gx)  # zero iff untransformed

        return go

    @compute
    def k(f, img):
        y, x = thread_idx("y", "x")
        img[y, x] = f(y * 2.0, x * 2.0)  # transformed coords; ambient unchanged

    img = T(np.zeros((2, 3)), ("y", "x"))
    k(probe(), img)
    Y, X = np.meshgrid(np.arange(2.0), np.arange(3.0), indexing="ij")
    np.testing.assert_allclose(img.to_numpy(), Y + X)  # (2y - y) + (2x - x)


# --- P8: the two derivative operators, one engine ----------------------------


def test_with_respect_to_a_local_value():
    """LIVE (P8 first movement): value-space, d(computed)/d(upstream local),
    forward-seeded over the one table at lower time — inside a kernel's f.
    The ORIGINAL pinned spelling: the numpy-authority oracle is IEEE
    non-trapping, so the 0/0 pole at the origin flows as nan (like a
    device) and the assertion simply skips that row."""
    from pdum.dsl import jit

    @jit()
    def go(y, x):
        d = sqrt(y * y + x * x)  # noqa: F821
        dd_dy = with_respect_to(d, y)  # noqa: F821 — forward seed y=1
        return dd_dy

    @compute
    def k(f, img):
        y, x = thread_idx("y", "x")
        img[y, x] = f(y, x)

    img = T(np.zeros((3, 3)), ("y", "x"))
    with np.errstate(invalid="ignore", divide="ignore"):
        k(go, img)
    Y, X = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(img.to_numpy()[1:], (Y / np.sqrt(Y * Y + X * X))[1:], rtol=1e-9)


def test_value_and_grad_wrt_ambient_is_fwidth():
    """LIVE (P8): function-space value + gradient wrt a declared argument
    set. With wrt = the ambient thread coordinates this IS dFdx/dFdy —
    S.4's 'fwidth is the wrt-ambient derivative' — and analytic AA at the
    shader's top level is its first consumer. The kernel destructures the
    tuple result (the pattern declares the structure); clamp inlines into
    the kernel by ordinary capture-and-call (one body language). The
    center sits at 7.5 — pixel centers off the signed-distance pole, as
    in real rasterization."""
    from pdum.dsl import jit, value_and_grad

    def circle(cy, cx, r):
        @jit()
        def go(y, x):
            d = sqrt((y - cy) * (y - cy) + (x - cx) * (x - cx))
            return d - r  # signed distance

        return go

    g = value_and_grad(circle(7.5, 7.5, 5.0), wrt=("y", "x"))

    @compute
    def aa_shader(f, img):
        y, x = thread_idx("y", "x")
        v, (dy, dx) = f(y, x)
        w = sqrt(dy * dy + dx * dx)  # fwidth — analytic, no 2x2 quad
        img[y, x] = clamp(v / w + 0.5, 0.0, 1.0)  # one-pixel edge

    img = T(np.zeros((16, 16)), ("y", "x"))
    aa_shader(g, img)
    a = img.to_numpy()
    assert a.min() == 0.0 and a.max() == 1.0  # interior black, exterior white
    assert ((a > 0.0) & (a < 1.0)).any()  # ...and the analytic AA band between


def test_value_and_grad_stages_inside_the_kernel_body():
    """LIVE (P8, owner question answered): pass f DIRECTLY and compute
    value_and_grad INSIDE the kernel — f is compile-time (a handle, not a
    tensor), so the host-evaluation rule stages the transform at lower
    time. The BASE parameter rides the rebind channel and the transform
    re-applies per launch: a warm hit (same fp, new captured values)
    computes with the NEW circle, never a stale one."""
    from pdum.dsl import events, jit, value_and_grad

    def circle(cy, cx, r):
        @jit()
        def go(y, x):
            d = sqrt((y - cy) * (y - cy) + (x - cx) * (x - cx))
            return d - r

        return go

    @compute
    def aa_shader(f, img):
        y, x = thread_idx("y", "x")
        g = value_and_grad(f, wrt=("y", "x"))  # staged: f's identity is compile-time
        v, (dy, dx) = g(y, x)
        w = sqrt(dy * dy + dx * dx)
        img[y, x] = clamp(v / w + 0.5, 0.0, 1.0)

    img = T(np.zeros((16, 16)), ("y", "x"))
    aa_shader(circle(7.5, 7.5, 5.0), img)
    big = img.to_numpy().copy()
    with events.forbid("kernel.miss"):  # warm hit: new values, same shape...
        aa_shader(circle(7.5, 7.5, 2.0), img)
    small = img.to_numpy()
    assert small.sum() > big.sum()  # ...and the SMALLER circle rendered (no stale capture)
    for a in (big, small):
        assert a.min() == 0.0 and a.max() == 1.0 and ((a > 0.0) & (a < 1.0)).any()


@P8
def test_derivative_type_law_records_mirror_their_value():
    """The differentiated value may be a scalar, a record, or a statically
    sized float tensor; the RESULT HAS THE SAME TYPE (per-field for
    records)."""
    from dataclasses import dataclass

    from pdum.dsl import jit
    from pdum.dsl.surfaces import record  # noqa: F401

    @dataclass(frozen=True)
    class RG:
        r: float
        g: float

    @jit()
    def go(y, x):
        c = RG(y * x, y + x)
        dc = with_respect_to(c, y)  # noqa: F821 — RG(d r/dy, d g/dy) = RG(x, 1)
        return dc.r + dc.g

    @compute
    def k(f, img):
        y, x = thread_idx("y", "x")
        img[y, x] = f(y, x)

    img = T(np.zeros((2, 3)), ("y", "x"))
    k(go, img)
    Y, X = np.meshgrid(np.arange(2.0), np.arange(3.0), indexing="ij")
    np.testing.assert_allclose(img.to_numpy(), X + 1.0)


# --- P8: buffer reads — the third dialect extension --------------------------


@P8
def test_element_reads_at_computed_indices():
    """The kernel dialect reads buffers at COMPUTED integer indices —
    a value-language load, gradient-free through the indices (the carrier
    discipline). This is the fuzz/texture door."""

    @compute
    def gather_diag(tex, img):
        (y,) = thread_idx("y")
        img[y] = tex[y, y]  # computed index read (not just thread coords)

    tex = T(np.arange(9.0).reshape(3, 3), ("y", "x"))
    img = T(np.zeros(3), ("y",))
    gather_diag(tex, img)
    np.testing.assert_allclose(img.to_numpy(), [0.0, 4.0, 8.0])


@P8
def test_neighborhood_static_loop_unrolls_to_scalar_ops():
    """A statically-known loop over element loads — 'converted into a
    series of scalar operations'. (The tensor-tier stencil view is the
    SAME computation at a different altitude; descent may convert.)"""

    @compute
    def box3(tex, img):
        (y,) = thread_idx("y")
        acc = 0.0
        for dy in range(3):
            acc = acc + tex[y + dy]
        img[y] = acc / 3.0

    tex = T(np.arange(5.0), ("y",))
    img = T(np.zeros(3), ("y",))
    box3(tex, img)
    np.testing.assert_allclose(img.to_numpy(), [1.0, 2.0, 3.0])


@P8
def test_fuzz_as_a_combinator_closes_over_a_buffer():
    """The combinator FORM of fuzz: a device function closing over a noise
    texture — buffer closure at the f tier (arg-rooted buffer slots)."""
    from pdum.dsl import jit

    def fuzz(ny, nx):
        def apply(f):
            @jit()
            def go(y, x):
                return f(y + ny[y, x], x + nx[y, x])  # closed-over buffers

            return go

        return apply  # Comb(apply) once blessed into the dsl tier

    rng = np.random.default_rng(0)
    ny = T(rng.standard_normal((8, 8)), ("y", "x"))
    nx = T(rng.standard_normal((8, 8)), ("y", "x"))
    assert fuzz(ny, nx) is not None


# --- P8: taps inside device-function/combinator bodies -----------------------


@P8
def test_taps_inside_combinator_bodies_with_validity():
    """A claimed BINDING inside a combinator body — tagless, the naming
    law; applied once it is valid, applied twice its name goes non-unique
    and the site is INVALIDATED. Arrives when function-valued ARGUMENTS
    inline (today they dispatch per element through the oracle, so their
    bindings are opaque; captured helpers already claim — test_kernel)."""
    from pdum.dsl import jit

    def scale(s):
        @jit()
        def go(y, x):
            yprime = y * s
            return yprime + x * s

        return go

    @compute
    def k(f, img):
        y, x = thread_idx("y", "x")
        img[y, x] = f(y, x)

    img = T(np.zeros((2, 2)), ("y", "x"))
    yp = T(np.zeros((2, 2)), ("y", "x"))
    k[config(taps={"yprime": yp})](scale(2.0), img)
    np.testing.assert_allclose(yp.to_numpy(), np.arange(2.0)[:, None] * 2.0 * np.ones((1, 2)))


@P8
def test_record_taps_land_as_struct_tensors():
    """isBits record-valued bindings claim as struct-element tensors —
    the structured encoding is the memory shape (200 §4)."""


# --- P8: the graphics tier (S.4 amendment) -----------------------------------


@P8
def test_quad_from_vertex_index_pairs_with_the_compute_zoo_f():
    """The whole S.4 flow in one spelling: a vertex shader with NO vertex
    buffers (corners computed from the raw ambient), a fragment shader
    that normalizes into f's space and calls THE SAME f as the compute
    zoo, PSO pairing (its own composition, never |), and rendering
    through the reference interpolator. Return is MANDATORY (position /
    color0); everything else is claimed by naming it."""
    from pdum.dsl import jit
    from pdum.tl.graphics import fragment, pair, position, render, vertex  # noqa: F821

    def circle(cy, cx, r):
        @jit()
        def go(y, x):
            d = sqrt((y - cy) * (y - cy) + (x - cx) * (x - cx))  # noqa: F821
            return 1.0 if d > r else 0.0

        return go

    @vertex
    def quad():
        vid = vertex_index()  # noqa: F821 — raw ambient; two triangles, six ids
        u = 1.0 if (vid == 1 or vid == 3 or vid == 4) else 0.0
        v = 1.0 if (vid == 2 or vid == 4 or vid == 5) else 0.0  # claimed varyings
        return position(u * 2.0 - 1.0, v * 2.0 - 1.0)

    @fragment
    def shade(f, varying):
        y = varying.v * 23.0  # normalize into f's 24x40 space
        x = varying.u * 39.0
        return f(y, x)  # color0 — the SAME f as the compute zoo

    img = T(np.zeros((24, 40)), ("y", "x"))
    pso = pair(quad, shade)  # the PAIR is the artifact unit
    render(pso, circle(12.0, 20.0, 8.0), target=img)  # encodable under the hood
    assert img.to_numpy().min() == 0.0 and img.to_numpy().max() == 1.0


@P8
def test_subset_pairing_shares_one_fragment_artifact():
    """The boundary is a record TYPE and strings never cross it: the
    fragment's required record is INFERRED from the fields it touches;
    pairing checks produced ⊇ required — so two vertex shaders with
    different superset interfaces share ONE fragment artifact, and
    adding a varying breaks no existing pairing."""
    from pdum.dsl import events
    from pdum.tl.graphics import fragment, pair, position, vertex  # noqa: F821

    @vertex
    def lean():
        vid = vertex_index()  # noqa: F821
        u = 1.0 if vid == 1 else 0.0
        return position(u, u)

    @vertex
    def rich():
        vid = vertex_index()  # noqa: F821
        u = 1.0 if vid == 1 else 0.0
        w = u * 3.0  # an EXTRA varying: a superset interface  # noqa: F841
        return position(u, u)

    @fragment
    def shade(varying):
        return varying.u  # requires exactly {u} — inferred from use

    pair(lean, shade)
    with events.forbid("fragment.miss"):
        pair(rich, shade)  # superset producer: the same fragment artifact


@P8
def test_flat_is_the_sole_interpolation_annotation():
    """Interpolation is declared at the vertex claim site — perspective-
    correct by default, flat(...) the one opt-out — and is a production
    detail EXCLUDED from the interface type the fragment pairs against."""
    from pdum.tl.graphics import flat, position, vertex  # noqa: F821

    @vertex
    def vs():
        vid = vertex_index()  # noqa: F821
        u = 0.5  # perspective-corrected by default
        pick = flat(vid)  # noqa: F841 — provoking vertex's value, no interpolation
        return position(u, u)

    assert vs.varyings() is not None  # both sites listed; flat-ness not in the type


@P8
def test_fragment_taps_bind_render_buffers_mrt():
    """A claimed binding in the fragment — or inside the f it calls — binds
    to a second render target at the pass: MRT, G-buffers for free. The
    bound NAME SET specializes the pair; the buffers are invocation data."""
    from pdum.dsl import jit
    from pdum.tl.graphics import fragment, pair, position, render, vertex  # noqa: F821
    from pdum.tl.kernel import config

    def shaded(cy, cx, r):
        @jit()
        def go(y, x):
            dist = sqrt((y - cy) * (y - cy) + (x - cx) * (x - cx))  # noqa: F821
            return 1.0 if dist > r else 0.0

        return go

    @vertex
    def quad():
        vid = vertex_index()  # noqa: F821
        u = 1.0 if (vid == 1 or vid == 3 or vid == 4) else 0.0
        v = 1.0 if (vid == 2 or vid == 4 or vid == 5) else 0.0
        return position(u * 2.0 - 1.0, v * 2.0 - 1.0)

    @fragment
    def shade(f, varying):
        return f(varying.v * 23.0, varying.u * 39.0)

    img = T(np.zeros((24, 40)), ("y", "x"))
    gbuf = T(np.zeros((24, 40)), ("y", "x"))
    pso = pair(quad, shade)
    render(pso[config(taps={"dist": gbuf})], shaded(12.0, 20.0, 8.0), target=img)
    assert gbuf.to_numpy().max() > 0.0  # the f-interior distance field, second target


@P8
def test_textures_are_runtime_objects_recognized_not_tensors():
    """A texture is a proper RUNTIME object (the wgpu-py texture) that the
    type system RECOGNIZES as its own leaf kind — never a dressed-up
    tensor. It may be BUILT from a tensor (the upload door). v1: exactly
    ONE format — rgba8unorm-srgb over the existing FormatEncoding; the
    format registry is later work. Sampling takes the SAMPLER (the
    interpolation object); explicit-LOD mips first; auto-LOD is ANALYTIC
    (log2 footprint from the wrt-ambient gradient), no 2x2 quad."""
    from pdum.dsl import jit
    from pdum.tl.graphics import sample, sampler, upload  # noqa: F821

    src = T(np.zeros((16, 16)), ("y", "x"))
    tex = upload(src)  # a wgpu texture, rgba8unorm-srgb — NOT a Tensor
    assert not isinstance(tex, Tensor)  # recognized in the type system, never impersonated
    smp = sampler(filter="linear", address="clamp")

    def lookup(t, s):
        @jit()
        def go(y, x):
            return sample(t, s, (y / 16.0, x / 16.0), lod=0)

        return go

    assert lookup(tex, smp) is not None


# --- L4: shared memory — both committed forms --------------------------------


@L4
def test_shared_memory_config_linked_and_kernel_side_static():
    """Two spellings, one mechanism. CONFIG-LINKED: the launch declares the
    region; the kernel binds it by name. KERNEL-SIDE STATIC: the kernel
    declares a compile-time-shaped tile inline. Barriers are the token
    mechanism made explicit."""
    from pdum.tl.kernel import shared

    @compute
    def k_linked(img):
        t1 = shared_bind("t1")  # noqa: F821 — binds config(shared_mem=shared(t1=...))
        y, x = thread_idx("y", "x")
        t1[y, x] = img[y, x] * 2.0
        barrier()  # noqa: F821 — token-threaded synchronization
        img[y, x] = t1[y, x]

    @compute
    def k_static(img):
        tile = shared_alloc(ty=16, tx=16)  # noqa: F821 — statically typed, compile-time known
        y, x = thread_idx("y", "x")
        tile[y, x] = img[y, x]
        barrier()  # noqa: F821
        img[y, x] = tile[y, x]

    img = T(np.zeros((16, 16)), ("y", "x"))
    k_linked[config(shared_mem=shared(t1=(("ty", 16), ("tx", 16))))](img)
    k_static(img)


# --- P9: data-dependent indexing joins as take/scatter_add -------------------


@P9
def test_data_dependent_stores_are_scatter_add():
    """Kernel subscripts at NON-thread indices become the take/scatter_add
    pair (200 §1.9) — deterministic by addition, first-wins at ties."""
