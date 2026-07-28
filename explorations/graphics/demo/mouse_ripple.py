"""mouse_ripple -- one program, two GPU backends, a mouse in the loop.

    python mouse_ripple.py                        # a window, WebGPU
    python mouse_ripple.py --on metal             # a window, Metal
    python mouse_ripple.py --offscreen 8          # headless: both backends, verified

A spinning rippled cylinder that the CURSOR drives. Five scalars move
with the mouse and every one of them takes the same road to the GPU:

    mouse x  ->  the rotation angle          (a VERTEX-stage capture)
    mouse y  ->  the ripple phase and depth  (two COMPUTE-stage captures)
    mouse xy ->  a spotlight in clip space   (two FRAGMENT-stage captures)

That road is the CAPTURED-SCALAR channel. A plain Python float read by a
shader body becomes an ``abi.slot`` in the lowered IR and four bytes in a
staging buffer -- so moving the mouse rewrites 28 bytes across three
buffers and touches nothing else. The program is lowered once,
translated once per target, and compiled into one pipeline per target
format; every frame after that is refresh-bind-encode. The demo ASSERTS
this rather than claiming it: ``--offscreen`` fails if a single extra
lowering, translation or pipeline happens during the animated run.

Read in this order:

  §1  THE PROGRAM     the part a user writes -- mesh, kernel, shaders
  §2  THE BACKEND     `--on webgpu` / `--on metal`, a preview of the
                      ruled user-facing spelling
  §3  THE FRAME       read mouse, refresh slots, encode, present
  §4  THE MODES       windowed, and headless with the verifications

The mechanical parts live next door so this file stays a story:
``program.py`` (target-neutral: residency, the shared expression
generator, the one-time lowering, the slot channel), ``wgsl_glue.py``
and ``msl_glue.py`` (one shell and one runtime each), ``metal_window.py``
(AppKit + CAMetalLayer), ``pngw.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _paths  # noqa: F401,E402 -- puts this worktree's packages on sys.path
import numpy as np  # noqa: E402
import program as P  # noqa: E402
import verify  # noqa: E402
from pdum.dsl import jit, value_and_grad  # noqa: E402
from pdum.dsl.intrinsics import clamp  # noqa: F401,E402 -- inlines by capture-and-call
from pdum.dsl.markers import cos, exp, sin, sqrt  # noqa: F401,E402 -- bare in shader bodies
from pdum.tl import Tensor, compute, global_idx, thread_idx  # noqa: F401,E402 -- bodies' globals
from pdum.tl.graphics import fragment, pair, position, vertex  # noqa: E402
from pngw import write_gray  # noqa: E402

TAU = 6.283185307179586
SEGMENTS = 16
SIZE = (64, 96)  # (H, W) for the headless target; the window picks its own
ATOL = 2e-3  # conformance/test_conformance_render.py's cylinder golden


# ===========================================================================
# §1  THE PROGRAM
# ===========================================================================

# The five mouse-driven scalars. They are ordinary module globals: a
# shader body that reads one captures it UNMARKED, which is the whole
# uniform doctrine -- no annotation, no parameter, no buffer. Rebinding
# one before a frame is a warm relaunch with a fresh value, not a new
# program. (The rotation angle is the exception and takes the other door;
# see `spun` below.)
_PHASE = 0.0  # ripple phase   <- mouse y
_AMP = 0.08  # ripple depth   <- mouse y
_MX = 0.0  # spotlight x    <- mouse x, in NDC
_MY = 0.0  # spotlight y    <- mouse y, in NDC

MESH_DT = np.dtype([("theta", "<f8"), ("h", "<f8")])


def cylinder_mesh(segments: int = SEGMENTS) -> Tensor:
    """The side surface as a RECORD vertex buffer: two triangles per
    segment, each element a (theta, h) record. The structured dtype IS
    the memory shape -- fields by name, no anonymous columns, one host
    buffer."""
    verts = []
    for s in range(segments):
        t0, t1 = s * TAU / segments, (s + 1) * TAU / segments
        verts += [(t0, -1.0), (t1, -1.0), (t0, 1.0), (t1, -1.0), (t1, 1.0), (t0, 1.0)]
    return Tensor.from_numpy(np.array(verts, dtype=MESH_DT), ("vertex_id",))


@compute
def ripple(theta_in, h_in, theta_out, h_out):
    """The geometry ripple, over the record's FIELD VIEWS.

    Four arguments, two host buffers: ``mesh.field("theta")`` and
    ``mesh.field("h")`` are two strided views of ONE allocation, and so
    are the outputs. The residency layer keeps them that way all the way
    to the device -- one buffer, one binding, indexed at its real stride
    -- so no repack happens anywhere in this program.
    """
    (i,) = global_idx("vertex_id")
    theta_out[i] = theta_in[i]
    h_out[i] = h_in[i] + sin(theta_in[i] * 3.0 + _PHASE) * _AMP


def spun(angle: float):
    """The rotation as a uniform: the world transform is built IN the
    vertex shader from the captured angle, never as a host-side matrix.

    ``spun`` mints a fresh closure per call, so the frame loop hands the
    pipeline a DIFFERENT function object every frame. That is safe and it
    is the interesting case: the angle rides the slot channel and never
    enters the IR, so one compiled pipeline serves every angle -- but
    nothing in the library checks that the new closure is layout-
    compatible with the compiled one, so ``P.check_swap`` does.
    """

    @vertex
    def vs(verts):
        (vid,) = thread_idx("vertex_id")
        p = verts[vid]  # the record element: fields by NAME
        world = p.theta + angle  # rotation -> world space, in-shader
        px = sin(world) * 0.85
        py = p.h * 0.7 + cos(world) * 0.12  # a hint of tilt
        u = p.theta * (1.0 / TAU)  # noqa: F841 -- a claimed varying
        v = p.h * 0.5 + 0.5  # noqa: F841 -- a claimed varying
        return position(px, py)

    return vs


def stripes(k: float):
    @jit()
    def g(u, v):
        return sin(u * k)  # periodic in u -- continuous across the seam by construction

    return g


G = value_and_grad(stripes(3.0 * TAU), wrt=("u", "v"))


@fragment
def shade(f, varying):
    """Analytic antialiasing plus a cursor spotlight.

    ``f`` is the striped pattern paired with its own gradient
    (``value_and_grad``), so the edge is one screen-footprint wide with
    no 2x2 quad and no ``fwidth``.

    ``varying.px`` and ``varying.py`` are free: EVERY binding in a vertex
    body is a claimed varying (the tagless law), so the clip-space
    position the vertex shader computed for ``position(px, py)`` is
    already interpolated and available here. The spotlight is just the
    distance from that to the cursor -- no extra plumbing, no second
    uniform block, nothing added to the vertex shader.
    """
    s, (du, dv) = f(varying.u, varying.v)
    w = sqrt(du * du + dv * dv) * 0.7 + 1e-6
    base = clamp(s / w * 0.5 + 0.5, 0.0, 1.0)  # the analytic edge
    dx = varying.px - _MX
    dy = varying.py - _MY
    spot = exp(-(dx * dx + dy * dy) * 6.0)
    return clamp(base * (0.45 + 0.85 * spot) + 0.18 * spot, 0.0, 1.0)


@dataclass
class Mouse:
    """The canonical cursor: (u, v) in the unit square, v = 0 at the TOP.

    Every window backend converts into this -- rendercanvas reports
    top-left-origin pixels, AppKit reports bottom-left-origin points --
    so §3 below never learns which windowing system it is running under.

    The ``float()`` coercions are NOT decoration. A captured scalar's
    environment fingerprint includes its Python TYPE, and ``float`` and
    ``numpy.float64`` are different fingerprints -- so a cursor whose
    coordinates went through numpy would key a different artifact and
    silently recompile the pipeline every frame. The first version of
    this demo did exactly that, and ``P.check_swap`` is what found it.
    That is a sharp edge worth naming: nothing warns you, and without
    the guard the only symptom is that the frame loop stops being warm.
    """

    u: float = 0.5
    v: float = 0.5

    def __post_init__(self) -> None:
        self.set(self.u, self.v)

    def set(self, u: float, v: float) -> None:
        self.u = float(min(max(u, 0.0), 1.0))
        self.v = float(min(max(v, 0.0), 1.0))


def bind_mouse(mouse: Mouse):
    """Push the cursor into the five captured scalars and return the
    vertex shader for this frame. This is the ENTIRE per-frame user-level
    act: four global assignments and one closure."""
    g = globals()
    g["_PHASE"] = mouse.v * TAU * 1.5
    g["_AMP"] = 0.02 + mouse.v * 0.20
    g["_MX"] = mouse.u * 2.0 - 1.0
    g["_MY"] = 1.0 - mouse.v * 2.0
    return spun((mouse.u - 0.5) * TAU).fn


def build_program():
    """The scene, lowered ONCE. Everything below this line is a backend's
    business; nothing above it mentions a device."""
    mesh = cylinder_mesh()
    n = mesh.layout.dim("vertex_id").size
    rippled = Tensor.from_numpy(np.zeros(n, dtype=MESH_DT), ("vertex_id",))
    args = (mesh.field("theta"), mesh.field("h"), rippled.field("theta"), rippled.field("h"))
    pso = pair(spun(0.0), shade)
    low = P.lower_program(
        ripple,
        args,
        labels=("mesh.theta", "mesh.h", "rippled.theta", "rippled.h"),
        pso=pso,
        vs_args=(rippled,),
        fs_args=(G,),
    )
    return mesh, low


# ===========================================================================
# §2  THE BACKEND -- a preview of the ruled user-facing spelling
# ===========================================================================
#
# 282 §7 rules that selecting a target names a PAIR: a GENERATOR (the
# language a program is written into) and a RUNTIME (the API that
# compiles and launches it). They are separable -- the same MSL would
# serve a metal-cpp harness, the same Metal runtime would launch a
# precompiled .metallib -- and they are 1:1 here only because we wrote
# both halves of both columns.
#
# These classes are DELIBERATELY EMPTY. They are a preview of the shape,
# not the shape: the real thing lands in the package API after the L4
# stratification merge, where a pair also has to key the artifact cache
# (spike_metal FAIL-5: `wgpu_artifact` and `metal_artifact` share an
# identical region key today and would collide the moment either of them
# started caching). Nothing here implies a transfer, either -- on unified
# memory a tensor's bytes are host memory and Metal device memory at the
# same time.


@dataclass(frozen=True)
class Generator:
    """A program -> shader source translator."""


@dataclass(frozen=True)
class Runtime:
    """A device API: compile, allocate, bind, encode, submit."""


@dataclass(frozen=True)
class WgslGenerator(Generator):
    pass


@dataclass(frozen=True)
class MslGenerator(Generator):
    pass


@dataclass(frozen=True)
class WgpuRuntime(Runtime):
    pass


@dataclass(frozen=True)
class MetalRuntime(Runtime):
    pass


@dataclass(frozen=True)
class Pair:
    generator: Generator
    runtime: Runtime


webgpu = Pair(WgslGenerator(), WgpuRuntime())
metal = Pair(MslGenerator(), MetalRuntime())

BY_NAME = {"webgpu": webgpu, "metal": metal}


def glue_for(p: Pair):
    """Today's stand-in for what the pair will eventually carry itself."""
    import msl_glue
    import wgsl_glue

    return {webgpu: wgsl_glue, metal: msl_glue}[p]


# ===========================================================================
# §3  THE FRAME
# ===========================================================================


def frame(engine, mouse: Mouse) -> None:
    """Read the mouse, refresh the slot channel, and that is the update.

    ``engine.update`` re-extracts the captured scalars from their live
    closures and globals, packs them through the staging plan, and writes
    them: 3 slot buffers, 28 bytes total for this program. No lowering,
    no translation, no pipeline, no geometry upload. The caller then
    encodes -- into an offscreen attachment or a drawable, same method.

    Note ``circular_mouse`` builds the cursor from ``np.cos``; ``Mouse``
    coerces back to ``float`` for the reason in its docstring.
    """
    engine.update(bind_mouse(mouse))


def circular_mouse(i: int, n: int) -> Mouse:
    """The headless cursor: a circle, so every frame moves all five
    scalars and no two frames are alike."""
    t = TAU * i / max(n, 1)
    return Mouse(0.5 + 0.34 * np.cos(t), 0.5 + 0.34 * np.sin(t))


# ===========================================================================
# §4  THE MODES
# ===========================================================================


def reference_frame(mouse: Mouse, size=SIZE) -> np.ndarray:
    """The numpy reference render of one frame -- the golden.

    Deliberately shares nothing with the device path: a fresh mesh, the
    ordinary kernel launch through the numpy executor, and
    ``graphics.render``'s reference rasterizer. It re-lowers both shader
    stages on every call, which is exactly what the frame loop exists not
    to do.
    """
    from pdum.tl.graphics import render

    vs_fn_angle = (mouse.u - 0.5) * TAU
    bind_mouse(mouse)
    mesh = cylinder_mesh()
    n = mesh.layout.dim("vertex_id").size
    rippled = Tensor.from_numpy(np.zeros(n, dtype=MESH_DT), ("vertex_id",))
    ripple(mesh.field("theta"), mesh.field("h"), rippled.field("theta"), rippled.field("h"))
    img = Tensor.from_numpy(np.zeros(size), ("y", "x"))
    render(pair(spun(vs_fn_angle), shade), rippled, G, target=img)
    return img.to_numpy()


def render_sequence(p: Pair, low, n_frames: int, outdir: Path):
    """Build once, animate n_frames offscreen, save PNGs. Returns the
    frames and the compile-once evidence."""
    glue = glue_for(p)
    name = p.generator.__class__.__name__.replace("Generator", "").lower()
    engine = glue.Engine(low, "r32float")
    off = engine.offscreen(SIZE)

    before = (len(P.LOWERINGS), len(P.TRANSLATIONS), len(glue.PIPELINES))
    frames = []
    for i in range(n_frames):
        mouse = circular_mouse(i, n_frames)
        frame(engine, mouse)
        frames.append(off.draw(engine))
        write_gray(outdir / f"{name}_{i:02d}.png", np.flipud(frames[-1]))
    # A hero shot from the SAME engine into a 9x larger attachment.
    # Resolution never entered the shader, the pipeline or the artifact
    # key, so this needs no second anything -- one more Offscreen and one
    # more encode.
    big = engine.offscreen((192, 288))
    frame(engine, circular_mouse(3, 8))
    write_gray(outdir / f"{name}_hero_192x288.png", np.flipud(big.draw(engine)))

    after = (len(P.LOWERINGS), len(P.TRANSLATIONS), len(glue.PIPELINES))

    # THE ASSERTION. A whole animated run with five scalars changing every
    # frame -- and a 9x resolution change -- must not lower, translate or
    # compile anything.
    assert before == after, f"{name}: the frame loop recompiled: {before} -> {after}"
    return engine, frames, before


def check_window_variant(p: Pair, low) -> str:
    """The windowed pipeline is a SECOND compiled artifact of the same
    program: the fragment's color0 is a scalar, so the 1-to-4 channel
    surface expansion is baked into the generated source and
    bgra8unorm-srgb is genuinely not r32float. Resolution, by contrast,
    keys nothing -- which is why a resizable window is free.

    Headless, we can still compile that variant and (on Metal) run a real
    CAMetalLayer drawable through the real encode path."""
    glue = glue_for(p)
    engine = glue.Engine(low, "bgra8unorm-srgb")
    frame(engine, Mouse(0.7, 0.35))
    if p is metal:
        from metal_window import make_layer

        layer = make_layer(engine.rt.device, 320, 240, glue._PIXEL["bgra8unorm-srgb"], scale=1.0)
        drawable = layer.nextDrawable()
        if drawable is None:
            return "srgb pipeline compiled; CAMetalLayer gave no drawable (headless)"
        cmd = engine.rt.queue.commandBuffer()
        engine.encode(cmd, drawable.texture(), clear=(0.02, 0.02, 0.03, 1.0))
        cmd.commit()
        cmd.waitUntilCompleted()
        return "srgb pipeline compiled; a real CAMetalLayer drawable took a real encoded frame"
    import wgpu

    tex = engine.device.create_texture(
        size=(320, 240, 1),
        format="bgra8unorm-srgb",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    enc = engine.device.create_command_encoder()
    engine.encode(enc, tex.create_view(), clear=(0.02, 0.02, 0.03, 1.0))
    engine.device.queue.submit([enc.finish()])
    return "srgb pipeline compiled and encoded into a bgra8unorm-srgb attachment"


def render_wgpu_frame(mouse: Mouse, size=SIZE):
    """The SAME frame through conformance/wgsl_executor.render_wgpu -- the
    existing one-shot device path, which re-lowers and re-compiles per
    call and repacks the record buffer into a contiguous WGSL struct
    array. If it misses the reference on the same pixels by the same
    amount, the tolerance is a property of the scene's AA ramp and not of
    anything this demo does differently.

    The ripple runs on the DEVICE here too. Running it on the host
    instead (the ordinary numpy launch) makes this frame differ from the
    demo's by ~1.8e-07 -- f64 ``sin`` narrowed to f32 is not f32 ``sin``
    -- which is a real and easily-missed way to make a "device golden"
    that is quietly a host golden."""
    from wgsl_executor import render_wgpu, wgpu_artifact

    bind_mouse(mouse)
    mesh = cylinder_mesh()
    n = mesh.layout.dim("vertex_id").size
    rippled = Tensor.from_numpy(np.zeros(n, dtype=MESH_DT), ("vertex_id",))
    args = (mesh.field("theta"), mesh.field("h"), rippled.field("theta"), rippled.field("h"))
    wgpu_artifact(P.artifact_of(ripple, *args)).launch(args)
    return render_wgpu(pair(spun((mouse.u - 0.5) * TAU), shade), rippled, G, shape=size)


def run_offscreen(n_frames: int, only: Pair | None, dump: bool) -> int:
    outdir = HERE / "frames"
    outdir.mkdir(exist_ok=True)
    pairs = [only] if only else [webgpu, metal]
    _mesh, low = build_program()
    print(f"lowered once: {P.LOWERINGS} | draw {low.draw_count} vertices | varyings {list(low.varyings)}")
    print(f"slot channel: compute {len(low.art.uniforms)}, vertex {len(low.v_slots)}, fragment {len(low.f_slots)}")

    results, engines = {}, {}
    for p in pairs:
        name = "webgpu" if p is webgpu else "metal"
        engine, frames, _ = render_sequence(p, low, n_frames, outdir)
        results[name], engines[name] = frames, engine
        glue = glue_for(p)
        print(f"\n[{name}] {len(glue.PIPELINES)} pipelines built, {n_frames} frames drawn, 0 rebuilt in the loop")
        print(f"[{name}] slot bytes written per frame: {engine.slot_bytes}")
        print(f"[{name}] resident buffers: {[r.label for r in engine.res.residents()]}")
        print(f"[{name}] windowed variant: {check_window_variant(p, low)}")
        if dump:
            print(f"\n----- {name} compute -----\n{engine.c_source}")
            print(f"\n----- {name} render -----\n{engine.r_source}")

    # The three comparisons, and what each one's bar means, are in
    # verify.py -- the bars differ and that distinction matters more than
    # the numbers do.
    verify.against_reference(results, lambda i: reference_frame(circular_mouse(i, n_frames)), n_frames, ATOL)
    m0 = circular_mouse(0, n_frames)
    verify.against_one_shot(results, reference_frame(m0), render_wgpu_frame(m0), ATOL)
    verify.cross_backend(results)
    verify.source_diff(engines)

    distinct = {name: len({f.tobytes() for f in fs}) for name, fs in results.items()}
    print(f"\ndistinct frames per backend: {distinct} (the mouse is actually moving the image)")
    print(f"lowerings for the WHOLE run: {len(P.LOWERINGS)}")
    print(f"translations: {P.TRANSLATIONS}")
    print("  (two per backend are the frame loop's; two more are the bgra8unorm-srgb")
    print("   windowed variant, which is a genuinely different compiled artifact)")
    print(f"PNGs in {outdir}")
    return 0


def run_window(p: Pair, width: int, height: int) -> int:
    _mesh, low = build_program()
    mouse = Mouse()
    glue = glue_for(p)
    glue.run_window(low, mouse, lambda: bind_mouse(mouse), width=width, height=height,
                    title="pdum -- mouse_ripple")
    return 0


def main(argv) -> int:
    on, n_off, dump, size = None, 0, False, (900, 600)
    args = list(argv)
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--on":
            i += 1
            on = BY_NAME[args[i]]
        elif a.startswith("--on="):
            on = BY_NAME[a.split("=", 1)[1]]
        elif a == "--offscreen":
            i += 1
            n_off = int(args[i])
        elif a.startswith("--offscreen="):
            n_off = int(a.split("=", 1)[1])
        elif a == "--dump":
            dump = True
        elif a.startswith("--size="):
            w, h = a.split("=", 1)[1].split("x")
            size = (int(w), int(h))
        else:
            print(__doc__)
            return 2
        i += 1
    if n_off:
        return run_offscreen(n_off, on, dump)
    return run_window(on or webgpu, *size)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
