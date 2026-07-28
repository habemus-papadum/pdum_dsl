"""THE SEAM LEDGER -- did runtime-vs-backend carve the code at its joints?

280's working definitions, under test:

    runtime = device management, host<->device transfer, launch,
              events/profiling
    backend = IR -> executable code, which a runtime launches

VERDICT, up front. The definitions are GOOD -- they carved cleanly
enough that `msl_backend.py` imports no Metal and `metal_runtime.py`
imports no pdum, and neither was awkward to write. But they are
INCOMPLETE in a specific, repeatable way: three things wanted to live in
both, one thing wanted to live in neither, and one thing turned out to
be co-owned by the pair rather than by either member. Those five are the
finding, and they are itemized as FAIL-1..FAIL-5 below.

The single most useful sentence this spike can offer 280: *the boundary
between backend and runtime is not a line, it is a NEGOTIATED CONTRACT,
and the contract has at least four clauses* -- the binding table, the
launch geometry, the guard/exactness convention, and the device
representation of the staging plan. A `Backend` that returns `str` and a
`Runtime` that takes `str` cannot express any of them. What the backend
must return is (source, launch-contract), and this spike's `meta` dict
is a crude first draft of exactly that.

The ledger below is DATA, not prose, so it can be checked against the
files. Run this module to print it.
"""

from __future__ import annotations

# --- (c) SHARED / target-neutral --------------------------------------------
# Written once conceptually, currently duplicated per backend. Each entry:
# (what, wgsl site, metal site, note)
SHARED = [
    (
        "region walking (op order, store token order)",
        "pdum.tl.dialect.walk_region",
        "same import",
        "genuinely shared already -- the one thing nobody copied",
    ),
    (
        "marker tables (_INFIX/_CMP/_FNS/_CORE_INFIX)",
        "wgsl_executor.py:45-48",
        "msl_backend.py:86-89",
        "BYTE-IDENTICAL dicts, copied. 3rd copy overall (_Gen is the 2nd).",
    ),
    (
        "the expression walker (go/operand/_expr, bool set, CSE by node id)",
        "wgsl_executor.py:103-192",
        "msl_backend.py:139-228",
        "structure identical; only leaf spellings differ (see rowdiff.py: 100%)",
    ),
    (
        "buffer index arithmetic (row-major strides over a param's own dims)",
        "wgsl_executor.py:87-101",
        "msl_backend.py:127-137",
        "pure lattice/layout math -- no target in it at all",
    ),
    (
        "writable determination (tensor_params vs writable, taps write)",
        "wgsl_executor.py:210-213",
        "msl_backend.py:253-258 + metal_executor.py:65",
        "artifact reasoning; both backends re-derive it",
    ),
    (
        "uniform slot collection + ordering (kernel uniforms + fn-arg blocks)",
        "wgsl_executor.py:76-79",
        "msl_backend.py:113-117",
        "the staging PLAN read; identical",
    ),
    (
        "staging unpack at launch (struct.unpack_from over meta['slots'])",
        "wgsl_executor.py:267",
        "metal_executor.py:88",
        "identical; 210 says the plan IS the ABI and neither invents it -- true here",
    ),
    (
        "launch geometry arithmetic (lattice extents -> thread counts)",
        "wgsl_executor.py:277-282",
        "metal_executor.py:93-95",
        "SHARED in principle; the WGSL copy FUSES it with workgroup rounding (FAIL-2)",
    ),
    (
        "host repack (to_numpy(order=) + ascontiguousarray f32)",
        "wgsl_executor.py:260",
        "metal_executor.py:71-72",
        "VERBATIM the same line. 89-100% of a launch (unified.py part 3).",
    ),
    (
        "writeback discipline (writable set -> Tensor.from_numpy -> _store)",
        "wgsl_executor.py:287-290",
        "metal_executor.py:104-112",
        "identical but for how bytes arrive",
    ),
    (
        "launch protocol: staging pack, fn-marker rebind, overlap refusal",
        "kernel.py:1082-1113",
        "same -- unmodified",
        "THE WIN: swapping the executor column reuses all of it, untouched",
    ),
    (
        "artifact/cache key discipline",
        "kernel.py:107,1191 (_EXECUTOR_FP, ARTIFACTS)",
        "msl_backend.py METAL_FP",
        "DECORATIVE on both sides -- see FAIL-5",
    ),
]

# --- (a) BACKEND: source emission, differs per target LANGUAGE ---------------
BACKEND = [
    ("float literal rendering", "_lit: '1.5'", "_lit: '1.5f' (msl_backend.py:230-238)"),
    ("numeric cast spelling", "f32(x) / i32(x)", "float(x) / int(x) (msl_backend.py:136,168)"),
    ("value declaration form", "let v: f32 = e;", "float v = e; (msl_backend.py:148-149)"),
    ("buffer declaration", "@group/@binding module-scope", "[[buffer(i)]] entry params (msl_backend.py:252-262)"),
    ("read-only spelling", "var<storage, read>", "const device (msl_backend.py:259)"),
    ("entry point", "@compute fn main(...)", "kernel void main0(...) (msl_backend.py:276)"),
    ("thread id builtin", "@builtin(global_invocation_id) vec3<u32>", "[[thread_position_in_grid]] uint3"),
    ("preamble", "(none)", "#include <metal_stdlib>; using namespace metal;"),
]

# --- (b) RUNTIME: device/queue/buffers/dispatch/sync, differs per API --------
RUNTIME = [
    ("device acquisition", "graphics.py:718-733 (singleton, no options)", "metal_runtime.py:93-101 (constructible)"),
    ("source -> pipeline", "create_shader_module + create_compute_pipeline", "metal_runtime.py:111-132"),
    ("buffer alloc + upload", "create_buffer_with_data", "metal_runtime.py:134-141"),
    ("ZERO-COPY adoption", "NO EQUIVALENT", "metal_runtime.py:143-154 (newBufferWithBytesNoCopy:)"),
    ("resource binding", "create_bind_group vs a pipeline layout", "setBuffer:offset:atIndex: -- no bind-group OBJECT"),
    ("encode + dispatch", "encoder/pass/dispatch_workgroups", "metal_runtime.py:176-208"),
    ("submit + sync", "queue.submit (+ implicit wait in read_buffer)", "commit + waitUntilCompleted"),
    ("readback", "queue.read_buffer -- submit/map/wait round trip",
     "metal_runtime.py:160-174 -- a VIEW; none needed under adoption"),
    ("GPU timing", "timestamp queries; needs a feature at device creation",
     "metal_runtime.py:210-215 -- GPUStartTime/GPUEndTime, free"),
]

# --- WHERE THE DEFINITIONS FAILED TO CARVE ----------------------------------
FAILURES = [
    (
        "FAIL-1",
        "Workgroup size is BACKEND in WGSL and RUNTIME in Metal.",
        "In WGSL `@workgroup_size(8,8,1)` is shader text -- backend output, and "
        "210 records the rule as 'workgroup size is pipeline-creation-time; "
        "dispatch dimensions are launcher data'. In Metal the threadgroup size "
        "is an argument to dispatchThreadgroups: and appears nowhere in the "
        "source. The SAME decision sits on opposite sides of the line. This "
        "spike had to route it through `meta['threadgroup']` "
        "(msl_backend.py:264-267,286-294) because a backend that returns only "
        "`str` cannot express Metal. 210's sentence is WebGPU-shaped, not general.",
    ),
    (
        "FAIL-2",
        "The bounds guard is emitted SOURCE whose necessity is a RUNTIME choice.",
        "`if (gid.x >= N) return;` exists only because WebGPU dispatches whole "
        "workgroups, so the last one overhangs. Metal's dispatchThreads: "
        "(non-uniform threadgroups) launches the exact grid and makes the guard "
        "dead code. Proven, not asserted: `exact_grid=True` omits the guard, and "
        "differential.py reports guard-vs-exact BITWISE EQUAL on all 11 subjects. "
        "So a line of backend output is or is not needed depending on which "
        "runtime entry point the launcher will call. Neither side can decide it "
        "alone -- it is a clause in a contract between them. Relatedly, the WGSL "
        "executor FUSES 'extents -> threads' (shared, target-neutral) with "
        "'threads -> workgroups' (runtime) at wgsl_executor.py:277-282; splitting "
        "them (metal_executor.py:93-95) is what made this visible.",
    ),
    (
        "FAIL-3",
        "The binding table is co-owned, and WebGPU makes the backend know a "
        "runtime rule.",
        "wgsl_executor.py:215 reads: `layout='auto' prunes unused bindings: U "
        "exists only when slots do`. That is BACKEND code encoding a WebGPU "
        "pipeline-layout inference rule -- emit a binding the shader doesn't "
        "read and the runtime's bind group won't match. Metal has no bind-group "
        "object at all: buffers are set by index, and an unused argument is "
        "harmless. So 'resource binding' is a joint backend/runtime concern in "
        "WebGPU and a pure runtime concern in Metal. Any Backend interface must "
        "therefore return the binding table as DATA (this spike: `meta['slots']`, "
        "`meta['n_params']`), never leave it implicit in the text.",
    ),
    (
        "FAIL-4",
        "The device representation of the staging plan belongs to NEITHER, so "
        "both invent it.",
        "210 is explicit: 'the plan IS the ABI; both renderer and launcher read "
        "it, neither invents layout.' In fact the plan describes HOST staging "
        "bytes (offset, struct fmt) and says nothing about the device side, so "
        "each backend independently decides the uniforms arrive as a flat f32 "
        "array (wgsl_executor.py:216 `array<f32>`; msl_backend.py:262 `const "
        "device float*`) -- and both silently narrow i32/i64 slots to f32 en "
        "route (wgsl_executor.py:269, metal_executor.py:88 both "
        "`asarray(uvals, dtype=np.float32)`), which 210 forbids: 'narrowing is "
        "declared, never silent'. This is a hole in the SHARED tier, not a "
        "backend or runtime concern, and it is where spike_runner's H3 "
        "(one staging-plan object) actually bites. With a third backend the "
        "invention has now happened three times.",
    ),
    (
        "FAIL-5",
        "The executor column has no cache key, so a third backend makes the "
        "existing collision concrete.",
        "kernel.py:1191 keys compiled executors as `(region.key, "
        "_EXECUTOR_FP)`. But `wgpu_artifact` and `metal_artifact` are both just "
        "`replace(art, executor=compile_X(art))` -- they never consult that "
        "cache, so `WGPU_FP` (wgsl_executor.py:38, declared and never used) and "
        "`METAL_FP` are decorative. VERIFIED: two `wgpu_artifact(art)` calls "
        "return different executor objects (a full recompile each time, which "
        "spike_runner measured at 8-9 ms per distinct pipeline), and the WGSL "
        "and MSL artifacts share an identical `region.key`, so if they DID key "
        "by `(region.key, _EXECUTOR_FP)` today they would collide. Backend "
        "identity must be part of the content key before a second device "
        "backend ships -- which is now.",
    ),
    (
        "FAIL-0",
        "And one that is neither backend nor runtime nor shared: the TARGET "
        "NUMERIC CONTRACT.",
        "Metal's `tanh` returns NaN for |x| >= 44.36 (= log(FLT_MAX)/2; it is "
        "computed from exp(2x), and MTLMathModeSafe does NOT fix it -- tested). "
        "The translation row is correct in both languages, the runtime never "
        "touches arithmetic, and the reference says tanh saturates to 1.0. So "
        "the disagreement lives in the target's MATH LIBRARY and has no code "
        "home under either definition. 210 has a numeric-policy section and it "
        "is the right place, but nothing in the backend/runtime split owns "
        "enforcing it. See differential.py subject `tanh_wide`.",
    ),
]

# --- the user-visible program diff ------------------------------------------
PROGRAM_DIFF = """
TODAY, test-side, the entire diff between 'run on WebGPU' and 'run on Metal':

    -from wgsl_executor import wgpu_artifact
    +from metal_executor import metal_artifact
    ...
    -wgpu_artifact(art).launch(dev_args, {})
    +metal_artifact(art).launch(dev_args, {})

ONE IDENTIFIER. Kernel source, arguments, staging, uniform capture,
fn-argument splicing, the overlap refusal, the writeback -- all identical
and all reused unmodified from kernel.py:1082-1113. That is the strongest
positive result in this spike: the executor column really is the seam,
and it held for a target whose memory model, binding model, dispatch
model and source language all differ.

But `metal_artifact(art)` is a TEST-SIDE spelling, and it should not be
the user-facing one, for two reasons: it needs an `_Artifact`, which no
public API hands out (spike_runner H2 -- `_invoke` keys, compiles AND
launches, so the artifact never escapes), and it reads as a conversion
function, which is exactly the 'magic compile function' 270 refuses.

PROPOSED user-facing spelling, per no-magic (270) and the existing
bracket precedent (`kernel[config(taps=...)]` is already the committed
way to say launch-time policy):

    from pdum.tl.runtime import metal, webgpu       # explicit, importable

    kernel[on(metal)](src, dst)                     # per-launch selection
    art = kernel.artifact(src, dst)                 # the missing public door
    art.on(metal).launch((src, dst))                # 270's incremental chain

Three properties that fall out of what this spike actually built:

  1. `on(...)` must KEY the artifact cache -- `(region.key, backend_fp,
     runtime_fp)`, not today's `(region.key, _EXECUTOR_FP)` (FAIL-5).
  2. `on(...)` names a PAIR, not one thing. This spike's Metal path is
     MSL-the-backend plus Metal-the-runtime, and they are genuinely
     separable (the same MSL text would serve a metal-cpp harness, and
     the same Metal runtime would launch a precompiled .metallib). The
     1:1 assumption in 280's 'usually 1:1' held here, but only because
     we wrote both halves; the spelling should not bake it in.
  3. Nothing in `on(...)` should imply a transfer, because on this
     device there isn't one. `Buffer.device` being a bare string label
     with no registry (spike_runner) stops being cosmetic the moment two
     real runtimes exist: with adoption, a tensor's memory can be
     simultaneously host memory and Metal device memory, and 'which
     device is this on' becomes a genuine question the type has no way
     to answer.
"""


def main():
    print(__doc__)
    print("=" * 78)
    print("(c) SHARED / TARGET-NEUTRAL -- written once in principle, copied today")
    print("=" * 78)
    for what, w, m, note in SHARED:
        print(f"\n  {what}")
        print(f"      wgsl : {w}")
        print(f"      metal: {m}")
        print(f"      note : {note}")

    print("\n" + "=" * 78)
    print("(a) BACKEND -- source emission, differs per target LANGUAGE")
    print("=" * 78)
    for what, w, m in BACKEND:
        print(f"\n  {what}\n      wgsl : {w}\n      msl  : {m}")

    print("\n" + "=" * 78)
    print("(b) RUNTIME -- device/queue/buffers/dispatch/sync, differs per API")
    print("=" * 78)
    for what, w, m in RUNTIME:
        print(f"\n  {what}\n      wgpu : {w}\n      metal: {m}")

    print("\n" + "=" * 78)
    print("WHERE THE DEFINITIONS FAILED TO CARVE")
    print("=" * 78)
    for tag, headline, body in FAILURES:
        print(f"\n{tag}: {headline}")
        for line in _wrap(body, 74):
            print("    " + line)

    print("\n" + "=" * 78)
    print("THE USER-VISIBLE PROGRAM DIFF")
    print("=" * 78)
    print(PROGRAM_DIFF)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(text.split()), width)


if __name__ == "__main__":
    main()
