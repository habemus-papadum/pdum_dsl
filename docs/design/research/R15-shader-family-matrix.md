# R15 — Shader-family feature matrix (WGSL / CUDA / MSL / Python twin)

*Research agent survey, 2026-07-12 backend detour. Tests the two-tier
dialect hypothesis: shader-core → compute-family → per-target packs, with
fragment-only forced out as its own family. Verdict: hypothesis holds.
Consumed by `070_backends-notes.md`.*

## 1. Feature matrix

| # | Feature | WGSL | CUDA C++ | MSL | Py-twin |
|---|---|---|---|---|---|
| 1a | Vector types | `vec2/3/4<f32,i32,u32,bool>`; `vecNf/i/u` aliases | `float2/3/4`, `int4`, `uint4`… structs; NO operators (need `helper_math.h`); float4 = 16B aligned | `float2/3/4`, `half4`, `int4`…; full operators; also `packed_float3` | Class w/ `__getattr__`; numpy-backed; easy |
| 1b | Construction | `vec3f(x,y,z)`, splat `vec3f(1.0)`, from-vec `vec4f(v.xyz, 1)` | `make_float4(x,y,z,w)` only; no splat/composite ctors | Full GLSL-style ctors incl. splat + mixed vec/scalar | Emulate WGSL ctor set |
| 1c | Read swizzle | Full (`v.xyz`, `v.wzyx`, rgba) — CONFIRMED | NONE — `.x/.y/.z/.w` only; no multi-swizzle — CONFIRMED (intentional, per NVIDIA) | Full swizzles — CONFIRMED | `__getattr__` trivial |
| 1d | Write swizzle | NOT allowed (`v.xz = …` illegal in WGSL 1.0; single `.x =` OK) — CONFIRMED | No | YES incl. write masks (`v.xw = …`), C++ ext | `__setattr__` trivial — must be restricted to the intersection |
| 2 | Matrices | `mat2x2..4x4<f32>` native (f16 w/ ext); col-major; `m*v`, `v*m`, `m*m` | NONE native — CONFIRMED. Ecosystem: hand-rolled structs, glm, CUTLASS; Warp emits its own `mat_t` template | `float2x2..4x4`, `half` variants; col-major cols ctor; `simdgroup_float8x8` for MMA | Emit own struct on CUDA (Warp precedent); numpy col-major twin |
| 3 | Shared/workgroup mem | `var<workgroup>`; size = const OR `override` expr (pipeline-creation time; override-sized arrays legal ONLY in workgroup space) — CONFIRMED; NO dispatch-time bytes | static `__shared__` + `extern __shared__ []` dynamic bytes as 3rd `<<<>>>` launch arg | static `threadgroup` decls + dynamic: kernel param `[[threadgroup(n)]]` sized by `setThreadgroupMemoryLength` at encode | Plain arrays per virtual workgroup |
| 4 | Barriers | `workgroupBarrier()`, `storageBarrier()`, `textureBarrier()` — workgroup-scoped; no grid sync | `__syncthreads()` (+and/or/count), `__syncwarp(mask)`, fences `__threadfence[_block/_system]`, coop-groups `grid.sync()` | `threadgroup_barrier(mem_flags::…)`, `simdgroup_barrier` | Phase-split execution (run all lanes to barrier, continue) |
| 5 | Atomics | `atomic<i32/u32>` ONLY — CONFIRMED; storage+workgroup; load/store/add/sub/min/max/and/or/xor/xchg/CAS-weak; single implicit relaxed order | Rich: int/uint/ull/float/double/half2/bf162 add; min/max incl. ull; scopes `_block`/`_system`; shared+global | `atomic_int/uint/bool` (ulong min/max MSL 2.4; `atomic_float` load/store/add/sub/xchg MSL 3.0); `memory_order_relaxed` ONLY — CONFIRMED; device+threadgroup | Sequential exec ⇒ atomics = plain ops |
| 6 | Subgroup/warp/simd | `subgroups` extension — shipped Chrome 134 (2025); wgpu `SUBGROUP`/`SUBGROUP_BARRIER`; ops: add/mul/min/max/and/or/xor, incl/excl prefix add+mul, ballot, all/any, elect, broadcast(First), shuffle/up/down/xor; NO masks; `subgroup_size` runtime-varying (4–128) | `__shfl_sync/_up/_down/_xor`, `__ballot_sync`, `__all/any_sync`, `__reduce_*_sync` (sm_80); EXPLICIT mask required (Volta+); warp = 32 | `simd_shuffle(_up/_down/_xor)`, `simd_broadcast(_first)`, `simd_sum/product/min/max/and/or/xor`, prefix incl/excl, `simd_ballot`, `simd_all/any`, `simd_is_first`; + quad ops; width 32 (Apple), 32/64 (AMD) | Lane-loop + numpy reduction over virtual subgroup; fixed twin width (e.g. 32) |
| 7a | Compute builtins | `global_invocation_id`, `local_invocation_id/_index`, `workgroup_id`, `num_workgroups`, `subgroup_invocation_id/_size` | `threadIdx`, `blockIdx`, `blockDim`, `gridDim`; global id COMPUTED | `thread_position_in_grid`, `thread_position_in_threadgroup`, `thread_index_in_threadgroup`, `threadgroup_position_in_grid`, `threadgroups_per_grid`, `threads_per_threadgroup`, `thread_index_in_simdgroup` | Loop indices of the virtual grid |
| 7b | Fragment builtins | `@builtin(position)`, `front_facing`, `sample_index/mask`, `frag_depth` out; `discard`; `dpdx/dpdy/fwidth` (+Coarse/Fine); `textureSample*` fragment-only, uniform control flow | N/A — no raster pipeline (`tex2D` exists but no implicit derivatives/discard) | `[[position]]`, `[[front_facing]]`, `[[sample_id]]`, `[[depth(...)]]`; `discard_fragment()`; `dfdx/dfdy/fwidth`; `sample()` implicit-lod fragment-only | Rasterizer twin: per-quad eval for derivatives; feasible but a separate interpreter mode |
| 8a | f16 | `enable f16;` + device feature `shader-f16` | `__half/__half2` via `cuda_fp16.h`, arith sm_53+ | `half` native since MSL 1.0 | `np.float16` |
| 8b | i64/u64 | ABSENT from WGSL core — CONFIRMED (wgpu native-only `SHADER_INT64`; gpuweb#5152 open) | full `long long` | `long/ulong` MSL 2.2/2.3+ | Python int + explicit 64-bit mask |
| 8c | int div/mod | DEFINED: `x/0==x`, `x%0==0`, `INT_MIN/-1==INT_MIN`, `%` = trunc, sign-of-dividend; signed +,-,* WRAP (naga polyfills on MSL/HLSL/SPIR-V) — CONFIRMED | UB on /0 and INT_MIN/-1; signed overflow UB; `%` trunc | UB (C++ rules); `%` trunc | Python `//`,`%` are FLOOR — must reimplement trunc + wrap via masking |
| 8d | fast-math/NaN | Impls MAY ASSUME NaN/Inf absent → indeterminate; no strict IEEE — CONFIRMED (gpuweb#2776) | IEEE by default; fast-math opt-in; FMA contraction on by default | fast-math ON BY DEFAULT (`MTLCompileOptions.fastMathEnabled=true`); opt-out `-fno-fast-math`, `metal::precise` — CONFIRMED | Faithful IEEE — twin is *stricter* than every GPU target |

## 2. Layer assignment

| Feature | Layer | Justification |
|---|---|---|
| vecN types, ctors, read-swizzle, single-component lvalue | shader-core | All 4 targets express it; CUDA needs framework-emitted operators/swizzle lowering (helper_math precedent) — spelling, not semantics |
| Write-swizzle (multi-component lvalue) | EXCLUDE from core | Only MSL has it; WGSL 1.0 forbids — core takes the intersection; lower `v.xz=` sugar to whole-vector rebuild if ever offered |
| matNxN, mat*vec, col-major | shader-core | Native in WGSL/MSL; on CUDA emit own struct (Warp does exactly this) — core owns layout so twin matches |
| Workgroup mem (static) + barriers | compute-family | Identical semantics, 3 spellings; a per-target string table (tinygrad's `smem_prefix`/`barrier` fields prove it) |
| Workgroup mem (dynamic size) | compute-family w/ target constraint | shared-bytes is a LAUNCH knob on CUDA/MSL but pipeline-creation `override` on WGSL ⇒ model as a *specialization constant* to stay portable |
| Atomics: i32/u32 {add,sub,min,max,and,or,xor,xchg,CAS}, relaxed, shared+global | compute-family | Exact WGSL set = the 3-way intersection |
| Float/i64 atomics, memory scopes/orders | per-target packs | CUDA-rich, MSL-partial, WGSL-none |
| Subgroup {broadcast(First), shuffle/up/down/xor, reduce, prefix, ballot, all/any, elect} | compute-family OPTIONAL capability | WGSL extension set ≈ the intersection (designed that way); gate like Slang capability atoms; masks/quad-ops stay per-target |
| Thread coordinates | compute-family | Pure renaming; CUDA global-id is a 1-line lowering |
| Fragment: position, front_facing, frag_depth, discard, derivatives, textureSample, vertex I/O | fragment-only family | CUDA's missing raster stage proves these can't sit in shader-core; WGSL/MSL align near 1:1 |
| f16 | shader-core behind capability flag | All 3 support; all 3 gate it |
| i64 | compute-family capability, WGSL-excluded | Absent from WGSL core; capability flag, never core |
| Int div/mod semantics | shader-core DEFINES policy | Pick: WGSL-defined semantics everywhere (guard-polyfill on CUDA/MSL, naga precedent) or "checked in twin, unchecked on GPU" |

## 3. Top 5 semantic traps for a shared core

1. **Floor vs trunc division/modulo**: Python `//`/`%` floor; ALL three shader
   targets truncate with sign-of-dividend `%`. The Py twin must never use
   native `//`/`%` on ints — else twin and GPU silently disagree on any
   negative operand.
2. **Div-by-zero & INT_MIN/-1**: defined in WGSL (`x/0==x`, `x%0==0`), UB in
   CUDA/MSL. Either polyfill-guard non-WGSL targets (naga's approach, a select
   per div) or declare it framework-UB and make the twin *raise* — pick one,
   document in core.
3. **NaN/fast-math asymmetry**: MSL fast-math ON by default, WGSL may assume
   NaN/Inf never exist, CUDA is IEEE. NaN-dependent code passes the twin and
   CUDA, breaks on Metal/WebGPU. Core must forbid NaN-as-data or force
   `-fno-fast-math`/`precise` per target.
4. **Subgroup model mismatch**: CUDA demands explicit masks (Volta+), WGSL/MSL
   use the implicit active set; width 32 vs 32/64 vs runtime 4–128. Core
   subgroup ops must be maskless, width-agnostic (expose `subgroup_size`),
   marked non-uniform-control-flow-unsafe.
5. **Shared-memory sizing time**: dispatch-time bytes (CUDA/Metal) vs
   pipeline-creation `override` (WGSL). Config bracket must model dynamic
   shared size as a *specialization constant* — WGSL binds at pipeline build
   (cache pipelines per size), CUDA/MSL at launch. (Bonus: signed overflow —
   WGSL wraps, C++ UB; twin must mask to 32 bits.)

## 4. Prior-art factoring

- **NVIDIA Warp**: ONE `wp.*` namespace (vec/mat/quat/atomics) over CUDA +
  compiled-C++ CPU twin; owns all type lowering — validates "core owns types,
  targets own spelling."
- **Taichi**: warp intrinsics in `ti.simt.warp.*` (CUDA-only, explicit masks),
  `ti.simt.subgroup` separate — evidence subgroups resist core placement.
- **tinygrad** (`renderer/cstyle.py`): single uop IR; per-target renderers are
  mostly *string tables* (`smem_prefix`, `barrier`, `code_for_workitem`) — the
  compute family IS shared machinery; only spelling is per-target.
- **Slang**: capability atoms checked at type-check time — the mature endpoint
  of "per-target packs as declared capabilities."
- **naga/tint**: their polyfill layers (int-div guards, atomic translation,
  relaxed-only MSL atomics) are a catalog of exactly the corner cases a shared
  core must legislate; naga's 994 golden snapshot outputs are the best
  cross-target corpus.

**Verdict**: hypothesis holds. Shader-core (types/ops/numeric policy) +
compute-family (launch/shared/barriers/atomics/coords, with subgroups and
i64/float-atomics as capability flags INSIDE the family) covers ~90%;
per-target packs stay small; fragment-only is forced as a separate family by
CUDA's absence of a raster stage.

Sources: WGSL spec; gpuweb#2776/#1431/#1774/#5152; wgpu PR#7012, issue#4385;
webgpufundamentals WGSL reference; Chrome 134 subgroups; wgpu-types Features;
CUDA programming guide; NVIDIA forums no-swizzle; helper_math.h; MSL spec
(atomics §6.15, simdgroup §6.9, threadgroup §4); MTLCompileOptions.fastMathEnabled;
setThreadgroupMemoryLength; Warp basics + functions; Taichi SIMT + RFC#4631;
tinygrad cstyle.py; Slang capabilities; naga/tint repos.
