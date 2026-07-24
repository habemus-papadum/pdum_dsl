# R16 — Invocation & runtime semantics (outputs, config, chaining, streams)

*Research agent survey, 2026-07-12 backend detour. Consumed by
`070_backends-notes.md`. Honors the owner decisions: no implicit output
allocation via program analysis; explicit outs or registered shape rules;
multi-output DPS; ping-pong chaining.*

## Findings

**Q1 — Output passing/allocation in prior art**
- **CuPy RawKernel**: `kernel(grid, block, args, shared_mem=0)` or
  `kernel[grid, block](args)`; caller pre-allocates outs, passes as args;
  returns nothing. CuPy **ElementwiseKernel**: optional explicit out,
  auto-allocates otherwise, and **returns the out array either way** — the
  Pythonic DPS-with-return precedent (NumPy ufunc `out=` discipline).
- **Triton**: caller allocates (`torch.empty_like`), passes pointer;
  `kernel[grid](x, y, out, n, BLOCK_SIZE=1024)`; grid is a **callable over
  meta-params** (`lambda meta: (cdiv(n, meta['BLOCK_SIZE']),)`) — precedent
  for grid-derived-by-rule; constexpr meta-params key compilation.
- **Warp**: `wp.launch(kernel, dim=..., inputs=[...], outputs=[...],
  device=..., stream=...)`; caller allocates; returns None (hostile to
  chaining/destructuring).
- **MLX metal_kernel**: output names at construction; `output_shapes`/
  `output_dtypes` at call — the bridge allocates; returns a list.
- **JAX Pallas**: `out_shape=ShapeDtypeStruct` (list ⇒ multiple), kernel body
  is pure DPS (`o_ref[...] =`), host call returns arrays. Taichi: fields
  pre-declared globally — wrong fit.
- Best fit: **explicit `out=` returning the outs (ufunc discipline) +
  MLX/Pallas-style declared shapes sourced from a REGISTERED RULE rather than
  per-call kwargs**. Confirms the owner lean: nobody infers output shapes from
  kernel bodies; everyone either takes buffers or takes declared shapes.

**Q2 — Launch-config vocabulary**
- CUDA: `<<<grid, block, sharedMemBytes, stream>>>` — all four runtime-flexible.
- WGSL/WebGPU: `dispatchWorkgroups(count)` runtime; `@workgroup_size` is
  shader-creation-time (override constants still fixed at pipeline creation =
  artifact-side); **no dynamic shared memory confirmed** — `var<workgroup>`
  sizes creation-fixed (override-sized at most; 16 KiB default limit).
- Metal: `dispatchThreadgroups(tgPerGrid, threadsPerThreadgroup)` — both at
  encode time; `setThreadgroupMemoryLength(len, index)` at encode;
  `maxTotalThreadsPerThreadgroup` is the pipeline-creation constraint.
- So `block` is runtime-flexible on CUDA/Metal but codegen-bound on WGSL;
  `smem` is runtime on CUDA/Metal, nonexistent-as-dynamic on WGSL.

**Q3 — Ping-pong chaining**
- Canonical pattern (WebGPU Game-of-Life codelab): two buffers + two bind
  groups selected by `step % 2`; after N stages the result lives in `buf[N%2]`.
- Sync: WebGPU — each dispatch is its own synchronization scope; as-if-serial
  within a submission, no explicit barriers exist or are needed
  (gpuweb#4433/#4434). CUDA — same-stream ordering. Metal — encode order
  within one command buffer. Orchestration compiles to "encode all stages
  under one ordering token"; barriers are never a user concern on any target.
- Shape changes across stages break ping-pong ⇒ per-edge intermediates owned
  by the pipeline's cache entry (allocated once, reused while shapes hold).

**Q4 — Streams/queues**
- DLPack `__dlpack__(stream=...)`: consumer passes its stream int; **producer
  inserts the sync**; `-1` = no sync; CPU ⇒ None. The boundary contract, not
  the launch contract.
- CuPy: ambient current stream. MLX: `stream=` kwarg or ambient `mx.stream`.
  wgpu: exactly ONE queue per device — the component is meaningless there.
- Conclusion: stream is bridge-ambient by default; we need only an optional
  pass-through token. It rides the **leaves channel** (launcher-bound, never
  byte-packed), so `launch(staging, leaves)` is untouched.

**Q5 — Return story**
- CuPy ops/ufuncs return the out array; JAX returns arrays; Warp returns None;
  wgpu-py `compute_with_buffers` declares outs as a dict and returns a dict.
  Returning **the destination objects themselves** is the Pythonic consensus;
  async-safe because stream/queue ordering makes later consumers see the writes.

## Recommended invocation surface

```python
kernel[{"grid": g, "block": 64}](x, out=o)      # explicit DPS; returns o
a, b = kernel[cfg](x, out=(a, b))                # N-destination DPS; returns tuple in out order
a, b = kernel[cfg](x)                            # ONLY if an out-shape rule is registered; else
                                                 # MissingRule("out_shape", kernel, loc) — loud
(b1, b2) > step1[c1] | step2[c2] | step3[c3]     # orchestrated ping-pong (below)
```

**Config schema** (one vocabulary, owned by the compute role; positional order
`[grid, block, smem, stream]`, named via dict per the bracket contract; §3c
strip→value→type pipeline per component):

| component | meaning | default mode | CUDA | Metal | WGSL |
|---|---|---|---|---|---|
| `grid` | workgroup/block/threadgroup **count** | **strip** (runtime; leaves channel) | grid dim | threadgroupsPerGrid | dispatchWorkgroups |
| `block` | threads per group | **value-specialize** (artifact key) | blockDim (runtime-lowered from key) | threadsPerThreadgroup at encode | `@workgroup_size` — codegen by necessity |
| `smem` | dynamic shared bytes | **strip** on CUDA/Metal; WGSL: **refuse** (or value-specialize into a creation-fixed `override`) | sharedMem arg | setThreadgroupMemoryLength | no dynamic form |
| `stream` | ordering token | **strip, never keys anything**; absent ⇒ bridge ambient | stream | command buffer/queue | omit — one queue |

Per-kernel overrides per §3c: a CUDA-only kernel may demote `block` to strip
(runtime blockDim); the WGSL schema **refuses** that override loudly — the
documented asymmetry. `grid` may be omitted when a grid rule is registered
(Triton's `grid(meta)` precedent): `grid = ceil_div(launch_domain, block)`,
launch domain defaulting to the out-shape rule's domain.

**Out-shape rules** (registration, not analysis): a rule column
`out_shape(arg_types, arg_shapes, config) -> Type-with-shapes`, registered per
kernel or per role. Feeds the existing `ResultPlan` seam exactly as designed
(040 §3b); destinations reused across calls while shapes hold. `out=` always
wins over the rule; both paths return the destinations (bridge arrays /
`DeviceValue`s — no new wrapper around user-supplied buffers).

**Chaining/ping-pong** — the reserved `orchestrate` semantics tag produces an
**encode plan**: stage k's out-slots wired to k+1's in-slots, all stages under
one ordering token (one wgpu command encoder / one CUDA stream / one
MTLCommandBuffer). Intermediate policy, decided at plan build:
- All inter-stage shapes+dtypes equal AND the applied value is a 2-tuple of
  buffers ⇒ **swap schedule**: stage k reads `buf[k%2]`, writes
  `buf[(k+1)%2]`; `>` returns `(result, scratch) = (buf[N%2], buf[(N-1)%2])` —
  re-applying continues the ping-pong naturally; parity is the plan's problem,
  never the user's.
- Shapes differ ⇒ user pair refused loudly ("ping-pong needs uniform edges —
  apply a single input and let per-edge intermediates be allocated");
  single-input form allocates per-edge device-resident intermediates via each
  stage's out-shape rule, owned by the pipeline's cache entry.
- Materialization stays a boundary act: `>` yields `DeviceValue`s; host bytes
  only at a materializer terminal (`| collect`).

**Seam mapping**: bracket payload → schema splits components into key-side
(value-specialized, e.g. `block`) and stripped runtime data; stripped
components are appended to the leaves channel in schema order, so
`FastRecord.launch(staging, leaves)` is unchanged — the launcher closure peels
`grid`/`smem`/`stream` off the tail. `__dlpack__(stream=...)` with
producer-side sync is implemented at the `DeviceValue`/materializer boundary
only.

## Alternatives rejected
- **Implicit allocation without a rule**: requires program analysis no
  surveyed system does; MLX/Pallas both demand declared shapes. Rejected per
  the owner lean — the rule form preserves the ergonomics.
- **Per-call `output_shapes=` kwarg (MLX-style)**: shape arithmetic belongs in
  one registered place; the rule + ResultPlan cache enables destination reuse.
- **Warp's `inputs=/outputs=` list kwargs**: returns None — kills
  destructuring and pipeline threading.
- **Returning None from DPS calls**: breaks chaining and `a, b =`; ufunc
  return-the-out discipline costs nothing.
- **`domain` (total threads) as the schema's first component**: WebGPU only
  has group counts; `grid` (count) is the portable primitive; total-thread
  convenience is what a grid rule provides.
- **Stream as a `launch()` parameter**: changes the frozen hit-path contract
  for a component two of three targets default ambiently; leaves channel
  already exists for launcher-bound non-byte values.
- **A `Swap`/`PingPong` explicit combinator**: unnecessary — buffer-pair
  application + the orchestrate plan's uniform-edge check; revisit only for
  `repeat(step, n)` time-iteration.

Sources: CuPy kernel + CUDA-API guides; Triton vector-add tutorial; Warp
basics + concurrency; MLX custom-Metal-kernels + streams; JAX Pallas
quickstart; WGSL spec (compute-shader workgroups); MTLComputeCommandEncoder
docs; gpuweb#4433/#4434; WebGPU Game-of-Life codelab; array-API `__dlpack__`;
wgpu-py utils `compute_with_buffers`. Local seams: 040 §3b/§3c, 010
§2.11-2.12/§4.2, kernel/pack.py, combinators.py.
