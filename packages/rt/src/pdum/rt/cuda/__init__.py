"""The CUDA column — the NAME, ahead of the column (283 §5).

Importing this package is safe and registers NOTHING. That is the whole
point: ``rt.cuda`` is spellable today, so a program can name its target
before the column exists, and ``acquire``/``executor_for`` refuse in the
registry's own voice — naming the arrival rather than reporting an
attribute error.

The column is the L4 team's, on their hardware, on their schedule. 283
§5 is the pattern, clause by clause, and it is theirs to bend: the
generator is one CUDA ``emit.Dialect`` (casts spell ``float(x)``,
literals take ``f``, declarations are C, vectors are ``float4``) with a
COMPOSED ambient leaf row — ``blockIdx.x * blockDim.x + threadIdx.x``
rather than a single builtin, which is exactly why the leaf spellings
are dialect hooks; the shell is ``extern "C" __global__`` with no
binding declarations at all, the binding table degenerating to
parameter order; the contract's guard clause is ``"emitted"``
(WebGPU-shaped — whole blocks), and its thread size lands in BOTH the
launch call and ``__launch_bounds__``, which is the supersession
ruling's cleanest exhibit. Arrival duties: the math-rows survey (exp,
sinh/cosh, pow — fast-math flags are LICENSES, never defaults) and the
ledger's CUDA column.
"""

from __future__ import annotations
