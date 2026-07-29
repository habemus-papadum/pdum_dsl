"""The launch contract — what a generator RETURNS beside source (283 §4).

A backend cannot return only source text (210, the supersession
bullet): the clauses below are NEGOTIATED per target, never universal
structure, and each earned its seat empirically:

- ``thread_size`` SPECIALIZES (owner-ruled): it keys the artifact and a
  change recompiles — whether it lands in source text (WGSL), in the
  dispatch call (Metal ``dispatchThreadgroups:``), or both (CUDA
  blockDim + ``__launch_bounds__``). Block/grid counts are launcher
  data everywhere and appear nowhere here.
- ``guard`` is a dispatch-granularity clause: ``"emitted"`` where the
  runtime launches whole groups (WebGPU, CUDA) and the overhang guard
  is real code; ``"exact"`` where the runtime launches exact grids
  (Metal ``dispatchThreads:``) and the guard is dead. Proven
  bitwise-immaterial where both apply.
- ``bindings`` is DATA, one table PER STAGE — WGSL shares one index
  space across stages, Metal indexes vertex and fragment tables
  separately (the same logical resource carries two indices), CUDA
  degenerates to parameter order. Implicit binding-in-text does not
  survive three targets.
- ``slots`` is the staging plan's DEVICE representation (staging.py):
  declared carriers, declared narrowing — never invented per backend.
- ``math`` records which target-numeric-contract rows (mathrows.py)
  were applied, so an artifact says what it substituted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Binding:
    """One resource binding: DATA, never implicit in emitted text."""

    name: str  # the logical resource ("buf0", "U", "vb1", a texture)
    index: int  # the index IN THIS STAGE's table
    space: str = "storage"  # storage | uniform | texture | sampler | param
    writable: bool = False


@dataclass(frozen=True)
class LaunchContract:
    thread_size: tuple[int, ...]  # SPECIALIZES — joins the artifact key
    guard: str  # "emitted" | "exact"
    bindings: tuple[tuple[Binding, ...], ...] = ()  # one tuple per stage
    slots: tuple = ()  # device slot rows (staging.device_slots)
    math: tuple[str, ...] = field(default=())  # applied mathrows, by name

    def __post_init__(self):
        if self.guard not in ("emitted", "exact"):
            raise ValueError(f'guard is "emitted" or "exact", got {self.guard!r}')

    def key(self) -> tuple:
        """The contract's contribution to the artifact content key —
        exactly the specializing clause, nothing launcher-shaped."""
        return ("threads", self.thread_size)
