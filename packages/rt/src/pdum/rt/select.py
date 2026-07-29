"""Selection — the ruled spelling (282 §7, 283 §3).

Selecting a target names a PAIR: a GENERATOR (a program → source
translator for one target language) and a RUNTIME (a device API:
compile, allocate, bind, encode, submit). They are separable — the
same MSL would serve a metal-cpp harness; the same Metal runtime would
launch a precompiled .metallib — and 1:1 today only because we wrote
both halves of both columns. The user-facing names (``webgpu``,
``metal``, ``cuda``) are pre-assembled instances; an unusual pairing
is constructed explicitly when someone needs one.

The classes are DELIBERATELY EMPTY (owner-ruled). What they carry is
IDENTITY: ``fp()`` is the pair's contribution to the artifact content
key (the FAIL-5 lesson — two columns' artifacts share a region key and
must never share a cache row). Nothing in a Pair implies a transfer:
under adoption/managed memory a tensor's bytes are host and device
memory at once; residency is the buffer's affair, resolved against the
registry, never the selection's.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Generator:
    """A program → shader/kernel source translator, one target language."""

    def fp(self) -> tuple:
        # Identity is the TYPE today (the classes are empty by ruling);
        # this becomes content-derived the day a generator gains state.
        return (type(self).__name__,)


@dataclass(frozen=True)
class Runtime:
    """A device API: acquire, compile, allocate, bind, encode, submit."""

    def fp(self) -> tuple:
        return (type(self).__name__,)


@dataclass(frozen=True)
class WgslGenerator(Generator):
    pass


@dataclass(frozen=True)
class MslGenerator(Generator):
    pass


@dataclass(frozen=True)
class CudaGenerator(Generator):
    pass


@dataclass(frozen=True)
class WgpuRuntime(Runtime):
    pass


@dataclass(frozen=True)
class MetalRuntime(Runtime):
    pass


@dataclass(frozen=True)
class CudaRuntime(Runtime):
    pass


@dataclass(frozen=True)
class Pair:
    generator: Generator
    runtime: Runtime

    def fp(self) -> tuple:
        return self.generator.fp() + self.runtime.fp()


webgpu = Pair(WgslGenerator(), WgpuRuntime())
metal = Pair(MslGenerator(), MetalRuntime())
# The CUDA column is the L4 team's (283 §5): the NAME exists so programs
# can spell it today; acquire/compile refuse toward its arrival.
cuda = Pair(CudaGenerator(), CudaRuntime())


def on(p: Pair) -> Pair:
    """The selection door: ``kernel[on(metal)]`` / ``art.on(metal)``.

    Today it validates and returns the pair; it exists so the spelling
    is stable before the kernel-tier bracket integration lands."""
    if not isinstance(p, Pair):
        raise TypeError(f"on(...) takes a Pair (webgpu, metal, cuda, or an explicit Pair), got {type(p).__name__}")
    return p
