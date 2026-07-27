"""Tiled matmul — the canonical capacity-constrained case (200 §7 P9;
LEVELS K-D), handed to L4 as its benchmark.

TILING IS LAYOUT, NOT SEMANTICS: every tile dim is DECLARED via split
(m -> (mo, mi), n -> (no, ni), k -> (ko, ki)), the contraction is the
same ONE visible line over both k parts, and the result merges back —
so the tiled program IS the plain matmul by construction (the
fuse-as-elision seed: L4's tile/bind moves are certified rewrites over
exactly this shape, and the descent objective — minimize parent-memory
traffic under child tile capacity — prices THIS program). The MAC count
pins the standard figure m·n·k under fuse_mac. Non-divisible tails are
an L4 adversarial input family (BOUNDARIES); the L0 entry states the
divisible case."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..compute import contract
from ..lifting import lift_step
from ..tensor import Tensor
from .zoo_common import ZooModel


@dataclass(frozen=True)
class GemmConfig:
    m: int = 8
    n: int = 6
    k: int = 4
    mi: int = 4  # tile widths (the inner dims); outer = size // inner
    ni: int = 3
    ki: int = 2


def make_tiled_matmul(cfg: GemmConfig):
    MO, MI = cfg.m // cfg.mi, cfg.mi
    NO, NI = cfg.n // cfg.ni, cfg.ni
    KO, KI = cfg.k // cfg.ki, cfg.ki

    def gemm(a, b):
        at = a.split("m", mo=MO, mi=MI).split("k", ko=KO, ki=KI)
        bt = b.split("k", ko=KO, ki=KI).split("n", no=NO, ni=NI)
        c = contract(at, bt, axis=("ko", "ki"))  # tiles ride; ONE line
        return c.merge(("mo", "mi"), "m").merge(("no", "ni"), "n")

    return gemm


def tiled_matmul(cfg: GemmConfig = GemmConfig(), seed: int = 13) -> ZooModel:
    rng = np.random.default_rng(seed)
    inputs = {
        "a": Tensor.from_numpy(rng.standard_normal((cfg.m, cfg.k)), ("m", "k")),
        "b": Tensor.from_numpy(rng.standard_normal((cfg.k, cfg.n)), ("k", "n")),
    }
    ls = lift_step(make_tiled_matmul(cfg), **{k: v.layout for k, v in inputs.items()})
    return ZooModel(ls.program, inputs, ls.outputs[0], lambda inp: inp["a"] @ inp["b"], ("m", "n"))
