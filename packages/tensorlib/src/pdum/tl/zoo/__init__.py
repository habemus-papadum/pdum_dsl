"""The model zoo — L0 surface programs that keep the machinery honest.

Each entry is a builder returning a ZooModel: an IR Program, its input
tensors (weights and activations are both just inputs — parameters are not
special at L0), the output var, and a pure-numpy reference the program must
match. Entries double as regression tests now and as benchmarks for every
later level (peak memory, sharding, fusion, scheduling).

Chosen to SPAN mechanisms, not to crawl the architecture gallery:
- gpt2: LayerNorm + causal MHA + GELU MLP + residuals (the baseline canon)
- llama_block: RMSNorm, RoPE (pairs as born-structured dims — no splits),
  GQA (kv heads repeated by declaration), SwiGLU
- attention variants: sliding-window mask, output gating, QK-norm, and the
  online-softmax (flash) accumulator as a composite reducer whose backward
  is DERIVED
- physics: 2D heat (fold + Dirichlet ghosts via pad), 1D FDTD on a charted
  staggered grid (exact half-integer charts, recharted differences)

Recorded boundaries (LEVELS.md): MoE routing / top-k (data-dependent
gather), KV-cache decode (mutation), dynamic shapes.
"""

from .attention import flash_attention, gated_attention, qknorm_attention, sliding_attention
from .gpt2 import GPT2Config, gpt2
from .llama import LlamaConfig, llama_block
from .megatron import MegatronConfig, megatron_block
from .physics import fdtd1d_staggered, heat2d
from .zoo_common import ZooModel

__all__ = [
    "GPT2Config",
    "LlamaConfig",
    "MegatronConfig",
    "ZooModel",
    "megatron_block",
    "fdtd1d_staggered",
    "flash_attention",
    "gated_attention",
    "gpt2",
    "heat2d",
    "llama_block",
    "qknorm_attention",
    "sliding_attention",
]
