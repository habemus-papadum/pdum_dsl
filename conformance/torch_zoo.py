"""Idiomatic PyTorch zoo — the BASELINE column (310), kept distinct from the
translated CHECK column (torch_evaluator.py).

These are the implementations a fluent torch author would write today —
F.layer_norm, F.scaled_dot_product_attention, einsum, index_add — NOT ports
of the reference interpreter. They exist to be benchmarked against: the
translated column must eventually meet them on performance, and they
cross-check the zoo's numpy denotation from a third, independently-authored
angle. Each takes the entry's own name-keyed inputs (numpy_inputs, as torch
tensors on one device) and returns the output in the entry's `order`.

Numerics: same f64 math, different operation ORDER (fused kernels,
reassociated reductions), so agreement is asserted under a stated tolerance
rather than the interpreter columns' 1e-9/1e-12. Ties in moe's top-k
routing would break the stable first-wins law — measure-zero with
continuous logits (the entries seed f64 gaussians), noted here once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pdum.tl.zoo import GemmConfig, GPT2Config, LlamaConfig, MoEConfig


def _causal_attend(q, k, v, scale=None):
    # (t, e) x (s, e) x (s, o): the fused idiom — sdpa builds the causal
    # mask and the softmax; math backend serves f64 on both devices
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def gpt2_baseline(inp: dict, cfg: GPT2Config = GPT2Config()) -> torch.Tensor:
    h = inp["wte"][inp["ids"]] + inp["wpe"]
    for i in range(cfg.layers):
        p = f"h.{i}."
        a = F.layer_norm(h, (cfg.d,), inp[p + "attn.ln1g"], inp[p + "attn.ln1b"], cfg.eps)
        q = torch.einsum("td,dhk->htk", a, inp[p + "attn.wq"])
        k = torch.einsum("sd,dhk->hsk", a, inp[p + "attn.wk"])
        v = torch.einsum("sd,dhk->hsk", a, inp[p + "attn.wv"])
        ctx = _causal_attend(q, k, v, scale=1.0 / cfg.hk**0.5)
        h = h + torch.einsum("htk,hkd->td", ctx, inp[p + "attn.wo"])
        a2 = F.layer_norm(h, (cfg.d,), inp[p + "mlp.ln2g"], inp[p + "mlp.ln2b"], cfg.eps)
        m = F.gelu(a2 @ inp[p + "mlp.w1"] + inp[p + "mlp.b1"], approximate="tanh")
        h = h + m @ inp[p + "mlp.w2"] + inp[p + "mlp.b2"]
    hf = F.layer_norm(h, (cfg.d,), inp["lnfg"], inp["lnfb"], cfg.eps)
    return hf @ inp["wte"].T  # tied head


def llama_baseline(inp: dict, cfg: LlamaConfig = LlamaConfig()) -> torch.Tensor:
    x, om = inp["x"], inp["omega"]
    T, C = x.shape[0], cfg.c

    def rms(v, g):
        return v * torch.rsqrt((v * v).mean(-1, keepdim=True) + cfg.eps) * g

    a = rms(x, inp["rms1g"])
    ang = torch.arange(T, dtype=x.dtype, device=x.device)[:, None] * om[None, :]
    cs, sn = torch.cos(ang), torch.sin(ang)
    q = torch.einsum("td,dgrcu->tgrcu", a, inp["wq"])
    k = torch.einsum("sd,dgcu->sgcu", a, inp["wk"])
    q0 = q[..., 0] * cs[:, None, None, :] - q[..., 1] * sn[:, None, None, :]
    q1 = q[..., 0] * sn[:, None, None, :] + q[..., 1] * cs[:, None, None, :]
    k0 = k[..., 0] * cs[:, None, :] - k[..., 1] * sn[:, None, :]
    k1 = k[..., 0] * sn[:, None, :] + k[..., 1] * cs[:, None, :]
    # GQA as sdpa heads: (g, r) query heads over g kv heads, K/V expanded on r
    qh = torch.cat([q0, q1], dim=-1).permute(1, 2, 0, 3).reshape(cfg.g * cfg.r, T, 2 * C)
    kh = torch.cat([k0, k1], dim=-1).permute(1, 0, 2)[:, None].expand(cfg.g, cfg.r, T, 2 * C).reshape(qh.shape)
    vv = torch.einsum("sd,dgk->gsk", a, inp["wv"])
    vh = vv[:, None].expand(cfg.g, cfg.r, T, cfg.kv).reshape(cfg.g * cfg.r, T, cfg.kv)
    ctx = _causal_attend(qh, kh, vh, scale=1.0 / (2 * C) ** 0.5)
    ctx = ctx.reshape(cfg.g, cfg.r, T, cfg.kv).permute(2, 0, 1, 3)
    h = x + torch.einsum("tgrk,grkd->td", ctx, inp["wo"])
    a2 = rms(h, inp["rms2g"])
    return h + (F.silu(a2 @ inp["w1"]) * (a2 @ inp["w3"])) @ inp["w2"]


def sliding_baseline(inp: dict, W: int = 2) -> torch.Tensor:
    q, k, v = inp["q"], inp["k"], inp["v"]
    t = torch.arange(q.shape[0], device=q.device)
    mask = (t[None, :] <= t[:, None]) & (t[:, None] - t[None, :] < W)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)


def gated_baseline(inp: dict) -> torch.Tensor:
    q, k, v = inp["q"], inp["k"], inp["v"]
    ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
    return torch.sigmoid(q @ inp["wg"]) * ctx


def qknorm_baseline(inp: dict, eps: float = 1e-6) -> torch.Tensor:
    def rms(x, g):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * g

    qn, kn = rms(inp["q"], inp["gq"]), rms(inp["k"], inp["gk"])
    return F.scaled_dot_product_attention(qn, kn, inp["v"], is_causal=True, scale=1.0)


def flash_baseline(inp: dict) -> torch.Tensor:
    return F.scaled_dot_product_attention(inp["q"], inp["k"], inp["v"], is_causal=True, scale=1.0)


def heat2d_baseline(inp: dict, T: int = 3, alpha: float = 0.1) -> torch.Tensor:
    # explicit Euler as a conv: the 5-point Laplacian over a zero-padded field
    u = inp["u0"][None, None]
    lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=u.dtype, device=u.device)
    w = lap[None, None]
    for _ in range(T):
        u = u + alpha * F.conv2d(F.pad(u, (1, 1, 1, 1)), w)
    return u[0, 0]


def gemm_baseline(inp: dict, cfg: GemmConfig = GemmConfig()) -> torch.Tensor:
    return inp["a"] @ inp["b"]


def moe_baseline(inp: dict, cfg: MoEConfig = MoEConfig()) -> torch.Tensor:
    """Capacity-factor top-k routing, the vectorized idiom: one-hot prefix
    sums for slot assignment, index_add into the expert buffer, batched
    expert matmuls, gather back. Overflowed choices drop (weight 0)."""
    x = inp["x"]
    E, K, CAP = cfg.e, cfg.k, cfg.cap
    logits = x @ inp["wr"]
    gate_vals, choice = torch.topk(logits, K, dim=1)
    gates = F.softmax(gate_vals, dim=1)
    onehot = F.one_hot(choice, E)  # (t, c, e)
    flat = onehot.reshape(-1, E)  # (t*c, e), t-major c-minor: arrival order
    pos = (torch.cumsum(flat, dim=0) - flat).reshape(onehot.shape)  # earlier arrivals at e
    pos = (pos * onehot).sum(-1)  # (t, c): my slot at my expert
    keep = pos < CAP
    dest = torch.where(keep, choice * CAP + pos, torch.zeros_like(pos))
    vals = x[:, None, :] * keep[..., None].to(x.dtype)  # (t, c, d)
    buf = torch.zeros(E * CAP, x.shape[1], dtype=x.dtype, device=x.device)
    buf.index_add_(0, dest.reshape(-1), vals.reshape(-1, x.shape[1]))
    h = F.gelu(torch.einsum("ecd,edm->ecm", buf.reshape(E, CAP, -1), inp["w1"]), approximate="tanh")
    y2 = torch.einsum("ecm,emd->ecd", h, inp["w2"]).reshape(E * CAP, -1)
    back = y2[dest.reshape(-1)].reshape(*dest.shape, -1)  # (t, c, d)
    return (back * (gates * keep.to(x.dtype))[..., None]).sum(1)


# name -> (the zoo entry under its default config, the baseline at the same
# config): the harness's pairing. trainer/fdtd/megatron have no baseline yet —
# the CHECK column covers them; a baseline lands when a benchmark wants it.
BASELINES = {
    "gpt2": gpt2_baseline,
    "llama": llama_baseline,
    "sliding": sliding_baseline,
    "gated": gated_baseline,
    "qknorm": qknorm_baseline,
    "flash": flash_baseline,
    "heat2d": heat2d_baseline,
    "gemm": gemm_baseline,
    "moe": moe_baseline,
}
