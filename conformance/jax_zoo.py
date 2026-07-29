"""Idiomatic jax zoo — the second BASELINE column (310), distinct from the
translated CHECK column (jax_evaluator.py).

What a fluent jax author writes today: jnp einsum, jax.nn activations and
softmax, ``.at[].add`` scatters, ``lax.top_k`` — the flax-style spellings,
independently authored from each entry's math, never ports of an
interpreter. The functions here are the MATH; ``jax.jit`` is the caller's
wrap (the benchmark rig applies it — jit-compiled is what "idiomatic jax"
means at timing time, and the compile lands in warmup, outside the timed
loop). Correctness is asserted eager AND jitted paths share one trace.

Same tolerance law as torch_zoo.py: f64 both sides (x64 on), stated
tolerance covers operation-order drift only. lax.top_k tie-behavior vs the
stable first-wins law is measure-zero with continuous logits.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # before any array is built

import jax.numpy as jnp  # noqa: E402 — after the x64 switch, by design

from pdum.tl.zoo import GemmConfig, GPT2Config, LlamaConfig, MoEConfig  # noqa: E402


def _layernorm(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(v + eps) * g + b


def _causal_softmax(sc, axis=1):
    T = sc.shape[0]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    mask = mask.reshape(mask.shape + (1,) * (sc.ndim - 2))
    return jax.nn.softmax(jnp.where(mask, sc, -jnp.inf), axis=axis)


def gpt2_baseline(inp: dict, cfg: GPT2Config = GPT2Config()) -> jax.Array:
    h = inp["wte"][inp["ids"]] + inp["wpe"]
    for i in range(cfg.layers):
        p = f"h.{i}."
        a = _layernorm(h, inp[p + "attn.ln1g"], inp[p + "attn.ln1b"], cfg.eps)
        q = jnp.einsum("td,dhk->thk", a, inp[p + "attn.wq"]) / jnp.sqrt(cfg.hk)
        k = jnp.einsum("sd,dhk->shk", a, inp[p + "attn.wk"])
        v = jnp.einsum("sd,dhk->shk", a, inp[p + "attn.wv"])
        pr = _causal_softmax(jnp.einsum("thk,shk->tsh", q, k))
        ctx = jnp.einsum("tsh,shk->thk", pr, v)
        h = h + jnp.einsum("thk,hkd->td", ctx, inp[p + "attn.wo"])
        a2 = _layernorm(h, inp[p + "mlp.ln2g"], inp[p + "mlp.ln2b"], cfg.eps)
        m = jax.nn.gelu(a2 @ inp[p + "mlp.w1"] + inp[p + "mlp.b1"], approximate=True)
        h = h + m @ inp[p + "mlp.w2"] + inp[p + "mlp.b2"]
    hf = _layernorm(h, inp["lnfg"], inp["lnfb"], cfg.eps)
    return hf @ inp["wte"].T  # tied head


def llama_baseline(inp: dict, cfg: LlamaConfig = LlamaConfig()) -> jax.Array:
    x, om = inp["x"], inp["omega"]
    T, C = x.shape[0], cfg.c

    def rms(v, g):
        return v * jax.lax.rsqrt((v * v).mean(-1, keepdims=True) + cfg.eps) * g

    a = rms(x, inp["rms1g"])
    ang = jnp.arange(T)[:, None] * om[None, :]
    cs, sn = jnp.cos(ang), jnp.sin(ang)
    q = jnp.einsum("td,dgrcu->tgrcu", a, inp["wq"])
    k = jnp.einsum("sd,dgcu->sgcu", a, inp["wk"])
    q0 = q[..., 0] * cs[:, None, None, :] - q[..., 1] * sn[:, None, None, :]
    q1 = q[..., 0] * sn[:, None, None, :] + q[..., 1] * cs[:, None, None, :]
    k0 = k[..., 0] * cs[:, None, :] - k[..., 1] * sn[:, None, :]
    k1 = k[..., 0] * sn[:, None, :] + k[..., 1] * cs[:, None, :]
    sc = jnp.einsum("tgrc,sgc->tsgr", q0, k0) + jnp.einsum("tgrc,sgc->tsgr", q1, k1)
    pr = _causal_softmax(sc / jnp.sqrt(2 * C))
    vv = jnp.einsum("sd,dgk->sgk", a, inp["wv"])
    ctx = jnp.einsum("tsgr,sgk->tgrk", pr, vv)
    h = x + jnp.einsum("tgrk,grkd->td", ctx, inp["wo"])
    a2 = rms(h, inp["rms2g"])
    return h + (jax.nn.silu(a2 @ inp["w1"]) * (a2 @ inp["w3"])) @ inp["w2"]


def sliding_baseline(inp: dict, W: int = 2) -> jax.Array:
    q, k, v = inp["q"], inp["k"], inp["v"]
    t = jnp.arange(q.shape[0])
    mask = (t[None, :] <= t[:, None]) & (t[:, None] - t[None, :] < W)
    return jax.nn.softmax(jnp.where(mask, q @ k.T, -jnp.inf), axis=1) @ v


def gated_baseline(inp: dict) -> jax.Array:
    q, k, v = inp["q"], inp["k"], inp["v"]
    ctx = _causal_softmax(q @ k.T) @ v
    return jax.nn.sigmoid(q @ inp["wg"]) * ctx


def qknorm_baseline(inp: dict, eps: float = 1e-6) -> jax.Array:
    def rms(x, g):
        return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + eps) * g

    sc = rms(inp["q"], inp["gq"]) @ rms(inp["k"], inp["gk"]).T
    return _causal_softmax(sc) @ inp["v"]


def flash_baseline(inp: dict) -> jax.Array:
    q, k, v = inp["q"], inp["k"], inp["v"]
    return _causal_softmax(q @ k.T) @ v


def heat2d_baseline(inp: dict, T: int = 3, alpha: float = 0.1) -> jax.Array:
    # padded shifted slices, not convolve2d: XLA's GPU conv accumulates below
    # f64 (measured 1.6e-8 abs at zoo shapes) and the columns stay honest f64
    u = inp["u0"]
    for _ in range(T):
        up = jnp.pad(u, 1)
        nsum = up[:-2, 1:-1] + up[2:, 1:-1] + up[1:-1, :-2] + up[1:-1, 2:]
        u = u + alpha * (nsum - 4.0 * u)
    return u


def gemm_baseline(inp: dict, cfg: GemmConfig = GemmConfig()) -> jax.Array:
    return inp["a"] @ inp["b"]


def moe_baseline(inp: dict, cfg: MoEConfig = MoEConfig()) -> jax.Array:
    """Capacity-factor top-k routing, the vectorized jax idiom: one-hot
    prefix sums for slots, ``.at[].add`` into the expert buffer, batched
    expert einsums, gather back. Overflowed choices drop (weight 0)."""
    x = inp["x"]
    E, K, CAP = cfg.e, cfg.k, cfg.cap
    logits = x @ inp["wr"]
    gate_vals, choice = jax.lax.top_k(logits, K)
    gates = jax.nn.softmax(gate_vals, axis=1)
    onehot = jax.nn.one_hot(choice, E, dtype=jnp.int64)  # (t, c, e)
    flat = onehot.reshape(-1, E)  # (t*c, e), t-major c-minor: arrival order
    pos = ((jnp.cumsum(flat, axis=0) - flat).reshape(onehot.shape) * onehot).sum(-1)
    keep = pos < CAP
    dest = jnp.where(keep, choice * CAP + pos, 0)
    vals = x[:, None, :] * keep[..., None].astype(x.dtype)  # (t, c, d)
    buf = jnp.zeros((E * CAP, x.shape[1]), dtype=x.dtype)
    buf = buf.at[dest.reshape(-1)].add(vals.reshape(-1, x.shape[1]))
    h = jax.nn.gelu(jnp.einsum("ecd,edm->ecm", buf.reshape(E, CAP, -1), inp["w1"]), approximate=True)
    y2 = jnp.einsum("ecm,emd->ecd", h, inp["w2"]).reshape(E * CAP, -1)
    back = y2[dest.reshape(-1)].reshape(*dest.shape, -1)  # (t, c, d)
    return (back * (gates * keep.astype(x.dtype))[..., None]).sum(1)


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
