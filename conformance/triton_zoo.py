"""Idiomatic Triton twins of the tile flagships — the tile tier's BASELINE
column (320 §8), and the apprenticeship: where Triton's real performance
idioms live (tl.dot, block loops, masks) becomes measured knowledge.

Hand-written from each flagship's math at REALISTIC sizes (tl.dot wants
>=16 per dim — the toy flagship shapes are the translator's subjects, not
these). Sizes are runtime arguments with masked edges, the standard Triton
discipline. ``tl.dot(..., input_precision="ieee")`` keeps the conformance
comparison honest f32; a perf run may flip to tensor-core defaults — that
choice belongs to the rig, not the twins.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        at = tl.load(a + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        bt = tl.load(b + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(at, bt, input_precision="ieee")
    tl.store(c + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def gemm_triton(a: np.ndarray, b: np.ndarray, BM=32, BN=32, BK=32) -> np.ndarray:
    M, K = a.shape
    K2, N = b.shape
    ta = torch.as_tensor(a, dtype=torch.float32).cuda().contiguous()
    tb = torch.as_tensor(b, dtype=torch.float32).cuda().contiguous()
    tc = torch.empty((M, N), dtype=torch.float32, device="cuda")
    _gemm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](ta, tb, tc, M, N, K, BM, BN, BK)
    torch.cuda.synchronize()
    return tc.cpu().numpy()


@triton.jit
def _flash_kernel(q, k, v, o, T, D: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr):
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    rd = tl.arange(0, D)
    qb = tl.load(q + rt[:, None] * D + rd[None, :], mask=rt[:, None] < T, other=0.0)
    m = tl.full((BT,), float("-inf"), dtype=tl.float32)
    den = tl.zeros((BT,), dtype=tl.float32)
    acc = tl.zeros((BT, D), dtype=tl.float32)
    for s0 in range(0, T, BS):
        rs = s0 + tl.arange(0, BS)
        kb = tl.load(k + rs[:, None] * D + rd[None, :], mask=rs[:, None] < T, other=0.0)
        sc = tl.dot(qb, tl.trans(kb), input_precision="ieee")
        keep = (rs[None, :] <= rt[:, None]) & (rs[None, :] < T)
        sc = tl.where(keep, sc, float("-inf"))
        m_new = tl.maximum(m, tl.max(sc, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(sc - m_new[:, None])
        den = den * alpha + tl.sum(p, axis=1)
        vb = tl.load(v + rs[:, None] * D + rd[None, :], mask=rs[:, None] < T, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, vb, input_precision="ieee")
        m = m_new
    tl.store(o + rt[:, None] * D + rd[None, :], acc / den[:, None], mask=rt[:, None] < T)


def flash_triton(q: np.ndarray, k: np.ndarray, v: np.ndarray, BT=32, BS=32) -> np.ndarray:
    T, D = q.shape
    tq = torch.as_tensor(q, dtype=torch.float32).cuda().contiguous()
    tk = torch.as_tensor(k, dtype=torch.float32).cuda().contiguous()
    tv = torch.as_tensor(v, dtype=torch.float32).cuda().contiguous()
    to = torch.empty((T, D), dtype=torch.float32, device="cuda")
    _flash_kernel[(triton.cdiv(T, BT),)](tq, tk, tv, to, T, D, BT, BS)
    torch.cuda.synchronize()
    return to.cpu().numpy()


@triton.jit
def _stencil_kernel(u, o, X, Y, alpha, BX: tl.constexpr, BY: tl.constexpr):
    # u is the (X+2, Y+2) halo; the interior (X, Y) writes to o. One fused
    # chain: nsum - 4*center, center + alpha*lap — five loads, one store.
    px, py = tl.program_id(0), tl.program_id(1)
    rx = px * BX + tl.arange(0, BX)
    ry = py * BY + tl.arange(0, BY)
    keep = (rx[:, None] < X) & (ry[None, :] < Y)
    hy = Y + 2
    base = (rx[:, None] + 1) * hy + (ry[None, :] + 1)
    c = tl.load(u + base, mask=keep, other=0.0)
    nsum = (
        tl.load(u + base + hy, mask=keep, other=0.0)
        + tl.load(u + base - hy, mask=keep, other=0.0)
        + tl.load(u + base + 1, mask=keep, other=0.0)
        + tl.load(u + base - 1, mask=keep, other=0.0)
    )
    tl.store(o + rx[:, None] * Y + ry[None, :], c + alpha * (nsum - 4.0 * c), mask=keep)


def stencil_triton(u: np.ndarray, alpha: float = 0.1, BX=32, BY=32) -> np.ndarray:
    X, Y = u.shape[0] - 2, u.shape[1] - 2
    tu = torch.as_tensor(u, dtype=torch.float32).cuda().contiguous()
    to = torch.empty((X, Y), dtype=torch.float32, device="cuda")
    _stencil_kernel[(triton.cdiv(X, BX), triton.cdiv(Y, BY))](tu, to, X, Y, alpha, BX, BY)
    torch.cuda.synchronize()
    return to.cpu().numpy()
