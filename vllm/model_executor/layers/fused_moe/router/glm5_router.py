# SPDX-License-Identifier: Apache-2.0
"""GLM5_ROUTER_KERNEL=fused (local, NOT FOR UPSTREAM): GLM-5.3 MoE router for small token counts on gfx1030.

gate_gemv:   router logits = x @ W^T for fp16 x / fp16 W, fp32 accumulation in a fixed order, rounded to fp16 like
             today's F.linear output, then fp32. GLM5_ROUTER_GEMV=exact (default): 16x16 tl.dot tiles, K in order; its
             logits matched rocBLAS bitwise on every real-hidden-state batch tested (card 10). =skinny: 2 experts per
             program, all tokens, K chunks of 1024 summed as a tree then in order; ~7x faster, but differs from rocBLAS
             by 1 fp16 ulp in most batches (so weights, and rarely the chosen set, differ from today).
select:      sigmoid + correction bias + top-k (n_group == 1) + renormalize + routed scaling in one Triton program
             per token. Top-k = repeated arg-max, ties to the lowest index: the same set and order as the stable
             descending sort of GLM5_MOE_TOPK_STABLE. Weights are the unbiased sigmoid scores of the chosen experts,
             / their sum, * routed_scaling_factor (the grouped_topk op order).
No torch.compile, no buffer sized by max_num_batched_tokens. Callers fall back to the old path outside
1 <= tokens <= GLM5_ROUTER_KERNEL_MAX_TOKENS (default 16).
"""
import os

import torch

from vllm.triton_utils import tl, triton


def enabled() -> bool:
    return os.environ.get("GLM5_ROUTER_KERNEL") == "fused"


def max_tokens() -> int:
    return int(os.environ.get("GLM5_ROUTER_KERNEL_MAX_TOKENS", "16"))


@triton.jit
def _gate_gemv_kernel(x_ptr, w_ptr, out_ptr, M, E, K, sxm, swe, som,
                      BM: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr):
    pe = tl.program_id(0)
    pm = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    re = pe * BE + tl.arange(0, BE)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BE), dtype=tl.float32)
    for k0 in range(0, K, BK):
        xk = tl.load(x_ptr + rm[:, None] * sxm + (k0 + rk)[None, :],
                     mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        wk = tl.load(w_ptr + re[:, None] * swe + (k0 + rk)[None, :],
                     mask=(re[:, None] < E) & ((k0 + rk)[None, :] < K), other=0.0)
        acc += tl.dot(xk, tl.trans(wk), out_dtype=tl.float32)
    # today: F.linear returns fp16 logits, then .to(float32)
    res = acc.to(tl.float16).to(tl.float32)
    tl.store(out_ptr + rm[:, None] * som + re[None, :], res, mask=(rm[:, None] < M) & (re[None, :] < E))


@triton.jit
def _gate_gemv_skinny_kernel(x_ptr, w_ptr, out_ptr, M, E, K, sxm, swe, som,
                             BMP: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr):
    pe = tl.program_id(0)
    rm = tl.arange(0, BMP)
    re = pe * BE + tl.arange(0, BE)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BMP, BE), dtype=tl.float32)
    for k0 in range(0, K, BK):
        xk = tl.load(x_ptr + rm[:, None] * sxm + (k0 + rk)[None, :],
                     mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0).to(tl.float32)
        wk = tl.load(w_ptr + re[:, None] * swe + (k0 + rk)[None, :],
                     mask=(re[:, None] < E) & ((k0 + rk)[None, :] < K), other=0.0).to(tl.float32)
        acc += tl.sum(xk[:, None, :] * wk[None, :, :], axis=2)
    res = acc.to(tl.float16).to(tl.float32)
    tl.store(out_ptr + rm[:, None] * som + re[None, :], res, mask=(rm[:, None] < M) & (re[None, :] < E))


def gate_gemv(x: torch.Tensor, w: torch.Tensor, variant: str | None = None) -> torch.Tensor:
    """fp32 [M, E] router logits (fp16 precision, like today's gate) for fp16 x [M, K] and W [E, K]."""
    M, K = x.shape
    E = w.shape[0]
    x = x.contiguous()
    out = torch.empty((M, E), dtype=torch.float32, device=x.device)
    variant = variant or os.environ.get("GLM5_ROUTER_GEMV", "exact")
    if variant == "skinny" and M <= 16:
        BE, BK = 2, 1024
        _gate_gemv_skinny_kernel[(triton.cdiv(E, BE),)](
            x, w, out, M, E, K, x.stride(0), w.stride(0), out.stride(0),
            BMP=triton.next_power_of_2(M), BE=BE, BK=BK, num_warps=2)
        return out
    BM, BE, BK = 16, 16, 128
    _gate_gemv_kernel[(triton.cdiv(E, BE), triton.cdiv(M, BM))](
        x, w, out, M, E, K, x.stride(0), w.stride(0), out.stride(0), BM=BM, BE=BE, BK=BK, num_warps=2)
    return out


@triton.jit
def _select_kernel(logit_ptr, bias_ptr, w_out, id_out, E, sl, sw, si, scale,
                   TOPK: tl.constexpr, EP: tl.constexpr, RENORM: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, EP)
    valid = e < E
    x = tl.load(logit_ptr + m * sl + e, mask=valid, other=0.0)
    s = tl.sigmoid(x)
    b = tl.load(bias_ptr + e, mask=valid, other=0.0)
    key = tl.where(valid, s + b, float("-inf"))
    kk = tl.arange(0, TOPK)
    ids = tl.zeros((TOPK,), dtype=tl.int32)
    ws = tl.zeros((TOPK,), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        mx = tl.max(key, axis=0)
        cand = tl.where(key == mx, e, EP)
        sel = tl.min(cand, axis=0)
        wsel = tl.sum(tl.where(e == sel, s, 0.0), axis=0)
        ids = tl.where(kk == j, sel, ids)
        ws = tl.where(kk == j, wsel, ws)
        key = tl.where(e == sel, float("-inf"), key)
    if RENORM:
        ws = ws / tl.sum(ws, axis=0)
    ws = ws * scale
    tl.store(w_out + m * sw + kk, ws)
    tl.store(id_out + m * si + kk, ids)


def select(logits: torch.Tensor, bias: torch.Tensor, topk: int, renormalize: bool,
           routed_scaling_factor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(topk_weights fp32 [M, topk], topk_ids int32 [M, topk]) for sigmoid scoring, n_group == 1."""
    M, E = logits.shape
    logits = logits.contiguous()
    w = torch.empty((M, topk), dtype=torch.float32, device=logits.device)
    ids = torch.empty((M, topk), dtype=torch.int32, device=logits.device)
    _select_kernel[(M,)](logits, bias, w, ids, E, logits.stride(0), w.stride(0), ids.stride(0),
                         float(routed_scaling_factor), TOPK=topk, EP=triton.next_power_of_2(E),
                         RENORM=renormalize, num_warps=4)
    return w, ids
