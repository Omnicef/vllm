# SPDX-License-Identifier: Apache-2.0
"""Fused mHC for gfx1030 (GLM5_MHC_KERNEL=fused): the post block, the next pre block and its RMSNorm in two
Triton kernels instead of ~156 torch kernels (20 Sinkhorn iterations of elementwise ops dominate those).

K1 (_post_mix_kernel), program (token, split): for its K slice of the flattened residual, the new residual
    residual_cur[j, h] = sum_i comb[i, j] * residual[i, h] + post[j] * x[h]   (fp32, i in order, stored in the
    residual dtype), then the split-K partial of mixes = residual_cur_flat @ fn^T and of sum(residual_cur^2).
    The K split is exactly triton_mix's (same _splits_for, K_PER_SPLIT, BLOCK_K, loop), so the mixes equal the
    current path's GLM5_MHC_KERNEL=triton mixes whenever residual_cur does.
K2 (_finalize_kernel), program token: partials reduced in index order; rsqrt; pre / post sigmoids; comb softmax
    plus Sinkhorn with the current code's iteration count, epsilon and row/column order, in registers;
    layer_input = sum_i pre_i * residual_cur_i rounded to the residual dtype; RMSNorm as the vllm_c kernel
    (fp32 variance, one rounding of x * rsqrt * w).
Variants: pre only (layer 0: K1 reads the residual, no post), post only (last layer: K1 without the mix), fused
post+pre. fp32 accumulation, fixed reduction order, no atomics, grids depend on the token count only.
"""
import torch

from vllm.model_executor.kernels.mhc.triton_mix import _splits_for
from vllm.triton_utils import tl, triton

_BLOCK_K = 256
_BLOCK_D = 1024


@triton.jit
def _post_mix_kernel(x_ptr, res_ptr, post_ptr, comb_ptr, fn_ptr, rout_ptr, part_ptr, sq_ptr,
                     H, K, N, K_PER_SPLIT,
                     HC: tl.constexpr, HAS_POST: tl.constexpr, DO_MIX: tl.constexpr,
                     S: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    lane = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sq = tl.zeros((), dtype=tl.float32)
    k0 = s * K_PER_SPLIT
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        f0 = k0 + kk
        if HAS_POST:
            j = f0 // H
            h = f0 - j * H + lane
            v = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for i in tl.static_range(HC):
                c = tl.load(comb_ptr + t * HC * HC + i * HC + j)
                v += c * tl.load(res_ptr + t * K + i * H + h).to(tl.float32)
            pt = tl.load(post_ptr + t * HC + j) * tl.load(x_ptr + t * H + h).to(tl.float32)
            vr = (v + pt).to(rout_ptr.dtype.element_ty)
            tl.store(rout_ptr + t * K + f0 + lane, vr)
            xv = vr.to(tl.float32)
        else:
            xv = tl.load(res_ptr + t * K + f0 + lane).to(tl.float32)
        if DO_MIX:
            wv = tl.load(fn_ptr + offs_n[:, None] * K + f0 + lane[None, :],
                         mask=n_mask[:, None], other=0.0)
            acc += tl.sum(wv * xv[None, :], axis=1)
            sq += tl.sum(xv * xv, axis=0)
    if DO_MIX:
        tl.store(part_ptr + (t * S + s) * BLOCK_N + offs_n, acc)
        tl.store(sq_ptr + t * S + s, sq)


@triton.jit
def _finalize_kernel(part_ptr, sq_ptr, scale_ptr, base_ptr, res_ptr, nw_ptr,
                     post_out_ptr, comb_out_ptr, li_ptr,
                     H, K, rms_eps, pre_eps, sk_eps, post_mult, norm_eps,
                     HC: tl.constexpr, S: tl.constexpr, BLOCK_N: tl.constexpr, SINKHORN: tl.constexpr,
                     HAS_NORM: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    i4 = tl.arange(0, HC)
    pre_m = tl.zeros((HC,), dtype=tl.float32)
    post_m = tl.zeros((HC,), dtype=tl.float32)
    comb_m = tl.zeros((HC, HC), dtype=tl.float32)
    sq = tl.zeros((), dtype=tl.float32)
    for s in range(S):                                   # fixed order 0..S-1, as triton_mix stage 2
        p = part_ptr + (t * S + s) * BLOCK_N
        pre_m += tl.load(p + i4)
        post_m += tl.load(p + HC + i4)
        comb_m += tl.load(p + 2 * HC + i4[:, None] * HC + i4[None, :])
        sq += tl.load(sq_ptr + t * S + s)
    rs = tl.rsqrt(sq / K + rms_eps)
    pre = tl.sigmoid(pre_m * rs * tl.load(scale_ptr) + tl.load(base_ptr + i4)) + pre_eps
    post = tl.sigmoid(post_m * rs * tl.load(scale_ptr + 1) + tl.load(base_ptr + HC + i4)) * post_mult
    cl = comb_m * rs * tl.load(scale_ptr + 2) + tl.load(base_ptr + 2 * HC + i4[:, None] * HC + i4[None, :])
    e = tl.exp(cl - tl.max(cl, axis=1)[:, None])
    comb = e / tl.sum(e, axis=1)[:, None] + sk_eps                  # softmax(dim=-1) + eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + sk_eps)           # / (sum(dim=-2) + eps)
    for _ in range(SINKHORN - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + sk_eps)      # rows (dim=-1)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + sk_eps)      # columns (dim=-2)
    tl.store(post_out_ptr + t * HC + i4, post)
    tl.store(comb_out_ptr + t * HC * HC + i4[:, None] * HC + i4[None, :], comb)

    ss = tl.zeros((), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_D):
        h = h0 + tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for i in tl.static_range(HC):
            pi = tl.sum(tl.where(i4 == i, pre, 0.0), axis=0)
            acc += pi * tl.load(res_ptr + t * K + i * H + h).to(tl.float32)
        li = acc.to(li_ptr.dtype.element_ty)
        tl.store(li_ptr + t * H + h, li)
        lf = li.to(tl.float32)
        ss += tl.sum(lf * lf, axis=0)
    if HAS_NORM:
        rn = tl.rsqrt(ss / H + norm_eps)
        for h0 in range(0, H, BLOCK_D):
            h = h0 + tl.arange(0, BLOCK_D)
            lf = tl.load(li_ptr + t * H + h).to(tl.float32)
            w = tl.load(nw_ptr + h).to(tl.float32)
            tl.store(li_ptr + t * H + h, (lf * rn * w).to(li_ptr.dtype.element_ty))


def fused_supported(residual: torch.Tensor, fn: torch.Tensor | None = None) -> bool:
    hc, h = residual.shape[-2], residual.shape[-1]
    return (residual.dtype in (torch.float16, torch.bfloat16) and residual.is_cuda
            and hc & (hc - 1) == 0 and h % _BLOCK_D == 0 and h % _BLOCK_K == 0
            and (fn is None or (fn.dtype == torch.float32 and fn.shape[1] == hc * h
                                and fn.shape[0] <= triton.next_power_of_2(2 * hc + hc * hc))))


def _k1(x, res, post, comb, fn, rout, do_mix):
    T, HC, H = res.shape
    K = HC * H
    S = _splits_for(T)
    kps = triton.cdiv(triton.cdiv(K, S), _BLOCK_K) * _BLOCK_K
    assert K % kps == 0 and kps % _BLOCK_K == 0 and H % _BLOCK_K == 0
    N = fn.shape[0] if do_mix else 1
    BLOCK_N = triton.next_power_of_2(N) if do_mix else 1
    part = torch.empty((T, S, BLOCK_N), dtype=torch.float32, device=res.device) if do_mix else res
    sq = torch.empty((T, S), dtype=torch.float32, device=res.device) if do_mix else res
    has_post = x is not None
    _post_mix_kernel[(T, S)](
        x if has_post else res, res, post if has_post else res, comb if has_post else res,
        fn if do_mix else res, rout if has_post else res, part, sq, H, K, N, kps,
        HC=HC, HAS_POST=has_post, DO_MIX=do_mix, S=S, BLOCK_N=BLOCK_N, BLOCK_K=_BLOCK_K, num_warps=4)
    return part, sq, S, BLOCK_N


def _k2(part, sq, S, BLOCK_N, res_cur, scale, base, rms_eps, pre_eps, sk_eps, post_mult, sinkhorn,
        norm_weight, norm_eps):
    T, HC, H = res_cur.shape
    post = torch.empty((T, HC, 1), dtype=torch.float32, device=res_cur.device)
    comb = torch.empty((T, HC, HC), dtype=torch.float32, device=res_cur.device)
    li = torch.empty((T, H), dtype=res_cur.dtype, device=res_cur.device)
    _finalize_kernel[(T,)](
        part, sq, scale, base, res_cur, norm_weight if norm_weight is not None else li, post, comb, li,
        H, HC * H, rms_eps, pre_eps, sk_eps, post_mult, norm_eps,
        HC=HC, S=S, BLOCK_N=BLOCK_N, SINKHORN=sinkhorn, HAS_NORM=norm_weight is not None,
        BLOCK_D=_BLOCK_D, num_warps=4)
    return post, comb, li


def _flat(residual):
    hc, h = residual.shape[-2], residual.shape[-1]
    return residual.reshape(-1, hc, h).contiguous()


def mhc_fused_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                       hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight=None, norm_eps=0.0):
    """-> residual_cur, post_mix [..., hc, 1], comb_mix [..., hc, hc], layer_input (normed if norm_weight)."""
    outer = residual.shape[:-2]
    res = _flat(residual)
    T, HC, H = res.shape
    rout = torch.empty_like(res)
    part, sq, S, BN = _k1(x.reshape(T, H).contiguous(), res, post_layer_mix.reshape(T, HC).contiguous().float(),
                          comb_res_mix.reshape(T, HC, HC).contiguous().float(), fn.contiguous(), rout, True)
    post, comb, li = _k2(part, sq, S, BN, rout, hc_scale.contiguous(), hc_base.contiguous(), rms_eps,
                         hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps)
    return (rout.view(*outer, HC, H), post.view(*outer, HC, 1), comb.view(*outer, HC, HC),
            li.view(*outer, H))


def mhc_fused_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                  sinkhorn_repeat, norm_weight=None, norm_eps=0.0):
    """Pre only (no incoming post): -> post_mix, comb_mix, layer_input."""
    outer = residual.shape[:-2]
    res = _flat(residual)
    T, HC, H = res.shape
    part, sq, S, BN = _k1(None, res, None, None, fn.contiguous(), None, True)
    post, comb, li = _k2(part, sq, S, BN, res, hc_scale.contiguous(), hc_base.contiguous(), rms_eps,
                         hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps)
    return post.view(*outer, HC, 1), comb.view(*outer, HC, HC), li.view(*outer, H)


def mhc_fused_post(x, residual, post_layer_mix, comb_res_mix):
    """Post only (last layer): -> residual_cur."""
    outer = residual.shape[:-2]
    res = _flat(residual)
    T, HC, H = res.shape
    rout = torch.empty_like(res)
    _k1(x.reshape(T, H).contiguous(), res, post_layer_mix.reshape(T, HC).contiguous().float(),
        comb_res_mix.reshape(T, HC, HC).contiguous().float(), None, rout, False)
    return rout.view(*outer, HC, H)
