# SPDX-License-Identifier: Apache-2.0
"""Deterministic mHC mixing GEMM for the torch fallback (GLM5_MHC_KERNEL=triton).

mixes[M, N] = x[M, K] @ fn[N, K]^T with N = 2n + n^2 = 24, K = n * hidden = 16384, fp32.

Split-K in two fixed-order stages, so the result is deterministic by construction:
  stage 1: program (m, s) sums its K slice for all N columns in a fixed loop -> part[m, s, :]
  stage 2: program m adds part[m, 0..S-1, :] in index order -> out[m, :]
No atomics, no data-dependent order. S is chosen from M only, so a given M always
reduces the same way. Small M needs many K slices to occupy the GPU; large M does not.
"""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mix_partial_kernel(x_ptr, w_ptr, part_ptr, K, N, K_PER_SPLIT,
                        stride_xm, stride_wn, S: tl.constexpr,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    m = tl.program_id(0)
    s = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    k0 = s * K_PER_SPLIT
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k0 + kk + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        xv = tl.load(x_ptr + m * stride_xm + offs_k, mask=k_mask, other=0.0)
        wv = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                     mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.sum(wv * xv[None, :], axis=1)
    tl.store(part_ptr + (m * S + s) * BLOCK_N + offs_n, acc)


@triton.jit
def _mix_reduce_kernel(part_ptr, out_ptr, N, stride_om, S: tl.constexpr, BLOCK_N: tl.constexpr):
    m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in range(S):                       # fixed order: 0, 1, ..., S-1
        acc += tl.load(part_ptr + (m * S + s) * BLOCK_N + offs_n)
    tl.store(out_ptr + m * stride_om + offs_n, acc, mask=offs_n < N)


def _splits_for(M: int) -> int:
    # enough programs to cover the GPU (36 CUs on gfx1030) at small M; one slice per
    # row once M alone fills it. Depends on M only -> same reduction order every call.
    if M <= 4:
        return 64
    if M <= 64:
        return 16
    return 1


def mhc_mix_gemm(x: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """x [M, K] fp32 (contiguous rows), fn [N, K] fp32 -> [M, N] fp32."""
    assert x.dtype == torch.float32 and fn.dtype == torch.float32
    M, K = x.shape
    N = fn.shape[0]
    if M == 0:
        return torch.empty((0, N), dtype=torch.float32, device=x.device)
    x = x.contiguous()
    fn = fn.contiguous()
    S = _splits_for(M)
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_K = 256
    k_per_split = triton.cdiv(triton.cdiv(K, S), BLOCK_K) * BLOCK_K
    part = torch.empty((M, S, BLOCK_N), dtype=torch.float32, device=x.device)
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    _mix_partial_kernel[(M, S)](x, fn, part, K, N, k_per_split, x.stride(0), fn.stride(0),
                                S=S, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    _mix_reduce_kernel[(M,)](part, out, N, out.stride(0), S=S, BLOCK_N=BLOCK_N, num_warps=1)
    return out
