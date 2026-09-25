#!/usr/bin/env python3
"""Tile sweep for triton_mix at M in {1, 3, 16}: splits S, BLOCK_K, num_warps. Median us/call."""
import statistics, torch
import vllm.model_executor.kernels.mhc.triton_mix as tm
from vllm.triton_utils import triton

K, N, DEV = 16384, 24, "cuda:0"
fn = (torch.randn(N, K, device=DEV) * 0.02).float()


def run(x, S, BK, W):
    M = x.shape[0]; BN = 32
    kps = triton.cdiv(triton.cdiv(K, S), BK) * BK
    part = torch.empty((M, S, BN), dtype=torch.float32, device=DEV)
    out = torch.empty((M, N), dtype=torch.float32, device=DEV)
    tm._mix_partial_kernel[(M, S)](x, fn, part, K, N, kps, x.stride(0), fn.stride(0), S=S, BLOCK_N=BN, BLOCK_K=BK, num_warps=W)
    tm._mix_reduce_kernel[(M,)](part, out, N, out.stride(0), S=S, BLOCK_N=BN, num_warps=1)
    return out


def t(x, *a):
    for _ in range(10): run(x, *a)
    torch.cuda.synchronize(); ts = []
    for _ in range(100):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); run(x, *a); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
    return statistics.median(ts)


for M in (1, 3, 16):
    x = torch.randn(M, K, device=DEV).float()
    res = sorted((t(x, S, BK, W), S, BK, W) for S in (16, 32, 64, 128, 256) for BK in (64, 128, 256) for W in (1, 2, 4, 8))
    print(f"M={M}: best " + "; ".join(f"{us:.1f}us S={S} BK={BK} W={W}" for us, S, BK, W in res[:4]), flush=True)
