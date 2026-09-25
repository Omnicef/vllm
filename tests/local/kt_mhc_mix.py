#!/usr/bin/env python3
"""Device-only kernel time (torch.profiler, sum of kernel durations / calls) for the three mHC mix paths."""
import torch
from torch.profiler import ProfilerActivity, profile
from vllm.model_executor.kernels.mhc.triton_mix import mhc_mix_gemm
K, N, DEV = 16384, 24, "cuda:0"
fn = (torch.randn(N, K, device=DEV) * 0.02).float()
def det(f):
    def g(x):
        p = torch.are_deterministic_algorithms_enabled(); torch.use_deterministic_algorithms(True, warn_only=True)
        try: return f(x)
        finally: torch.use_deterministic_algorithms(p, warn_only=True)
    return g
P = {"rocblas-default": lambda x: torch.matmul(x, fn.t()), "rocblas-det": det(lambda x: torch.matmul(x, fn.t())),
     "triton": lambda x: mhc_mix_gemm(x, fn)}
for M in (1, 3, 16, 2048):
    x = torch.randn(M, K, device=DEV).float()
    row = []
    for n, f in P.items():
        for _ in range(10): f(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(50): f(x)
            torch.cuda.synchronize()
        us = sum(e.device_time_total for e in p.key_averages() if e.device_time_total > 0 and e.key != "ProfilerStep*") / 50
        row.append(f"{n} {us:8.1f}us")
    print(f"M={M:5d}: " + " | ".join(row), flush=True)
