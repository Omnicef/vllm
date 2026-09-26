#!/usr/bin/env python3
"""The GLM-5.3 indexer head-weight GEMM (attention.py: weights = torch.mm(hidden.float(), W_fp32), [M,6144]x[6144,32])
on one card: is stock torch.mm fp32 bitwise repeatable, and is the deterministic Triton GEMM (mhc_mix_gemm) a
drop-in? 20 repeats per M, max rel diff vs a float64 reference, time per call."""
import time
import torch
from vllm.model_executor.kernels.mhc.triton_mix import mhc_mix_gemm

DEV, K, N = "cuda:0", 6144, 32
g = torch.Generator(device="cpu").manual_seed(0)
w = (torch.randn(N, K, generator=g) * 0.02).to(DEV, torch.bfloat16)
wt = w.t().contiguous().float()              # the stock cached [K, N] fp32
wn = w.float().contiguous()                   # [N, K] fp32 for mhc_mix_gemm
for M in (128, 512, 2048):
    x = torch.randn(M, K, generator=g).to(DEV, torch.bfloat16)
    ref = (x.double() @ w.double().t())
    for name, fn in (("torch.mm fp32 (stock)", lambda: torch.mm(x.float(), wt)),
                     ("mhc_mix_gemm (Triton)", lambda: mhc_mix_gemm(x.float().contiguous(), wn))):
        out = fn(); torch.cuda.synchronize()
        reps = [fn() for _ in range(20)]; torch.cuda.synchronize()
        distinct = len({hash(r.cpu().numpy().tobytes()) for r in [out] + reps})
        t0 = time.perf_counter()
        for _ in range(50): fn()
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / 50 * 1e3
        rel = float((out.double() - ref).abs().max() / ref.abs().max())
        print(f"M {M:5d} {name}: distinct outputs over 21 calls {distinct}, max rel diff vs fp64 {rel:.1e}, {ms:.3f} ms/call", flush=True)
