#!/usr/bin/env python3
"""Phase 7a bench on one GPU: the mHC mixing GEMM x[M,16384] fp32 @ fn[24,16384]^T.

Paths: rocblas-default (torch.matmul), rocblas-det (torch.use_deterministic_algorithms, as GLM5_MHC_DET=1),
triton (vllm.model_executor.kernels.mhc.triton_mix). For M in {1, 3, 16, 2048}: per-call time (hip events,
median of 200 after warm-up), bitwise-distinct results over 50 runs, max |diff| vs an fp64 reference, and
the GPU kernel name(s) each path launches (torch.profiler).
"""
import statistics, torch
from torch.profiler import ProfilerActivity, profile
from vllm.model_executor.kernels.mhc.triton_mix import mhc_mix_gemm

K, N, DEV = 16384, 24, "cuda:0"
torch.manual_seed(0)
fn = (torch.randn(N, K, device=DEV) * 0.02).float()   # fp32, the checkpoint's hc_*_fn dtype


def det(f):
    def g(x):
        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            return f(x)
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=True)
    return g


PATHS = {
    "rocblas-default": lambda x: torch.matmul(x, fn.t()),
    "rocblas-det": det(lambda x: torch.matmul(x, fn.t())),
    "triton": lambda x: mhc_mix_gemm(x, fn),
}


def timeit(f, x, iters=200):
    for _ in range(20): f(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); f(x); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


def kernels(f, x):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        f(x); torch.cuda.synchronize()
    return sorted({e.name[:70] for e in p.events() if e.device_type.name == "CUDA"})


print(f"{'M':>5s} {'path':16s} {'us/call':>9s} {'distinct/50':>11s} {'max|diff| vs fp64':>18s}  kernels")
for M in (1, 3, 16, 2048):
    x = (torch.randn(M, K, device=DEV) * 1.0).to(torch.bfloat16).float()   # bf16 residual cast to fp32, as in the model
    ref = (x.double() @ fn.double().t())
    for name, f in PATHS.items():
        outs = [f(x).clone() for _ in range(50)]
        distinct = len({o.cpu().numpy().tobytes() for o in outs})
        err = (outs[0].double() - ref).abs().max().item()
        us = timeit(f, x)
        print(f"{M:5d} {name:16s} {us:9.1f} {distinct:11d} {err:18.3e}  {' | '.join(kernels(f, x))}", flush=True)
