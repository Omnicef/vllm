#!/usr/bin/env python3
"""mhc_pre_torch end to end: GLM5_MHC_KERNEL=triton vs GLM5_MHC_DET=1 vs an fp64 reference of the same function."""
import os, torch
import importlib; mk = importlib.import_module("vllm.model_executor.kernels.mhc.torch")
DEV = "cuda:0"; n, H = 4, 4096
torch.manual_seed(1)
fn = (torch.randn(2 * n + n * n, n * H, device=DEV) * 0.02).to(torch.bfloat16).float()
scale = torch.tensor([1.0, 0.5, 0.25], device=DEV); base = torch.randn(2 * n + n * n, device=DEV) * 0.1
args = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=2.0, sinkhorn_repeat=20)
def run(mode, res, f=fn, sc=scale, b=base):
    os.environ.pop("GLM5_MHC_KERNEL", None); os.environ.pop("GLM5_MHC_DET", None)
    if mode == "triton": os.environ["GLM5_MHC_KERNEL"] = "triton"
    if mode == "det": os.environ["GLM5_MHC_DET"] = "1"
    return mk.mhc_pre_torch(res, f, sc, b, **args)
for M in (0, 1, 3, 2048):
    res = torch.randn(M, n, H, device=DEV).to(torch.bfloat16)
    tri = [run("triton", res) for _ in range(20)]
    same = all(all(torch.equal(a, b) for a, b in zip(tri[0], t)) for t in tri[1:])
    det = run("det", res)
    # (the function asserts bf16/fp16 input, so the fp64 reference lives in bench_mhc_mix.py at GEMM level)
    if M:
        d = [(tri[0][i].float() - det[i].float()).abs().max().item() for i in range(3)]
        print(f"M={M}: triton 20/20 bitwise identical={same}  |triton - det| post={d[0]:.2e} comb={d[1]:.2e} layer_input={d[2]:.2e}")
    else:
        print(f"M=0: ok shapes {[tuple(t.shape) for t in tri[0]]}")
