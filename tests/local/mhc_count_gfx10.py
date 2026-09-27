"""One card: kernels launched and GPU time per GLM-5.3 mHC fused post+pre call (the ROCm path: mhc_post_torch +
mhc_pre_torch with GLM5_MHC_KERNEL=triton + RMSNorm), fp16 residual, at decode token counts. Captured replay time
is how decode runs; x90 calls per step (45 layers x attn + ffn)."""
import os
os.environ["GLM5_MHC_KERNEL"] = "triton"
import torch
from torch.profiler import profile, ProfilerActivity
import vllm.models.glm5next  # noqa: F401
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch

DEV, HC, H = "cuda:0", 4, 4096
g = torch.Generator().manual_seed(0)
fn = (torch.randn(2 * HC + HC * HC, HC * H, generator=g) * 0.01).to(DEV)
scale = torch.tensor([1.0, 1.0, 1.0], device=DEV); base = torch.zeros(2 * HC + HC * HC, device=DEV)
nw = torch.ones(H, device=DEV, dtype=torch.float16)


def rms(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def call(x, res, post, comb):
    r = mhc_post_torch(x, res, post, comb)
    p, c, li = mhc_pre_torch(r, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
    return r, p, c, rms(li, nw)


for T in (1, 3, 24):
    x = torch.randn(T, H, device=DEV, dtype=torch.float16)
    res = torch.randn(T, HC, H, device=DEV, dtype=torch.float16)
    post = torch.rand(T, HC, 1, device=DEV); comb = torch.rand(T, HC, HC, device=DEV)
    for _ in range(3):
        call(x, res, post, comb)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call(x, res, post, comb); torch.cuda.synchronize()
    kern = [e for e in prof.events() if e.device_type.name == "CUDA"]
    names = {}
    for e in kern:
        k = e.name.split("<")[0][:60]; names[k] = names.get(k, 0) + 1
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        call(x, res, post, comb)
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = call(x, res, post, comb)
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); gr.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    med = sorted(ts)[25]
    busy = sum(e.device_time for e in kern) / 1000.0 if kern and hasattr(kern[0], "device_time") else float("nan")
    print(f"T={T}: {len(kern)} kernels per call | captured replay {med * 1000:.0f} us/call -> x90 {90 * med:.1f} ms/step "
          f"| summed kernel time (eager profile) {busy * 1000:.0f} us")
    if T == 3:
        for k, v in sorted(names.items(), key=lambda x: -x[1])[:12]:
            print(f"    {v:4d}  {k}")
