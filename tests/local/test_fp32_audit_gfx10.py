#!/usr/bin/env python3
"""fp32 / reduction-GEMM audit of the served GLM-5.3 forward on one card (glm53-v030, GLM5_MHC_KERNEL=triton).
Engine shapes (hidden 4096, hc_mult 4, 288 experts, index 32 heads x 128), decode M = 1 / 3 and a 512-row prefill
chunk. Each op: 21 calls on the same inputs, count of distinct outputs (1 = bitwise repeatable)."""
import torch
from vllm.model_executor.kernels.mhc.triton_mix import mhc_mix_gemm

DEV, H, HC, E = "cuda:0", 4096, 4, 288
g = torch.Generator(device="cpu").manual_seed(0)
r = lambda *s, dt=torch.bfloat16, sc=1.0: (torch.randn(*s, generator=g) * sc).to(DEV, dt)
w_idx = r(32, H, sc=0.02); w_idx_t = w_idx.t().contiguous().float(); w_idx_nk = w_idx.float().contiguous()
w_rt_bf = r(E, H, sc=0.02); w_rt_f32 = w_rt_bf.float()
w_gate = r(128, H, sc=0.02)
w_mix = r(3 * HC + HC * HC - 2 * HC + 2 * HC, HC * H, dt=torch.float32, sc=0.01)   # 24 x 16384
ops = {
    "indexer head weights fp32 torch.mm (stock)": lambda x: torch.mm(x.float(), w_idx_t),
    "indexer head weights mhc_mix_gemm (knob)": lambda x: mhc_mix_gemm(x.float().contiguous(), w_idx_nk),
    "router tier 4 torch.mm bf16->fp32 out_dtype": lambda x: torch.mm(x, w_rt_bf.T, out_dtype=torch.float32),
    "router tier 5 F.linear fp32": lambda x: torch.nn.functional.linear(x.float(), w_rt_f32),
    "kpool gate score F.linear bf16": lambda x: torch.nn.functional.linear(x, w_gate),
}
mix = lambda xm: mhc_mix_gemm(xm, w_mix)
def post(comb, res):
    return torch.einsum("...ij,...ih->...jh", comb, res)
print(f"{'op':48s} " + "  ".join(f"M={m:<4d}" for m in (1, 3, 512)))
for name, fn in ops.items():
    row = []
    for M in (1, 3, 512):
        x = r(M, H)
        outs = [fn(x) for _ in range(21)]; torch.cuda.synchronize()
        row.append(len({o.contiguous().view(torch.uint8).cpu().numpy().tobytes() for o in outs}))
    print(f"{name:48s} " + "  ".join(f"{c:>2d}/21 " for c in row), flush=True)
row = []
for M in (1, 3, 512):
    xm = r(M, HC * H, dt=torch.float32)
    outs = [mix(xm) for _ in range(21)]; torch.cuda.synchronize()
    row.append(len({o.contiguous().view(torch.uint8).cpu().numpy().tobytes() for o in outs}))
print(f"{'mHC mixing mhc_mix_gemm [M,16384]x[24,16384]':48s} " + "  ".join(f"{c:>2d}/21 " for c in row), flush=True)
row = []
for M in (1, 3, 512):
    comb = r(M, HC, HC, dt=torch.float32); res = r(M, HC, H, dt=torch.float32)
    outs = [post(comb, res) for _ in range(21)]; torch.cuda.synchronize()
    row.append(len({o.contiguous().view(torch.uint8).cpu().numpy().tobytes() for o in outs}))
print(f"{'mHC post einsum fp32 [M,4,4]x[M,4,4096]':48s} " + "  ".join(f"{c:>2d}/21 " for c in row), flush=True)
