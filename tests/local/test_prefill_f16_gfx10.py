"""GLM5_INDEXER_PREFILL_F16=1 vs the current prefill scoring (fp8_mqa_logits_torch), one card, prefill shapes
M=512 rows, H=32 heads, D=128, N keys 2656 (10.6k ctx) and 8192 (32k). Real-ish fp8 values (randn, e4m3), real
dtypes. Reports time per call, max |diff| vs the current path, and each path's error against an exact fp32
reference (fp8 values are exact in fp32; fp32 GEMM), plus the rows whose k-th score (k=512) is tied across the
cutoff in each path."""
import os
import torch
import vllm.models.glm5next  # noqa: F401
from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch

DEV, M, H, D, K = "cuda:0", 512, 32, 128, 512
g = torch.Generator().manual_seed(0)
F8 = current_platform.fp8_dtype()
for N in (2656, 8192):
    q = (torch.randn(M, H, D, generator=g) * 1.5).to(F8).to(DEV)
    k = (torch.randn(N, D, generator=g) * 1.5).to(F8).to(DEV)
    sc = (torch.rand(N, generator=g) * 0.01 + 0.001).to(DEV)
    w = (torch.rand(M, H, generator=g) * 0.1).to(DEV)
    ks = torch.zeros(M, dtype=torch.int32, device=DEV)
    ke = torch.clamp(torch.arange(M, device=DEV, dtype=torch.int32) + N - M + 1, max=N)
    run = lambda: fp8_mqa_logits_torch(q, (k, sc), w, ks, ke)
    out, t = {}, {}
    for mode in ("0", "1"):
        os.environ["GLM5_INDEXER_PREFILL_F16"] = mode
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); o = run(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
        out[mode], t[mode] = o, sorted(ts)[10]
    rep_ok = torch.equal(out["1"], run())
    ref = (torch.einsum("mhd,nd->hmn", q.float(), k.float()) * sc).relu()
    ref = (ref * w.unsqueeze(-1).transpose(0, 1)).sum(0)
    mask = torch.arange(N, device=DEV)[None, :] < ke[:, None]
    ref = ref.masked_fill(~mask, float("-inf"))
    fin = mask
    def err(x):
        d = (x - ref)[fin].abs()
        return float(d.max()), float((d / ref[fin].abs().clamp(min=1e-6)).median())
    def ties(x):
        v = x.topk(K, dim=-1).values[:, -1:]
        above = (x > v).sum(-1); tied = (x == v).sum(-1)
        return int((tied > K - above).sum())
    dmax = float((out["1"] - out["0"])[fin].abs().max())
    e0, e1 = err(out["0"]), err(out["1"])
    print(f"N={N}: time current {t['0']:.2f} ms, f16 {t['1']:.2f} ms | max |f16 - current| {dmax:.3e} "
          f"(|score| median {float(ref[fin].abs().median()):.3e}) | vs exact fp32: current max {e0[0]:.2e} "
          f"median rel {e0[1]:.1e}, f16 max {e1[0]:.2e} median rel {e1[1]:.1e} | rows tied at the k={K} cutoff: "
          f"current {ties(out['0'])}, f16 {ties(out['1'])}, exact {ties(ref)} | f16 repeat identical {rep_ok}")
