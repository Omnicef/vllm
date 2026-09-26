#!/usr/bin/env python3
"""GLM-5.3 routed MoE (int4 W4A16, fused_experts) at the chunk-0 shapes on one card: bitwise repeatability over 21
calls with the shipped V620 configs (E=288,N=256: SPLIT_K=1 at every M), plus time per call. Hidden 4096, per-rank
intermediate 2048/8 = 256, 288 experts, top-8, group 128. Topk ids sorted per row (GLM5_MOE_SORTED)."""
import time
import torch
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_op

DEV, K, N, E, TOPK, G = "cuda:0", 4096, 256, 288, 8, 128
g = torch.Generator(device="cpu").manual_seed(0)
w1 = torch.randint(0, 256, (E, 2 * N, K // 2), generator=g, dtype=torch.uint8).to(DEV)
w2 = torch.randint(0, 256, (E, K, N // 2), generator=g, dtype=torch.uint8).to(DEV)
s1 = (torch.rand(E, 2 * N, K // G, generator=g) * 0.01).to(DEV, torch.bfloat16)
s2 = (torch.rand(E, K, N // G, generator=g) * 0.01).to(DEV, torch.bfloat16)
for M in (128, 512):
    x = torch.randn(M, K, generator=g).to(DEV, torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(M)]).sort(dim=1).values.to(DEV, torch.int32)
    wts = torch.softmax(torch.randn(M, TOPK, generator=g), -1).to(DEV, torch.float32)
    fn = lambda: fused_experts_op(x, w1, w2, wts, ids, use_int4_w4a16=True, w1_scale=s1, w2_scale=s2,
                                  block_shape=[0, G])
    fn(); torch.cuda.synchronize()
    outs = [fn() for _ in range(21)]; torch.cuda.synchronize()
    distinct = len({o.contiguous().view(torch.uint8).cpu().numpy().tobytes() for o in outs})
    t0 = time.perf_counter()
    for _ in range(20): fn()
    torch.cuda.synchronize()
    print(f"M {M:4d}: fused_experts int4 W4A16 (shipped config, SPLIT_K=1): distinct {distinct}/21, "
          f"{(time.perf_counter() - t0) / 20 * 1e3:.2f} ms/call, finite {bool(torch.isfinite(outs[0]).all())}", flush=True)
