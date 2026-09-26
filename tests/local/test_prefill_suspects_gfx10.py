#!/usr/bin/env python3
"""Prefill-path suspects for the >17k-token engine fault, one card: HIP top_k_per_row_prefill, the kpool
expand_pools_and_append_tail, and the torch prefill logits, at 512 query rows (MAX_BATCHED 512) and N live pooled
keys around 4,096 and at the dying step (4,480 / 4,608). Each N runs 3x with a device sync; a fault aborts the process.
Also checks the top-k output against torch.topk (set equality per row) and the expand output bounds."""
import torch
from vllm.platforms import current_platform
import vllm.models.glm5next  # noqa: F401
from vllm.models.glm5next.amd.ops.kpool_compress import expand_pools_and_append_tail
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch
import vllm._custom_ops  # noqa: F401  (loads torch.ops._C)

DEV, FP8, H, D, M, KP, TOPK = "cuda:0", current_platform.fp8_dtype(), 32, 128, 512, 4, 512
g = torch.Generator(device="cpu").manual_seed(0)
for N in (4095, 4096, 4097, 4200, 4480, 4608):
    q = torch.randn(M, H, D, generator=g).to(DEV).to(FP8); w = torch.randn(M, H, generator=g).to(DEV)
    k = torch.randn(N, D, generator=g).to(DEV).to(FP8); s = torch.rand(N, generator=g).to(DEV) + 0.1
    ke = torch.arange(N - M + 1, N + 1, dtype=torch.int32, device=DEV)      # causal: row i sees N-M+1+i pools
    ks = torch.zeros(M, dtype=torch.int32, device=DEV)
    ok = True
    for _ in range(3):
        logits = fp8_mqa_logits_torch(q, (k, s[:, None]), w, ks, ke)
        idx = torch.full((M, TOPK), -1, dtype=torch.int32, device=DEV)
        torch.ops._C.top_k_per_row_prefill(logits, ks, ke, idx, M, logits.stride(0), logits.stride(1), TOPK)
        seq = (ke * KP + 2).to(torch.int32)                                   # tokens, with a 2-token tail
        out = expand_pools_and_append_tail(idx, seq, KP)
        torch.cuda.synchronize()
        ref = torch.topk(logits, TOPK, dim=1).indices
        same = all(set(idx[r].tolist()) == set(ref[r].tolist()) for r in (0, M // 2, M - 1))
        inb = bool(((out < seq[:, None]) | (out == -1)).all())
        ok &= same and inb
    print(f"N {N}: logits {tuple(logits.shape)}, top-k matches torch.topk {same}, expand in bounds {inb}, 3/3 no fault -> {'ok' if ok else 'MISMATCH'}", flush=True)
print("done")
