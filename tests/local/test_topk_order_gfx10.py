#!/usr/bin/env python3
"""HIP top_k_per_row_prefill on one card at the indexer's chunk-6 shape (512 rows, causal row ends 1,968-2,479 pools
... here 512 rows over N=2480 columns, k=512): 21 calls on identical logits -> distinct raw outputs vs distinct
row-sorted outputs (GLM5_SORT_TOPK's sort), and whether the selected sets ever differ."""
import torch
import vllm._custom_ops  # noqa: F401

DEV, M, N, K = "cuda:0", 512, 2480, 512
g = torch.Generator().manual_seed(0)
logits = torch.randn(M, N, generator=g).to(DEV)
ks = torch.zeros(M, dtype=torch.int32, device=DEV)
ke = torch.arange(N - M + 1, N + 1, dtype=torch.int32, device=DEV)
raws = []
for _ in range(21):
    idx = torch.full((M, K), -1, dtype=torch.int32, device=DEV)
    torch.ops._C.top_k_per_row_prefill(logits, ks, ke, idx, M, logits.stride(0), logits.stride(1), K)
    raws.append(idx.clone())
torch.cuda.synchronize()
key = lambda t: t.cpu().numpy().tobytes()
srt = [torch.sort(torch.where(r < 0, torch.full_like(r, 1 << 30), r), dim=-1).values for r in raws]
print(f"raw outputs distinct {len({key(r) for r in raws})}/21; row-sorted outputs distinct {len({key(s) for s in srt})}/21; "
      f"sets always equal {all(torch.equal(srt[0], s) for s in srt)}; rows whose order differs run1 vs run2 "
      f"{int((raws[0] != raws[1]).any(-1).sum())}/{M}")
