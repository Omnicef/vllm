#!/usr/bin/env python3
"""GLM-5.3 expert selection on one card: real layer-10 gate weight + e_score_correction_bias, served router GEMM
(torch.mm bf16 -> fp32 out_dtype), noaux_tc selection = torch.topk(sigmoid(logits) + bias, 8) via the real
grouped_topk (n_group 1, topk_group 1, renormalize, scale 2.5). 21 calls on the same logits: distinct expert sets;
rows with an exact tie at the 8th/9th cut-off; sigmoid saturation. Candidate GLM5_MOE_TOPK_STABLE: selection by a
stable descending sort (ties -> lowest expert index)."""
import time
import torch
from safetensors import safe_open
import vllm.models.glm5next  # noqa: F401
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk

DEV = "cuda:0"
with safe_open("/models/GLM-5.3-Flash-AWQ-W4A16/model-00002-of-00009.safetensors", "pt") as f:
    W = f.get_tensor("model.language_model.layers.10.mlp.gate.weight").to(DEV)
    bias = f.get_tensor("model.language_model.layers.10.mlp.gate.e_score_correction_bias").to(DEV).float()
print(f"gate {W.dtype}{list(W.shape)}, bias {bias.dtype}{list(bias.shape)} range [{float(bias.min()):.4f}, {float(bias.max()):.4f}], "
      f"distinct bias values {bias.unique().numel()}/288")


def stable_topk_ids(logits, k=8):
    scores = logits.sigmoid() + bias[None]
    return torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :k].to(torch.int32)


g = torch.Generator().manual_seed(0)
for scale in (1.0, 4.0):
    h = (torch.randn(512, 4096, generator=g) * scale).to(DEV, torch.bfloat16)
    logits = torch.mm(h, W.T, out_dtype=torch.float32)
    sc = logits.sigmoid() + bias[None]
    s_sorted = sc.sort(dim=-1, descending=True).values
    ties = int((s_sorted[:, 7] == s_sorted[:, 8]).sum())
    sat = float((logits.sigmoid() == 1.0).float().mean())
    runs = []
    for _ in range(21):
        w, ids = grouped_topk(h, logits, topk=8, renormalize=True, num_expert_group=1, topk_group=1,
                              scoring_func="sigmoid", routed_scaling_factor=2.5, e_score_correction_bias=bias)
        runs.append(ids.sort(dim=-1).values.cpu())
    torch.cuda.synchronize()
    distinct = len({r.numpy().tobytes() for r in runs})
    rows_vary = int(torch.stack(runs).ne(runs[0][None]).any(0).any(-1).sum())
    st = [stable_topk_ids(logits).sort(dim=-1).values.cpu() for _ in range(21)]
    st_distinct = len({r.numpy().tobytes() for r in st})
    agree = int((st[0] == runs[0]).all(-1).sum())
    t0 = time.perf_counter()
    for _ in range(50): stable_topk_ids(logits)
    torch.cuda.synchronize(); t_st = (time.perf_counter() - t0) / 50 * 1e3
    t0 = time.perf_counter()
    for _ in range(50): torch.topk(logits.sigmoid() + bias[None], 8, dim=-1, sorted=True)
    torch.cuda.synchronize(); t_tk = (time.perf_counter() - t0) / 50 * 1e3
    print(f"scale {scale}: logits range [{float(logits.min()):.1f}, {float(logits.max()):.1f}], sigmoid==1.0 {sat:.3%}, "
          f"rows tied at the 8/9 cut-off {ties}/512 | grouped_topk: distinct sets {distinct}/21, rows varying {rows_vary} | "
          f"stable sort: distinct {st_distinct}/21, rows equal to grouped_topk run 1 {agree}/512, {t_st:.3f} vs topk {t_tk:.3f} ms",
          flush=True)
