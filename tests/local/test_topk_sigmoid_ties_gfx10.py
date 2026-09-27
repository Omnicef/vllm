#!/usr/bin/env python3
"""glm53-main routing for GLM-5.3 on one card: n_group=1/topk_group=1 is degenerate, so main's router factory picks
FusedTopKBiasRouter, which (aiter off, sigmoid) runs ops.topk_sigmoid. Same real layer-10 gate + correction bias and
the same tied-score setup as tests/local/test_moe_topk_det_gfx10.py: 21 calls, distinct expert sets, rows tied at the
8/9 cut-off, and agreement with a stable descending sort (ties -> lowest expert index)."""
import torch
from safetensors import safe_open
import vllm.models.glm5next  # noqa: F401
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias

DEV = "cuda:0"
with safe_open("/models/GLM-5.3-Flash-AWQ-W4A16/model-00002-of-00009.safetensors", "pt") as f:
    W = f.get_tensor("model.language_model.layers.10.mlp.gate.weight").to(DEV)
    bias = f.get_tensor("model.language_model.layers.10.mlp.gate.e_score_correction_bias").to(DEV).float()
g = torch.Generator().manual_seed(0)
for scale in (1.0, 4.0):
    h = (torch.randn(512, 4096, generator=g) * scale).to(DEV, torch.bfloat16)
    logits = torch.mm(h, W.T, out_dtype=torch.float32)
    sc = logits.sigmoid() + bias[None]
    s_sorted = sc.sort(dim=-1, descending=True).values
    ties = int((s_sorted[:, 7] == s_sorted[:, 8]).sum())
    runs = []
    for _ in range(21):
        w, ids = fused_topk_bias(h, logits, "sigmoid", bias, 8, True, routed_scaling_factor=2.5)
        runs.append(ids.sort(dim=-1).values.cpu())
    torch.cuda.synchronize()
    distinct = len({r.numpy().tobytes() for r in runs})
    stable = torch.sort(sc, dim=-1, descending=True, stable=True).indices[:, :8].sort(dim=-1).values.cpu().to(runs[0].dtype)
    agree = int((stable == runs[0]).all(-1).sum())
    tied_rows = (s_sorted[:, 7] == s_sorted[:, 8]).nonzero().flatten().cpu()
    agree_tied = int((stable[tied_rows] == runs[0][tied_rows]).all(-1).sum()) if len(tied_rows) else 0
    print(f"scale {scale}: rows tied at 8/9 {ties}/512 | fused_topk_bias (ops.topk_sigmoid): distinct sets {distinct}/21 | "
          f"equal to stable-sort choice {agree}/512 rows (tied rows {agree_tied}/{len(tied_rows)})", flush=True)
