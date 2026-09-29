"""Hook points: GateLinear.forward and GroupedTopKRouter._compute_routing with GLM5_ROUTER_KERNEL=fused vs unset.
Flag on + M <= 16 -> fused path, equal to today's; M = 24 -> today's path; flag off -> today's path."""
import os, types
os.environ["GLM5_MOE_TOPK_STABLE"] = "1"
import torch, torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
from vllm.model_executor.layers.fused_moe.router import glm5_router as G
dev = "cuda"; torch.manual_seed(0)
W = (torch.randn(288, 4096, device=dev) * 0.02).half(); B = torch.randn(288, device=dev) * 0.01
gate = types.SimpleNamespace(weight=W, bias=None, out_dtype=torch.float32, allow_ll_bf16_gemm=False,
                             allow_fp32_router_gemm=False, allow_bf16x3_router_gemm=False, allow_cublas_router_gemm=False)
router = GroupedTopKRouter(top_k=8, global_num_experts=288, num_expert_group=1, topk_group=1, renormalize=True,
                           scoring_func="sigmoid", routed_scaling_factor=2.5, e_score_correction_bias=B)
calls = {"gemv": 0, "select": 0}
og, os_ = G.gate_gemv, G.select
G.gate_gemv = lambda *a, **k: (calls.__setitem__("gemv", calls["gemv"] + 1), og(*a, **k))[1]
G.select = lambda *a, **k: (calls.__setitem__("select", calls["select"] + 1), os_(*a, **k))[1]
ok = True
for flag in ("fused", None):
    if flag: os.environ["GLM5_ROUTER_KERNEL"] = flag
    else: os.environ.pop("GLM5_ROUTER_KERNEL", None)
    for M in (1, 3, 16, 24):
        x = torch.randn(M, 4096, device=dev).half()
        calls.update(gemv=0, select=0)
        if flag and M <= 16:
            lg, _ = GateLinear.forward(gate, x)
        else:
            lg = F.linear(x, W).to(torch.float32)          # Tier 5 (what GateLinear does with the flag off)
        w, i = router._compute_routing(x, lg, torch.int32)
        ref_w, ref_i = router.__class__.__mro__[0]._compute_routing.__wrapped__(router, x, lg, torch.int32) if False else (None, None)
        os.environ.pop("GLM5_ROUTER_KERNEL", None)
        rw, ri = router._compute_routing(x, F.linear(x, W).to(torch.float32), torch.int32)   # today's path
        if flag: os.environ["GLM5_ROUTER_KERNEL"] = flag
        expect = (1, 1) if (flag and M <= 16) else (0, 0)
        good = (calls["gemv"], calls["select"]) == expect and torch.equal(i, ri) and torch.equal(w, rw)
        ok &= good
        print(f"flag={flag} M={M}: fused calls gemv/select {calls['gemv']}/{calls['select']} (expect {expect}), "
              f"ids == today {torch.equal(i, ri)}, weights == today {torch.equal(w, rw)} -> {'ok' if good else 'FAIL'}")
print(f"PASS router hooks: {ok}")
