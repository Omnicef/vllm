"""Router crash repro: engine-style dummy inputs through GLM5_ROUTER_KERNEL=fused select (and gate GEMV), eager and
inside a captured graph, with HIP_LAUNCH_BLOCKING=1: zeros, NaN, +inf, -inf, mixed, torch.empty garbage, padded sizes
1..16. Every expert index must be in [0, 287]."""
import os, sys
os.environ["GLM5_MOE_TOPK_STABLE"] = "1"
import torch
from vllm.model_executor.layers.fused_moe.router import glm5_router as G
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk
dev = "cuda"; E = 288
B = torch.randn(E, device=dev) * 0.01
W = (torch.randn(E, 4096, device=dev) * 0.02).half()
def cases(M):
    g = torch.randn(M, E, device=dev)
    nan = g.clone(); nan[:, ::7] = float("nan")
    allnan = torch.full((M, E), float("nan"), device=dev)
    pinf = g.clone(); pinf[:, 5:20] = float("inf")
    ninf = torch.full((M, E), float("-inf"), device=dev)
    mixed = torch.full((M, E), float("-inf"), device=dev); mixed[:, 10] = float("nan"); mixed[:, 20] = 1.0
    asc = (torch.arange(E, device=dev, dtype=torch.float32)[None] / 100).repeat(M, 1); asc[:, 285] = float("nan")
    return {"zeros": torch.zeros(M, E, device=dev), "some NaN": nan, "all NaN": allnan, "+inf": pinf,
            "all -inf": ninf, "-inf + NaN + one finite": mixed, "ascending with NaN at 285": asc, "garbage (torch.empty)": torch.empty(M, E, device=dev)}
bad = 0; mism = 0
for M in (1, 2, 3, 4, 8, 16):
    for name, lg in cases(M).items():
        w, i = G.select(lg, B, 8, True, 2.5); torch.cuda.synchronize()
        oob = int(((i < 0) | (i >= E)).sum())
        rw, ri = grouped_topk(torch.empty(M, 1, device=dev), lg, 8, True, 1, 1, "sigmoid", 2.5, B)
        same_ids = torch.equal(i, ri)
        nanrow = bool(torch.isnan(lg).any())
        same_w = torch.equal(w, rw) or torch.equal(torch.nan_to_num(w, nan=-7.0), torch.nan_to_num(rw, nan=-7.0))
        bad += oob > 0; mism += not (same_ids and same_w)
        if oob or not (same_ids and same_w):
            print(f"M={M} {name}: out-of-range ids {oob} (max {int(i.max())}), ids == today {same_ids}", flush=True)
# gate GEMV on NaN / inf hidden + capture with garbage
x = torch.full((4, 4096), float("nan"), device=dev).half()
lg = G.gate_gemv(x, W); w, i = G.select(lg, B, 8, True, 2.5); torch.cuda.synchronize()
bad += int(((i < 0) | (i >= E)).sum()) > 0
xs = torch.empty(16, 4096, device=dev, dtype=torch.float16)
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    G.select(G.gate_gemv(xs[:4], W), B, 8, True, 2.5)
torch.cuda.current_stream().wait_stream(st)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    ow, oi = G.select(G.gate_gemv(xs[:4], W), B, 8, True, 2.5)
xs.fill_(float("nan")); gr.replay(); torch.cuda.synchronize()
bad += int(((oi < 0) | (oi >= E)).sum()) > 0
print(f"cases with out-of-range expert ids: {bad}; cases where ids differ from today's path: {mism}")
print(f"PASS all ids in range: {bad == 0}")
