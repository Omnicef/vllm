"""grouped_topk (torch.compile dynamic=True) as served: first call (the trace) at TRACE tokens, like the profile run at
max_num_batched_tokens; then GPU time per call at 3/4 tokens inside a captured graph (1000 replays).
Run once per TRACE value in a fresh process (fresh dynamo / inductor state; inductor cache disabled)."""
import os, sys, time
os.environ["GLM5_MOE_TOPK_STABLE"] = "1"; os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
import torch
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk
TRACE = int(sys.argv[1]); dev = "cuda"
bias = torch.randn(288, device=dev) * 0.01
def call(m):
    x = torch.randn(m, 4096, device=dev, dtype=torch.float16)
    g = torch.randn(m, 288, device=dev, dtype=torch.float32)
    return x, g
x, g = call(TRACE)
grouped_topk(x, g, 8, True, 1, 1, "sigmoid", 2.5, bias)          # the trace
res = {}
for m in (3, 4):
    x, g = call(m)
    for _ in range(3): grouped_topk(x, g, 8, True, 1, 1, "sigmoid", 2.5, bias)
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        grouped_topk(x, g, 8, True, 1, 1, "sigmoid", 2.5, bias)
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(10): out = grouped_topk(x, g, 8, True, 1, 1, "sigmoid", 2.5, bias)
    for _ in range(5): gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(100): gr.replay()
    e1.record(); torch.cuda.synchronize()
    res[m] = e0.elapsed_time(e1) / 1000 * 1000   # us per call (100 replays x 10 calls)
kern = sorted({k for k in os.listdir("/tmp/torchinductor_root") if True})[:0] if os.path.isdir("/tmp/torchinductor_root") else []
print(f"traced at {TRACE}: GPU us per call at 3 tokens {res[3]:.1f}, at 4 tokens {res[4]:.1f}", flush=True)
