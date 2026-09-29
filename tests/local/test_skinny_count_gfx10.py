"""GLM5_MOE_SKINNY_COUNT=1: eager calls and graph captures counted per token count; replays not; reset clears;
Worker RPC methods registered. Random int4 weights (counting only)."""
import os
os.environ["GLM5_MOE_SKINNY_COUNT"] = "1"
import torch
from vllm.model_executor.layers.fused_moe import glm5_moe_skinny as S
from vllm.v1.worker.gpu_worker import Worker
DEV, E, N, K, G, T = "cuda:0", 8, 256, 512, 128, 4
w1 = torch.randint(0, 255, (E, 2 * N, K // 2), dtype=torch.uint8, device=DEV)
w2 = torch.randint(0, 255, (E, K, N // 2), dtype=torch.uint8, device=DEV)
s1 = torch.rand(E, 2 * N, K // G, device=DEV).half() * 0.01
s2 = torch.rand(E, K, N // G, device=DEV).half() * 0.01
def call(M, out=None):
    x = torch.randn(M, K, device=DEV).half() if out is None else X
    ti = torch.arange(M * T, device=DEV, dtype=torch.int32).view(M, T) % E
    tw = torch.full((M, T), 0.25, device=DEV)
    return S.moe_skinny(x, w1, w2, s1, s2, tw, ti, G, 10.0, out=out)
ok = S.count_dump() == {"eager": {}, "capture": {}}
call(1); call(1); call(3)
ok &= S.count_dump() == {"eager": {1: 2, 3: 1}, "capture": {}}
X = torch.randn(2, K, device=DEV).half(); out = torch.empty(2, K, device=DEV).half()
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    call(2, out)
torch.cuda.current_stream().wait_stream(st)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    call(2, out)
for _ in range(5):
    g.replay()
torch.cuda.synchronize()
d = S.count_dump(); print(d)
ok &= d == {"eager": {1: 2, 3: 1, 2: 1}, "capture": {2: 1}}
S.count_reset(); ok &= S.count_dump() == {"eager": {}, "capture": {}}
ok &= hasattr(Worker, "glm5_skinny_count_dump") and hasattr(Worker, "glm5_skinny_count_reset")
print(f"PASS skinny counter: {ok}")
