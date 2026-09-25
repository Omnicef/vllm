#!/usr/bin/env python3
"""GLM5_PROF_EVENTS mechanism on one GPU: nested blocks (layer > attn, mlp) captured in a graph,
read after each replay by glm5_prof_events.collect(); children must sum to <= the layer, all > 0,
and the layer total must match the replay's wall time measured outside the graph."""
import json, os, glob
os.environ["GLM5_PROF_EVENTS"] = "1"; os.environ["GLM5_PROF_EVENTS_OUT"] = "/tmp"
import torch
from vllm.utils import glm5_prof_events as pe
DEV = "cuda:0"
class Attn(torch.nn.Module):
    def __init__(s): super().__init__(); s.w = torch.nn.Linear(2048, 2048)
    def forward(s, x): return s.w(x)
class MLP(torch.nn.Module):
    def __init__(s): super().__init__(); s.a = torch.nn.Linear(2048, 8192); s.b = torch.nn.Linear(8192, 2048)
    def forward(s, x): return s.b(torch.relu(s.a(x)))
class Layer(torch.nn.Module):
    def __init__(s): super().__init__(); s.self_attn = Attn(); s.mlp = MLP()
    def forward(s, x): x = x + s.self_attn(x); return torch.tanh(x) + s.mlp(x)   # tanh = "the rest" (mHC stand-in)
layers = torch.nn.ModuleList([Layer() for _ in range(3)]).to(DEV)
for i, L in enumerate(layers):
    pe.attach(L, f"L{i}"); pe.attach(L.self_attn, f"L{i}.kda"); pe.attach(L.mlp, f"L{i}.moe")
x = torch.randn(64, 2048, device=DEV)
def fwd(x):
    for L in layers: x = L(x)
    return x
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): fwd(x)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = fwd(x)
pe._acc.clear()
walls = []
for _ in range(100):
    o0, o1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    o0.record(); g.replay(); o1.record(); pe.collect(); walls.append(o0.elapsed_time(o1))
pe.dump()
d = json.load(open(glob.glob(f"/tmp/glm5_prof_events.{os.getpid()}.json")[0]))
blk = {k: v[0] / v[1] for k, v in d["blocks_ms_sum_count"].items()}
for k in sorted(blk): print(f"  {k:8s} {blk[k]*1000:8.1f} us  (n={d['blocks_ms_sum_count'][k][1]})")
layers_sum = sum(blk[f"L{i}"] for i in range(3)); wall = sum(walls) / len(walls)
kids_ok = all(blk[f"L{i}.kda"] + blk[f"L{i}.moe"] <= blk[f"L{i}"] * 1.02 and min(blk[f"L{i}.kda"], blk[f"L{i}.moe"]) > 0 for i in range(3))
print(f"layers sum {layers_sum*1000:.1f} us vs replay wall {wall*1000:.1f} us ({100*layers_sum/wall:.0f} %); children within layer: {kids_ok}")
print("PASS" if kids_ok and 0.8 < layers_sum / wall <= 1.02 and d["replays"] == 100 else "FAIL")
