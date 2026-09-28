import os, json, glob
os.environ["GLM5_STEP_TIMING"] = "1"; os.environ["GLM5_PROF_EVENTS_OUT"] = "/tmp"
import torch
from vllm.utils import glm5_prof_events as pe
x = torch.randn(4096, 4096, device="cuda")
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    y = x @ x
torch.cuda.current_stream().wait_stream(st)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = x @ x
for i in range(400):
    tok = pe.step_begin("Mgr@1"); g.replay(); pe.step_end(tok)
torch.cuda.synchronize(); pe.step_dump()
d = json.load(open(glob.glob("/tmp/glm5_steptime.*.json")[0]))["Mgr@1"]
print("replays", d["n"], "gpu samples", len(d["gpu"]), "median gpu ms", sorted(d["gpu"])[len(d["gpu"]) // 2],
      "gap samples", len(d["gap"]))
