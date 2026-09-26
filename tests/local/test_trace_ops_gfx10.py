#!/usr/bin/env python3
"""GLM5_TRACE_OPS hook mechanics on a toy stack: hooks fire per submodule, only while pos0 >= 0, hashes stable."""
import os
import torch
os.makedirs("/root/.cache/vllm", exist_ok=True)
log = "/root/.cache/vllm/ops-trace.log"
if os.path.exists(log): os.remove(log)
import vllm.models.glm5next  # noqa: F401
from vllm.models.glm5next.nvidia import model as M
layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4)) for _ in range(2)]).cuda()
M._glm5_ops_register(layers, [1])
x = torch.randn(3, 8, device="cuda")
M._GLM5_OPS["pos0"] = 0; layers[1](x); layers[0](x)
M._GLM5_OPS["pos0"] = -1; layers[1](x)          # decode-like: must not log
M._GLM5_OPS["pos0"] = 0; layers[1](x)
lines = open(log).read().splitlines()
print(len(lines), "lines (want 8: 4 submodules x 2 logged calls, layer 1 only)")
print("\n".join(l[:110] for l in lines[:4]))
print("repeat call hashes identical:", lines[:4] == lines[4:8])
