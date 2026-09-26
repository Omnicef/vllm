#!/usr/bin/env python3
"""GLM5_TOPK_TRACE helper: eager prefill hashes + in-graph decode ring, flushed at the next request (one card)."""
import hashlib, os, tempfile
import torch
d = tempfile.mkdtemp(); os.environ["GLM5_TOPK_TRACE"] = d
import vllm.models.glm5next  # noqa: F401
from vllm.model_executor.layers.sparse_attn_indexer_kpool import _glm5_topk_trace
DEV, W, L = "cuda:0", 2051, ["m.layers.3.k", "m.layers.7.k"]
buf = torch.zeros(512, W, dtype=torch.int32, device=DEV)
def prefill(pos0):
    for i, nm in enumerate(L):
        buf[:512] = torch.arange(512 * W, device=DEV, dtype=torch.int32).view(512, W) + i + pos0
        _glm5_topk_trace(nm, buf, 512, torch.arange(pos0, pos0 + 512, device=DEV), True)
prefill(0); prefill(512)
step = torch.zeros(1, dtype=torch.int32, device=DEV)
def decode():
    for i, nm in enumerate(L):
        buf[:3] = step * 10 + i
        _glm5_topk_trace(nm, buf, 3, None, False)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    decode()
want = {}
for k in range(1, 4):
    step.fill_(k); g.replay(); torch.cuda.synchronize()
    for i, nm in enumerate(L):
        want[(k, nm)] = hashlib.sha256(torch.full((3, W), k * 10 + i, dtype=torch.int32).numpy().tobytes()).hexdigest()[:16]
prefill(0)
lines = open(os.path.join(d, "topk-trace.log")).read().splitlines()
dec = {(int(l.split("step=")[1].split(",")[0]), l.split("layer=")[1].split(",")[0]): l.split("sha=")[1]
       for l in lines if "stage=decode" in l}
pre = [l for l in lines if "stage=prefill" in l]
print(f"prefill lines {len(pre)} (want 6), decode lines {len(dec)} (want 6)")
print("decode shas match:", all(dec.get(k) == v for k, v in want.items()))
