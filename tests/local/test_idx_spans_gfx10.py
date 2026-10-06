"""One card: indexer sub-op spans (GLM5_PROF_EVENTS inline ranges) inside a captured decode graph. The capture-safe
paged logits + HIP top-k + tie knob + sort, captured with spans on; replay, collect(), and check (1) every sub-op
name is present with a positive time, (2) the logits and selection equal the same graph captured with spans off."""
import os, sys, json, glob, importlib
os.environ["GLM5_PROF_EVENTS"] = sys.argv[1] if len(sys.argv) > 1 else "1"
os.environ["GLM5_PROF_EVENTS_OUT"] = "/tmp"
os.environ["GLM5_INDEXER_DESHUFFLE"] = "1"; os.environ["GLM5_TOPK_TIES"] = "stable"; os.environ["GLM5_SORT_TOPK"] = "1"
import torch
import vllm.models.glm5next  # noqa: F401
from vllm.utils import glm5_prof_events as pe
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fp8_paged_mqa_logits_decode_torch
import vllm.models.glm5next.amd.sparse_indexer as m
from vllm.platforms import current_platform

DEV, B, H, D, BS, NB, MAXLEN, K = "cuda:0", 3, 32, 128, 32, 400, 8192, 512
g = torch.Generator().manual_seed(0)
vals = (torch.randn(NB, BS * D, generator=g) * 2).to(current_platform.fp8_dtype()).view(torch.uint8)
scl = (torch.rand(NB, BS, generator=g) + 0.5).view(torch.uint8).reshape(NB, BS * 4)
kv = torch.cat([vals, scl], 1).reshape(NB, BS, 1, D + 4).to(DEV)
q = (torch.randn(B, 1, H, D, generator=g)).to(current_platform.fp8_dtype()).to(DEV)
w = torch.rand(B, H, generator=g).to(DEV)
ctx = torch.tensor([5000, 7000, 8000], dtype=torch.int32, device=DEV)
bt = torch.randperm(NB, generator=g)[: MAXLEN // BS].repeat(B, 1).to(torch.int32).to(DEV)


def step():
    pe.reset_seq()
    lg = _fp8_paged_mqa_logits_decode_torch(q, kv, w, ctx, bt, MAXLEN)
    t = torch.empty(B, K, dtype=torch.int32, device=DEV)
    tk = pe.begin("topk")
    torch.ops._C.top_k_per_row_decode(lg, 1, ctx, t, B, lg.stride(0), lg.stride(1), K)
    pe.end(tk)
    tt = pe.begin("ties")
    m._glm5_topk_ties(t, lg, torch.zeros(B, dtype=torch.int32, device=DEV), m._glm5_decode_row_end(ctx, 1, B), K)
    pe.end(tt)
    ts = pe.begin("sort_topk")
    m._glm5_sort_pools(t)
    pe.end(ts)
    return lg, t


st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    step()
torch.cuda.current_stream().wait_stream(st)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    lg, t = step()
for _ in range(20):
    gr.replay(); pe.collect()
torch.cuda.synchronize()
torch.save({"lg": lg.cpu(), "t": t.cpu()}, "/tmp/spans-%s.pt" % os.environ["GLM5_PROF_EVENTS"])
if pe.ENABLED:
    subs = {}
    for name, (ms, cnt) in pe._acc.items():
        if name.startswith("sub."):
            sub = name.split(".")[1]
            subs[sub] = subs.get(sub, 0.0) + ms / cnt
    print("per-replay ms by sub-op:", {k: round(v, 3) for k, v in sorted(subs.items())})
    need = {"read_deshuffle", "dequant", "qk_score", "weighting", "topk", "ties", "sort_topk"}
    print("all sub-ops present with time > 0:", need <= set(subs) and all(subs[k] > 0 for k in need))
