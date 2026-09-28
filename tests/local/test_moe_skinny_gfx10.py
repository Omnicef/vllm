#!/usr/bin/env python3
"""GLM5_MOE_SKINNY vs the Triton WNA16 MoE path (fused_experts -> invoke_fused_moe_wna16_triton_kernel, SwiGLU limit
through gemm1_clamp_limit), one card. Real GLM-5.3 layer-10 routed experts (rank-0 shard: 256 of 2048 intermediate
rows), int4 group 128, and real hidden states (phase-18 layer inputs).

  per expert    top-1 routing to each of 288 experts (weight 1): output vs Triton, rel. to max |Triton|
  top-8         random top-8 routing, tokens 1 / 3 / 6 / 8 / 15 / 16
  clamp         hidden states scaled so gate/up exceed +-10: skinny (limit 10) vs Triton (limit 10) within tolerance;
                negative control skinny without the limit must differ
  repeat        21 calls bitwise identical; graph replay equals eager
"""
import glob, json, os
import torch
from safetensors import safe_open
import vllm.models.glm5next  # noqa: F401
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe import glm5_moe_skinny as S

DEV, E, NFULL, NR, K, TOPK, GS, LIM = "cuda:0", 288, 2048, 256, 4096, 8, 128, 10.0
MD = "/models/GLM-5.3-Flash-AWQ-W4A16/"
IDX = json.load(open(MD + "model.safetensors.index.json"))["weight_map"]
L = 10
handles = {}


def get(name):
    f = IDX[name]
    if f not in handles:
        handles[f] = safe_open(MD + f, "pt")
    return handles[f].get_tensor(name)


def packed(e, proj, rows):
    p = f"model.language_model.layers.{L}.mlp.experts.{e}.{proj}."
    return get(p + "weight_packed")[rows], get(p + "weight_scale")[rows].half()


w13, s13, w2, s2 = [], [], [], []
for e in range(E):
    g, gs = packed(e, "gate_proj", slice(0, NR)); u, us = packed(e, "up_proj", slice(0, NR))
    d, ds = packed(e, "down_proj", slice(None))
    w13.append(torch.cat([g, u])); s13.append(torch.cat([gs, us]))
    w2.append(d[:, : NR // 8]); s2.append(ds[:, : NR // GS])
W13 = torch.stack(w13).contiguous().view(torch.uint8).to(DEV)       # [E, 2N, K/2] N-first bytes
S13 = torch.stack(s13).contiguous().to(DEV)                          # [E, 2N, K/G]
W2 = torch.stack(w2).contiguous().view(torch.uint8).to(DEV)          # [E, K, N/2]
S2 = torch.stack(s2).contiguous().to(DEV)                            # [E, K, N/G]
print("weights", tuple(W13.shape), tuple(S13.shape), tuple(W2.shape), tuple(S2.shape))
hid = torch.cat([torch.load(f, map_location="cpu", weights_only=False)["hidden_in"].half()
                 for f in sorted(glob.glob("/dumps/dsa-*-idx.pt"))[:4]])[:512].to(DEV)


def triton(x, tw, ti, limit):
    qc = FusedMoEQuantConfig.make(quant_dtype=None, w1_scale=S13, w2_scale=S2, block_shape=[0, GS],
                                  weight_dtype="int4", gemm1_clamp_limit=limit)
    return fused_experts(x, W13, W2, tw, ti, quant_config=qc)


def skinny(x, tw, ti, limit):
    return S.moe_skinny(x, W13, W2, S13, S2, tw, ti, GS, limit)


def rel(a, b):
    return float((a.float() - b.float()).abs().max()) / max(float(b.float().abs().max()), 1e-6)


TOL = 5e-3
ok = True
# per expert, top-1
worst = 0.0
for e0 in range(0, E, 16):
    x = hid[:16]
    ti = (torch.arange(16, device=DEV, dtype=torch.int32)[:, None] + e0) % E
    tw = torch.ones(16, 1, device=DEV)
    worst = max(worst, rel(skinny(x, tw, ti, LIM), triton(x, tw, ti, LIM)))
print(f"per expert (top-1, all 288 experts, real hidden): max rel err {worst:.2e} -> {'ok' if worst <= TOL else 'FAIL'}")
ok &= worst <= TOL
# top-8
g = torch.Generator(device=DEV).manual_seed(0)
for M in (1, 3, 6, 8, 15, 16):
    x = hid[M: 2 * M].contiguous()
    ti = torch.stack([torch.randperm(E, generator=g, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.softmax(torch.randn(M, TOPK, generator=g, device=DEV), -1)
    r = rel(skinny(x, tw, ti, LIM), triton(x, tw, ti, LIM))
    ok &= r <= TOL
    print(f"top-8 M={M:2d} real hidden: max rel err {r:.2e} -> {'ok' if r <= TOL else 'FAIL'}")
# clamp beyond the limit. fused_experts (the functional Triton path) ignores gemm1_clamp_limit, so the clamp
# reference is the serving activation op itself, torch.ops._C.silu_and_mul_with_clamp, on dequantized weights with the
# Triton path's fp16 intermediates; that reference is first checked against the real Triton kernel without a limit.
def deq(wbytes, scale, rows):
    w = wbytes[rows].contiguous().view(torch.int32)                  # [r, K/8] k-sequential nibbles
    sh = torch.arange(0, 32, 4, device=DEV, dtype=torch.int32)
    q = ((w.unsqueeze(-1) >> sh) & 0xF).reshape(w.shape[0], -1).float() - 8.0
    return q * scale[rows].float().repeat_interleave(GS, dim=1)


def ref(x, tw, ti, limit):
    out = torch.zeros(x.shape[0], K, device=DEV)
    for m in range(x.shape[0]):
        for k in range(ti.shape[1]):
            e = int(ti[m, k])
            gu = (x[m].float() @ deq(W13[e], S13[e], slice(None)).t()).half().view(1, -1)
            act = torch.empty(1, NR, dtype=torch.float16, device=DEV)
            torch.ops._C.silu_and_mul_with_clamp(act, gu, float(limit if limit else 1e30), 1.0, 0.0)
            d = (act.float() @ deq(W2[e], S2[e], slice(None)).t()).half()
            out[m] += (d.float() * float(tw[m, k])).view(-1)
    return out.half()


x = (hid[:4] * 400).contiguous()
ti = torch.stack([torch.randperm(E, generator=g, device=DEV)[:TOPK] for _ in range(4)]).to(torch.int32)
tw = torch.softmax(torch.randn(4, TOPK, generator=g, device=DEV), -1)
r_ref_tri = rel(ref(x, tw, ti, None), triton(x, tw, ti, None))
R = ref(x, tw, ti, LIM)
r_on = rel(skinny(x, tw, ti, LIM), R)
r_off = rel(skinny(x, tw, ti, None), R)
r_refoff = rel(ref(x, tw, ti, None), R)
good = r_ref_tri <= TOL and r_on <= TOL and r_off > 10 * TOL and r_refoff > 10 * TOL
print(f"clamp (inputs x400, limit 10): reference(no limit) vs real Triton kernel {r_ref_tri:.2e} (validates the "
      f"reference) | skinny(limit) vs reference(limit) {r_on:.2e} | controls: skinny(no limit) {r_off:.2e}, "
      f"reference(no limit) {r_refoff:.2e} vs reference(limit) -> {'ok' if good else 'FAIL'}")
ok &= good
# repeat + graph
x = hid[:3].contiguous()
ti = torch.stack([torch.randperm(E, generator=g, device=DEV)[:TOPK] for _ in range(3)]).to(torch.int32)
tw = torch.softmax(torch.randn(3, TOPK, generator=g, device=DEV), -1)
o = skinny(x, tw, ti, LIM)
rep = all(torch.equal(o, skinny(x, tw, ti, LIM)) for _ in range(20))
out = torch.empty_like(o)
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    S.moe_skinny(x, W13, W2, S13, S2, tw, ti, GS, LIM, out=out)
torch.cuda.current_stream().wait_stream(st)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    S.moe_skinny(x, W13, W2, S13, S2, tw, ti, GS, LIM, out=out)
out.zero_(); gr.replay(); torch.cuda.synchronize()
geq = torch.equal(out, o)
print(f"21x bitwise {rep} | graph replay == eager {geq}")
ok &= rep and geq
print(f"PASS skinny MoE vs Triton: {ok}")
