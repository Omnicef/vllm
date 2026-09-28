"""Captured time per MoE layer (routed experts, real layer-10 weights, rank-0 shard) at 3 / 6 / 15 tokens:
skinny vs the Triton W4A16 path with (a) the current V620 config and (b) the tuned config from the 0d sweep
(results.jsonl; tokens without their own tuned entry use the nearest tuned size). x42 MoE layers per decode step."""
import glob, json
import torch
from safetensors import safe_open
import vllm.models.glm5next  # noqa: F401
from vllm.model_executor.layers.fused_moe import fused_experts, override_config
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_configs
from vllm.model_executor.layers.fused_moe import glm5_moe_skinny as S

DEV, E, NR, K, TOPK, GS = "cuda:0", 288, 256, 4096, 8, 128
MD = "/models/GLM-5.3-Flash-AWQ-W4A16/"
IDX = json.load(open(MD + "model.safetensors.index.json"))["weight_map"]
H = {}
def get(n):
    f = IDX[n]
    if f not in H:
        H[f] = safe_open(MD + f, "pt")
    return H[f].get_tensor(n)
w13, s13, w2, s2 = [], [], [], []
for e in range(E):
    p = f"model.language_model.layers.10.mlp.experts.{e}."
    w13.append(torch.cat([get(p + "gate_proj.weight_packed")[:NR], get(p + "up_proj.weight_packed")[:NR]]))
    s13.append(torch.cat([get(p + "gate_proj.weight_scale")[:NR], get(p + "up_proj.weight_scale")[:NR]]).half())
    w2.append(get(p + "down_proj.weight_packed")[:, : NR // 8]); s2.append(get(p + "down_proj.weight_scale")[:, : NR // GS].half())
W13 = torch.stack(w13).contiguous().view(torch.uint8).to(DEV); S13 = torch.stack(s13).contiguous().to(DEV)
W2 = torch.stack(w2).contiguous().view(torch.uint8).to(DEV); S2 = torch.stack(s2).contiguous().to(DEV)
qc = FusedMoEQuantConfig.make(quant_dtype=None, w1_scale=S13, w2_scale=S2, block_shape=[0, GS], weight_dtype="int4")
cur = get_moe_configs(E, NR, "int4_w4a16")
tuned = {r["tokens"]: r["best"] for r in map(json.loads, open("/tune/results.jsonl")) if r["quant"] == "int4_w4a16"}
hid = torch.load(sorted(glob.glob("/dumps/dsa-*-idx.pt"))[0], map_location="cpu", weights_only=False)["hidden_in"].half().to(DEV)


def cap(fn, reps=50):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2] * 1000


gen = torch.Generator(device=DEV).manual_seed(0)
for M in (3, 6, 15):
    x = hid[:M].contiguous()
    ti = torch.stack([torch.randperm(E, generator=gen, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.softmax(torch.randn(M, TOPK, generator=gen, device=DEV), -1)
    c_cur = cur[min(cur, key=lambda k: abs(int(k) - M))]
    tk = min(tuned, key=lambda k: abs(k - M)); c_tun = tuned[tk]
    def tri(c):
        def f():
            with override_config(c):
                fused_experts(x, W13, W2, tw, ti, quant_config=qc)
        return f
    t_cur, t_tun = cap(tri(c_cur)), cap(tri(c_tun))
    t_sk = cap(lambda: S.moe_skinny(x, W13, W2, S13, S2, tw, ti, GS, 10.0))
    print(f"M={M:2d}: Triton current {t_cur:7.1f} us | Triton tuned (from M={tk}) {t_tun:7.1f} us | skinny {t_sk:6.1f} us"
          f" | x42 layers: {42 * t_cur / 1000:5.1f} / {42 * t_tun / 1000:5.1f} / {42 * t_sk / 1000:4.1f} ms per step")
