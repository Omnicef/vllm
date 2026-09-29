"""GLM5_ROUTER_KERNEL=fused vs today's router on one card.
Today: gate = F.linear(fp16 x, fp16 W) -> fp16 -> .to(fp32); top-k = grouped_topk (torch.compile, GLM5_MOE_TOPK_STABLE=1,
sigmoid, n_group 1, renormalize, x2.5). Fused: glm5_router.gate_gemv + glm5_router.select.
Checks: select vs grouped_topk on identical logits (random / fp16-rounded / planted exact ties / saturated scores):
ids and weights bitwise; full path (gate + select) on real hidden states with the real layer-10 gate + bias; 21 repeat
calls bitwise; graph replay == eager; GPU time per layer (graph, 1000 calls) at 1 / 3 / 24 / 96 tokens."""
import glob, json, os
os.environ["GLM5_MOE_TOPK_STABLE"] = "1"
import torch
import torch.nn.functional as F
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk
from vllm.model_executor.layers.fused_moe.router import glm5_router as G

dev = "cuda"
MD = "/models/GLM-5.3-Flash-AWQ-W4A16/"
IDX = json.load(open(MD + "model.safetensors.index.json"))["weight_map"]
def get(n):
    return safe_open(MD + IDX[n], "pt").get_tensor(n)
L = 10
W = get(f"model.language_model.layers.{L}.mlp.gate.weight").to(dev, torch.float16).contiguous()
B = get(f"model.language_model.layers.{L}.mlp.gate.e_score_correction_bias").to(dev, torch.float32).contiguous()
print(f"gate W {tuple(W.shape)} (checkpoint dtype bf16 -> served fp16), bias {tuple(B.shape)} fp32, "
      f"distinct bias values {B.unique().numel()}")

def today_topk(lg):
    return grouped_topk(torch.empty(lg.shape[0], 1, device=dev), lg, 8, True, 1, 1, "sigmoid", 2.5, B)
def fused_topk(lg):
    return G.select(lg, B, 8, True, 2.5)
def same(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

ok = True
g = torch.Generator(device=dev).manual_seed(0)
# 1. identical logits
cases = {}
for M in (1, 3, 4, 16, 24, 96, 512):
    cases[f"random fp32 M={M}"] = torch.randn(M, 288, device=dev, generator=g) * 3
    cases[f"fp16-rounded M={M}"] = (torch.randn(M, 288, device=dev, generator=g) * 3).half().float()
t = (torch.randn(64, 288, device=dev, generator=g) * 3).half().float()
t[:, 10:40] = 2.0; t[:, 100:130] = 2.0          # exact ties across the cut-off (bias repeats too)
cases["planted exact ties M=64"] = t
cases["saturated sigmoid (logits 30..40) M=64"] = 30 + 10 * torch.rand(64, 288, device=dev, generator=g)
for name, lg in cases.items():
    a, b = today_topk(lg), fused_topk(lg)
    r = same(a, b); ok &= r
    extra = "" if r else (f" ids equal {torch.equal(a[1], b[1])}, max |dw| {float((a[0] - b[0]).abs().max()):.2e}")
    print(f"select {name}: ids+weights bitwise {r}{extra}")

# 2. real hidden states, full path
hid = torch.cat([torch.load(f, map_location="cpu", weights_only=False)["hidden_in"].half()
                 for f in sorted(glob.glob("/dumps/dsa-*-idx.pt"))[:8]]).to(dev)
tot = same_ids = same_all = 0; max_dl = 0.0
for M in (1, 3, 4, 16):
    for s0 in range(0, min(hid.shape[0], 256) - M, max(M, 16)):
        x = hid[s0:s0 + M].contiguous()
        lt = F.linear(x, W).to(torch.float32)
        lf = G.gate_gemv(x, W)
        max_dl = max(max_dl, float((lt - lf).abs().max()))
        a, b = today_topk(lt), fused_topk(lf)
        tot += 1; same_ids += torch.equal(a[1], b[1]); same_all += same(a, b)
lt = F.linear(hid[:16].contiguous(), W).to(torch.float32); lf = G.gate_gemv(hid[:16].contiguous(), W)
print(f"real hidden (layer {L} gate): {tot} batches (M 1/3/4/16): logits bitwise equal "
      f"{torch.equal(lt, lf)} (max |d| over all batches {max_dl:.2e}); same experts {same_ids}/{tot}, "
      f"same experts+weights {same_all}/{tot}")
# skinny GEMV variant: how often does routing differ from today on real hidden states (per token)?
tok = diff_set = diff_w = 0; maxdw = 0.0
for s0 in range(0, hid.shape[0] - 16, 16):
    x = hid[s0:s0 + 16].contiguous()
    a = today_topk(F.linear(x, W).to(torch.float32)); b = fused_topk(G.gate_gemv(x, W, "skinny"))
    sa, sb = a[1].sort(-1).values, b[1].sort(-1).values
    tok += 16; diff_set += int((sa != sb).any(-1).sum()); diff_w += int((a[0] != b[0]).any(-1).sum())
    same_set = (sa == sb).all(-1)
    if same_set.any():
        maxdw = max(maxdw, float((a[0][same_set] - b[0][same_set]).abs().max()))
print(f"skinny GEMV vs today, {tok} real tokens: different expert set {diff_set}, different weights {diff_w}, "
      f"max |dw| where the set agrees {maxdw:.1e}; deterministic: "
      f"{all(torch.equal(G.gate_gemv(hid[:3].contiguous(), W, 'skinny'), G.gate_gemv(hid[:3].contiguous(), W, 'skinny')) for _ in range(20))}")
# selection on today's logits must be exact regardless of the GEMV
a, b = today_topk(lt), fused_topk(lt)
ok &= same(a, b)
print(f"real hidden, select on today's logits: bitwise {same(a, b)}")

# 3. repeats + graph
x = hid[:3].contiguous()
def full(x):
    return G.select(G.gate_gemv(x, W), B, 8, True, 2.5)
o = full(x)
rep = all(same(o, full(x)) for _ in range(20))
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    full(x)
torch.cuda.current_stream().wait_stream(st)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    og = full(x)
gr.replay(); torch.cuda.synchronize()
geq = same(o, og)
ok &= rep and geq
print(f"21 calls bitwise {rep}; graph replay == eager {geq}")

# 4. timing per layer (graph of 10 calls, 100 replays)
def gtime(fn):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(10): fn()
    for _ in range(5): gr.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record()
    for _ in range(100): gr.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)          # ms per 1000 calls = us per call
grouped_topk(torch.empty(512, 1, device=dev), torch.randn(512, 288, device=dev), 8, True, 1, 1, "sigmoid", 2.5, B)
for M in (1, 3, 24, 96):
    x = hid[:M].contiguous(); lg = F.linear(x, W).float()
    tg = gtime(lambda: F.linear(x, W).to(torch.float32)); fg = gtime(lambda: G.gate_gemv(x, W, "exact"))
    fs = gtime(lambda: G.gate_gemv(x, W, "skinny"))
    tt = gtime(lambda: today_topk(lg)); ft = gtime(lambda: fused_topk(lg))
    print(f"M={M:3d}: gate today {tg:6.1f} us, exact {fg:6.1f}, skinny {fs:6.1f}{' (M>16: exact path)' if M > 16 else ''} | "
          f"top-k today {tt:6.1f} us, fused {ft:6.1f} | router today {tg + tt:6.1f} -> exact {fg + ft:6.1f} / "
          f"skinny {fs + ft:6.1f} us per layer", flush=True)
print(f"PASS fused router (select exact, repeat, graph): {ok}")
