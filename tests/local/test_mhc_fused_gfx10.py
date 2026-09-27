#!/usr/bin/env python3
"""GLM5_MHC_KERNEL=fused vs the current ROCm path (mhc_post_torch + mhc_pre_torch with the triton mixing GEMM +
the vllm_c RMSNorm), one card, fp16 residual, real GLM-5.3 mHC weights (layers 1-10) and real hidden states
(phase-18 indexer inputs, layer 3) as residual seeds and stand-in sublayer outputs.

  variants: pre only (layer 0), post + pre (every other boundary), post only (last layer)
  token counts: 1, 3 (MTP verify), 8, 32 (decode batches), 512 (prefill chunk)
  tolerance: fp32 post/comb mixes rtol 1e-4 (atol 1e-6), fp16 residual / layer input rtol 2e-3 (atol 1e-5)
  repeat: 21 calls of the fused path bitwise identical
  chain: 20 consecutive boundaries (layers 1..10, attn then ffn weights), fused and current each fed their own
         outputs; the difference per step
  timing: captured replay per call, fused vs current, x90 calls per decode step
"""
import glob, os, sys
os.environ["GLM5_MHC_KERNEL"] = "triton"          # the current path's mixing GEMM
import torch
from safetensors import safe_open
import vllm.models.glm5next  # noqa: F401
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm.model_executor.kernels.mhc import triton_fused as F

DEV, HC, H = "cuda:0", 4, 4096
EPS, SK, PM, RMS_EPS, NORM_EPS = 1e-6, 20, 2.0, 1e-5, 1e-5
import json
MD = "/models/GLM-5.3-Flash-AWQ-W4A16/"
IDX = json.load(open(MD + "model.safetensors.index.json"))["weight_map"]


def get(name):
    with safe_open(MD + IDX[name], "pt") as f:
        return f.get_tensor(name)


W = {}
for L in range(1, 11):
    for part in ("attn", "ffn"):
        p = f"model.language_model.layers.{L}.hc_{part}_"
        W[(L, part)] = (get(p + "fn").float().to(DEV), get(p + "scale").float().to(DEV),
                        get(p + "base").float().to(DEV))
    W[(L, "attn_norm")] = get(f"model.language_model.layers.{L}.input_layernorm.weight").half().to(DEV)
    W[(L, "ffn_norm")] = get(f"model.language_model.layers.{L}.post_attention_layernorm.weight").half().to(DEV)
hid = []
for fp in sorted(glob.glob("/dumps/dsa-*-idx.pt"))[:40]:
    d = torch.load(fp, map_location="cpu", weights_only=False)
    if d.get("hidden_in") is not None and d["hidden_in"].shape[0] >= 128:
        hid.append(d["hidden_in"].half())
HID = torch.cat(hid)[:4096].to(DEV)
print(f"real hidden rows {HID.shape[0]}, |h| mean {HID.float().abs().mean():.3f}")


def rms_c(x, w):
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x.contiguous(), w, NORM_EPS)
    return out


def ref_pre(res, fn, sc, bs, nw):
    p, c, li = mhc_pre_torch(res, fn, sc, bs, RMS_EPS, EPS, EPS, PM, SK)
    return p, c, rms_c(li, nw)


def ref_post_pre(x, res, post, comb, fn, sc, bs, nw):
    r = mhc_post_torch(x, res, post, comb)
    return (r, *ref_pre(r, fn, sc, bs, nw))


def fus_pre(res, fn, sc, bs, nw):
    return F.mhc_fused_pre(res, fn, sc, bs, RMS_EPS, EPS, EPS, PM, SK, nw, NORM_EPS)


def fus_post_pre(x, res, post, comb, fn, sc, bs, nw):
    return F.mhc_fused_post_pre(x, res, post, comb, fn, sc, bs, RMS_EPS, EPS, EPS, PM, SK, nw, NORM_EPS)


def rel(a, b):
    a, b = a.float(), b.float()
    m = b.abs() > 1e-3
    return float(((a - b).abs()[m] / b.abs()[m]).max()) if m.any() else 0.0


def close(a, b, fp16):
    return torch.allclose(a.float(), b.float(), rtol=2e-3 if fp16 else 1e-4, atol=1e-5 if fp16 else 1e-6)


ok = True
fn, sc, bs = W[(1, "attn")]; nw = W[(1, "attn_norm")]
fn2, sc2, bs2 = W[(1, "ffn")]; nw2 = W[(1, "ffn_norm")]
for T in (1, 3, 8, 32, 512):
    res0 = HID[:T].unsqueeze(1).expand(T, HC, H).contiguous()
    x = HID[T: 2 * T].contiguous() if 2 * T <= HID.shape[0] else HID[:T].flip(0).contiguous()
    rp = ref_pre(res0, fn, sc, bs, nw); fp = fus_pre(res0, fn, sc, bs, nw)
    g1 = close(fp[0], rp[0], False) and close(fp[1], rp[1], False) and close(fp[2], rp[2], True)
    rpp = ref_post_pre(x, res0, rp[0], rp[1], fn2, sc2, bs2, nw2)
    fpp = fus_post_pre(x, res0, rp[0], rp[1], fn2, sc2, bs2, nw2)
    g2 = (close(fpp[0], rpp[0], True) and close(fpp[1], rpp[1], False) and close(fpp[2], rpp[2], False)
          and close(fpp[3], rpp[3], True))
    ro = mhc_post_torch(x, res0, rp[0], rp[1]); fo = F.mhc_fused_post(x, res0, rp[0], rp[1])
    g3 = close(fo, ro, True)
    reps = [fus_post_pre(x, res0, rp[0], rp[1], fn2, sc2, bs2, nw2) for _ in range(21)]
    g4 = all(all(torch.equal(a, b) for a, b in zip(reps[0], r)) for r in reps[1:])
    reps = [fus_pre(res0, fn, sc, bs, nw) for _ in range(21)]
    g4 &= all(all(torch.equal(a, b) for a, b in zip(reps[0], r)) for r in reps[1:])
    ok &= g1 and g2 and g3 and g4
    print(f"T={T:3d} pre: post {rel(fp[0], rp[0]):.1e} comb {rel(fp[1], rp[1]):.1e} li {rel(fp[2], rp[2]):.1e} "
          f"-> {'ok' if g1 else 'FAIL'} | post+pre: res {rel(fpp[0], rpp[0]):.1e} post {rel(fpp[1], rpp[1]):.1e} "
          f"comb {rel(fpp[2], rpp[2]):.1e} li {rel(fpp[3], rpp[3]):.1e} -> {'ok' if g2 else 'FAIL'} | "
          f"post only {rel(fo, ro):.1e} -> {'ok' if g3 else 'FAIL'} | 21x bitwise {g4} | "
          f"res bytes equal {100 * float((fpp[0] == rpp[0]).float().mean()):.2f}%")

def l2(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm())


# chain: 20 boundaries, each path fed its own outputs. Control: the current path against itself with the seed
# residual moved by one fp16 ulp (the growth rounding alone produces through this recurrence)
for T in (1, 32):
    res = HID[:T].unsqueeze(1).expand(T, HC, H).contiguous()
    resu = torch.nextafter(res, torch.full_like(res, float("inf")))          # +1 ulp everywhere
    rP, rC, _ = ref_pre(res, *W[(1, "attn")], W[(1, "attn_norm")])
    fP, fC, _ = fus_pre(res, *W[(1, "attn")], W[(1, "attn_norm")])
    uP, uC, _ = ref_pre(resu, *W[(1, "attn")], W[(1, "attn_norm")])
    rR, fR, uR = res, res, resu
    line, cline = [], []
    step = 0
    for L in range(1, 11):
        for part, npart in (("ffn", "ffn_norm"), ("attn", "attn_norm")):
            if step == 20:
                break
            Lw = L if part == "ffn" else L + 1
            if (Lw, part) not in W:
                Lw = L
            x = HID[(step + 2) * T % (HID.shape[0] - T):][:T].contiguous()
            rR, rP, rC, rli = ref_post_pre(x, rR, rP, rC, *W[(Lw, part)], W[(Lw, npart)])
            fR, fP, fC, fli = fus_post_pre(x, fR, fP, fC, *W[(Lw, part)], W[(Lw, npart)])
            uR, uP, uC, uli = ref_post_pre(x, uR, uP, uC, *W[(Lw, part)], W[(Lw, npart)])
            step += 1
            line.append(f"{step}:{l2(fR, rR):.1e}/{l2(fli, rli):.1e}")
            cline.append(f"{step}:{l2(uR, rR):.1e}/{l2(uli, rli):.1e}")
    print(f"chain T={T} fused vs current (step: residual / layer-input relative L2): " + " ".join(line))
    print(f"chain T={T} control, current +1ulp seed vs current:                     " + " ".join(cline))


def cap_time(fn_, reps=50):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn_()
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn_()
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); gr.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2]


for T in (1, 3, 8, 32, 512):
    res0 = HID[:T].unsqueeze(1).expand(T, HC, H).contiguous()
    x = HID[T: 2 * T].contiguous() if 2 * T <= HID.shape[0] else HID[:T].contiguous()
    post = torch.rand(T, HC, 1, device=DEV); comb = torch.rand(T, HC, HC, device=DEV)
    tr = cap_time(lambda: ref_post_pre(x, res0, post, comb, fn2, sc2, bs2, nw2))
    tf = cap_time(lambda: fus_post_pre(x, res0, post, comb, fn2, sc2, bs2, nw2))
    print(f"timing T={T:3d}: current {tr * 1000:6.0f} us, fused {tf * 1000:6.1f} us per call "
          f"-> x90 current {90 * tr:5.1f} ms, fused {90 * tf:5.2f} ms per step")
print(f"PASS fused mHC: {ok}")
