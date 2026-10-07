#!/usr/bin/env python3
"""Dynamic draft depth (upstream #57053): the verify width changes between steps. One card, no model.

With a batch-size schedule the drafter proposes K = 2, 3 or 4 tokens per step, so the target verifies 3, 4 or 5 rows
per sequence; the width is uniform within a step and changes between steps, and each width replays its own decode
graph over the same caches and state. Each section below captures one graph per width (3 / 4 / 5) over shared static
buffers and caches, then replays them in a mixed order that covers every width transition (3->3, 3->4, ..., 5->5),
against an eager twin and an independent reference. Production env: GLM5_INDEXER_DESHUFFLE / GLM5_TOPK_TIES=stable /
GLM5_TOPK_TIES_BOUND / GLM5_SORT_TOPK / GLM5_FP16_KPOOL as in the launcher; fp16 activations.

  indexer  fused paged MQA logits (GLM5_INDEXER_KERNEL=fused) + HIP top-k + stable ties + sort on flattened verify
           rows (8 sequences x width, row lengths as _glm5_decode_row_end): replay == eager (logits and selection
           bitwise); eager vs the capture-safe torch logits: score error <= 1e-5 x max, selection overlap >= 99.5 %,
           0 real selection errors (as test_indexer_fused_gfx10)
  kpool    kpool_decode_update_and_maybe_write_cache_batched (decode tail ring 8 = sized for K up to 4) over a run
           with per-request random acceptance, i.e. rejected drafts that complete pools and are redone: replay ==
           eager (kv cache and tail ring bitwise); every pool whose four tokens were all accepted == the prefill
           writer (kpool_compress_and_write_cache) on the accepted keys, bitwise
  kda      the KDA spec path as kda.py calls it (causal_conv1d_update with num_accepted_tokens over one conv slot per
           request, fused_recurrent_kda with K+1 per-position state slots): replay == eager (outputs and both states
           bitwise); accepted-token outputs and the final recurrent state vs the plain decode path fed only the
           accepted tokens one step at a time (max rel diff <= 2e-3)
Negative controls: one perturbed accepted key / input must be detected by the kpool and kda references.

  python3 test_dynk_varwidth_gfx10.py
"""
import os, random, sys
for _k, _v in (("GLM5_INDEXER_DESHUFFLE", "1"), ("GLM5_TOPK_TIES", "stable"), ("GLM5_TOPK_TIES_BOUND", "1"),
               ("GLM5_SORT_TOPK", "1"), ("GLM5_FP16_KPOOL", "1")):
    os.environ[_k] = _v
import torch
import vllm.models.glm5next  # noqa: F401  (import order: breaks the indexer <-> model import cycle)
import vllm.models.glm5next.amd.sparse_indexer as m
from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fp8_paged_mqa_logits_decode_torch as torch_logits
from vllm.v1.attention.ops import glm5_indexer_fused as FI
from vllm.models.glm5next.amd.ops.kpool_compress import (
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
)
from vllm.models.glm5next.amd.ops.third_party.kda import fused_recurrent_kda
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

DEV = "cuda:0"
F8 = current_platform.fp8_dtype()
WIDTHS = (3, 4, 5)                                  # verify rows per sequence = K + 1, K = 2 / 3 / 4
rnd = random.Random(57053)
ORDER = [x for a in WIDTHS for b in WIDTHS for x in (a, b)] + [rnd.choice(WIDTHS) for _ in range(22)]
TRANS = {(a, b) for a, b in zip(ORDER, ORDER[1:])}
assert TRANS == {(a, b) for a in WIDTHS for b in WIDTHS}, TRANS
g = torch.Generator().manual_seed(0)
print(f"step widths ({len(ORDER)} steps, all {len(TRANS)} transitions): {ORDER}", flush=True)


def capture(fn):
    """warm-up on a side stream (compiles Triton), then capture one call"""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = fn()
    torch.cuda.synchronize()
    return gr, out


# ---------------------------------------------------------------------------------------------------- indexer
H, D, BS, K = 32, 128, 32, 512
S, MAXLEN, NB, CTX = 8, 8192, 4096, 2560             # 8 sequences, 32k-token pool width, 2560 pools of keys


def tile(vals):
    p = vals.shape[0]
    return vals.view(p, BS // 16, 16, D // 16, 16).transpose(2, 3).reshape(p, BS * D)


def make_cache(keys_u8, scales, num_blocks):
    n = keys_u8.shape[0]
    pages = (n + BS - 1) // BS
    vals = torch.zeros(pages * BS, D, dtype=torch.uint8); vals[:n] = keys_u8
    sc = torch.ones(pages * BS, dtype=torch.float32); sc[:n] = scales
    perm = torch.randperm(num_blocks, generator=g)[:pages]
    cache = torch.zeros(num_blocks, BS * D + BS * 4, dtype=torch.uint8)
    cache[perm, : BS * D] = tile(vals.view(pages, BS, D))
    cache[perm, BS * D:] = sc.view(pages, BS).contiguous().view(torch.uint8).view(pages, BS * 4)
    return cache.view(num_blocks, BS, 1, D + 4), perm.to(torch.int32)


def selection(logits, lens):
    rows = logits.shape[0]
    t = torch.empty(rows, K, dtype=torch.int32, device=DEV)
    torch.ops._C.top_k_per_row_decode(logits, 1, lens, t, rows, logits.stride(0), logits.stride(1), K)
    if m._glm5_need_sort_after_ties(m._glm5_topk_ties(t, logits, torch.zeros(rows, dtype=torch.int32, device=DEV),
                                                      m._glm5_decode_row_end(lens, 1, rows), K)):
        m._glm5_sort_pools(t)
    return t


def vs_torch(q, cache, w, lens, bt, lg, sel, tol=1e-5):
    a = torch_logits(q, cache, w, lens, bt, MAXLEN)
    sa = selection(a, lens)
    worst, real, minov, ok = 0.0, 0, 1.0, True
    for r in range(a.shape[0]):
        L = int(lens[r]); ar, br = a[r, :L], lg[r, :L]
        scale = float(ar.abs().max()) or 1.0
        err = float((ar - br).abs().max()) / scale; worst = max(worst, err)
        ok &= err <= tol and bool(torch.isinf(a[r, L:]).all()) and bool(torch.isinf(lg[r, L:]).all())
        A = set(x for x in sa[r].tolist() if x >= 0); B = set(x for x in sel[r].tolist() if x >= 0)
        minov = min(minov, len(A & B) / max(len(A), 1))
        if L > K:
            kth = float(ar.topk(K).values[-1])
            real += sum(1 for x in A ^ B if abs(float(ar[x]) - kth) > tol * scale)
    return ok and minov >= 0.995 and real == 0, worst, minov, real


keys = (torch.randn(CTX, D, generator=g) * 1.5).to(F8).view(torch.uint8)
icache, perm = make_cache(keys, torch.rand(CTX, generator=g) * 0.01 + 0.001, NB)
icache = icache.to(DEV)
MAXROWS = S * max(WIDTHS)
q_s = torch.zeros(MAXROWS, 1, H, D, device=DEV).to(F8)
w_s = torch.zeros(MAXROWS, H, device=DEV)
lens_s = torch.zeros(MAXROWS, dtype=torch.int32, device=DEV)
bt_s = torch.zeros(MAXROWS, MAXLEN // BS, dtype=torch.int32, device=DEV)
bt_s[:, : perm.numel()] = perm.to(DEV)


def istep(w):
    rows = S * w
    seq = torch.randint(300, CTX + 1, (S,), generator=g, dtype=torch.int32)
    lens = m._glm5_decode_row_end(seq, w, rows).to(torch.int32)
    q = (torch.randn(rows, 1, H, D, generator=g) * 1.5).to(F8)
    wt = torch.rand(rows, H, generator=g) * 0.1
    q_s[:rows].copy_(q.to(DEV)); w_s[:rows].copy_(wt.to(DEV)); lens_s[:rows].copy_(lens.to(DEV))
    return rows


def ifn(rows):
    def f():
        lg = FI.fused_paged_mqa_logits(q_s[:rows], icache, w_s[:rows], lens_s[:rows], bt_s[:rows], MAXLEN)
        return lg, selection(lg, lens_s[:rows])
    return f


igraphs = {}
for w in WIDTHS:
    istep(w)                                         # valid inputs for the warm-up
    igraphs[w] = capture(ifn(S * w))
i_ok, i_worst, i_minov, i_real, i_same = True, 0.0, 1.0, 0, 0
for w in ORDER:
    rows = istep(w)
    gr, (glg, gsel) = igraphs[w]
    gr.replay(); torch.cuda.synchronize()
    elg, esel = ifn(rows)()
    same = torch.equal(glg, elg) and torch.equal(gsel, esel)
    good, worst, minov, real = vs_torch(q_s[:rows], icache, w_s[:rows], lens_s[:rows], bt_s[:rows], elg, esel)
    i_same += same; i_ok &= same and good
    i_worst, i_minov, i_real = max(i_worst, worst), min(i_minov, minov), i_real + real
print(f"indexer: replay == eager {i_same}/{len(ORDER)} steps | vs torch: max rel score err {i_worst:.1e}, min "
      f"overlap {100 * i_minov:.2f} %, real selection errors {i_real} -> {'ok' if i_ok else 'FAIL'}", flush=True)

# ---------------------------------------------------------------------------------------------------- kpool tail
KP, HD, PAGE, R, NP = 4, 128, 32, 8, 64              # 8 requests, 64 pools (256 tokens) each
RING = KP * 2                                        # kpool 4 + max K 4 = span 8 -> ring 8 (as get_kv_cache_spec)
NPAGES = R * NP // PAGE
LMAX = NP * KP
ktrue = torch.randn(R, LMAX, HD, generator=g).to(torch.float16)
strue = torch.randn(R, LMAX, HD, generator=g).to(torch.float16)
ape = (0.1 * torch.randn(KP, HD, generator=g)).to(DEV)


def kcaches():
    return (torch.zeros(NPAGES, PAGE, HD + 4, dtype=torch.uint8, device=DEV),
            torch.zeros(R + 1, 2, RING, HD, dtype=torch.bfloat16, device=DEV))


kv_g, tail_g = kcaches()
kb = {w: dict(key=torch.zeros(R, w, HD, dtype=torch.float16, device=DEV),
              score=torch.zeros(R, w, HD, dtype=torch.float16, device=DEV),
              tslot=torch.full((R, w), -1, dtype=torch.int32, device=DEV),
              slot=torch.full((R, w), -1, dtype=torch.int32, device=DEV),
              pos=torch.full((R, w), -1, dtype=torch.int32, device=DEV)) for w in WIDTHS}


def kfn(kv, tail, b):
    return lambda: kpool_decode_update_and_maybe_write_cache_batched(
        kv, tail, b["tslot"], b["key"], b["score"], ape, b["slot"], b["pos"], KP, HD, round_scale=True)


kgraphs = {w: capture(kfn(kv_g, tail_g, kb[w]))[0] for w in WIDTHS}   # inert inputs (-1): warm-up writes nothing
kv_g.zero_(); tail_g.zero_()
kv_e, tail_e = kcaches()
p = [0] * R
k_same = 0
for w in ORDER:
    acc = [rnd.randint(1, w) for _ in range(R)]
    key = torch.randn(R, w, HD, generator=g).to(torch.float16)        # rejected drafts keep these random keys
    score = torch.randn(R, w, HD, generator=g).to(torch.float16)
    pos = torch.tensor([[p[r] + t for t in range(w)] for r in range(R)], dtype=torch.int32)
    for r in range(R):
        key[r, : acc[r]] = ktrue[r, p[r]: p[r] + acc[r]]
        score[r, : acc[r]] = strue[r, p[r]: p[r] + acc[r]]
    rr = torch.arange(R, dtype=torch.int32)[:, None]
    tslot = (rr + 1) * RING + pos % RING
    slot = torch.where(pos % KP == KP - 1, rr * NP + pos // KP, torch.full_like(pos, -1))
    b = kb[w]
    for n, v in (("key", key), ("score", score), ("tslot", tslot), ("slot", slot), ("pos", pos)):
        b[n].copy_(v.to(DEV))
    kgraphs[w].replay()
    kfn(kv_e, tail_e, {n: v.to(DEV) for n, v in (("key", key), ("score", score), ("tslot", tslot),
                                                   ("slot", slot), ("pos", pos))})()
    torch.cuda.synchronize()
    k_same += torch.equal(kv_g, kv_e) and torch.equal(tail_g, tail_e)
    p = [p[r] + acc[r] for r in range(R)]
assert max(p) + max(WIDTHS) <= LMAX, p


def prefill_ref(kt):
    ref = torch.zeros(NPAGES, PAGE, HD + 4, dtype=torch.uint8, device=DEV)
    for r in range(R):
        n = p[r] // KP
        kpool_compress_and_write_cache(ref, kt[r, : n * KP].view(n, KP, HD).to(DEV),
                                       strue[r, : n * KP].view(n, KP, HD).to(DEV), ape,
                                       r * NP + torch.arange(n, dtype=torch.int64, device=DEV),
                                       pool_size=KP, head_dim=HD, round_scale=True)
    return ref


DIM = torch.arange(HD, device=DEV)


def pool_bytes(kv, loc):                            # one entry of the preshuffled page: 128 K bytes + 4 scale bytes
    flat = kv.reshape(-1)
    base = (loc // PAGE) * PAGE * (HD + 4); tok = loc % PAGE
    koff = base + (tok // 16) * 16 * HD + (DIM // 16) * 256 + (tok % 16) * 16 + DIM % 16
    soff = base + HD * PAGE + tok * 4
    return torch.cat([flat[koff], flat[soff: soff + 4]])


def pools_equal(a, b):
    return sum(torch.equal(pool_bytes(a, r * NP + j), pool_bytes(b, r * NP + j)) for r in range(R)
               for j in range(p[r] // KP))


npools = sum(x // KP for x in p)
ref = prefill_ref(ktrue)
k_match = pools_equal(kv_g, ref)
kbad = ktrue.clone(); kbad[0, 9] += 0.5                                # pool 2 of request 0
k_neg = npools - pools_equal(kv_g, prefill_ref(kbad))
k_ok = k_same == len(ORDER) and k_match == npools and k_neg >= 1
print(f"kpool: replay == eager {k_same}/{len(ORDER)} steps | complete accepted pools == prefill writer {k_match}/"
      f"{npools} (accepted tokens per request {p}) | negative control: {k_neg} pool(s) differ -> "
      f"{'ok' if k_ok else 'FAIL'}", flush=True)

# ---------------------------------------------------------------------------------------------------- KDA
KH, KD, CW, MAXK = 8, 128, 4, max(WIDTHS) - 1        # per-rank heads at TP8 (64 / 8), head dim, conv width
L = KH * KD; CDIM = 3 * L; SL = CW - 1 + MAXK        # conv state width k - 1 + num_spec
B, NSLOT = 8, 1 + 8 * (MAXK + 1)
LB = -5.0
cw = (0.3 * torch.randn(CDIM, CW, generator=g)).to(torch.float16).to(DEV)
cb = (0.1 * torch.randn(CDIM, generator=g)).to(torch.float16).to(DEV)
a_log = (0.5 * torch.randn(KH, generator=g)).to(DEV)
dt_bias = (0.1 * torch.randn(KH * KD, generator=g)).to(DEV)
KLMAX = 200
xtrue = (0.5 * torch.randn(B, KLMAX, CDIM, generator=g)).to(torch.float16)
gtrue = torch.randn(B, KLMAX, KH, KD, generator=g).to(torch.float16)
btrue = torch.randn(B, KLMAX, KH, generator=g).to(torch.float16)


def kda_states():
    conv = (torch.zeros(NSLOT, CDIM, SL, dtype=torch.float16, device=DEV) if is_conv_state_dim_first()
            else torch.zeros(NSLOT, SL, CDIM, dtype=torch.float16, device=DEV).transpose(-1, -2))
    return conv, torch.zeros(NSLOT, KH, KD, KD, dtype=torch.float32, device=DEV)


def rearr(x):
    return x.reshape(1, -1, KH, KD)


def kda_spec(conv, rec, x, gg, bb, qsl, idx, nacc):
    """kda.py's spec branch (pure spec-verify step)"""
    y = causal_conv1d_update(x, conv, cw, cb, activation="silu", conv_state_indices=idx[:, 0],
                             num_accepted_tokens=nacc, query_start_loc=qsl, max_query_len=idx.size(-1))
    q, k, v = y.split(L, dim=-1)
    o, _ = fused_recurrent_kda(q=rearr(q), k=rearr(k), v=rearr(v), g=gg, beta=bb, initial_state=rec,
                               use_qk_l2norm_in_kernel=True, cu_seqlens=qsl, ssm_state_indices=idx,
                               num_accepted_tokens=nacc, out=None, sigmoid_beta=True, a_log=a_log, g_bias=dt_bias,
                               compute_gate=True, lower_bound=LB)
    return o


def kda_plain(conv, rec, x, gg, bb, slot):
    """kda.py's plain-decode branch, one token for one request"""
    y = causal_conv1d_update(x, conv, cw, cb, activation="silu", conv_state_indices=slot)
    q, k, v = y.split(L, dim=-1)
    o, _ = fused_recurrent_kda(q=rearr(q), k=rearr(k), v=rearr(v), g=gg, beta=bb, initial_state=rec,
                               use_qk_l2norm_in_kernel=True,
                               cu_seqlens=torch.tensor([0, 1], dtype=torch.int32, device=DEV), ssm_state_indices=slot,
                               out=None, sigmoid_beta=True, a_log=a_log, g_bias=dt_bias, compute_gate=True,
                               lower_bound=LB)
    return o


spec_idx = (1 + torch.arange(B * (MAXK + 1), dtype=torch.int32, device=DEV)).view(B, MAXK + 1)
nacc_s = torch.ones(B, dtype=torch.int32, device=DEV)
kd = {w: dict(x=torch.zeros(B * w, CDIM, dtype=torch.float16, device=DEV),
              g=torch.zeros(1, B * w, KH, KD, dtype=torch.float16, device=DEV),
              b=torch.zeros(1, B * w, KH, dtype=torch.float16, device=DEV),
              qsl=(torch.arange(B + 1, dtype=torch.int32) * w).to(DEV)) for w in WIDTHS}
conv_g, rec_g = kda_states()
dgraphs = {w: capture(lambda w=w: kda_spec(conv_g, rec_g, kd[w]["x"], kd[w]["g"], kd[w]["b"], kd[w]["qsl"],
                                           spec_idx, nacc_s)) for w in WIDTHS}
conv_g.zero_(); rec_g.zero_(); nacc_s.fill_(1)
conv_e, rec_e = kda_states()
nacc_e = torch.ones(B, dtype=torch.int32, device=DEV)
c = [0] * B
outs: list[list[torch.Tensor]] = [[] for _ in range(B)]                        # accepted-token outputs from the graph run, in order
d_same = 0
for w in ORDER:
    acc = [rnd.randint(1, w) for _ in range(B)]
    x = (0.5 * torch.randn(B, w, CDIM, generator=g)).to(torch.float16)
    gg = torch.randn(B, w, KH, KD, generator=g).to(torch.float16)
    bb = torch.randn(B, w, KH, generator=g).to(torch.float16)
    for r in range(B):
        x[r, : acc[r]] = xtrue[r, c[r]: c[r] + acc[r]]
        gg[r, : acc[r]] = gtrue[r, c[r]: c[r] + acc[r]]
        bb[r, : acc[r]] = btrue[r, c[r]: c[r] + acc[r]]
    xd, gd, bd = x.reshape(B * w, CDIM).to(DEV), gg.reshape(1, B * w, KH, KD).to(DEV), bb.reshape(1, B * w, KH).to(DEV)
    kd[w]["x"].copy_(xd); kd[w]["g"].copy_(gd); kd[w]["b"].copy_(bd)
    gr, go = dgraphs[w]
    gr.replay()
    eo = kda_spec(conv_e, rec_e, xd.clone(), gd, bd, kd[w]["qsl"], spec_idx, nacc_e)
    torch.cuda.synchronize()
    d_same += torch.equal(go, eo) and torch.equal(conv_g, conv_e) and torch.equal(rec_g, rec_e)
    for r in range(B):
        outs[r].append(go[0, r * w: r * w + acc[r]].float().cpu())
    a = torch.tensor(acc, dtype=torch.int32, device=DEV)
    nacc_s.copy_(a); nacc_e.copy_(a)
    c = [c[r] + acc[r] for r in range(B)]
assert max(c) <= KLMAX, c


def reference(xt, req):
    conv, rec = kda_states()
    slot = torch.tensor([1], dtype=torch.int32, device=DEV)
    o = [kda_plain(conv, rec, xt[req, j][None].to(DEV), gtrue[req, j].view(1, 1, KH, KD).to(DEV),
                   btrue[req, j].view(1, 1, KH).to(DEV), slot)[0, 0].float().cpu() for j in range(c[req])]
    return torch.stack(o), rec[1].clone()


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-6))


d_out, d_state = 0.0, 0.0
for r in range(B):
    ro, rs = reference(xtrue, r)
    d_out = max(d_out, rel(torch.cat(outs[r]), ro))
    d_state = max(d_state, rel(rec_g[spec_idx[r, int(nacc_s[r]) - 1]], rs))
xbad = xtrue.clone(); xbad[0, 5] += 1.0
_, rs_bad = reference(xbad, 0)
d_neg = rel(rec_g[spec_idx[0, int(nacc_s[0]) - 1]], rs_bad)
d_ok = d_same == len(ORDER) and d_out <= 2e-3 and d_state <= 2e-3 and d_neg > 2e-3
print(f"kda: replay == eager {d_same}/{len(ORDER)} steps | vs plain decode on accepted tokens (per request {c}): "
      f"outputs max rel {d_out:.1e}, final state max rel {d_state:.1e} | negative control {d_neg:.1e} -> "
      f"{'ok' if d_ok else 'FAIL'}", flush=True)

ok = i_ok and k_ok and d_ok
print(f"PASS dynamic verify width (indexer {i_ok}, kpool tail {k_ok}, kda spec state {d_ok}): {ok}")
sys.exit(0 if ok else 1)
