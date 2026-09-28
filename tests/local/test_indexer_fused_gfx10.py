#!/usr/bin/env python3
"""GLM5_INDEXER_KERNEL=fused vs the capture-safe torch decode logits (_fp8_paged_mqa_logits_decode_torch), one card.

Inputs: (a) synthetic pages (random e4m3 values, random scales, shuffled block tables), (b) real indexer keys and
queries from the phase-18 layer-3 dumps (k_quant / k_scale written into tiled pages, the dump's last query rows as
decode queries). Page size 32 pools, 16x16 tiled (GLM5_INDEXER_DESHUFFLE=1), head dim 128, 32 heads,
max_model_len 8192 pools (32k tokens). Rows 1 / 3 / 8 / 32, contexts ~1k / 10k / 30k tokens (256 / 2560 / 7680
pools).

Checks:
  scores   valid positions |fused - torch| <= 1e-5 * max|torch| per row; positions >= length are -inf in both
  select   HIP top_k_per_row_decode (k=512) + GLM5_TOPK_TIES=stable + sort on both: per-row overlap >= 99.5 %, and
           every candidate in one set but not the other has a torch score within the score tolerance of that row's
           k-th score (a cutoff reshuffle); anything else is a real error (count must be 0)
  repeat   21 fused calls bitwise identical
  negative the fused kernel with the de-shuffle turned off must fail the score check (the check sees layout errors)
  graph    captured replay equals eager
  time     captured, fused vs torch, per call, at each context; torch also at max_model_len 65536 (262k tokens)
"""
import glob, os
os.environ["GLM5_INDEXER_DESHUFFLE"] = "1"; os.environ["GLM5_TOPK_TIES"] = "stable"; os.environ["GLM5_SORT_TOPK"] = "1"
import torch
import vllm.models.glm5next  # noqa: F401
import vllm.model_executor.layers.sparse_attn_indexer_kpool as m
from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fp8_paged_mqa_logits_decode_torch as torch_logits
from vllm.v1.attention.ops import glm5_indexer_fused as F

DEV, H, D, BS, K = "cuda:0", 32, 128, 32, 512
F8 = current_platform.fp8_dtype()
g = torch.Generator().manual_seed(0)


def tile(vals):      # [pages, BS, D] uint8 token-major -> the writer's 16x16 tiled page layout
    p = vals.shape[0]
    return vals.view(p, BS // 16, 16, D // 16, 16).transpose(2, 3).reshape(p, BS * D)


def make_cache(keys_u8, scales, num_blocks):
    """keys_u8 [N, D] uint8 (fp8 bits), scales [N] fp32 -> kv_cache [num_blocks, BS, 1, D+4] and a block table."""
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
    m._glm5_topk_ties(t, logits, torch.zeros(rows, dtype=torch.int32, device=DEV),
                      m._glm5_decode_row_end(lens, 1, rows), K)
    m._glm5_sort_pools(t)
    return t


def compare(name, q, cache, w, lens, bt, maxlen, tol=1e-5):
    a = torch_logits(q, cache, w, lens, bt, maxlen)
    b = F.fused_paged_mqa_logits(q, cache, w, lens, bt, maxlen)
    ok, worst, real, minov = True, 0.0, 0, 1.0
    sa, sb = selection(a, lens), selection(b, lens)
    for r in range(a.shape[0]):
        L = int(lens[r])
        ar, br = a[r, :L], b[r, :L]
        scale = float(ar.abs().max()) or 1.0
        err = float((ar - br).abs().max()) / scale
        worst = max(worst, err)
        ok &= err <= tol and bool(torch.isinf(a[r, L:]).all()) and bool(torch.isinf(b[r, L:]).all())
        A = set(x for x in sa[r].tolist() if x >= 0); B = set(x for x in sb[r].tolist() if x >= 0)
        ov = len(A & B) / max(len(A), 1); minov = min(minov, ov)
        if L > K:
            kth = float(ar.topk(K).values[-1])
            for x in A ^ B:
                if abs(float(ar[x]) - kth) > tol * scale:
                    real += 1
    ok &= minov >= 0.995 and real == 0
    return ok, worst, minov, real, a, b


ok = True
NB = 4096
maxlen = 8192
# (a) synthetic
for rows in (1, 3, 8, 32):
    for ctx_pools in (256, 2560, 7680):
        keys = (torch.randn(ctx_pools, D, generator=g) * 1.5).to(F8).view(torch.uint8)
        scales = torch.rand(ctx_pools, generator=g) * 0.01 + 0.001
        cache, perm = make_cache(keys, scales, NB)
        cache = cache.to(DEV)
        bt = torch.zeros(rows, maxlen // BS, dtype=torch.int32); bt[:, : perm.numel()] = perm
        bt = bt.to(DEV)
        q = (torch.randn(rows, 1, H, D, generator=g) * 1.5).to(F8).to(DEV)
        w = (torch.rand(rows, H, generator=g) * 0.1).to(DEV)
        lens = torch.tensor([ctx_pools - (r % 3) for r in range(rows)], dtype=torch.int32, device=DEV)
        good, worst, minov, real, a, b = compare("syn", q, cache, w, lens, bt, maxlen)
        rep = all(torch.equal(b, F.fused_paged_mqa_logits(q, cache, w, lens, bt, maxlen)) for _ in range(20))
        ok &= good and rep
        print(f"synthetic rows {rows:2d} ctx {ctx_pools * 4:5d} tok: max rel score err {worst:.1e} | min overlap "
              f"{100 * minov:.2f}% | real selection errors {real} | 21x bitwise {rep} -> {'ok' if good and rep else 'FAIL'}")
# negative control: layout ignored
os.environ["GLM5_INDEXER_DESHUFFLE"] = "0"
bad = F.fused_paged_mqa_logits(q, cache, w, lens, bt, maxlen)
os.environ["GLM5_INDEXER_DESHUFFLE"] = "1"
L0 = int(lens[0]); neg = float((bad[0, :L0] - a[0, :L0]).abs().max()) / float(a[0, :L0].abs().max())
print(f"negative control (de-shuffle off): max rel score err {neg:.2e} -> {'detected' if neg > 1e-5 else 'NOT DETECTED'}")
ok &= neg > 1e-5
# (b) real keys / queries from the phase-18 dumps
dumps = sorted(glob.glob("/dumps/dsa-*-idx.pt"), key=lambda f: int(f.split("dsa-")[1].split("-")[0]))
real_done = 0
for fp in dumps[::-1]:
    d = torch.load(fp, map_location="cpu", weights_only=False)
    if d.get("stage") != "scored_prefill" or d["k_quant"].shape[0] < 2048:
        continue
    kq = d["k_quant"].contiguous().view(torch.uint8); ks = d["k_scale"].float().reshape(-1)
    n = kq.shape[0]
    cache, perm = make_cache(kq, ks, NB); cache = cache.to(DEV)
    rows = 3
    bt = torch.zeros(rows, maxlen // BS, dtype=torch.int32); bt[:, : perm.numel()] = perm; bt = bt.to(DEV)
    q = d["q"][-rows:].contiguous().view(rows, 1, H, D).to(DEV)
    w = d["weights"][-rows:].float().to(DEV)
    lens = torch.tensor([n - 2, n - 1, n], dtype=torch.int32, device=DEV)
    good, worst, minov, real, a, b = compare("real", q, cache, w, lens, bt, maxlen)
    ok &= good
    print(f"real (dump {fp.rsplit('/', 1)[-1]}, {n} pools): max rel score err {worst:.1e} | min overlap "
          f"{100 * minov:.2f}% | real selection errors {real} -> {'ok' if good else 'FAIL'}")
    real_done += 1
    if real_done == 3:
        break


def cap_time(fn_, reps=30):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn_()
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = fn_()
    gr.replay(); torch.cuda.synchronize()
    eq = torch.equal(out, fn_())
    ts = []
    for _ in range(reps):
        x, y = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        x.record(); gr.replay(); y.record(); torch.cuda.synchronize(); ts.append(x.elapsed_time(y))
    return sorted(ts)[reps // 2], eq


for ctx_pools in (256, 2560, 7680):
    keys = (torch.randn(ctx_pools, D, generator=g) * 1.5).to(F8).view(torch.uint8)
    cache, perm = make_cache(keys, torch.rand(ctx_pools, generator=g) * 0.01 + 0.001, NB); cache = cache.to(DEV)
    for ml in (8192, 65536):
        bt = torch.zeros(1, ml // BS, dtype=torch.int32); bt[:, : perm.numel()] = perm; bt = bt.to(DEV)
        q = (torch.randn(1, 1, H, D, generator=g) * 1.5).to(F8).to(DEV); w = torch.rand(1, H, device=DEV) * 0.1
        lens = torch.tensor([ctx_pools], dtype=torch.int32, device=DEV)
        tt, _ = cap_time(lambda: torch_logits(q, cache, w, lens, bt, ml))
        tf, eq = cap_time(lambda: F.fused_paged_mqa_logits(q, cache, w, lens, bt, ml))
        ok &= eq
        print(f"time c=1 ctx {ctx_pools * 4:5d} tok, max_model_len {ml * 4:6d} tok: torch {tt:7.3f} ms, fused "
              f"{tf:6.3f} ms per layer (x11: {11 * tt:6.1f} -> {11 * tf:5.2f} ms) | replay == eager {eq}")
print(f"PASS fused indexer decode: {ok}")
