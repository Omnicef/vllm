#!/usr/bin/env python3
"""GLM-5.3 kpool indexer key cache on ROCm, end to end with the real kernels (one card, no model).

Real kernels: get_compressed_slot_mapping (writer slots), _kpool_compress_insert -> kpool_compress_and_write_cache
(the ROCm kpool writer, 16x16 preshuffle when page_size > 1), cp_gather_indexer_k_quant_cache_triton (prefill
gather), and the decode readers. Geometry as served: attention block 640 tokens (ROCm kernel block 640), index_kpool
4, pool pages of 32 (storage block 128 tokens), max_model_len 32768 -> block-table width 52. The 9,432-token needle
prompt is prefilled in the served chunks (1920 x4, 1280, 472) with blocks allocated per chunk, as the scheduler does.

Table arms (what the indexer metadata builder hands the writer and the gather):
  stock   indexer.py: 128 % 640 != 0, so the 640-token block table indexes 32-pool pages as-is
  expand  proposed P1: each 640-token block id b -> pool pages 5b..5b+4
Prefill check: the gathered keys against the writer's own compressed output (return_compressed), per 32-pool page
position: o = correct, 0 = all zero, = = identical to position 15 (aliased page), m = masked (scale 0, values not
stored), x = other. Decode check (expand table, so pages are right): logits against an fp32 reference built from the
writer's compressed output, for (a) upstream per_seq next_n==1, (b) the capture-safe loop, (c) the rows variant at
next_n 3, each as-is and with the pages de-shuffled (proposed P2), plus 20 repeats bitwise.
"""
import sys
import torch
import torch.nn.functional as F
from vllm.platforms import current_platform
import vllm.models.glm5next  # noqa: F401  (import order: breaks the indexer <-> model import cycle)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.model_executor.layers.sparse_attn_indexer_kpool import _kpool_compress_insert
from vllm.models.glm5next.amd.ops.kpool_compress import kpool_compress_and_write_cache
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    cp_gather_indexer_k_quant_cache_triton,
    indexer_k_quant_and_cache_triton,
    fp8_paged_mqa_logits_torch_per_seq,
    _fp8_paged_mqa_logits_decode_torch,
    _fp8_paged_mqa_logits_rows_torch,
)

DEV, FP8 = "cuda:0", current_platform.fp8_dtype()
HD, KP, PG, BLK, H = 128, 4, 32, 640, 32
F5 = BLK // (PG * KP)                        # pool pages per attention block (5)
WIDTH = -(-32768 // BLK)                     # 52, as served
MAXLEN_POOLS = 32768 // KP
CHUNKS = [1920, 1920, 1920, 1920, 1280, 472]
T = sum(CHUNKS); P = T // KP                 # 9432 tokens, 2358 pools
NBLK, ROWS = 64, 4
g = torch.Generator(device="cpu").manual_seed(0)
ids = (torch.randperm(NBLK - 1, generator=g)[: -(-T // BLK)] + 1).tolist()   # 15 block ids, never the null block 0

k = torch.randn(T, HD, generator=g).to(DEV, torch.bfloat16)
gate = torch.randn(T, HD, generator=g).to(DEV, torch.bfloat16)
ape = (0.1 * torch.randn(KP, HD, generator=g)).to(DEV)
ref_v, ref_s = kpool_compress_and_write_cache(
    torch.zeros(1, PG, HD + 4, dtype=torch.uint8, device=DEV), k.view(P, KP, HD), gate.view(P, KP, HD), ape,
    torch.zeros(P, dtype=torch.int64, device=DEV), pool_size=KP, head_dim=HD, round_scale=True,
    return_compressed=True, write_cache=False)


def table(nblocks, arm):
    t = torch.zeros(ROWS, WIDTH, dtype=torch.int32)
    t[0, :nblocks] = torch.tensor(ids[:nblocks], dtype=torch.int32)
    if arm == "expand":                      # P1: 640-token block ids -> 32-pool page ids
        t = (t[:, :, None] * F5 + torch.arange(F5, dtype=torch.int32)).flatten(1)
    return t.to(DEV)


def prefill(arm):
    cache = torch.zeros(NBLK * F5, PG, HD + 4, dtype=torch.uint8, device=DEV)
    end = 0
    for n in CHUNKS:
        start, end = end, end + n
        bt = table(-(-end // BLK), arm)[:1]  # blocks allocated up to this chunk; [num_reqs] view of the row buffer
        qsl = torch.tensor([0, n], dtype=torch.int32, device=DEV)
        sl = torch.tensor([end], dtype=torch.int32, device=DEV)
        slots = get_compressed_slot_mapping(n, qsl, sl, bt, PG, KP)
        _kpool_compress_insert(k[start:end], gate[start:end], ape, cache, slots, KP, HD, round_scale=True)
    torch.cuda.synchronize()
    bt = table(-(-T // BLK), arm)[:1]
    kq = torch.full((P, HD), 0x7F, dtype=torch.uint8, device=DEV).view(FP8)
    ks = torch.full((P,), -1.0, device=DEV)
    cp_gather_indexer_k_quant_cache_triton(cache, kq, ks, bt, torch.tensor([0, P], dtype=torch.int32, device=DEV),
                                           token_to_seq=torch.zeros(P, dtype=torch.int32, device=DEV))
    return cache, kq, ks


def classify(kq, ks):
    b, rb, out = kq.view(torch.uint8), ref_v.view(torch.uint8), []
    for p in range(-(-P // PG)):
        s = slice(p * PG, min((p + 1) * PG, P))
        if torch.equal(b[s], rb[s]) and torch.equal(ks[s], ref_s[s]): out.append("o")
        elif (b[s] == 0).all() and (ks[s] == 0).all(): out.append("0")
        elif (ks[s] == 0).all() and (b[s] == 0x7F).all(): out.append("m")
        elif p != 15 and b[s].shape == b[15 * PG:16 * PG].shape and torch.equal(b[s], b[15 * PG:16 * PG]) \
                and torch.equal(ks[s], ks[15 * PG:16 * PG]): out.append("=")
        else: out.append("x")
    return "".join(out)


def deshuffle(cache):                        # P2 prototype: undo the writer's 16x16 tile order, page by page
    n = cache.shape[0]
    flat = cache.view(n, -1).clone()
    v = flat[:, : PG * HD].view(n, PG // 16, HD // 16, 16, 16).permute(0, 1, 3, 2, 4).reshape(n, PG * HD)
    flat[:, : PG * HD] = v
    return flat.view_as(cache)


print(f"geometry: block {BLK} tok, kpool {KP}, page {PG} pools, width {WIDTH}; prompt {T} tok = {P} pools; ids {ids}")
caches = {}
for arm in ("stock", "expand"):
    cache, kq, ks = prefill(arm)
    caches[arm] = cache
    pat = classify(kq, ks)
    nf = int((~torch.isfinite(kq.float())).sum())
    print(f"prefill {arm:6s}: pages {pat}  (correct {pat.count('o')}/{len(pat)}, non-finite values {nf})")

# decode: one verify query at the end of the prompt, expand table (right pages) so only the layout is under test
q3 = torch.randn(1, 3, H, HD, generator=g).to(DEV).to(FP8)
w3 = torch.randn(3, H, generator=g).to(DEV)
cl = torch.tensor([P], dtype=torch.int32, device=DEV)
bt = table(-(-T // BLK), "expand")[:1]
vf, sf = ref_v.float(), ref_s


def ref_row(j, limit):
    r = (F.relu(q3[0, j].float() @ vf.T) * w3[j][:, None]).sum(0) * sf
    return r[:limit]


def score(name, rows):                       # rows: list of (logits_row, ref_row)
    rel, nan, rec = 0.0, 0, []
    for got, ref in rows:
        got = got[: ref.numel()]
        nan += int((~torch.isfinite(got)).sum())
        rel = max(rel, float(((got - ref).abs().max() / ref.abs().max()).nan_to_num(float("inf"))))
        kk = min(512, ref.numel())
        rec.append(len(set(torch.topk(got.nan_to_num(-1e30), kk).indices.tolist())
                       & set(torch.topk(ref, kk).indices.tolist())) / kk)
    return f"{name:34s} max rel diff {rel:9.2e}  non-finite {nan:5d}  top-512 recall {min(rec):.3f}"


results = {}
for label, cache in (("as-is", caches["expand"]), ("de-shuffled (P2)", deshuffle(caches["expand"]))):
    kv4 = cache.unsqueeze(-2)
    runs = {
        "(a) per_seq next_n=1": lambda: fp8_paged_mqa_logits_torch_per_seq(q3[:, 2:3], kv4, w3[2:3], cl, bt, MAXLEN_POOLS),
        "(b) capture-safe next_n=1": lambda: _fp8_paged_mqa_logits_decode_torch(q3[:, 2:3], kv4, w3[2:3], cl, bt, MAXLEN_POOLS),
        "(c) rows next_n=3": lambda: _fp8_paged_mqa_logits_rows_torch(q3, kv4, w3, cl, bt, MAXLEN_POOLS),
    }
    for name, fn in runs.items():
        out = fn()
        rows = [(out[0], ref_row(2, P))] if out.shape[0] == 1 else [(out[j], ref_row(j, P - 2 + j)) for j in range(3)]
        rep = all(torch.equal(fn(), out) for _ in range(20))
        print(score(f"{name} [{label}]", rows) + f"  20x bitwise {'yes' if rep else 'NO'}")
        results[(name, label)] = rows

# stock table on decode as well: aliasing, not layout
kv4 = deshuffle(caches["stock"]).unsqueeze(-2)
bts = table(-(-T // BLK), "stock")[:1]
out = _fp8_paged_mqa_logits_decode_torch(q3[:, 2:3], kv4, w3[2:3], cl, bts, MAXLEN_POOLS)
print(score("(b) de-shuffled, STOCK table", [(out[0], ref_row(2, P))]))

# the stock DeepSeek-V3.2/V4 ROCm writer (per-token fp8, block 64): same SHUFFLE rule, same plain next_n==1 read
BS, NB, N = 64, 48, 2900
cache64 = torch.zeros(NB, BS, HD + 4, dtype=torch.uint8, device=DEV)
kd = torch.randn(N, HD, generator=g).to(DEV, torch.bfloat16)
bt64 = (torch.randperm(NB - 1, generator=g)[: -(-N // BS)] + 1).to(DEV, torch.int32)[None]
tok = torch.arange(N, device=DEV)
indexer_k_quant_and_cache_triton(kd, cache64, (bt64[0, tok // BS] * BS + tok % BS).long(), HD, "ue8m0")
gq = torch.empty(N, HD, device=DEV, dtype=FP8); gs = torch.empty(N, device=DEV)
cp_gather_indexer_k_quant_cache_triton(cache64, gq, gs, bt64, torch.tensor([0, N], dtype=torch.int32, device=DEV),
                                       token_to_seq=torch.zeros(N, dtype=torch.int32, device=DEV))
ref64 = (F.relu(q3[0, 2].float() @ gq.float().T) * w3[2][:, None]).sum(0) * gs
cln = torch.tensor([N], dtype=torch.int32, device=DEV)
PG_SAVE, PG = PG, BS                         # deshuffle() reads the page size from PG
for label, c in (("as-is", cache64), ("de-shuffled (P2)", deshuffle(cache64))):
    o = fp8_paged_mqa_logits_torch_per_seq(q3[:, 2:3], c.unsqueeze(-2), w3[2:3], cln, bt64, 8192)
    print(score(f"DSV4 writer, (a) per_seq [{label}]", [(o[0], ref64)]))
PG = PG_SAVE

ok_pre = classify(*prefill("expand")[1:]) == "o" * (-(-P // PG))
ok_dec = all(float(((g_[: r.numel()] - r).abs().max() / r.abs().max())) < 1e-4
             for (_, l_), rows in results.items() if l_.startswith("de-") for g_, r in rows)
print(f"PASS P1 prefill (expand) {ok_pre}; PASS P2 decode (de-shuffled, all three readers) {ok_dec}")
sys.exit(0 if ok_pre and ok_dec else 1)
