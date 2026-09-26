#!/usr/bin/env python3
"""GLM-5.3 kpool indexer key cache on ROCm, end to end with the real kernels (one card, no model).

Real kernels: get_compressed_slot_mapping (writer slots), _kpool_compress_insert -> kpool_compress_and_write_cache
(the ROCm kpool writer, 16x16 preshuffle when page_size > 1), cp_gather_indexer_k_quant_cache_triton (prefill
gather), and the decode readers. Geometry as served (and as in the 6d dumps): attention block 640 tokens (ROCm kernel
block 640), index_kpool 4, pool pages of 32 (storage block 128 tokens), max_model_len 32768 -> block-table width 52;
the 9,432-token needle prompt prefilled in the served chunks (1920 x4, 1280, 472), blocks allocated per chunk.

Fixes under test are the branch's own code, toggled by their knobs:
  P1 GLM5_INDEXER_TABLE=expand   glm5_block_table_expand_factor / glm5_expand_block_table (indexer.py)
  P2 GLM5_INDEXER_DESHUFFLE=1    glm5_indexer_values_token_major inside the torch paged readers
Prefill: gathered keys against the writer's own compressed output, per 32-pool page position: o = correct,
0 = all zero, = = identical to position 16 (aliased page), m = masked (scale 0, values not stored), x = other.
Decode (right pages): logits of (a) upstream per_seq next_n==1, (b) the capture-safe loop, (c) rows at next_n 3,
knob off and on, against an fp32 reference built from the writer's output; 20 repeats bitwise; graph capture of
(c) with the expanded table in a fixed buffer, replayed at two lengths against eager.
Writer inputs (pre-pool k, gate score, ape) are not in the dumps, so they are synthetic; with a dump path as argv[1],
the decode queries and head weights are the dump's last three query rows (layer 3).

  python3 test_kpool_layout_gfx10.py [dump_dir]
"""
import glob, os, sys
import torch
import torch.nn.functional as F
from vllm.platforms import current_platform
import vllm.models.glm5next  # noqa: F401  (import order: breaks the indexer <-> model import cycle)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.mla.indexer import glm5_block_table_expand_factor, glm5_expand_block_table
from vllm.models.glm5next.nvidia.sparse_indexer import _kpool_compress_insert
from vllm.models.glm5next.amd.ops.kpool_compress import kpool_compress_and_write_cache
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    cp_gather_indexer_k_quant_cache_triton,
    indexer_k_quant_and_cache_triton,
    fp8_paged_mqa_logits_torch_per_seq,
    _fp8_paged_mqa_logits_decode_torch,
    _fp8_paged_mqa_logits_rows_torch,
    fp8_mqa_logits_torch,
)
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

DEV, FP8 = "cuda:0", current_platform.fp8_dtype()
HD, KP, PG, BLK, H = 128, 4, 32, 640, 32
WIDTH = -(-32768 // BLK)                     # 52, as served
MAXLEN_POOLS = 32768 // KP
CHUNKS = [1920, 1920, 1920, 1920, 1280, 472]
if os.environ.get("KP_TOKENS"):              # other geometries: KP_TOKENS=29976 KP_CHUNK=512 (the served chunking)
    _t, _c = int(os.environ["KP_TOKENS"]), int(os.environ.get("KP_CHUNK", "512"))
    CHUNKS = [_c] * (_t // _c) + ([_t % _c] if _t % _c else [])
T = sum(CHUNKS); P = T // KP                 # 9432 tokens, 2358 pools by default
NBLK, ROWS = -(-T // BLK) + 49, 4
g = torch.Generator(device="cpu").manual_seed(0)
ids = (torch.randperm(NBLK - 1, generator=g)[: -(-T // BLK)] + 1).tolist()   # 15 block ids, never the null block 0


def knob(name, val):
    if val is None: os.environ.pop(name, None)
    else: os.environ[name] = val


knob("GLM5_INDEXER_TABLE", "expand")
F5 = glm5_block_table_expand_factor(PG * KP, BLK)
knob("GLM5_INDEXER_TABLE", None)
assert F5 == BLK // (PG * KP) and glm5_block_table_expand_factor(PG * KP, BLK) == 1, F5
NPAGES = NBLK * F5

k = torch.randn(T, HD, generator=g).to(DEV, torch.bfloat16)
gate = torch.randn(T, HD, generator=g).to(DEV, torch.bfloat16)
ape = (0.1 * torch.randn(KP, HD, generator=g)).to(DEV)
ref_v, ref_s = kpool_compress_and_write_cache(
    torch.zeros(1, PG, HD + 4, dtype=torch.uint8, device=DEV), k.view(P, KP, HD), gate.view(P, KP, HD), ape,
    torch.zeros(P, dtype=torch.int64, device=DEV), pool_size=KP, head_dim=HD, round_scale=True,
    return_compressed=True, write_cache=False)


def table(nblocks, arm):
    t = torch.zeros(ROWS, WIDTH, dtype=torch.int32, device=DEV)
    t[0, :nblocks] = torch.tensor(ids[:nblocks], dtype=torch.int32)
    return glm5_expand_block_table(t, F5) if arm == "expand" else t


def prefill(arm):
    cache = torch.zeros(NPAGES, PG, HD + 4, dtype=torch.uint8, device=DEV)
    end = 0
    for n in CHUNKS:
        start, end = end, end + n
        bt = table(-(-end // BLK), arm)[:1]  # blocks allocated up to this chunk; [num_reqs] view of the row buffer
        qsl = torch.tensor([0, n], dtype=torch.int32, device=DEV)
        sl = torch.tensor([end], dtype=torch.int32, device=DEV)
        slots = get_compressed_slot_mapping(n, qsl, sl, bt, PG, KP)
        _kpool_compress_insert(k[start:end], gate[start:end], ape, cache, slots, KP, HD, round_scale=True)
    bt = table(-(-T // BLK), arm)[:1]
    kq, ks = WS_V[:P], WS_S[:P].view(torch.float32).view(-1)   # the engine's workspace, sliced per chunk
    kq.view(torch.uint8).fill_(0x7F); ks.fill_(-1.0)
    cp_gather_indexer_k_quant_cache_triton(cache, kq, ks, bt, torch.tensor([0, P], dtype=torch.int32, device=DEV),
                                           token_to_seq=torch.zeros(P, dtype=torch.int32, device=DEV))
    torch.cuda.synchronize()
    return cache, kq, ks


def classify(kq, ks):
    b, rb, out = kq.view(torch.uint8), ref_v.view(torch.uint8), []
    a16 = slice(16 * PG, 17 * PG)
    for p in range(-(-P // PG)):
        s = slice(p * PG, min((p + 1) * PG, P))
        if torch.equal(b[s], rb[s]) and torch.equal(ks[s], ref_s[s]): out.append("o")
        elif (b[s] == 0).all() and (ks[s] == 0).all(): out.append("0")
        elif (ks[s] == 0).all() and (b[s] == 0x7F).all(): out.append("m")
        elif p != 16 and b[s].shape == b[a16].shape and torch.equal(b[s], b[a16]) and torch.equal(ks[s], ks[a16]):
            out.append("=")
        else: out.append("x")
    return "".join(out)


# gather workspace sized as the engine does: get_max_prefill_buffer_size rows (max_model_len * 40), fp8 values
# + 4 scale bytes per row, one allocation sliced per chunk (sparse_attn_indexer_kpool / _gather_workspace_shapes)
class _MC:
    class model_config: max_model_len = 32768
WS_ROWS = get_max_prefill_buffer_size(_MC)
WS_V = torch.empty(WS_ROWS, HD, dtype=FP8, device=DEV)
WS_S = torch.empty(WS_ROWS, 4, dtype=torch.uint8, device=DEV)
print(f"gather workspace: {WS_ROWS} rows ({(WS_V.numel() + WS_S.numel()) / 2**20:.0f} MiB), this prompt needs {P}")
print(f"geometry: block {BLK} tok, kpool {KP}, page {PG} pools, expand factor {F5}, width {WIDTH}; "
      f"prompt {T} tok = {P} pools; ids {ids}")
caches, pats = {}, {}
for arm in ("stock", "expand"):
    cache, kq, ks = prefill(arm)
    caches[arm], pats[arm] = cache, classify(kq, ks)
    print(f"prefill {arm:6s}: pages {pats[arm]}  (correct {pats[arm].count('o')}/{len(pats[arm])}, "
          f"non-finite values {int((~torch.isfinite(kq.float())).sum())})")

# decode queries: the dump's last three query rows (layer 3) when given, else synthetic
src = "synthetic"
q3 = torch.randn(1, 3, H, HD, generator=g).to(DEV).to(FP8)
w3 = torch.randn(3, H, generator=g).to(DEV)
if len(sys.argv) > 1:
    best = None
    for f in glob.glob(sys.argv[1] + "/dsa-*-idx.pt"):
        x = torch.load(f, map_location="cpu", weights_only=False)
        if x.get("stage") == "scored_prefill" and "layers.3." in x.get("prefix", "") and \
                (best is None or int(x["cu_seqlen_ke"].max()) > int(best["cu_seqlen_ke"].max())):
            best = x
    r = int(torch.argmax(best["cu_seqlen_ke"]))
    q3 = best["q"][r - 2: r + 1][None].to(DEV).to(FP8)
    w3 = best["weights"][r - 2: r + 1].to(DEV).float()
    src = f"dump layer 3 rows {r - 2}..{r}"
cl = torch.tensor([P], dtype=torch.int32, device=DEV)
bt = table(-(-T // BLK), "expand")[:1]
vf, sf = ref_v.float(), ref_s


def ref_row(j, limit):
    return ((F.relu(q3[0, j].float() @ vf.T) * w3[j][:, None]).sum(0) * sf)[:limit]


def metrics(rows):                           # rows: list of (logits_row, ref_row)
    rel, nan, rec = 0.0, 0, []
    for got, ref in rows:
        got = got[: ref.numel()]
        nan += int((~torch.isfinite(got)).sum())
        rel = max(rel, float(((got - ref).abs().max() / ref.abs().max()).nan_to_num(float("inf"))))
        kk = min(512, ref.numel())
        rec.append(len(set(torch.topk(got.nan_to_num(-1e30), kk).indices.tolist())
                       & set(torch.topk(ref, kk).indices.tolist())) / kk)
    return rel, nan, min(rec)


def show(name, m, extra=""):
    print(f"{name:40s} max rel diff {m[0]:9.2e}  non-finite {m[1]:5d}  top-512 recall {m[2]:.3f}{extra}")


print(f"decode queries: {src}")
kv4 = caches["expand"].unsqueeze(-2)
runs = {
    "(a) per_seq next_n=1": lambda: fp8_paged_mqa_logits_torch_per_seq(q3[:, 2:3], kv4, w3[2:3], cl, bt, MAXLEN_POOLS),
    "(b) capture-safe next_n=1": lambda: _fp8_paged_mqa_logits_decode_torch(q3[:, 2:3], kv4, w3[2:3], cl, bt, MAXLEN_POOLS),
    "(c) rows next_n=3": lambda: _fp8_paged_mqa_logits_rows_torch(q3, kv4, w3, cl, bt, MAXLEN_POOLS),
}
fixed = {}
for val, label in ((None, "P2 off"), ("1", "P2 on")):
    knob("GLM5_INDEXER_DESHUFFLE", val)
    for name, fn in runs.items():
        out = fn()
        rows = [(out[0], ref_row(2, P))] if out.shape[0] == 1 else [(out[j], ref_row(j, P - 2 + j)) for j in range(3)]
        rep = all(torch.equal(fn(), out) for _ in range(20))
        m = metrics(rows)
        show(f"{name} [{label}]", m, f"  20x bitwise {'yes' if rep else 'NO'}")
        if val: fixed[name] = (m, rep)

# graph capture of (c) with P1+P2: fixed buffers, replay at two lengths, compare with eager
static_q, static_w = q3.clone(), w3.clone()
static_cl = torch.zeros(1, dtype=torch.int32, device=DEV)
bt_buf = torch.zeros(ROWS, WIDTH * F5, dtype=torch.int32, device=DEV)
knob("GLM5_INDEXER_TABLE", "expand")
tbl = torch.zeros(ROWS, WIDTH, dtype=torch.int32, device=DEV); tbl[0, :len(ids)] = torch.tensor(ids, dtype=torch.int32)
static_bt = glm5_expand_block_table(tbl[:1], glm5_block_table_expand_factor(PG * KP, BLK), bt_buf)
static_cl.fill_(P)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2): _fp8_paged_mqa_logits_rows_torch(static_q, kv4, static_w, static_cl, static_bt, MAXLEN_POOLS)
torch.cuda.current_stream().wait_stream(s)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    g_out = _fp8_paged_mqa_logits_rows_torch(static_q, kv4, static_w, static_cl, static_bt, MAXLEN_POOLS)
graph_ok = True
for L in (P, 1000):
    static_cl.fill_(L)
    graph.replay(); torch.cuda.synchronize()
    eager = _fp8_paged_mqa_logits_rows_torch(q3, kv4, w3, torch.tensor([L], dtype=torch.int32, device=DEV), bt, MAXLEN_POOLS)
    same = torch.equal(g_out, eager)
    m = metrics([(g_out[j], ref_row(j, L - 2 + j)) for j in range(3)])
    show(f"graph replay (c) len {L} [P1+P2]", m, f"  == eager {'yes' if same else 'NO'}")
    graph_ok &= same and m[0] < 1.5e-7 and m[2] == 1.0 and m[1] == 0

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
for val, label in ((None, "P2 off"), ("1", "P2 on")):
    knob("GLM5_INDEXER_DESHUFFLE", val)
    o = fp8_paged_mqa_logits_torch_per_seq(q3[:, 2:3], cache64.unsqueeze(-2), w3[2:3], cln, bt64, 8192)
    show(f"DSV4 writer, (a) per_seq [{label}]", metrics([(o[0], ref64)]))

# prefill logits transient at the served chunk shape (MAX_BATCHED 512 query rows x all pools so far x 32 heads)
M = min(512, T)
qm = torch.randn(M, H, HD, generator=g).to(DEV).to(FP8); wm = torch.randn(M, H, generator=g).to(DEV)
ksx = torch.zeros(M, dtype=torch.int32, device=DEV); kex = torch.full((M,), P, dtype=torch.int32, device=DEV)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
fp8_mqa_logits_torch(qm, (ref_v, ref_s[:, None]), wm, ksx, kex); torch.cuda.synchronize()
print(f"prefill logits transient: M {M} x N {P} x H {H}: peak +{(torch.cuda.max_memory_allocated() - base) / 2**20:.0f} MiB "
      f"(chunker budget checks M*N*4 = {M * P * 4 / 2**20:.0f} MiB against VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512)")

ok_pre = pats["expand"] == "o" * len(pats["expand"]) and pats["stock"].count("o") < len(pats["stock"])
ok_dec = all(m[0] < 1.5e-7 and m[1] == 0 and m[2] == 1.0 and rep for m, rep in fixed.values())
print(f"PASS P1 prefill {ok_pre}; PASS P2 decode {ok_dec}; PASS graph {graph_ok}")
sys.exit(0 if ok_pre and ok_dec and graph_ok else 1)
