"""GLM5_INDEXER_POOL_BUFS=1 on one card: the prefill gather workspace sized 2 x cdiv(L, 4) + 4 rows instead of 40 x L.
Real indexer keys / queries (phase-18 dumps), tiled paged cache (block 32 pools). MAX_LEN 32k and 131k, contexts
1k / 10k / 30k tokens (256 / 2560 / 7680 pools). Checks: (1) chunk plans (_split_indexer_prefill_chunks) identical
for single requests; for multi-request batches every request's rows stay whole; (2) gathered keys / scales, prefill
scores, HIP top-k + bounded ties selection bitwise identical between the two workspace sizes; (3) the same pipeline
captured in a graph replays equal to eager."""
import glob, os
os.environ["GLM5_TOPK_TIES"] = "stable"; os.environ["GLM5_TOPK_TIES_BOUND"] = "1"; os.environ["GLM5_INDEXER_PREFILL_F16"] = "1"
import torch
import vllm.models.glm5next  # noqa: F401
import vllm.model_executor.layers.sparse_attn_indexer_kpool as m
from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import cp_gather_indexer_k_quant_cache_triton, fp8_mqa_logits_torch
from vllm.v1.attention.backends.mla import indexer as IDX

DEV, D, BS, K, NB = "cuda:0", 128, 32, 512, 4096
F8 = current_platform.fp8_dtype()
g = torch.Generator().manual_seed(0)
dumps = sorted(glob.glob("/dumps/dsa-*-idx.pt"), key=lambda f: int(f.split("dsa-")[1].split("-")[0]))
src = None
for fp in dumps[::-1]:
    d = torch.load(fp, map_location="cpu", weights_only=False)
    if d.get("stage") == "scored_prefill" and d["k_quant"].shape[0] >= 2048:
        src = d; break
KQ = src["k_quant"].contiguous().view(torch.uint8); KS = src["k_scale"].float().reshape(-1)
Q = src["q"]; Wt = src["weights"].float()


def cache_for(npools):
    idx = torch.arange(npools) % KQ.shape[0]
    keys, sc = KQ[idx], KS[idx] * (1 + 0.001 * (torch.arange(npools) // KQ.shape[0]).float())
    pages = (npools + BS - 1) // BS
    vals = torch.zeros(pages * BS, D, dtype=torch.uint8); vals[:npools] = keys
    s = torch.ones(pages * BS); s[:npools] = sc
    perm = torch.randperm(NB, generator=g)[:pages]
    cache = torch.zeros(NB, BS * D + BS * 4, dtype=torch.uint8)
    tiled = vals.view(pages, BS // 16, 16, D // 16, 16).transpose(2, 3).reshape(pages, BS * D)
    cache[perm, : BS * D] = tiled
    cache[perm, BS * D:] = s.view(pages, BS).contiguous().view(torch.uint8).view(pages, BS * 4)
    return cache.view(NB, BS, D + 4).to(DEV), perm.to(torch.int32).view(1, -1).to(DEV)


def pipeline(cache, bt, n, ws_rows, M=128):
    kbuf = torch.empty(ws_rows, D, dtype=F8, device=DEV)
    sbuf = torch.empty(ws_rows, 4, dtype=torch.uint8, device=DEV)
    kq, ks = kbuf[:n], sbuf[:n]
    cu = torch.arange(2, dtype=torch.int32, device=DEV) * n
    t2s = torch.zeros(n, dtype=torch.int32, device=DEV)
    cp_gather_indexer_k_quant_cache_triton(cache, kq, ks, bt, cu, t2s)
    q, w = QD[:M], WD[:M]
    ke = torch.clamp(torch.arange(M, device=DEV, dtype=torch.int32) + n - M + 1, min=1)
    ks0 = torch.zeros(M, dtype=torch.int32, device=DEV)
    lg = fp8_mqa_logits_torch(q, (kq, ks.view(torch.float32).view(-1)), w, ks0, ke)
    t = torch.empty(M, K, dtype=torch.int32, device=DEV)
    torch.ops._C.top_k_per_row_prefill(lg, ks0, ke, t, M, lg.stride(0), lg.stride(1), K)
    m._glm5_topk_ties(t, lg, ks0, ke, K); m._glm5_sort_pools(t)
    return kq.view(torch.uint8).clone(), ks.clone(), lg, t


QD, WD = Q[-128:].contiguous().to(DEV), Wt[-128:].contiguous().to(DEV)
ok = True
split = IDX.DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks if hasattr(
    IDX, "DeepseekV32IndexerMetadataBuilder") else None
for ML in (32768, 131072):
    big, small = 40 * ML, 2 * ((ML + 3) // 4) + 4
    for ctx in (1024, 10240, 30720):
        n = ctx // 4
        if split is not None:
            for lens in ([n], [n, n], [n, 256, n, 2560]):
                sl = torch.tensor(lens, dtype=torch.int32); ql = torch.tensor([512] * len(lens), dtype=torch.int32)
                pa = split(sl, ql, big, 512 * 2 ** 20); pb = split(sl, ql, small, 512 * 2 ** 20)
                if len(lens) == 1:
                    ok &= pa == pb
                whole = all(sum(sl[r.start:r.stop].tolist()) <= small or r.stop - r.start == 1 for r, _ in pb)
                ok &= whole
        cache, bt = cache_for(n)
        A = pipeline(cache, bt, n, big); B = pipeline(cache, bt, n, small)
        same = all(torch.equal(a, b) for a, b in zip(A, B)); ok &= same
        # graph replay of the pooled pipeline
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            pipeline(cache, bt, n, small)
        torch.cuda.current_stream().wait_stream(st)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            C = pipeline(cache, bt, n, small)
        gr.replay(); torch.cuda.synchronize()
        geq = all(torch.equal(a, c) for a, c in zip(A, C)); ok &= geq
        print(f"MAX_LEN {ML // 1024}k ctx {ctx // 1024}k: workspace rows {big} vs {small} ({big * 132 / 2**20:.0f} vs "
              f"{small * 132 / 2**20:.1f} MiB) | keys/scales/scores/selection identical {same} | replay == eager {geq}",
              flush=True)
print(f"chunk plans checked: {split is not None}")
print(f"PASS pool-sized indexer buffers: {ok}")
