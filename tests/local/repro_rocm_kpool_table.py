#!/usr/bin/env python3
"""GLM-5.3-Flash kpool indexer key cache on ROCm: block-table granularity. Stock vLLM only, one GPU, no model.

Geometry as served: attention block 640 tokens (hybrid KDA page rule), index_kpool 4, pool pages of 32
(Glm5NextIndexerCache: storage block 128 tokens), max_model_len 32768 (table width 52). On ROCm the sparse-MLA and
indexer backends accept MultipleOf(16), so the kernel block stays 640, and the indexer metadata builder only converts
the table when storage % kernel == 0 (vllm/v1/attention/backends/mla/indexer.py):

    if kernel_block_size is not None and spec.block_size != kernel_block_size \
            and spec.block_size % kernel_block_size == 0:
        indexer_block_table = (block_table[:, ::factor] // factor)

128 % 640 != 0, so the 640-token table reaches the writer and the gather as if each entry named a 32-pool page.
This script writes a 9,432-token prompt in the served prefill chunks (1920 x4, 1280, 472; blocks allocated per
chunk) with the real writer and slot mapping, gathers it back with the real gather, and compares against the writer's
own compressed output, for (a) the table the builder produces and (b) a page-granular table (block b -> pages
5b..5b+4). Prints, per 32-pool page position, o = correct, 0 = all zero, m = masked (scale 0), x = wrong.
"""
import torch
from vllm.platforms import current_platform
import vllm.models.glm5next  # noqa: F401  (import order)
from vllm.models.glm5next.amd.ops.kpool_compress import kpool_compress_and_write_cache
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import cp_gather_indexer_k_quant_cache_triton

DEV, FP8 = "cuda", current_platform.fp8_dtype()
HD, KP, PG, BLK = 128, 4, 32, 640
F = BLK // (PG * KP)                          # 5 pool pages per attention block
WIDTH = -(-32768 // BLK)                      # 52
CHUNKS = [1920, 1920, 1920, 1920, 1280, 472]
T = sum(CHUNKS); P = T // KP
NBLK = 64
g = torch.Generator().manual_seed(0)
ids = (torch.randperm(NBLK - 1, generator=g)[: -(-T // BLK)] + 1).tolist()
k = torch.randn(T, HD, generator=g).to(DEV, torch.bfloat16)
gate = torch.randn(T, HD, generator=g).to(DEV, torch.bfloat16)
ape = (0.1 * torch.randn(KP, HD, generator=g)).to(DEV)
ref_v, ref_s = kpool_compress_and_write_cache(
    torch.zeros(1, PG, HD + 4, dtype=torch.uint8, device=DEV), k.view(P, KP, HD), gate.view(P, KP, HD), ape,
    torch.zeros(P, dtype=torch.int64, device=DEV), pool_size=KP, head_dim=HD, round_scale=True,
    return_compressed=True, write_cache=False)


def table(nblocks, page_granular):
    t = torch.zeros(4, WIDTH, dtype=torch.int32, device=DEV)
    t[0, :nblocks] = torch.tensor(ids[:nblocks], dtype=torch.int32)
    if page_granular:
        t = (t[:, :, None] * F + torch.arange(F, dtype=torch.int32, device=DEV)).flatten(1)
    return t[:1]


def run(page_granular):
    cache = torch.zeros(NBLK * F, PG, HD + 4, dtype=torch.uint8, device=DEV)
    end = 0
    for n in CHUNKS:
        start, end = end, end + n
        bt = table(-(-end // BLK), page_granular)
        slots = get_compressed_slot_mapping(n, torch.tensor([0, n], dtype=torch.int32, device=DEV),
                                            torch.tensor([end], dtype=torch.int32, device=DEV), bt, PG, KP)
        last = slots[KP - 1::KP]                          # the slot of each completed pool
        kpool_compress_and_write_cache(cache, k[start:end].view(-1, KP, HD), gate[start:end].view(-1, KP, HD),
                                       ape, last.contiguous(), pool_size=KP, head_dim=HD,
                                       write_mask=last >= 0, round_scale=True)
    kq = torch.full((P, HD), 0x7F, dtype=torch.uint8, device=DEV).view(FP8)
    ks = torch.full((P,), -1.0, device=DEV)
    cp_gather_indexer_k_quant_cache_triton(cache, kq, ks, table(-(-T // BLK), page_granular),
                                           torch.tensor([0, P], dtype=torch.int32, device=DEV),
                                           token_to_seq=torch.zeros(P, dtype=torch.int32, device=DEV))
    b, rb, out = kq.view(torch.uint8), ref_v.view(torch.uint8), []
    for p in range(-(-P // PG)):
        s = slice(p * PG, min((p + 1) * PG, P))
        if torch.equal(b[s], rb[s]) and torch.equal(ks[s], ref_s[s]): out.append("o")
        elif (b[s] == 0).all() and (ks[s] == 0).all(): out.append("0")
        elif (ks[s] == 0).all(): out.append("m")
        else: out.append("x")
    return "".join(out)


for name, pg in (("table as the builder produces it", False), ("page-granular table", True)):
    pat = run(pg)
    print(f"{name:34s}: {pat}  ({pat.count('o')}/{len(pat)} pages correct)")
