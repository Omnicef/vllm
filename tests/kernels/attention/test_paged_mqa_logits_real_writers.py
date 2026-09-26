# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged MQA logits on ROCm without AITER, against caches written by the real writers.

Writers: `indexer_k_quant_and_cache_triton` (block 1: flat pages; block 64: 16x16-tiled pages) and the
DeepSeek-V4 C4A compressor `compress_norm_rope_store_triton` (ratio 4, block 64: token-major pages).
Reference: the keys decoded back from the cache bytes by `cp_gather_indexer_k_quant_cache_triton` in the
layout that writer uses, then relu(q.k) weighted per head and scaled, in fp32.
Readers: `rocm_fp8_paged_mqa_logits` with the AITER module unavailable (the layout gate picks the portable
kernel or the torch fallback), and the torch fallback directly.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm indexer cache layouts"
)

HEADS, DIM, NBLK, MAXLEN = 32, 128, 64, 4096


def _reference_logits(q, w, kq, ks, ctx):
    """[B=1, 1, H, D] query against decoded keys [N, D] fp8 + scales [N] -> [MAXLEN] fp32."""
    k = kq[:ctx].float()
    s = ks[:ctx].float()
    ref = (torch.relu(q[0, 0].float() @ k.T) * w[0][:, None]).sum(0) * s
    out = torch.full((MAXLEN,), float("-inf"), device=q.device)
    out[:ctx] = ref
    return out


def _metrics(got, ref, ctx, k=32):
    g, r = got[:ctx].float(), ref[:ctx]
    rel = float((g - r).abs().max() / r.abs().max())
    kk = min(k, ctx)
    ov = len(set(torch.topk(g.nan_to_num(-1e30), kk).indices.tolist())
             & set(torch.topk(r, kk).indices.tolist())) / kk
    return rel, ov


def _decode(cache, block_table, ctx, layout):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        cp_gather_indexer_k_quant_cache_triton,
    )

    fp8 = current_platform.fp8_dtype()
    kq = torch.empty(ctx, DIM, dtype=fp8, device="cuda")
    ks = torch.empty(ctx, dtype=torch.float32, device="cuda")
    cp_gather_indexer_k_quant_cache_triton(
        cache, kq, ks, block_table,
        torch.tensor([0, ctx], dtype=torch.int32, device="cuda"),
        token_to_seq=torch.zeros(ctx, dtype=torch.int32, device="cuda"),
        cache_layout=layout,
    )
    return kq, ks


def _write_indexer(block_size, ctx, seed):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        indexer_k_quant_and_cache_triton,
    )

    g = torch.Generator().manual_seed(seed)
    nb = -(-ctx // block_size)
    num_blocks = max(NBLK, nb + 8)
    cache = torch.zeros(num_blocks, block_size, DIM + 4, dtype=torch.uint8, device="cuda")
    bt = (torch.randperm(num_blocks, generator=g)[:nb]).to("cuda", torch.int32)
    table = torch.zeros(1, MAXLEN // block_size, dtype=torch.int32, device="cuda")
    table[0, :nb] = bt
    k = torch.randn(ctx, DIM, generator=g).to("cuda", torch.bfloat16)
    tok = torch.arange(ctx, device="cuda")
    slots = (table[0, tok // block_size] * block_size + tok % block_size).long()
    indexer_k_quant_and_cache_triton(k, cache, slots, DIM, "ue8m0")
    return cache, table, "NORMAL" if block_size == 1 else "SHUFFLE"


def _write_c4a(ctx_pools, seed, block_size=64):
    """ctx_pools compressed keys through the real C4A compressor (ratio 4, overlap)."""
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        compress_norm_rope_store_triton,
    )

    g = torch.Generator().manual_seed(seed)
    ratio, rope = 4, 64
    n_tok = ctx_pools * ratio
    state_width = 2 * DIM                       # overlap: coff = 2
    state_bs = 16
    n_state_blocks = -(-n_tok // state_bs) + 1
    state = (torch.randn(n_state_blocks, state_bs, 2 * state_width, generator=g) * 0.5).to("cuda")
    state_table = torch.arange(n_state_blocks, dtype=torch.int32, device="cuda")[None]
    positions = torch.arange(n_tok, dtype=torch.int64, device="cuda")
    state_slots = positions.clone()
    cache = torch.zeros(NBLK, block_size, DIM + 4, dtype=torch.uint8, device="cuda")
    nb = -(-ctx_pools // block_size)
    bt = (torch.randperm(NBLK, generator=g)[:nb]).to("cuda", torch.int32)
    table = torch.zeros(1, MAXLEN // block_size, dtype=torch.int32, device="cuda")
    table[0, :nb] = bt
    pool = positions // ratio
    kv_slots = torch.where(
        (positions + 1) % ratio == 0,
        (table[0, pool // block_size] * block_size + pool % block_size).long(),
        torch.full_like(positions, -1),
    )
    cos_sin = torch.randn(n_tok, rope, generator=g).to("cuda", torch.float32)
    compress_norm_rope_store_triton(
        state_cache=state, num_actual=n_tok,
        token_to_req_indices=torch.zeros(n_tok, dtype=torch.int32, device="cuda"),
        positions=positions, slot_mapping=state_slots, block_table=state_table,
        block_size=state_bs, state_width=state_width, cos_sin_cache=cos_sin,
        kv_cache=cache, k_cache_metadata=SimpleNamespace(slot_mapping=kv_slots),
        pdl_kwargs={}, head_dim=DIM, rope_head_dim=rope, compress_ratio=ratio,
        overlap=True, use_fp4_cache=False,
        rms_norm_weight=torch.ones(DIM, dtype=torch.bfloat16, device="cuda"),
        rms_norm_eps=1e-6, quant_block=128, token_stride=DIM, scale_dim=4,
    )
    return cache, table, "NORMAL"


def _query(seed):
    g = torch.Generator().manual_seed(seed + 100)
    q = (torch.randn(1, 1, HEADS, DIM, generator=g) * 0.3).to("cuda").to(current_platform.fp8_dtype())
    w = (torch.rand(1, HEADS, generator=g) * 0.2).to("cuda")
    return q, w


CASES = [
    # writer, block, compress_ratio passed to the reader, expected reader
    ("indexer", 1, 1, "portable"),
    ("indexer", 64, 1, "torch"),
    ("c4a", 64, 4, "portable"),
]


@pytest.mark.parametrize("writer,block_size,compress_ratio,expect", CASES)
@pytest.mark.parametrize("ctx", [700, 2100])
def test_gate_reads_real_writer_pages(monkeypatch, writer, block_size, compress_ratio, expect, ctx):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    if writer == "c4a" and (mod._ON_GFX942 or mod._ON_GFX950):
        pytest.skip("C4A is served by the upstream Triton kernel on gfx942/950")
    monkeypatch.setattr(mod, "paged_mqa_logits_module", lambda: None)
    used = []
    real = mod.portable_fp8_paged_mqa_logits
    monkeypatch.setattr(mod, "portable_fp8_paged_mqa_logits",
                        lambda *a, **k: (used.append(1), real(*a, **k))[1])

    if writer == "indexer":
        cache, table, layout = _write_indexer(block_size, ctx, seed=ctx + block_size)
    else:
        cache, table, layout = _write_c4a(ctx, seed=ctx)
    kq, ks = _decode(cache, table, ctx, layout)
    q, w = _query(ctx)
    ref = _reference_logits(q, w, kq, ks, ctx)

    out = mod.rocm_fp8_paged_mqa_logits(
        q, cache.unsqueeze(-2), w, torch.tensor([ctx], dtype=torch.int32, device="cuda"),
        table, torch.zeros(8, 2, dtype=torch.int32, device="cuda"), MAXLEN,
        compress_ratio=compress_ratio,
    )[0]
    picked = "portable" if used else "torch"
    rel, ov = _metrics(out, ref, ctx)
    print(f"{writer} block {block_size} ratio {compress_ratio} ctx {ctx}: reader {picked}, rel {rel:.1e}, top-32 {ov:.3f}")
    assert picked == expect
    assert rel < 1e-4 and ov == 1.0, f"{picked} reader on {layout} pages: rel {rel:.2e}, top-32 overlap {ov:.3f}"


@pytest.mark.parametrize("ctx", [700, 2100])
def test_portable_kernel_on_tiled_pages_is_wrong(ctx):
    """Negative control for the gate: 16x16-tiled pages read by the portable (token-major) kernel."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    cache, table, layout = _write_indexer(64, ctx, seed=ctx + 64)
    kq, ks = _decode(cache, table, ctx, layout)
    q, w = _query(ctx)
    ref = _reference_logits(q, w, kq, ks, ctx)
    out = mod.portable_fp8_paged_mqa_logits(
        q, cache.unsqueeze(-2), w, torch.tensor([ctx], dtype=torch.int32, device="cuda"), table, MAXLEN
    )[0]
    rel, ov = _metrics(out, ref, ctx)
    print(f"negative: portable on tiled pages ctx {ctx}: rel {rel:.1e}, top-32 {ov:.3f}")
    assert ov < 0.5
