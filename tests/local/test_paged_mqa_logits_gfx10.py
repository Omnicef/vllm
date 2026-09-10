# SPDX-License-Identifier: Apache-2.0
"""Equivalence + capture-safety for the ROCm paged MQA indexer fallback.

Compares ``_fp8_paged_mqa_logits_decode_torch`` (capture-safe, on-device) with
``fp8_paged_mqa_logits_torch_per_seq`` (the per-sequence reference that reads
``context_lens[i].item()``), then captures the new path into a CUDA graph and
replays it with different sequence lengths -- which is the property that matters:
hoisting the lengths to the host would pass the equivalence check and still bake
the warm-up batch's lengths into the graph.

Shapes are GLM-5.3-Flash's indexer shapes: index_head_dim=128, index_n_heads=32
(config.json), and the attention block size vLLM selects on this model, 640
("Setting attention block size to 640 tokens to ensure that attention page size
is >= mamba page size").

Run on one GPU:
    python3 tests/local/test_paged_mqa_logits_gfx10.py
"""

import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _fp8_paged_mqa_logits_decode_torch,
    fp8_paged_mqa_logits_torch_per_seq,
)

DIM = 128        # index_head_dim
HEADS = 32       # index_n_heads
BLOCK = 640      # attention block size chosen for GLM-5.3 on this stack
DEV = "cuda:0"


def build(batch, lens, max_model_len, seed=0, two_d_lens=False):
    """Random cache/query/table at indexer shapes. Cache is filled through the
    same flat view both implementations read: per page, all fp8 values, then all
    fp32 scales."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    fp8 = current_platform.fp8_dtype()
    pages_per_seq = cdiv(max_model_len, BLOCK)
    num_blocks = batch * pages_per_seq + 4

    flat = torch.zeros(num_blocks, BLOCK * (DIM + 4), dtype=torch.uint8, device=DEV)
    vals = (torch.randn(num_blocks, BLOCK * DIM, generator=g, device=DEV) * 0.4).to(fp8)
    flat[:, : BLOCK * DIM] = vals.view(torch.uint8)
    scales = torch.rand(num_blocks, BLOCK, generator=g, device=DEV) * 0.2 + 0.05
    flat[:, BLOCK * DIM :] = scales.contiguous().view(torch.uint8).view(num_blocks, BLOCK * 4)
    kv_cache = flat.view(num_blocks, BLOCK, 1, DIM + 4)

    q = (torch.randn(batch, 1, HEADS, DIM, generator=g, device=DEV) * 0.4).to(fp8)
    weights = torch.rand(batch, HEADS, generator=g, device=DEV, dtype=torch.float32)

    # distinct pages per sequence; pad the tail with 0 and -1 to exercise clamping
    block_tables = torch.full((batch, pages_per_seq), -1, dtype=torch.int32, device=DEV)
    for i, L in enumerate(lens):
        n = cdiv(L, BLOCK)
        block_tables[i, :n] = torch.arange(
            i * pages_per_seq + 1, i * pages_per_seq + 1 + n, dtype=torch.int32, device=DEV
        )
        if n < pages_per_seq:
            block_tables[i, n::2] = 0

    cl = torch.tensor(lens, dtype=torch.int32, device=DEV)
    return q, kv_cache, weights, (cl[:, None] if two_d_lens else cl), block_tables


def compare(tag, batch, lens, max_model_len, ppc, two_d_lens=False):
    q, kv, w, cl, bt = build(batch, lens, max_model_len, seed=batch + ppc, two_d_lens=two_d_lens)
    ref = fp8_paged_mqa_logits_torch_per_seq(q, kv, w, cl, bt, max_model_len)
    new = _fp8_paged_mqa_logits_decode_torch(q, kv, w, cl, bt, max_model_len, pages_per_chunk=ppc)
    worst = 0.0
    for i, L in enumerate(lens):
        a, b = ref[i, :L], new[i, :L]
        assert torch.isfinite(b).all(), f"{tag}: seq {i} has non-finite inside context"
        d = (a - b).abs().max().item() if L else 0.0
        worst = max(worst, d)
        assert torch.allclose(a, b, rtol=1e-3, atol=5e-3), (
            f"{tag}: seq {i} len {L} mismatch, max|diff|={d:.3e}")
        assert torch.isneginf(new[i, L:]).all(), f"{tag}: seq {i} not -inf past len {L}"
    print(f"  PASS {tag:52s} batch={batch:<3} ppc={ppc} max|diff|={worst:.2e}")


def capture_replay():
    """The property a .tolist() hoist would fail: capture once, replay with new lengths."""
    batch, max_len, ppc = 4, 8 * BLOCK, 3
    lens_a = [BLOCK, 2 * BLOCK + 7, 5 * BLOCK - 1, 8 * BLOCK]
    lens_b = [3 * BLOCK - 5, BLOCK + 1, 8 * BLOCK, 2 * BLOCK]
    q, kv, w, cl, bt = build(batch, [max_len] * batch, max_len, seed=99)

    cl.copy_(torch.tensor(lens_a, dtype=torch.int32, device=DEV))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warm-up, required before capture
        for _ in range(3):
            _fp8_paged_mqa_logits_decode_torch(q, kv, w, cl, bt, max_len, pages_per_chunk=ppc)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _fp8_paged_mqa_logits_decode_torch(q, kv, w, cl, bt, max_len, pages_per_chunk=ppc)
    print("  PASS capture succeeded (no host sync on the decode path)")

    for tag, lens in (("replay lengths A", lens_a), ("replay lengths B", lens_b)):
        cl.copy_(torch.tensor(lens, dtype=torch.int32, device=DEV))
        graph.replay()
        torch.cuda.synchronize()
        ref = fp8_paged_mqa_logits_torch_per_seq(q, kv, w, cl, bt, max_len)
        for i, L in enumerate(lens):
            assert torch.allclose(ref[i, :L], out[i, :L], rtol=1e-3, atol=5e-3), (
                f"{tag}: seq {i} wrong after replay -- lengths were baked into the graph")
            assert torch.isneginf(out[i, L:]).all(), f"{tag}: seq {i} not -inf past {L}"
        print(f"  PASS {tag} honoured by the replayed graph")


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    print(f"device: {torch.cuda.get_device_name(0)}  fp8: {current_platform.fp8_dtype()}")
    print(f"shapes: dim={DIM} heads={HEADS} block_size={BLOCK}")

    ml4, ml8 = 4 * BLOCK, 8 * BLOCK
    # lengths deliberately both multiples and non-multiples of the page size
    compare("single seq, exact page multiple", 1, [2 * BLOCK], ml4, 2)
    compare("single seq, mid-page", 1, [2 * BLOCK + 137], ml4, 2)
    compare("single seq, one token", 1, [1], ml4, 2)
    compare("single seq, full context", 1, [ml4], ml4, 2)
    compare("batch 7, mixed lengths", 7,
            [1, BLOCK, BLOCK + 1, 2 * BLOCK - 1, 2 * BLOCK, 3 * BLOCK + 99, ml4], ml4, 2)
    compare("batch 7, 2D context_lens (call-site form)", 7,
            [BLOCK, 2 * BLOCK, 137, ml4, BLOCK + 5, 3 * BLOCK, 2 * BLOCK + 1], ml4, 3,
            two_d_lens=True)
    compare("batch 32, chunk boundary (ppc=8)", 32,
            [(i * 211 + 1) % ml8 or 1 for i in range(32)], ml8, 8)
    compare("batch 32, tail chunk not full (ppc=3)", 32,
            [(i * 211 + 1) % ml8 or 1 for i in range(32)], ml8, 3)
    compare("batch 32, all full context", 32, [ml8] * 32, ml8, 8)
    capture_replay()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
