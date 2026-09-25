# SPDX-License-Identifier: Apache-2.0
"""GLM5_INDEXER_SPEC=capture: paged MQA logits for MTP verification (next_n > 1), capture-safe.

Compares _fp8_paged_mqa_logits_rows_torch with a per-row reference: fp8_paged_mqa_logits_torch_per_seq's
next_n == 1 branch (page-major cache, reads .item()) applied to each verify row with that row's limit.
NOT the per_seq next_n > 1 branch: that one (from DeepGEMM's test) slices kv_cache[..., :dim] / [..., dim:]
as if value and scale were interleaved per token, but the indexer cache is page-major (all fp8 values,
then all fp32 scales; kpool_compress.py S_OFFSET_NBYTES_IN_PAGE = page_size * head_dim), so it reads value
bytes as scales (NaN / garbage). Its disagreement is reported, not asserted. At next_n 1/2/3, batch 1/4/8, live context 700/2049/2100/9000/31000, 1-D and
2-D context lengths, max_model_len 32768. Per case:
  * values: finite inside each row's limit, -inf past it, max |diff| within fp32 rounding
  * top-k (k = index_topk = 2048), tie-aware: every key strictly above the reference's cut-off score
    is selected by both; keys selected by only one side sit at the cut-off score (within rounding);
    both sides select the same count. (No lowest-index tie-break is implied or tested.)
  * bitwise identical across 20 runs
Then capture once and replay with two different sets of lengths.

Run on one GPU:  python3 tests/local/test_paged_mqa_logits_spec_gfx10.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from test_paged_mqa_logits_gfx10 import BLOCK, DEV, DIM, HEADS, build as build1   # same cache layout
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _fp8_paged_mqa_logits_rows_torch, fp8_paged_mqa_logits_torch_per_seq)

MAX_LEN, TOPK = 32768, 2048
REL_TOL = 2e-6        # fp32 rounding: the reference dequantizes before the dot, ours scales after


def build(batch, ctxs, next_n, seed, two_d):
    q1, kv, _, _, bt = build1(batch, ctxs, MAX_LEN, seed=seed)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    q = (torch.randn(batch, next_n, HEADS, DIM, generator=g, device=DEV) * 0.4).to(q1.dtype)
    w = torch.rand(batch * next_n, HEADS, generator=g, device=DEV, dtype=torch.float32)
    cl = torch.tensor(ctxs, dtype=torch.int32, device=DEV)
    if two_d:   # the kpool call site's form: one limit per verify row
        cl = cl[:, None] - next_n + 1 + torch.arange(next_n, device=DEV, dtype=torch.int32)[None, :]
    return q, kv, w, cl, bt


def limits_of(cl, next_n):
    if cl.dim() == 1:
        return (cl[:, None] - next_n + 1 + torch.arange(next_n, device=DEV, dtype=torch.int32)[None, :]).reshape(-1)
    return cl.reshape(-1)


def reference_rows(q, kv, w, cl, bt):
    """layout-correct reference: the next_n == 1 per-sequence path, once per verify row"""
    B, nn = q.shape[:2]
    lims = limits_of(cl, nn).view(B, nn)
    out = torch.empty(B * nn, MAX_LEN, device=DEV, dtype=torch.float32)
    for j in range(nn):
        r = fp8_paged_mqa_logits_torch_per_seq(q[:, j:j + 1].contiguous(), kv, w[j::nn].contiguous(),
                                               lims[:, j].contiguous(), bt, MAX_LEN)
        out[j::nn] = r
    return out


def topk_check(ref, new, lim):
    """tie-aware top-k agreement on one row; returns (#only-in-one-side, cut-off score)"""
    k = min(TOPK, lim)
    a, b = ref[:lim], new[:lim]
    ta, ia = torch.topk(a, k); tb, ib = torch.topk(b, k)
    cut = ta[-1].item()
    tol = REL_TOL * max(abs(cut), 1e-30) + 1e-12
    sa, sb = set(ia.tolist()), set(ib.tolist())
    above = set(torch.nonzero(a > cut + tol).flatten().tolist())
    assert above <= sa and above <= sb, "a key strictly above the cut-off is missing"
    for x in sa ^ sb:
        assert abs(a[x].item() - cut) <= tol, f"key {x} differs away from the cut-off ({a[x].item()} vs {cut})"
    assert len(sa) == len(sb) == k
    return len(sa ^ sb), cut


def case(next_n, batch, ctx, two_d):
    ctxs = [max(next_n, min(MAX_LEN, ctx - 37 * i)) for i in range(batch)]
    q, kv, w, cl, bt = build(batch, ctxs, next_n, seed=next_n * 1000 + batch * 10 + (ctx % 97), two_d=two_d)
    ref = reference_rows(q, kv, w, cl, bt)
    upstream = fp8_paged_mqa_logits_torch_per_seq(q, kv, w, cl, bt, MAX_LEN) if next_n > 1 else None
    outs = [_fp8_paged_mqa_logits_rows_torch(q, kv, w, cl, bt, MAX_LEN) for _ in range(20)]
    new = outs[0]
    assert all(torch.equal(new, o) for o in outs[1:]), "not bitwise identical across 20 runs"
    lims = limits_of(cl, next_n).tolist()
    worst_rel, tie_diffs = 0.0, 0
    for r, L in enumerate(lims):
        a, b = ref[r, :L], new[r, :L]
        assert torch.isfinite(b).all(), f"row {r}: non-finite inside limit {L}"
        assert torch.isneginf(new[r, L:]).all(), f"row {r}: not -inf past {L}"
        rel = ((a - b).abs().max() / a.abs().max().clamp_min(1e-30)).item()
        worst_rel = max(worst_rel, rel)
        assert rel <= REL_TOL * 50, f"row {r}: rel diff {rel:.2e}"
        tie_diffs += topk_check(ref[r], new[r], L)[0]
    up = ""
    if upstream is not None:
        nan = sum(int(torch.isnan(upstream[r, :L]).sum()) for r, L in enumerate(lims))
        up = f"  | upstream next_n>1 ref: {nan} NaN inside limits"
    print(f"  PASS next_n={next_n} batch={batch} ctx={ctx:5d} {'2D' if two_d else '1D'}  "
          f"max rel diff {worst_rel:.1e}  top-k keys differing at the cut-off {tie_diffs}  20/20 bitwise{up}", flush=True)
    return worst_rel


def capture_replay():
    next_n, batch = 3, 4
    lens_a = [700, 2049, 9000, 31000]; lens_b = [2100, 640, 31000, 1281]
    q, kv, w, cl, bt = build(batch, [MAX_LEN] * batch, next_n, seed=7, two_d=True)
    def set_lens(lens):
        cl.copy_(torch.tensor(lens, dtype=torch.int32, device=DEV)[:, None] - next_n + 1
                 + torch.arange(next_n, device=DEV, dtype=torch.int32)[None, :])
    set_lens(lens_a)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): _fp8_paged_mqa_logits_rows_torch(q, kv, w, cl, bt, MAX_LEN)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _fp8_paged_mqa_logits_rows_torch(q, kv, w, cl, bt, MAX_LEN)
    print("  PASS capture (next_n=3, batch 4): no host sync on the verify path")
    for tag, lens in (("A", lens_a), ("B", lens_b)):
        set_lens(lens); graph.replay(); torch.cuda.synchronize()
        ref = reference_rows(q, kv, w, cl, bt)
        for r, L in enumerate(limits_of(cl, next_n).tolist()):
            rel = ((ref[r, :L] - out[r, :L]).abs().max() / ref[r, :L].abs().max()).item()
            assert rel <= REL_TOL * 50 and torch.isneginf(out[r, L:]).all(), f"replay {tag}: row {r} wrong"
        print(f"  PASS replay with lengths {tag} {lens} honoured")


def main():
    print(f"device {torch.cuda.get_device_name(0)}; dim={DIM} heads={HEADS} block={BLOCK} max_len={MAX_LEN} topk={TOPK}")
    worst = 0.0
    for next_n in (1, 2, 3):
        for batch in (1, 4, 8):
            for ctx in (700, 2049, 2100, 9000, 31000):
                for two_d in (False, True):
                    worst = max(worst, case(next_n, batch, ctx, two_d))
    capture_replay()
    print(f"\nALL CHECKS PASSED (90 cases; worst relative diff {worst:.1e})")


if __name__ == "__main__":
    main()
