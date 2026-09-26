#!/usr/bin/env python3
"""Recall at the indexer, from real activations (GLM5_TRACE_DSA=all dumps of the 9.4k needle prefill).

For every indexer layer: take the query at the question (the prompt's last token, i.e. the last row of
the chunk whose key range reaches the end of the prompt) and ask whether the pools holding the planted
facts are in the selected top-512 pools (2048 tokens / kpool 4, plus the always-selected tail), under:
  engine    the engine's own selection (pools_raw: torch prefill logits -> HIP top_k_per_row_prefill)
  eng-topk  torch.topk over the engine's prefill logits (isolates the top-k kernel)
  decode    the capture-safe decode logits (_fp8_paged_mqa_logits_decode_torch) on the same keys, fp32
  ref32     an fp32 dense reference: sum_h w_h * relu(q_h . k) * k_scale, everything in fp32
Also: max relative difference of engine and decode logits against ref32, and each fact's rank.

  python3 recall_analysis.py <dump_dir> <prompt_token_count> <factA_tok_lo> <factA_tok_hi> <factB_tok_lo> <factB_tok_hi>
"""
import glob, re, sys, collections
import torch
from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fp8_paged_mqa_logits_decode_torch

import os; DEV = os.environ.get("DEV", "cuda:0")
d, ntok = sys.argv[1], int(sys.argv[2])
facts = {"A": (int(sys.argv[3]), int(sys.argv[4])), "B": (int(sys.argv[5]), int(sys.argv[6]))}
KP, SELECT = 4, 512
fact_pools = {k: set(range(lo // KP, hi // KP + 1)) for k, (lo, hi) in facts.items()}

by_layer = collections.defaultdict(list)
for f in glob.glob(d + "/dsa-*-idx.pt"):
    x = torch.load(f, map_location="cpu", weights_only=False)
    if x.get("stage") != "scored_prefill" or "prefix" not in x:
        continue
    m = re.search(r"layers\.(\d+)\.", x["prefix"])
    by_layer[int(m.group(1))].append((int(re.search(r"dsa-(\d+)-", f).group(1)), f, x))

def rank_of(scores, pools):
    order = torch.argsort(scores, descending=True)
    pos = torch.empty_like(order); pos[order] = torch.arange(order.numel())
    return min(int(pos[p]) + 1 for p in pools if p < scores.numel())

def paged_decode_logits(q, kq, ks, w, n):
    """lay the gathered keys out as a page-major indexer cache and run the decode path on them"""
    bs, D = 64, kq.shape[1]
    npg = (n + bs - 1) // bs
    flat = torch.zeros(npg, bs * (D + 4), dtype=torch.uint8, device=DEV)
    kpad = torch.zeros(npg * bs, D, dtype=kq.dtype, device=DEV); kpad[:n] = kq
    spad = torch.zeros(npg * bs, dtype=torch.float32, device=DEV); spad[:n] = ks
    flat[:, : bs * D] = kpad.view(npg, bs * D).view(torch.uint8)
    flat[:, bs * D:] = spad.view(npg, bs).contiguous().view(torch.uint8).view(npg, bs * 4)
    cache = flat.view(npg, bs, 1, D + 4)
    bt = torch.arange(npg, dtype=torch.int32, device=DEV)[None, :]
    cl = torch.tensor([n], dtype=torch.int32, device=DEV)
    return _fp8_paged_mqa_logits_decode_torch(q[None, None], cache, w[None], cl, bt, npg * bs)[0, :n]

print(f"prompt {ntok} tokens; fact pools A {sorted(fact_pools['A'])} B {sorted(fact_pools['B'])}; select {SELECT} pools + tail")
print(f"{'layer':>5s} {'pools':>6s} | {'engine':>8s} {'eng-topk':>8s} {'decode':>8s} {'ref32':>8s} | ranks A/B (ref32)  | rel diff eng/dec vs ref32")
for L in sorted(by_layer):
    runs = sorted(by_layer[L])
    # the chunk that holds the prompt's last token: largest key end, latest run
    run, f, x = max(runs, key=lambda t: (int(x_ke.max()) if (x_ke := t[2]["cu_seqlen_ke"]).numel() else 0, t[0]))
    ke = x["cu_seqlen_ke"]; r = int(torch.argmax(ke)); n = int(ke[r])      # last row, its key count (pools)
    lo = int(x["cu_seqlen_ks"][r])
    q = x["q"][r].to(DEV); w = x["weights"][r].to(DEV).float()
    kq = x["k_quant"][lo:n].to(DEV); ks = x["k_scale"][lo:n].to(DEV).float()
    eng = x["logits"][r, lo:n].to(DEV).float()
    ref = ((torch.einsum("hd,nd->hn", q.float(), kq.float()) * ks[None]).relu() * w[:, None]).sum(0)
    dec = paged_decode_logits(q, kq, ks, w, n - lo)
    sel_eng = set(int(p) - lo for p in x["pools_raw"][r].tolist() if p >= 0)
    tail = set(range(max(0, (n - lo) - 1), n - lo))                      # always-selected tail pool
    top = lambda s: set(torch.topk(s, min(SELECT, s.numel())).indices.tolist()) | tail
    sets = {"engine": sel_eng | tail, "eng-topk": top(eng), "decode": top(dec), "ref32": top(ref)}
    hit = {k: "".join(("A" if fact_pools["A"] & s else "-") + ("B" if fact_pools["B"] & s else "-") for _ in [0]) for k, s in sets.items()}
    rel = lambda a: ((a - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
    print(f"{L:5d} {n - lo:6d} | {hit['engine']:>8s} {hit['eng-topk']:>8s} {hit['decode']:>8s} {hit['ref32']:>8s} | "
          f"{rank_of(ref, fact_pools['A']):5d}/{rank_of(ref, fact_pools['B']):5d}        | {rel(eng):.1e} / {rel(dec):.1e}"
          f"  (engine vs its own topk: {len(sets['engine'] ^ sets['eng-topk'])} differ)")
print("columns: A/B = the fact's pools are inside that selection; '-' = missed")
