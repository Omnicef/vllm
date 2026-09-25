#!/usr/bin/env python3
"""Determinism walk-back from one traced launch (GLM5_TRACE=-1 + GLM5_TRACE_DSA=3), 5 identical 9k prefills.

  python3 walkback.py <dir>     (dir holds dsa-<run>-idx.pt, trace-r<N>-s-1.log)

1. Layer-3 indexer dumps grouped into runs (a run starts at position 0), per chunk.
   At the first scored chunk (chunk 6: positions 1920..2431), sha of every tensor per run, and exact ties at the
   top-k cut-off in the logits (row-wise: values equal to the k-th largest, where the tie straddles the cut).
2. Per-layer hidden-state shas (GLM5_TRACE) for chunks 5 and 6: first layer whose output differs across runs.
"""
import collections, glob, hashlib, re, sys
import torch

d = sys.argv[1]


def sha(t):
    if t is None:
        return "none"
    if not torch.is_tensor(t):
        return str(t)
    x = t.contiguous()
    if x.dtype.itemsize == 1:
        x = x.view(torch.uint8)
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()[:12]


dumps = []
for f in glob.glob(d + "/dsa-*-idx.pt"):
    n = int(re.search(r"dsa-(\d+)-idx", f).group(1))
    dumps.append((n, f))
dumps.sort()
runs, cur = [], None
for n, f in dumps:
    x = torch.load(f, map_location="cpu", weights_only=False)
    pos0 = int(x["positions"].min()) if "positions" in x and torch.is_tensor(x["positions"]) else None
    if x.get("stage") == "short_prefill" and pos0 == 0:
        cur = []
        runs.append(cur)
    if cur is not None:
        cur.append(x)
print(f"{len(dumps)} layer-3 dumps -> {len(runs)} runs, chunks per run {[len(r) for r in runs]}")
runs = runs[-5:]

# chunk 5 (last short) and chunk 6 (first scored): per-tensor shas across runs
for ci in (5, 6):
    xs = [r[ci] for r in runs if len(r) > ci]
    print(f"\nchunk {ci} stage {xs[0].get('stage')} tokens {xs[0].get('tokens')}")
    for k in ("hidden_in", "k_raw", "q", "k_quant", "k_scale", "weights", "logits", "pools_raw", "expanded",
              "causal_indices"):
        if k in xs[0]:
            hs = [sha(x.get(k)) for x in xs]
            print(f"  {k:14s} distinct {len(set(hs))}/{len(hs)}  {hs if len(set(hs)) > 1 else hs[0]}")

# ties at the cut-off in chunk 6 logits
x6 = [r[6] for r in runs if len(r) > 6]
k = int(x6[0].get("select_k", 512))
lg = x6[0]["logits"].float()
ks, ke = x6[0]["cu_seqlen_ks"], x6[0]["cu_seqlen_ke"]
rows_tied, total_tied = 0, 0
for i in range(lg.shape[0]):
    row = lg[i, int(ks[i]):int(ke[i])]
    if row.numel() <= k:
        continue
    v = torch.sort(row, descending=True).values
    kth = v[k - 1]
    above = int((row > kth).sum()); eq = int((row == kth).sum())
    if above + eq > k:
        rows_tied += 1; total_tied += eq
print(f"\nchunk 6 (run 1) ties at the cut-off (k={k}): {rows_tied} of {lg.shape[0]} rows straddle a tie, "
      f"{total_tied} tied values in those rows; rows with > k candidates: "
      f"{sum(1 for i in range(lg.shape[0]) if int(ke[i]) - int(ks[i]) > k)}")
if len(x6) > 1:
    same_logits = all(torch.equal(x6[0]["logits"], x["logits"]) for x in x6[1:])
    same_idx = all(torch.equal(x6[0]["pools_raw"], x["pools_raw"]) for x in x6[1:])
    print(f"logits bitwise equal across runs: {same_logits}; top-k indices equal: {same_idx}")

# per-layer hidden shas (GLM5_TRACE), files in forward order
files = sorted(glob.glob(d + "/trace-r*-s-1.log"), key=lambda f: int(re.search(r"trace-r(\d+)", f).group(1)))
fw = []
for f in files:
    L = {}
    for line in open(f):
        m = dict(kv.split("=", 1) for kv in line.strip().split(","))
        if "h" in m:
            L[m["layer"]] = (m.get("h"), m.get("pre_conv"), m.get("pre_ssm"))
        elif "pre_conv" in m:
            L["pre" + m["layer"]] = (m.get("pre_conv"), m.get("pre_ssm"))
    fw.append(L)
per = len(fw) // 5 if len(fw) >= 5 else 0
print(f"\n{len(fw)} traced prefill forwards -> {per} per run")
for ci in (5, 6):
    if per <= ci:
        continue
    fr = [fw[r * per + ci] for r in range(5)]
    first = None
    for lay in ["input"] + [str(i) for i in range(64)]:
        vals = [f.get(lay, (None,))[0] for f in fr]
        if vals[0] is None:
            continue
        if len(set(vals)) > 1:
            first = lay; break
    pre = None
    for lay in [str(i) for i in range(64)]:
        vals = [f.get("pre" + lay) for f in fr]
        if vals[0] is None:
            continue
        if len(set(vals)) > 1:
            pre = lay; break
    print(f"chunk {ci}: first layer whose output hidden differs across runs: {first}; "
          f"first KDA layer whose pre-state differs: {pre}; layer 2 output distinct "
          f"{len(set(f.get('2', (None,))[0] for f in fr))}/5")
