#!/usr/bin/env python3
"""Is the engine's pool selection a valid top-k of its own logits? Per layer, last row:
  below_cut  engine-selected pools whose logit is strictly below the 512th-largest logit (a valid top-k has 0,
             apart from ties exactly at the cut-off)
  ties       how many logits equal the cut-off value (bf16-rounded einsum creates ties)
  worst_rank the worst rank among engine-selected pools (valid top-k: <= 512 + ties)
  sel_n      how many valid (>= 0) pool ids the engine produced; dup = duplicates among them
"""
import glob, re, sys, collections, torch
d = sys.argv[1]
by = collections.defaultdict(list)
for f in glob.glob(d + "/dsa-*-idx.pt"):
    x = torch.load(f, map_location="cpu", weights_only=False)
    if "prefix" in x: by[int(re.search(r"layers\.(\d+)\.", x["prefix"]).group(1))].append(x)
print(f"{'layer':>5s} {'sel_n':>5s} {'dup':>4s} {'below_cut':>9s} {'ties':>5s} {'worst_rank':>10s} {'raw sorted?':>11s}  first raw ids")
for L in sorted(by):
    x = max(by[L], key=lambda t: int(t["cu_seqlen_ke"].max()))
    ke = x["cu_seqlen_ke"]; r = int(torch.argmax(ke)); n = int(ke[r]); lo = int(x["cu_seqlen_ks"][r])
    lg = x["logits"][r, lo:n].float()
    raw = x["pools_raw"][r]; sel = [int(p) - lo for p in raw.tolist() if p >= 0]
    cut = torch.topk(lg, 512).values[-1]
    order = torch.argsort(lg, descending=True); rank = torch.empty_like(order); rank[order] = torch.arange(order.numel())
    below = sum(1 for p in sel if 0 <= p < lg.numel() and lg[p] < cut)
    print(f"{L:5d} {len(sel):5d} {len(sel)-len(set(sel)):4d} {below:9d} {int((lg == cut).sum()):5d} "
          f"{max(int(rank[p]) + 1 for p in sel if 0 <= p < lg.numel()):10d} {str(raw[:len(sel)].tolist() == sorted(raw[:len(sel)].tolist())):>11s}  {raw[:6].tolist()}")
x = max(by[31], key=lambda t: int(t["cu_seqlen_ke"].max()))
print("dump keys:", sorted(x.keys())); print("pools_raw shape", tuple(x["pools_raw"].shape), "logits", tuple(x["logits"].shape), "dtype", x["logits"].dtype)
