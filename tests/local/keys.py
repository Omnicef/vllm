import glob, re, sys, collections, torch
d = sys.argv[1]
by = collections.defaultdict(list)
for f in glob.glob(d + "/dsa-*-idx.pt"):
    x = torch.load(f, map_location="cpu", weights_only=False)
    if "prefix" in x: by[int(re.search(r"layers\.(\d+)\.", x["prefix"]).group(1))].append(x)
def runs(mask):
    idx = mask.nonzero().flatten().tolist(); out = []; s = p = None
    for i in idx:
        if s is None: s = p = i
        elif i == p + 1: p = i
        else: out.append((s, p)); s = p = i
    if s is not None: out.append((s, p))
    return out
for L in (3, 31):
    x = max(by[L], key=lambda t: int(t["cu_seqlen_ke"].max()))
    ke = x["cu_seqlen_ke"]; r = int(torch.argmax(ke)); n = int(ke[r]); lo = int(x["cu_seqlen_ks"][r])
    kq = x["k_quant"][lo:n].float(); ks = x["k_scale"][lo:n].float()
    zero_k = (kq.abs().sum(1) == 0); zero_s = (ks == 0); nan_s = ~torch.isfinite(ks); nan_k = ~torch.isfinite(kq).all(1)
    print(f"layer {L}: {n-lo} pools; k all-zero {int(zero_k.sum())}, scale==0 {int(zero_s.sum())}, scale non-finite {int(nan_s.sum())}, k non-finite {int(nan_k.sum())}")
    rz = runs(zero_s | zero_k)
    print(f"   zero-key runs: {len(rz)}; first 12: {rz[:12]}; run lengths {collections.Counter(b - a + 1 for a, b in rz).most_common(6)}")
    print(f"   non-finite pools: {nan_s.nonzero().flatten().tolist()[:12]}  | fact A pools 114-117 zero? {bool((zero_s | zero_k)[114:118].all())}  B 2229-2232 zero? {bool((zero_s | zero_k)[2229:2233].all())}")
    print(f"   tokens in the chunk (dump 'tokens'): {x['tokens']}; ks/ke of last row: {lo}/{n}; k_quant rows total {x['k_quant'].shape[0]}")
