import glob, re, sys, collections, torch
d = sys.argv[1]; A = range(114, 118); B = range(2229, 2233)
by = collections.defaultdict(list)
for f in glob.glob(d + "/dsa-*-idx.pt"):
    x = torch.load(f, map_location="cpu", weights_only=False)
    if "prefix" in x: by[int(re.search(r"layers\.(\d+)\.", x["prefix"]).group(1))].append(x)
print("layer | engine: cut  #==cut #>cut #==0 | ref32: cut #==cut #>cut #==0 | A logit eng/ref (best pool) | B logit eng/ref")
for L in sorted(by):
    x = max(by[L], key=lambda t: int(t["cu_seqlen_ke"].max()))
    ke = x["cu_seqlen_ke"]; r = int(torch.argmax(ke)); n = int(ke[r]); lo = int(x["cu_seqlen_ks"][r])
    eng = x["logits"][r, lo:n].float()
    q = x["q"][r].float(); w = x["weights"][r].float(); kq = x["k_quant"][lo:n].float(); ks = x["k_scale"][lo:n].float()
    ref = ((torch.einsum("hd,nd->hn", q, kq) * ks[None]).relu() * w[:, None]).sum(0)
    def stats(s):
        c = torch.topk(s, 512).values[-1].item()
        return c, int((s == c).sum()), int((s > c).sum()), int((s == 0).sum())
    ce, ne, ge, ze = stats(eng); cr, nr, gr, zr = stats(ref)
    fa = max(A, key=lambda p: ref[p].item()); fb = max(B, key=lambda p: ref[p].item())
    print(f"{L:5d} | {ce:9.3g} {ne:5d} {ge:5d} {ze:5d} | {cr:9.3g} {nr:5d} {gr:5d} {zr:5d} | "
          f"{eng[fa].item():.3g}/{ref[fa].item():.3g} | {eng[fb].item():.3g}/{ref[fb].item():.3g}")
