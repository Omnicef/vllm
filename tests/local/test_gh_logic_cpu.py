"""CPU check of the GLM5_DSA_GPUHASH helpers: cutoff-tie counting, masked score hash, order sensitivity."""
import os, torch
os.environ["GLM5_DSA_GPUHASH"] = "1"
torch.cuda.is_current_stream_capturing = lambda: False
import vllm.models.glm5next  # noqa: F401  (import order as in serving)
import vllm.model_executor.layers.sparse_attn_indexer_kpool as m
m._GH["on"] = True                     # skip the Worker patch import on CPU
m._GH["fwd"] = 0
P = "model.layers.3.self_attn.indexer.k_cache"
ninf = -1e30
L = torch.full((4, 12), 0.5)
L[0, :6] = torch.tensor([5, 4, 4, 4, 1, 0.])      # k=3: kth 4, 1 above, 3 tied for 2 slots -> cross, 1 left out
L[1, :6] = torch.tensor([5, 4, 3, 2, 1, 0.])      # no tie
L[2, :6] = torch.tensor([5, 5, 4, 3, 1, 0.])      # tie above the cutoff only -> none
L[3, :6] = torch.tensor([7, 7, 7, 7, 7, 7.])      # but only 3 valid columns (<= k) -> skipped
ks = torch.tensor([0, 0, 0, 0], dtype=torch.int32); ke = torch.tensor([6, 6, 6, 3], dtype=torch.int32)
m.glm5_gh_scores(P, L, ks, ke, 3)
row = m._GH["buf"][0, 3]
assert int(row[3]) == 1 and int(row[4]) == 1 and int(row[5]) == 1, row.tolist()
a1 = int(row[0])
L2 = L.clone(); L2[:, 8:] = 123.0                  # differs only outside [ks, ke)
m._GH["buf"].zero_(); m.glm5_gh_scores(P, L2, ks, ke, 3)
assert int(m._GH["buf"][0, 3, 0]) == a1, "A must ignore invalid columns"
L3 = L.clone(); L3[1, 2] = 3.0000002               # one ulp-scale change inside
m._GH["buf"].zero_(); m.glm5_gh_scores(P, L3, ks, ke, 3)
assert int(m._GH["buf"][0, 3, 0]) != a1, "A must see valid changes"
t = torch.tensor([[1, 2, 3, -1]], dtype=torch.int32)
m._GH["buf"].zero_(); m.glm5_gh_add(P, 1, t); b1 = int(m._GH["buf"][0, 3, 1])
m._GH["buf"].zero_(); m.glm5_gh_add(P, 1, t[:, [1, 0, 2, 3]]); b2 = int(m._GH["buf"][0, 3, 1])
assert b1 != b2, "B must be order-sensitive"
m._GH["fwd"] = 64; m.glm5_gh_add(P, 1, t)          # out-of-range slot ignored
print("gh logic ok")
# dump path: buffer created under inference_mode (as in a forward), dumped and reset outside it
m._GH["buf"] = None; m._GH["fwd"] = 0
with torch.inference_mode():
    m.glm5_gh_add(P, 1, t)
os.makedirs("/root/.cache/vllm", exist_ok=True)
assert m.glm5_gh_dump() == 1 and int(m._GH["buf"].abs().sum()) == 0 and m._GH["fwd"] == -1
print("gh dump ok")
# rank-agreement columns + GLM5_GH_SAVE: forward 6, layer 11 kept on the device and written by the dump
os.environ["GLM5_GH_SAVE"] = "6:11"; os.environ["GLM5_GH_RANK"] = "1"
m._GH["buf"] = None; m._GH["fwd"] = 6
xin = torch.randn(4, 16)
m.glm5_gh_add("model.layers.11.", 7, xin)
m.glm5_gh_add("model.layers.11.self_attn.indexer", 9, xin * 2)
m.glm5_gh_add("model.layers.12.", 7, xin)                 # other layer: hashed, not saved
assert set(m._GH["save"]) == {7, 9} and int(m._GH["buf"][6, 11, 7]) != 0 and int(m._GH["buf"][6, 12, 7]) != 0
m.glm5_gh_dump()
import glob as _g
saved = torch.load(sorted(_g.glob("/root/.cache/vllm/gh-save-r0-q*.pt"))[-1])
assert torch.equal(saved[7], xin) and torch.equal(saved[9], xin * 2) and m._GH["save"] == {}
print("gh save ok")
