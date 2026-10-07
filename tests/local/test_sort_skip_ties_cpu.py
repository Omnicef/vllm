"""CPU: after the GLM5_TOPK_TIES=stable pass, GLM5_SORT_TOPK's sort is an identity, so skipping it is bitwise-identical.
HIP-like top-k selections (a correct top-k set in random order, random member among exact ties, -1 tail) go through
_glm5_topk_ties (torch path; the Triton bounded path is checked against it on GPU by test_ties_bound_gfx10), then the
old sort is applied to a copy: the two must be bitwise equal, on rows with exact ties at the k-th score, fully tied
rows, NaN scores, rows shorter than k and empty rows, for k = 8, 64 and 512 (production select_k)."""
import os
import sys
import types

os.environ["GLM5_TOPK_TIES"] = "stable"
os.environ.pop("GLM5_TOPK_TIES_BOUND", None)          # torch path on CPU
os.environ["GLM5_SORT_TOPK"] = "1"
os.environ.pop("GLM5_SORT_AFTER_TIES", None)
# Without a GPU the platform is not ROCm: glm5next/sparse_indexer.py would import the NVIDIA module and both modules
# would register op sparse_attn_indexer_kpool. Stub the NVIDIA one; on the ROCm box only the AMD module loads anyway.
_stub = types.ModuleType("vllm.models.glm5next.nvidia.sparse_indexer")
_stub.SparseAttnIndexerKpool = object
sys.modules[_stub.__name__] = _stub

import torch  # noqa: E402

import vllm.models.glm5next.amd.sparse_indexer as m  # noqa: E402

g = torch.Generator().manual_seed(20261006)


def hip_like(sc, rs, re_, k):
    """A correct top-k of each row's valid window in random order, random choice among exact ties, -1 tail."""
    rows, n = sc.shape
    t = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        lo, hi = int(rs[r]), int(re_[r])
        if hi <= lo:
            continue
        v = torch.nan_to_num(sc[r, lo:hi], nan=float("-inf"))
        noise = torch.rand(hi - lo, generator=g)                  # random tie-break
        order = sorted(range(hi - lo), key=lambda i: (-float(v[i]), float(noise[i])))[:k]
        perm = torch.randperm(len(order), generator=g)            # HIP writes the set in a varying order
        t[r, :len(order)] = torch.tensor(order, dtype=torch.int32)[perm]
    return t


cases, sensitive = 0, 0
for k in (8, 64, 512):
    for trial in range(40):
        rows = int(torch.randint(1, 9, (1,), generator=g))
        n = int(torch.randint(k // 2 + 1, 4 * k + 2, (1,), generator=g))
        kind = trial % 5
        sc = torch.randn(rows, n, generator=g)
        if kind == 1:
            sc = (sc * 2).round() / 2                               # many exact ties, incl. at the k-th score
        elif kind == 2:
            sc = torch.full((rows, n), 0.75)                        # fully tied rows
        elif kind == 3:
            sc[torch.rand(rows, n, generator=g) < 0.1] = float("nan")
        elif kind == 4:
            sc = torch.where(torch.rand(rows, n, generator=g) < 0.5, 1.0, -1.0)   # two-level ties
        rs = torch.randint(0, max(1, n // 4), (rows,), generator=g).to(torch.int32)
        re_ = torch.minimum(rs + torch.randint(0, n, (rows,), generator=g).to(torch.int32), torch.tensor(n))
        re_[0] = rs[0]                                              # one empty row
        t = hip_like(sc, rs, re_, k)
        raw = t.clone(); m._glm5_sort_pools(raw)
        sensitive += not torch.equal(raw, t)                      # negative control: unsorted input differs
        assert m._glm5_topk_ties(t, sc, rs, re_, k) is True
        ref = t.clone()
        m._glm5_sort_pools(ref)                                     # the old pipeline: ties, then sort
        assert torch.equal(t, ref), (k, trial, kind)
        assert not m._glm5_need_sort_after_ties(True)
        cases += 1

# the sort still runs when the ties pass did not
os.environ["GLM5_TOPK_TIES"] = ""
t = torch.tensor([[5, 1, -1, 3]], dtype=torch.int32)
t0 = t.clone()
assert m._glm5_topk_ties(t, torch.zeros(1, 8), torch.zeros(1, dtype=torch.int32), torch.full((1,), 8), 4) is False
assert torch.equal(t, t0) and m._glm5_need_sort_after_ties(False)
os.environ["GLM5_SORT_AFTER_TIES"] = "1"
assert m._glm5_need_sort_after_ties(True)                           # A/B knob forces the sort
assert sensitive > cases // 2, (sensitive, cases)        # the comparison can fail
print(f"PASS sort skip after stable ties is bitwise-identical: True ({cases} cases; negative control: {sensitive} raw HIP-order inputs differ from their sorted copy)")
