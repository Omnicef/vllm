#!/usr/bin/env python3
"""GLM5_TOPK_TIES=stable on one card, against the HIP top_k_per_row_prefill / top_k_per_row_decode.

Rows (prefill shape 512 x 2656, k 512; decode shape 3 x 8192, k 512, next_n 3):
  T  planted cutoff ties: k-10 distinct high scores, 40 candidates tied at the cutoff value, the rest lower
  U  all scores distinct (no ties)
  M  fewer valid candidates than k (ke - ks < k), large garbage scores in the masked tail
  W  the whole valid tail tied (every valid score equal, more than k of them)
Some prefill rows get ks > 0 (indices are relative to ks in the HIP convention).
Off (knob unset): tied rows (T, W) show >1 distinct selection over 21 calls. On: 1, equal to a CPU lowest-index
reference. U rows: identical to the HIP set. M rows: nothing outside [ks, ke) selected, same set as off.
Then the knob's cost per call at decode and prefill shapes.
"""
import os
import numpy as np
import torch
import vllm.models.glm5next  # noqa: F401  (import order as in serving)
import vllm.model_executor.layers.sparse_attn_indexer_kpool as m

DEV, K, CALLS = "cuda:0", 512, 21
g = torch.Generator().manual_seed(0)


def distinct_scores(n):
    return torch.randperm(n, generator=g).float() / n          # exact-distinct floats in [0, 1)


def build(rows, n, kinds, ks_nonzero=False):
    S = torch.empty(rows, n); ks = torch.zeros(rows, dtype=torch.int32); ke = torch.full((rows,), n, dtype=torch.int32)
    for r in range(rows):
        kind = kinds[r % len(kinds)]
        s0 = int(torch.randint(0, 64, (1,), generator=g)) if ks_nonzero and r % 3 == 0 else 0
        ks[r] = s0
        S[r] = distinct_scores(n) - 2.0                         # everything low by default
        if kind == "T":
            L = n - s0
            p = torch.randperm(L, generator=g) + s0
            S[r, p[: K - 10]] = 10.0 + distinct_scores(K - 10)   # clearly above
            S[r, p[K - 10: K + 30]] = 5.0                        # 40 tied at the cutoff, 10 slots for them
        elif kind == "U":
            S[r, s0:] = distinct_scores(n - s0)
        elif kind == "M":
            ke[r] = s0 + K // 2
            S[r, s0: ke[r]] = distinct_scores(int(ke[r]) - s0)
            S[r, ke[r]:] = 100.0                                 # garbage beyond ke must never be selected
        elif kind == "W":
            ke[r] = min(n, s0 + 2 * K)
            S[r, s0: ke[r]] = 3.0
    return S, ks, ke


def reference(S, ks, ke):
    out = torch.full((S.shape[0], K), -1, dtype=torch.int32)
    for r in range(S.shape[0]):
        a, b = int(ks[r]), int(ke[r])
        v = S[r, a:b].numpy()
        sel = sorted(np.lexsort((np.arange(b - a), -v))[:K].tolist())   # by score desc, then lowest index
        out[r, : len(sel)] = torch.tensor(sel, dtype=torch.int32)
    return out


def canon(t):
    big = torch.iinfo(t.dtype).max
    s = torch.where(t < 0, big, t).sort(dim=-1).values
    return torch.where(s == big, -1, s)


def run_prefill(S, ks, ke, knob):
    L = S.to(DEV); ks_d = ks.to(DEV); ke_d = ke.to(DEV)
    outs = []
    for _ in range(CALLS):
        t = torch.empty(S.shape[0], K, dtype=torch.int32, device=DEV)
        torch.ops._C.top_k_per_row_prefill(L, ks_d, ke_d, t, S.shape[0], L.stride(0), L.stride(1), K)
        if knob:
            m._glm5_topk_ties(t, L, ks_d, ke_d, K)
        outs.append(t.cpu())
    return outs


def run_decode(S, seq_lens, next_n, knob):
    L = S.to(DEV); sl = seq_lens.to(DEV)
    rows = S.shape[0]
    outs = []
    for _ in range(CALLS):
        t = torch.empty(rows, K, dtype=torch.int32, device=DEV)
        torch.ops._C.top_k_per_row_decode(L, next_n, sl, t, rows, L.stride(0), L.stride(1), K)
        if knob:
            m._glm5_topk_ties(t, L, torch.zeros(rows, dtype=torch.int32, device=DEV),
                              m._glm5_decode_row_end(sl, next_n, rows), K)
        outs.append(t.cpu())
    return outs


def check(name, S, ks, ke, kinds, off, on):
    ok = True
    ref = reference(S, ks, ke)
    rows = S.shape[0]
    for kind in sorted(set(kinds)):
        rr = [r for r in range(rows) if kinds[r % len(kinds)] == kind]
        d_off = len({canon(o[rr]).numpy().tobytes() for o in off})
        d_on = len({canon(o[rr]).numpy().tobytes() for o in on})
        eq_ref = all(torch.equal(canon(o[rr]), ref[rr]) for o in on)
        same_as_hip = all(torch.equal(canon(on[i][rr]), canon(off[i][rr])) for i in range(CALLS))
        outside = any(bool(((o[rr] >= 0) & (o[rr] >= (ke - ks)[rr, None])).any()) for o in on + off)
        if kind in ("T", "W"):
            good = d_off > 1 and d_on == 1 and eq_ref
        elif kind == "U":
            good = d_on == 1 and same_as_hip and eq_ref
        else:  # M
            good = d_on == 1 and same_as_hip and not outside and eq_ref
        ok &= good
        print(f"{name} {kind}: rows {len(rr)} | off distinct {d_off}/{CALLS} | on distinct {d_on}/{CALLS} | "
              f"on == lowest-index ref {eq_ref} | on == HIP set {same_as_hip} | masked selected {outside} "
              f"-> {'ok' if good else 'FAIL'}")
    return ok


def cost(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2]


os.environ["GLM5_TOPK_TIES"] = "stable"
ok = True
kinds = ["T", "U", "M", "W"]
S, ks, ke = build(512, 2656, kinds, ks_nonzero=True)
ok &= check("prefill", S, ks, ke, kinds, run_prefill(S, ks, ke, False), run_prefill(S, ks, ke, True))

# decode: 12 rows = 4 requests x next_n 3; seq_lens 1D (the kernel derives each row's length)
next_n, n = 3, 8192
dk = ["T", "T", "T", "U", "U", "U", "M", "M", "M", "W", "W", "W"]
seq_lens = torch.tensor([n, n, K // 2 + next_n - 1, 2 * K + next_n - 1], dtype=torch.int32)
Sd = torch.empty(12, n)
for r in range(12):
    kind = dk[r]
    Sd[r] = distinct_scores(n) - 2.0
    if kind == "T":
        p = torch.randperm(n - 2, generator=g)
        Sd[r, p[: K - 10]] = 10.0 + distinct_scores(K - 10); Sd[r, p[K - 10: K + 30]] = 5.0
    elif kind == "U":
        Sd[r] = distinct_scores(n)
    elif kind == "M":
        Sd[r, K // 2 + 2:] = 100.0                              # garbage past every M row's end (ends K/2..K/2+2)
    elif kind == "W":
        Sd[r] = 3.0
        Sd[r, 2 * K + 2:] = 100.0                              # W rows end at 2K..2K+2
ked = torch.stack([m._glm5_decode_row_end(seq_lens, next_n, 12)]).view(-1).to(torch.int32).cpu()
ksd = torch.zeros(12, dtype=torch.int32)
okd = check("decode", Sd, ksd, ked, dk, run_decode(Sd, seq_lens, next_n, False), run_decode(Sd, seq_lens, next_n, True))
ok &= okd

# cost of the knob alone, per call
for name, rows, n in (("decode c=1 MTP2", 3, 8192), ("decode c=8 MTP2", 24, 8192),
                      ("prefill 512 x 2656 (10.6k ctx)", 512, 2656), ("prefill 512 x 8192 (32k ctx)", 512, 8192)):
    L = torch.rand(rows, n, device=DEV)
    t = torch.empty(rows, K, dtype=torch.int32, device=DEV)
    rs = torch.zeros(rows, dtype=torch.int32, device=DEV); re_ = torch.full((rows,), n, dtype=torch.int32, device=DEV)
    torch.ops._C.top_k_per_row_prefill(L, rs, re_, t, rows, L.stride(0), L.stride(1), K)
    t0 = t.clone()
    hip = cost(lambda: torch.ops._C.top_k_per_row_prefill(L, rs, re_, t, rows, L.stride(0), L.stride(1), K))
    knob = cost(lambda: (t.copy_(t0), m._glm5_topk_ties(t, L, rs, re_, K)))
    # captured (how decode runs): the same knob replayed from a graph, and checked against the eager result
    t.copy_(t0); m._glm5_topk_ties(t, L, rs, re_, K); eager_out = t.clone()
    t.copy_(t0)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        m._glm5_topk_ties(t, L, rs, re_, K)                  # warm-up on the side stream
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        t.copy_(t0)
        m._glm5_topk_ties(t, L, rs, re_, K)
    gr.replay(); torch.cuda.synchronize()
    cap_ok = torch.equal(t, eager_out)
    graph = cost(gr.replay)
    print(f"cost {name}: HIP top-k {hip * 1000:.0f} us | knob eager {knob * 1000:.0f} us | knob captured "
          f"{graph * 1000:.0f} us (replay == eager {cap_ok}) | x11 layers eager {11 * knob:.2f} ms, captured "
          f"{11 * graph:.2f} ms")
    ok &= cap_ok
print(f"PASS GLM5_TOPK_TIES=stable: {ok}")
