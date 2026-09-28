"""GLM5_TOPK_TIES_BOUND=1 vs the unbounded tie step, one card. Logits width = MAX_LEN in pools: 8192 (32k tokens)
and 32768 (131k); row lengths 256 / 2560 / 7680 pools (1k / 10k / 30k tokens). Rows: decode 3 and 24 (starts 0),
prefill 512 (some rows start > 0). Scores with planted cutoff ties, fully tied tails and exact-distinct rows; HIP
top-k first, then both tie steps on copies: selection must be bitwise identical. Captured timing of both."""
import os
import torch
import vllm.models.glm5next  # noqa: F401
import vllm.model_executor.layers.sparse_attn_indexer_kpool as m

DEV, K = "cuda:0", 512
os.environ["GLM5_TOPK_TIES"] = "stable"
g = torch.Generator().manual_seed(0)


def make(rows, width, length, prefill):
    L = torch.randn(rows, width, generator=g)
    rs = torch.zeros(rows, dtype=torch.int32); re_ = torch.full((rows,), length, dtype=torch.int32)
    for r in range(rows):
        s0 = int(torch.randint(0, 64, (1,), generator=g)) if prefill and r % 3 == 0 else 0
        rs[r] = s0; re_[r] = min(width, s0 + length - (r % 3))
        a, b = s0, int(re_[r]); n = b - a
        kind = r % 3
        if kind == 0 and n > K:            # planted ties at the cutoff
            p = torch.randperm(n, generator=g)[: K + 30] + a
            L[r, p[: K - 10]] = 10.0 + torch.rand(K - 10, generator=g); L[r, p[K - 10:]] = 5.0
        elif kind == 1:                    # whole valid tail tied
            L[r, a:b] = 3.0
        L[r, b:] = 100.0                   # garbage past the end must never be selected
    return L.to(DEV), rs.to(DEV), re_.to(DEV)


def hip(L, rs, re_):
    t = torch.empty(L.shape[0], K, dtype=torch.int32, device=DEV)
    torch.ops._C.top_k_per_row_prefill(L, rs, re_, t, L.shape[0], L.stride(0), L.stride(1), K)
    return t


def run(L, rs, re_, t0, bound):
    os.environ["GLM5_TOPK_TIES_BOUND"] = "1" if bound else "0"
    t = t0.clone(); m._glm5_topk_ties(t, L, rs, re_, K); return t


def cap(fn, reps=30):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn()
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    gr.replay(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); gr.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2] * 1000


ok = True
for width in (8192, 32768):
    for length in (256, 2560, 7680):
        for rows, prefill in ((3, False), (24, False), (512, True)):
            L, rs, re_ = make(rows, width, length, prefill)
            t0 = hip(L, rs, re_)
            a, b = run(L, rs, re_, t0, False), run(L, rs, re_, t0, True)
            eq = torch.equal(a, b); ok &= eq
            line = f"MAX_LEN {width * 4 // 1024}k ctx {length * 4 // 1000}k rows {rows:3d}: identical {eq}"
            if rows in (3, 512):
                ta = cap(lambda: m._glm5_topk_ties(t0.clone(), L, rs, re_, K) if os.environ.__setitem__("GLM5_TOPK_TIES_BOUND", "0") is None else None)
                tb = cap(lambda: m._glm5_topk_ties(t0.clone(), L, rs, re_, K) if os.environ.__setitem__("GLM5_TOPK_TIES_BOUND", "1") is None else None)
                line += f" | captured unbounded {ta:7.0f} us, bounded {tb:6.0f} us (x11 layers {11 * ta / 1000:5.2f} -> {11 * tb / 1000:5.2f} ms)"
            print(line, flush=True)
print(f"PASS bounded tie step identical: {ok}")
