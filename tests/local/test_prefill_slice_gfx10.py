"""Sliced prefill scoring (fp8_mqa_logits_torch, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB slices) vs the original one-shot
function (pre-patch file), one card. Bitwise equality at 2048 rows x {10k, 30k} contexts (2560 / 7500 pools), both
scoring modes (GLM5_INDEXER_PREFILL_F16=1 default, and 0); peak memory of the call at 512 / 1024 / 2048 rows, 30k."""
import importlib.util, os
import torch
import vllm.models.glm5next  # noqa: F401
from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch as sliced
spec = importlib.util.spec_from_file_location("old_rams", "/old/rams_before_slice.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

DEV, H, D = "cuda:0", 32, 128
F8 = current_platform.fp8_dtype()
g = torch.Generator().manual_seed(0)


def inputs(M, N):
    q = (torch.randn(M, H, D, generator=g) * 1.5).to(F8).to(DEV)
    k = (torch.randn(N, D, generator=g) * 1.5).to(F8).to(DEV)
    sc = (torch.rand(N, 1, generator=g) * 0.01 + 0.001).to(DEV)
    w = (torch.rand(M, H, generator=g) * 0.1).to(DEV)
    ks = torch.zeros(M, dtype=torch.int32, device=DEV)
    ke = torch.clamp(torch.arange(M, device=DEV, dtype=torch.int32) + N - M + 1, min=1, max=N)
    return q, (k, sc), w, ks, ke


ok = True
for f16 in ("1", "0"):
    os.environ["GLM5_INDEXER_PREFILL_F16"] = f16
    for N in (2560, 7500):
        args = inputs(2048, N)
        a = old.fp8_mqa_logits_torch(*args)
        b = sliced(*args)
        rows = max(1, min(2048, 512 * 1024 * 1024 // (H * N * 4)))
        eq = torch.equal(a, b)
        d = float((a - b)[torch.isfinite(a)].abs().max()) if not eq else 0.0
        ok &= eq
        print(f"f16={f16} 2048 x {N} pools ({N * 4 // 1000}k ctx): slice rows {rows}, bitwise equal {eq}"
              + ("" if eq else f", max |diff| {d:.3e}"))
os.environ["GLM5_INDEXER_PREFILL_F16"] = "1"
N = 7500
for M in (512, 1024, 2048):
    args = inputs(M, N)
    res = {}
    for name, fn in (("one-shot (old)", old.fp8_mqa_logits_torch), ("sliced", sliced)):
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        try:
            out = fn(*args); torch.cuda.synchronize()
            res[name] = (torch.cuda.max_memory_allocated() - base) / 2**30
            del out
        except torch.OutOfMemoryError:
            res[name] = float("nan")
    print(f"peak scoring memory, {M} rows x 30k ctx: " + ", ".join(f"{k} {v:.2f} GiB" for k, v in res.items()))
print(f"PASS sliced prefill scoring bitwise: {ok}")
