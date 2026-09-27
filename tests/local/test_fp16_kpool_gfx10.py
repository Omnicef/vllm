#!/usr/bin/env python3
"""GLM5_FP16_KPOOL=1 on one card: does feeding the ROCm kpool kernels fp16 keys / gate scores change the pooled keys by
more than bf16 rounding itself does?

Same fp32 source for both paths. Baseline (a): bf16 path vs bf16 path with the source jittered below one bf16 ulp
(two equally valid bf16 roundings) -- how much the output moves under bf16 rounding alone. Test (b): fp16 path vs bf16
path. Metrics on the pooled keys: relative RMS difference of the dequantized values, share of identical fp8 bytes,
share of pools with equal scale. Pass: (b) no worse than 1.25x (a) in RMS, and identical-byte / equal-scale shares
no lower than (a) minus 2 points. Paths (real kernels): prefill = kpool_compress_and_write_cache; decode = tail seed
+ kpool_decode_update_and_maybe_write_cache_batched completing pools one token at a time.

Query path (fwht128_quant_fp8, the indexer query FWHT + fp8 quant): fp16 input x16 must give fp8 values and scales
bitwise equal to the bf16 path on x16.to(bfloat16), and fp16 must still be rejected with GLM5_FP16_KPOOL unset.
"""
import os
os.environ["GLM5_FP16_KPOOL"] = "1"
import torch
import vllm.models.glm5next  # noqa: F401
from vllm.platforms import current_platform
from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8
from vllm.models.glm5next.amd.ops.kpool_compress import (
    kpool_compress_and_write_cache,
    kpool_seed_tail_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
)

DEV, HD, KP, PG, FP8 = "cuda:0", 128, 4, 32, current_platform.fp8_dtype()
g = torch.Generator().manual_seed(0)
P = 4096
src_k = torch.randn(P * KP, HD, generator=g)
src_s = torch.randn(P * KP, HD, generator=g)
# sub-ulp jitter: at most a quarter of a bf16 ulp, so bf16(src) and bf16(src + jitter) are two valid roundings
jit_k = src_k + (torch.rand(src_k.shape, generator=g) - 0.5) * src_k.abs() * 2 ** -9
jit_s = src_s + (torch.rand(src_s.shape, generator=g) - 0.5) * src_s.abs() * 2 ** -9
ape = (0.1 * torch.randn(KP, HD, generator=g)).to(DEV)


def prefill(k, s, dt):
    v, sc = kpool_compress_and_write_cache(
        torch.zeros(1, PG, HD + 4, dtype=torch.uint8, device=DEV),
        k.to(DEV, dt).view(P, KP, HD), s.to(DEV, dt).view(P, KP, HD), ape,
        torch.zeros(P, dtype=torch.int64, device=DEV), pool_size=KP, head_dim=HD, round_scale=True,
        return_compressed=True, write_cache=False)
    return v, sc


def decode(k, s, dt, n_tok=66, prompt=6):
    k = k[:n_tok].to(DEV, dt); s = s[:n_tok].to(DEV, dt)
    tail = torch.zeros(2, 2, KP, HD, dtype=torch.bfloat16, device=DEV)
    cache = torch.zeros(2, 64, HD + 4, dtype=torch.uint8, device=DEV)
    kpool_compress_and_write_cache(cache, k[:4].view(1, KP, HD), s[:4].view(1, KP, HD), ape,
                                   torch.zeros(1, dtype=torch.int64, device=DEV), pool_size=KP, head_dim=HD,
                                   round_scale=True)
    tslots = torch.full((prompt,), -1, dtype=torch.int64, device=DEV)
    tslots[4:] = torch.arange(4, prompt, device=DEV) % KP
    kpool_seed_tail_cache(tail, k[:prompt], s[:prompt], tslots, KP, HD)
    for t in range(prompt, n_tok):
        pos = torch.tensor([[t]], dtype=torch.int32, device=DEV)
        slot = torch.tensor([[t // KP if t % KP == KP - 1 else -1]], dtype=torch.int32, device=DEV)
        kpool_decode_update_and_maybe_write_cache_batched(
            cache, tail, pos % KP, k[t].view(1, 1, HD), s[t].view(1, 1, HD), ape, slot, pos, KP, HD,
            round_scale=True)
    torch.cuda.synchronize()
    n_pools = n_tok // KP
    flat = cache[0].reshape(-1)
    v = flat[: 64 * HD].view(FP8).view(64, HD)[:n_pools]            # page-major values (tile order irrelevant here)
    sc = flat[64 * HD: 64 * HD + 64 * 4].view(torch.float32)[:n_pools]
    return v, sc


def metrics(a, b):
    (va, sa), (vb, sb) = a, b
    da, db = va.float() * sa[:, None], vb.float() * sb[:, None]
    rms = float((da - db).norm() / db.norm())
    return rms, float((va.view(torch.uint8) == vb.view(torch.uint8)).float().mean()), float((sa == sb).float().mean())


ok = True
for name, fn in (("prefill", prefill), ("decode", decode)):
    base = metrics(fn(jit_k, jit_s, torch.bfloat16), fn(src_k, src_s, torch.bfloat16))
    test = metrics(fn(src_k, src_s, torch.float16), fn(src_k, src_s, torch.bfloat16))
    good = test[0] <= 1.25 * base[0] and test[1] >= base[1] - 0.02 and test[2] >= base[2] - 0.02
    ok &= good
    print(f"{name}: (a) bf16 vs bf16-jitter rms {base[0]:.2e} bytes {base[1]:.3%} scales {base[2]:.3%} | "
          f"(b) fp16 vs bf16 rms {test[0]:.2e} bytes {test[1]:.3%} scales {test[2]:.3%} -> {'ok' if good else 'WORSE'}")
# query path: bitwise, over normal-range values plus fp16 extremes (large, tiny, subnormal, zero rows)
x16 = (torch.randn(4096 * 32, 128, generator=g) * torch.logspace(-4, 2, 4096 * 32)[:, None]).to(torch.float16)
x16[0] = 0; x16[1] = 60000; x16[2] = 6e-8
x16 = x16.to(DEV)
qa, sa = fwht128_quant_fp8(x16)
qb, sb = fwht128_quant_fp8(x16.to(torch.bfloat16))
q_ok = torch.equal(qa.view(torch.uint8), qb.view(torch.uint8)) and torch.equal(sa, sb)
print(f"query: fp16 vs bf16(x16) fp8 bytes equal {torch.equal(qa.view(torch.uint8), qb.view(torch.uint8))}, "
      f"scales equal {torch.equal(sa, sb)} -> {'ok' if q_ok else 'DIFFERENT'}")
os.environ.pop("GLM5_FP16_KPOOL")
try:
    fwht128_quant_fp8(x16)
    guard_ok = False
except AssertionError:
    guard_ok = True
os.environ["GLM5_FP16_KPOOL"] = "1"
print(f"query: fp16 rejected with GLM5_FP16_KPOOL unset: {guard_ok}")
ok &= q_ok and guard_ok
print(f"PASS fp16 kpool within bf16 rounding, query path bitwise: {ok}")
