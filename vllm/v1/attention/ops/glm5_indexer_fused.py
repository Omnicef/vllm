# SPDX-License-Identifier: Apache-2.0
"""Fused paged indexer decode logits for gfx1030 (GLM5_INDEXER_KERNEL=fused, local, NOT FOR UPSTREAM).

Replaces the capture-safe torch loop (_fp8_paged_mqa_logits_decode_torch: per 8-page chunk a gather, the 16x16 tile
de-shuffle copy, fp8->fp32, an fp32 bmm, relu, x head weights, sum over heads, x scales, mask, slice copy, with a
trip count fixed by max_model_len) by one Triton kernel:

  grid (rows, cdiv(width, BLOCK_P)), width = min(max_model_len, table_width * block_size). Program (b, i) reads the
  row's length from the device and returns at once if its block starts past it, so the work follows the context,
  not max_model_len; the grid itself is fixed (capture-safe). Otherwise, for its BLOCK_P positions: page ids from the
  block table (clamped as the torch path does), fp8 values loaded straight from the tiled page layout (the
  de-shuffle is address arithmetic), fp8 -> fp32, q.k over the head dim with fp32 accumulation for all heads
  (tl.dot, ieee fp32), relu, x head weights, sum over heads, x the per-position scale; positions past the length
  stay -inf (the output is pre-filled). Each score is produced by exactly one program: fixed reduction order.

Same output as the torch path: [rows, max_model_len] fp32, -inf outside [0, length). The HIP top-k reads it as today.
"""
import os

import torch

from vllm.triton_utils import tl, triton

_BLOCK_P = 64


@triton.jit
def _paged_logits_kernel(q_ptr, kv_u8_ptr, kv_f32_ptr, w_ptr, lens_ptr, bt_ptr, out_ptr,
                         num_blocks, bt_stride, out_stride, width,
                         H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, PAGE_BYTES: tl.constexpr,
                         DESHUFFLE: tl.constexpr, FP8_FNUZ: tl.constexpr, BLOCK_P: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    p0 = tl.program_id(1) * BLOCK_P
    limit = tl.load(lens_ptr + b)
    if p0 >= limit:
        return
    p = p0 + tl.arange(0, BLOCK_P)
    valid = (p < limit) & (p < width)
    page = tl.minimum(p // BS, (width - 1) // BS)                        # stay inside the table row
    blk = tl.load(bt_ptr + b * bt_stride + page, mask=valid, other=0).to(tl.int64)
    blk = tl.minimum(tl.maximum(blk, 0), num_blocks - 1)                  # as the torch path's clamp
    slot = p % BS
    d = tl.arange(0, D)
    if DESHUFFLE:   # writer layout: [BS/16, D/16, 16 (slot), 16 (dim)] per page
        off = ((slot[:, None] // 16) * (D // 16) + d[None, :] // 16) * 256 + (slot[:, None] % 16) * 16 + d[None, :] % 16
    else:
        off = slot[:, None] * D + d[None, :]
    raw = tl.load(kv_u8_ptr + blk[:, None] * PAGE_BYTES + off, mask=valid[:, None], other=0)
    if FP8_FNUZ:
        v = raw.to(tl.float8e4b8, bitcast=True).to(tl.float32)
    else:
        v = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scale = tl.load(kv_f32_ptr + blk * (PAGE_BYTES // 4) + (BS * D) // 4 + slot, mask=valid, other=0.0)
    h = tl.arange(0, H)
    qt = tl.load(q_ptr + b * H * D + h[None, :] * D + d[:, None])        # [D, H] fp32
    s = tl.dot(v, qt, input_precision="ieee")                            # [BLOCK_P, H] fp32
    w = tl.load(w_ptr + b * H + h)
    logit = tl.sum(tl.maximum(s, 0.0) * w[None, :], axis=1) * scale
    tl.store(out_ptr + b * out_stride + p, logit, mask=valid)


def fused_paged_mqa_logits(q: torch.Tensor, kv_cache: torch.Tensor, weights: torch.Tensor,
                           context_lens: torch.Tensor, block_tables: torch.Tensor,
                           max_model_len: int) -> torch.Tensor:
    """Same contract as _fp8_paged_mqa_logits_decode_torch: q [B, 1, H, D] fp8, kv_cache [num_blocks, block_size,
    1, D + 4] uint8 (values then fp32 scales per page), weights [>= B, H] fp32, context_lens [B] (or [B, 1]),
    block_tables [B, W] int32 -> [B, max_model_len] fp32, -inf outside [0, context_len)."""
    from vllm.platforms import current_platform

    batch, _, heads, dim = q.shape
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    page_bytes = block_size * dim + block_size * 4
    if context_lens.dim() > 1:
        context_lens = context_lens.squeeze(-1)
    lens = context_lens.to(device=q.device, dtype=torch.int32).contiguous()
    out = torch.full((batch, max_model_len), float("-inf"), device=q.device, dtype=torch.float32)
    width = min(max_model_len, block_tables.shape[1] * block_size)
    kv_flat = kv_cache.view(num_blocks, page_bytes)
    qf = q[:, 0].to(torch.float32).contiguous()
    w = weights[:batch].to(torch.float32).contiguous()
    bt = block_tables.to(torch.int32)
    deshuffle = block_size > 1 and os.environ.get("GLM5_INDEXER_DESHUFFLE") == "1"
    grid = (batch, triton.cdiv(width, _BLOCK_P))
    _paged_logits_kernel[grid](
        qf, kv_flat, kv_flat.view(torch.float32), w, lens, bt, out,
        num_blocks, bt.stride(0), out.stride(0), width,
        H=heads, D=dim, BS=block_size, PAGE_BYTES=page_bytes, DESHUFFLE=deshuffle,
        FP8_FNUZ=current_platform.is_fp8_fnuz(), BLOCK_P=_BLOCK_P, num_warps=4)
    return out


def fused_supported(q: torch.Tensor, kv_cache: torch.Tensor) -> bool:
    heads, dim, bs = q.shape[2], q.shape[3], kv_cache.shape[1]
    return (q.shape[1] == 1 and kv_cache.dtype == torch.uint8 and heads & (heads - 1) == 0 and heads >= 16
            and dim & (dim - 1) == 0 and dim >= 16 and (bs == 1 or bs % 16 == 0) and (bs * dim) % 4 == 0
            and bs <= _BLOCK_P * 1024)
