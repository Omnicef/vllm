# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

from typing import TYPE_CHECKING

import re

import torch

from vllm.triton_utils import tl, triton

import vllm.envs as envs
from vllm import _custom_ops  # noqa: F401  # registers the torch.ops._C kernels
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.models.glm5next.amd.ops import kpool_compress as kpool_ops
from vllm.models.glm5next.common.sparse_indexer import (
    RADIX_TOPK_WORKSPACE_SIZE,
    _build_decode_scatter_indices,
    _decode_topk_seq_lens,
    _fill_causal_indices,
    _fill_short_decode_causal_indices,
    _gather_workspace_shapes,
    _scatter_decode_tokens_by_request,
    kv_cache_as_quant_view,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

# kpool write helper: form pools from the current token batch and compress them
# into the index K cache via the fused Triton kernel.


def _kpool_compress_insert(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Pool ``kpool`` consecutive tokens into one fp8 K and write at pool slots.

    ``slot_mapping`` is pool-granular (compress_ratio == kpool on the spec):
    only the *last* token of each complete pool carries a valid (>=0) slot;
    intra-pool tokens are -1. Every position is treated as a pool-completion
    candidate and non-completions are masked off inside the kernel. Compacting
    the valid rows first costs two device syncs on the eager prefill path and
    buys nothing numerically. Assumes pool-aligned chunk starts.
    """
    n = slot_mapping.shape[0]
    # No pool can complete in a batch smaller than one pool; also keeps the
    # clamped gather indices below in bounds.
    if n < kpool:
        return
    pos = torch.arange(n, device=k.device)
    valid = slot_mapping >= 0
    # Drop pools whose start falls before the batch (leading padding); their
    # gate/k data is undefined anyway.
    write_mask = valid & (pos >= kpool - 1)
    offs = torch.arange(kpool, device=k.device)
    idx = (pos - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
    kpool_ops.kpool_compress_and_write_cache(
        kv_cache,
        k[idx],  # [n, kpool, head_dim]
        gate_score[idx],
        ape,
        slot_mapping.to(torch.int64),
        pool_size=kpool,
        head_dim=head_dim,
        write_mask=write_mask,
        round_scale=round_scale,
        write_cache=True,
        return_compressed=False,
    )


_GLM5_DSA_RUN = [0]

def _glm5_dsa_on(prefix) -> bool:
    import os as _os

    want = _os.environ.get("GLM5_TRACE_DSA")
    if not want:
        return False
    # "<idx>", "<idx>,<idx>,..." or "all" (every indexer layer; recall analysis 2026-09-25)
    if want != "all" and not any((".layers.%s." % w) in str(prefix) for w in want.split(",")):
        return False
    try:
        import torch.distributed as _dist

        if _dist.is_initialized() and _dist.get_rank() != 0:
            return False
    except Exception:
        pass
    return True

def _glm5_dsa_sha(t):
    import hashlib

    if t is None:
        return "none"
    x = t.detach().contiguous().cpu()
    if x.dtype.itemsize == 1:
        x = x.view(torch.uint8)
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]

def _glm5_dsa_dump(tag, d, fields):
    n = _GLM5_DSA_RUN[0]
    torch.save(d, "/root/.cache/vllm/dsa-%d-%s.pt" % (n, tag))
    with open("/root/.cache/vllm/dsa-trace.log", "a") as f:
        for k in fields:
            v = d[k]
            h = _glm5_dsa_sha(v) if torch.is_tensor(v) else str(v)
            f.write("run=%d,tag=%s,tokens=%s,field=%s,h=%s\n"
                    % (n, tag, d.get("tokens"), k, h))


# local diagnostics (NOT FOR UPSTREAM): GLM5_DSA_GPUHASH=1 -- for every eager (prefill) forward and every
# sparse-attention layer, device-side hashes of (A) the indexer scores before top-k (valid [ks, ke) columns only),
# (B) the selected pools after top-k + GLM5_SORT_TOPK, (C) the sparse-MLA output, and the cutoff-tie counts, added
# into one preallocated device buffer: no host sync inside the forward. The forward slot is a host counter bumped by
# Glm5NextModel.forward (eager only; graph replays never run it). The buffer reaches the host only through the
# worker method glm5_gpuhash_dump (POST /collective_rpc between requests), which appends it to
# /root/.cache/vllm/gh-r<rank>.jsonl and resets it.
# Columns: 0 A, 1 B, 2 C, 3 rows whose k-th score is tied across the cutoff, 4 tied candidates left out,
#          5 scored indexer calls, 6 attention tokens,
#          7 decoder-layer input x (the previous layer's all-reduced sublayer output), 8 residual streams in,
#          9 indexer input hidden state (every layer; rank-agreement check, 2026-09-27),
#          10 qr (indexer q-LoRA input), 11 q after wq_b, 12 kw (wk_weights_proj output), 13 raw prefill q.k GEMM
#          output (before scale / relu / weights; summed over row slices) (2026-09-28).
# Columns 7-13 are on when GLM5_GH_RANK=1 at launch or after POST /collective_rpc {"method": "glm5_gh_rank_set",
# "args": ["1"]} (and off again with "0"), so one request can be hashed without touching the timing tests.
# Columns 7-9 need GLM5_GH_RANK=1. GLM5_GH_SAVE=<fwd>:<layer> also keeps that forward/layer's x, residual and indexer input on the device; the dump
# writes them to /root/.cache/vllm/gh-save-r<rank>-q<n>.pt (offline max |rank r - rank 0|).
_GH = {"on": None, "buf": None, "fwd": -1, "save": {}, "q": 0, "rank": None, "cur": ""}
_GH_FWD, _GH_LAYERS, _GH_COLS = 64, 96, 14


def glm5_gh_on() -> bool:
    if _GH["on"] is None:
        import os as _os

        _GH["on"] = _os.environ.get("GLM5_DSA_GPUHASH") == "1"
        if _GH["on"]:
            from vllm.v1.worker.gpu_worker import Worker

            Worker.glm5_gpuhash_dump = lambda self: glm5_gh_dump()
            # POST /collective_rpc {"method": "glm5_gpuhash_set", "args": ["0"|"1"]}: pause / resume hashing
            # (e.g. around timing measurements in the same launch)
            Worker.glm5_gpuhash_set = lambda self, on: _GH.__setitem__("active", on == "1")
            Worker.glm5_gh_rank_set = lambda self, on: _GH.__setitem__("rank", on == "1")
    return _GH["on"] and _GH.get("active", True) and not torch.cuda.is_current_stream_capturing()


def glm5_gh_forward() -> None:
    if glm5_gh_on():
        _GH["fwd"] += 1


def _gh_slot(prefix, device):
    m = re.search(r"layers\.(\d+)\.", str(prefix))
    f = _GH["fwd"]
    if m is None or not 0 <= f < _GH_FWD or int(m.group(1)) >= _GH_LAYERS:
        return None
    if _GH["buf"] is None:
        _GH["buf"] = torch.zeros(_GH_FWD, _GH_LAYERS, _GH_COLS, dtype=torch.int64, device=device)
    return _GH["buf"][f, int(m.group(1))]


def _gh_hash(t):
    x = t.detach().contiguous().view(-1)
    x = x.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[x.dtype.itemsize]).to(torch.int64)
    w = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.int64) * 2654435761 % 2147483647
    return (x * w).sum()   # wraps in int64: deterministic, order-sensitive


def glm5_gh_add(prefix, col, t) -> None:
    if not glm5_gh_on():
        return
    if col >= 7:
        if _GH["rank"] is None:
            import os as _os

            _GH["rank"] = _os.environ.get("GLM5_GH_RANK") == "1"
        if not _GH["rank"]:                             # rank-agreement columns only on request
            return
    s = _gh_slot(prefix, t.device)
    if s is not None:
        s[col] += _gh_hash(t)
        if col == 2:
            s[6] += t.shape[0]
        if col >= 7:
            import os as _os

            want = _os.environ.get("GLM5_GH_SAVE", "")
            m = re.search(r"layers\.(\d+)\.", str(prefix))
            if want and m and want == "%d:%s" % (_GH["fwd"], m.group(1)):
                _GH["save"][col] = t.detach().clone()      # device copy, no host sync


def glm5_gh_scores(prefix, logits, ks, ke, k) -> None:
    """(A) hash of the valid scores, and the cutoff ties at the k-th score of each row."""
    if not glm5_gh_on():
        return
    s = _gh_slot(prefix, logits.device)
    if s is None:
        return
    cols = torch.arange(logits.shape[1], device=logits.device)
    valid = (cols[None, :] >= ks[:, None].long()) & (cols[None, :] < ke[:, None].long())
    s[0] += _gh_hash(torch.where(valid, logits, torch.zeros((), dtype=logits.dtype, device=logits.device)))
    s[5] += 1
    if logits.shape[1] <= k:
        return
    sc = torch.where(valid, logits.float(), float("-inf"))
    kth = sc.topk(k, dim=-1).values[:, -1:]
    above = (sc > kth).sum(-1)
    tied = ((sc == kth) & valid).sum(-1)
    free = k - above                               # slots left for the tied group
    cross = (valid.sum(-1) > k) & (tied > free)    # the tie straddles the cutoff
    s[3] += cross.sum()
    s[4] += torch.where(cross, tied - free, 0).sum()


def glm5_gh_dump() -> int:
    import json

    n = _GH["fwd"] + 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    rows = _GH["buf"][: max(n, 0)].cpu().tolist() if _GH["buf"] is not None else []
    with open("/root/.cache/vllm/gh-r%d.jsonl" % rank, "a") as f:
        f.write(json.dumps({"forwards": n, "buf": rows}) + "\n")
    if _GH["save"]:
        torch.save({c: v.cpu() for c, v in _GH["save"].items()},
                   "/root/.cache/vllm/gh-save-r%d-q%d.pt" % (rank, _GH["q"]))
        _GH["save"] = {}
    _GH["q"] += 1
    if _GH["buf"] is not None:
        with torch.inference_mode():   # the buffer is born in a forward (an inference tensor)
            _GH["buf"].zero_()
    _GH["fwd"] = -1
    return n


# local diagnostics (NOT FOR UPSTREAM): GLM5_TOPK_TRACE=<dir> records every indexer layer's final top-k
# indices. Prefill (eager): rank 0 appends one sha per (request, chunk, layer) to <dir>/topk-trace.log.
# Decode (captured in FULL_DECODE_ONLY graphs): each layer copies up to _TT_ROWS rows into a fixed device ring
# indexed by a device-side step counter (capture-safe); the ring is hashed per (step, layer) and cleared at the
# next request's first prefill chunk. A request starts where the prefill's first position is 0.
_TT_STEPS, _TT_ROWS, _TT_LAYERS = 320, 3, 64
_TT = {"ring": None, "ctr": None, "slot": {}, "req": 0, "chunk": 0}


def _glm5_topk_trace(prefix, buf, n, positions, is_prefill) -> None:
    import os as _os

    d = _os.environ.get("GLM5_TOPK_TRACE")
    if not d or n <= 0:
        return
    try:
        import torch.distributed as _dist

        rank0 = not _dist.is_initialized() or _dist.get_rank() == 0
    except Exception:
        rank0 = True
    capturing = torch.cuda.is_current_stream_capturing()
    slot = _TT["slot"].setdefault(str(prefix), len(_TT["slot"]))
    if slot >= _TT_LAYERS:
        return
    if _TT["ring"] is None:
        if capturing:
            return
        _TT["ring"] = torch.full((_TT_STEPS, _TT_LAYERS, _TT_ROWS, buf.shape[1]), -2,
                                 dtype=buf.dtype, device=buf.device)
        _TT["ctr"] = torch.zeros(1, dtype=torch.int64, device=buf.device)
    ring, ctr = _TT["ring"], _TT["ctr"]
    if not is_prefill:
        if slot == 0:
            ctr.add_(1).remainder_(_TT_STEPS)
        r = min(n, _TT_ROWS)
        ring[:, slot, :r].index_copy_(0, ctr, buf[:r].unsqueeze(0))
        return
    if capturing:
        return
    import hashlib

    new_req = slot == 0 and positions is not None and int(positions[:n].min()) == 0
    if slot == 0:
        if new_req:
            if rank0 and _TT["req"] > 0:
                steps = int(ctr.item())
                h = ring.cpu()
                with open(_os.path.join(d, "topk-trace.log"), "a") as f:
                    for st in range(1, steps + 1):
                        for nm, sl in _TT["slot"].items():
                            x = h[st, sl]
                            f.write("req=%d,stage=decode,step=%d,layer=%s,sha=%s\n" % (
                                _TT["req"], st, nm,
                                hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16]))
            ring.fill_(-2)
            ctr.zero_()
            _TT["req"] += 1
            _TT["chunk"] = 0
        else:
            _TT["chunk"] += 1
    if rank0:
        x = buf[:n].contiguous().cpu()
        with open(_os.path.join(d, "topk-trace.log"), "a") as f:
            f.write("req=%d,stage=prefill,step=%d,layer=%s,pos0=%d,sha=%s\n" % (
                _TT["req"], _TT["chunk"], prefix,
                int(positions[:n].min()) if positions is not None else -1,
                hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16]))


def _glm5_sort_pools(t) -> None:
    """GLM5_SORT_TOPK=1: sort selected pool/token ids per row, invalids at the tail."""
    import os as _os

    if _os.environ.get("GLM5_SORT_TOPK") != "1":
        return
    big = torch.iinfo(t.dtype).max
    srt = torch.where(t < 0, big, t).sort(dim=-1).values
    t.copy_(torch.where(srt == big, -1, srt))


def _glm5_topk_ties(t, logits, row_start, row_end, k) -> None:
    """local (NOT FOR UPSTREAM): GLM5_TOPK_TIES=stable makes the HIP top-k choice deterministic under exact ties at
    the k-th score (the HIP kernels pick a varying member of a tied group; phase 19b). Selection = every valid
    candidate scoring above the k-th score plus the lowest-index candidates tied with it, filling to k. Written back
    in the HIP convention: indices relative to row_start, ascending, -1 tail. Branchless and capture-safe: the same
    ops and shapes on every row; the k-th score is the minimum score of the HIP selection (the HIP set is right up to
    the choice among ties), so rows with fewer valid candidates than k keep exactly their HIP set."""
    import os as _os

    if _os.environ.get("GLM5_TOPK_TIES") != "stable":
        return
    if _os.environ.get("GLM5_TOPK_TIES_BOUND") == "1" and logits.dtype == torch.float32 and logits.stride(1) == 1:
        return _glm5_topk_ties_bounded(t, logits, row_start, row_end, k)
    rows, n = t.shape[0], logits.shape[1]
    cols = torch.arange(n, device=logits.device, dtype=torch.int32)
    rs = row_start[:rows].to(torch.int32).view(-1, 1)
    re_ = row_end[:rows].to(torch.int32).view(-1, 1)
    sc = torch.nan_to_num(logits[:rows].float(), nan=float("-inf"))
    valid = (cols >= rs) & (cols < re_)
    sel0 = t.to(torch.int64)
    g = sc.gather(1, (sel0 + rs).clamp(0, n - 1))
    kth = torch.where(sel0 >= 0, g, float("inf")).amin(dim=-1, keepdim=True)
    above = valid & (sc > kth)
    tied = valid & (sc == kth)
    free = k - above.sum(-1, keepdim=True, dtype=torch.int32)
    sel = above | (tied & (tied.cumsum(-1, dtype=torch.int32) <= free))
    dest = torch.where(sel, sel.cumsum(-1, dtype=torch.int32) - 1, k).to(torch.int64)
    out = torch.full((rows, k + 1), -1, dtype=t.dtype, device=t.device)
    out.scatter_(1, dest, (cols - rs).to(t.dtype))   # unselected all land in the dropped column k
    t.copy_(out[:, :k])


@triton.jit
def _glm5_ties_bounded_kernel(t_ptr, out_ptr, logits_ptr, rs_ptr, re_ptr, t_stride, out_stride, l_stride,
                              K: tl.constexpr, KPOW2: tl.constexpr, BLOCK: tl.constexpr):
    """One program per row: same selection as _glm5_topk_ties, but every loop runs over [row_start, row_end) only,
    read on the device (work follows the row length, not max_model_len). Passes: k-th score = min of the HIP
    selection; count above it; then, in column order, keep every above candidate and the lowest-index tied ones
    up to k, written ascending (relative to row_start); -1 tail."""
    r = tl.program_id(0).to(tl.int64)
    rs = tl.load(rs_ptr + r)
    re = tl.load(re_ptr + r)
    ninf = float("-inf")
    j = tl.arange(0, KPOW2)
    sel = tl.load(t_ptr + r * t_stride + j, mask=j < K, other=-1)
    g = tl.load(logits_ptr + r * l_stride + rs + tl.maximum(sel, 0), mask=(j < K) & (sel >= 0), other=0.0)
    g = tl.where(g != g, ninf, g)
    kth = tl.min(tl.where((j < K) & (sel >= 0), g, float("inf")), axis=0)
    above = tl.zeros((), dtype=tl.int32)
    for c0 in range(rs, re, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        v = tl.load(logits_ptr + r * l_stride + c, mask=c < re, other=ninf)
        v = tl.where(v != v, ninf, v)
        above += tl.sum(((c < re) & (v > kth)).to(tl.int32), axis=0)
    free = K - above
    base = tl.zeros((), dtype=tl.int32)
    tied_seen = tl.zeros((), dtype=tl.int32)
    for c0 in range(rs, re, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        valid = c < re
        v = tl.load(logits_ptr + r * l_stride + c, mask=valid, other=ninf)
        v = tl.where(v != v, ninf, v)
        tied = valid & (v == kth)
        trank = tied_seen + tl.cumsum(tied.to(tl.int32), axis=0)
        keep = (valid & (v > kth)) | (tied & (trank <= free))
        pos = base + tl.cumsum(keep.to(tl.int32), axis=0) - 1
        tl.store(out_ptr + r * out_stride + pos, (c - rs).to(tl.int32), mask=keep)
        base += tl.sum(keep.to(tl.int32), axis=0)
        tied_seen += tl.sum(tied.to(tl.int32), axis=0)
    for k0 in range(0, K, BLOCK):
        kk = k0 + tl.arange(0, BLOCK)
        tl.store(out_ptr + r * out_stride + kk, tl.full((BLOCK,), -1, tl.int32), mask=(kk < K) & (kk >= base))


def _glm5_topk_ties_bounded(t, logits, row_start, row_end, k) -> None:
    """GLM5_TOPK_TIES_BOUND=1: _glm5_topk_ties with loops bounded by each row's length (Triton, capture-safe)."""
    rows = t.shape[0]
    out = torch.empty_like(t)
    rs = row_start[:rows].to(torch.int32).contiguous()
    re_ = row_end[:rows].to(torch.int32).contiguous()
    _glm5_ties_bounded_kernel[(rows,)](t, out, logits, rs, re_, t.stride(0), out.stride(0), logits.stride(0),
                                       K=k, KPOW2=triton.next_power_of_2(k), BLOCK=1024, num_warps=4)
    t.copy_(out)


def _glm5_decode_row_end(seq_lens, next_n, rows):
    """Row lengths exactly as top_k_per_row_decode computes them (csrc/libtorch_stable/sampler.cu)."""
    if seq_lens.dim() == 2:
        return seq_lens.reshape(-1)[:rows].clamp(min=0)
    r = torch.arange(rows, device=seq_lens.device)
    return (seq_lens[r // next_n] - next_n + r % next_n + 1).clamp(min=0)


@eager_break_during_capture
def sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_pool_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    # kpool params (Plan-A: gate is consumed at write time and read back at
    # topk time to softmax-weight the pool).
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    # Paged tail cache (in-progress pool's raw K + gate score), replacing the
    # transient _DECODE_TAIL ring. tail_prefix resolves attn_metadata[tail_prefix]
    # for the tail group's token-granular slot_mapping. None on the dummy/profiling
    # path and when the tail cache is disabled.
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Reserve profiler-visible memory for the worst-case decode logits,
        # whose shape is [B * next_n, max_pool_len]. This profiling branch
        # returns before invoking the logits kernel itself.
        cfg = get_current_vllm_config_or_none()
        worst_decode_tokens = 0
        if cfg is not None:
            sched = cfg.scheduler_config
            num_spec = (
                cfg.speculative_config.num_speculative_tokens
                if cfg.speculative_config is not None
                else 0
            )
            worst_decode_tokens = min(
                sched.max_num_seqs * (num_spec + 1),
                sched.max_num_batched_tokens,
            )
        # float32 logits -> 4 bytes/element; uint8 sentinel so elems == bytes.
        decode_logits_elems = worst_decode_tokens * max_pool_len * 4
        prefill_cap_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        # local (2026-09-28): the ROCm torch prefill scoring holds one score slice (<= cap, sliced in
        # fp8_mqa_logits_torch) AND the chunk's logits (<= cap, by the chunker) at once, plus fp16 copies of the
        # gathered keys and of the chunk's queries; reserve all of it so the KV profile leaves room.
        if current_platform.is_rocm():
            n_q = cfg.scheduler_config.max_num_batched_tokens if cfg is not None else hidden_states.shape[0]
            q_heads = q_quant.shape[-2] if q_quant is not None and q_quant.dim() >= 2 else 64
            cast_bytes = total_seq_lens * head_dim * 2 + n_q * q_heads * head_dim * 2
            prefill_cap_elems = 2 * prefill_cap_elems + cast_bytes
        max_logits_elems = max(decode_logits_elems, prefill_cap_elems)
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return topk_indices_buffer
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        if index_kpool > 1 and gate_score is not None and compress_ape is not None:
            # kpool prefill write: pool kpool consecutive prefill tokens via
            # softmax(gate+ape)-weighted sum -> Hadamard -> fp8 -> pool slots.
            # Decode tokens (the first num_decode_tokens in the batch) cannot be
            # pooled here — their pool's earlier tokens are not in this batch —
            # so they are deferred to the tail-buffer kernel in has_decode.
            # compress_ratio == index_kpool makes slot_mapping pool-granular.
            n_prefill = num_tokens - num_decode_tokens
            if n_prefill > 0:
                # decode tokens are batched first; prefill tokens follow.
                prefill_slice = slice(num_decode_tokens, num_tokens)
                _kpool_compress_insert(
                    k[prefill_slice],
                    gate_score[prefill_slice],
                    compress_ape,
                    kv_cache,
                    slot_mapping[prefill_slice],
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
                # Persist each request's incomplete prefill pool so decode can
                # finish it, including after PD transfer. Tail slots use
                # ``pos % kpool`` within the request's tail block. Processing
                # only the batch's trailing tokens would miss all but the last
                # request in a multi-request prefill.
                if tail_kv_cache is not None and tail_prefix is not None:
                    tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                    if tail_meta is not None:
                        assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                        kpool_ops.kpool_seed_tail_cache(
                            tail_kv_cache,
                            k[prefill_slice],
                            gate_score[prefill_slice],
                            tail_meta.slot_mapping[prefill_slice],
                            index_kpool,
                            head_dim,
                        )
        else:
            # standard: per-token fp8 quant + scatter (all tokens).
            assert scale_fmt is not None
            from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
                indexer_k_quant_and_cache_triton,
            )

            indexer_k_quant_and_cache_triton(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Short sequences select every pool, so skip sparse scoring and fill
        # the top-k buffer with all causal token indices. The index-K cache was
        # already written above.
        n_prefill_sf = num_tokens - num_decode_tokens
        # Host-side short-prefill predicate: max_prefill_seq_len is computed
        # in the metadata builder (exact for prefill rows) and equals
        # positions[prefill_slice].max() + 1, so this replaces a
        # positions.max().item() device sync per layer. -1 (unknown metadata)
        # falls back to the device-side check.
        if prefill_metadata.max_prefill_seq_len >= 0:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and prefill_metadata.max_prefill_seq_len <= topk_tokens
            )
        else:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
                <= topk_tokens
            )
        if short_prefill:
            # short_prefill is only True when positions is not None (above),
            # but narrow explicitly for the indexer below.
            assert positions is not None
            _pos = positions[num_decode_tokens:num_tokens].to(torch.int32)
            _buf = topk_indices_buffer[num_decode_tokens:num_tokens]
            _fill_causal_indices(_buf, _pos)
            if _glm5_dsa_on(k_cache_prefix):
                # No scoring and no top-k happen on this branch: every pool is
                # selected, so the indexer cannot be a source of divergence for
                # prompts this short. Hash what it did produce.
                _GLM5_DSA_RUN[0] += 1
                _glm5_dsa_dump("idx", {
                    "stage": "short_prefill",
                    "prefix": str(k_cache_prefix),
                    "hidden_in": hidden_states[num_decode_tokens:num_tokens].detach().cpu(),
                    "tokens": int(num_tokens - num_decode_tokens),
                    "max_prefill_seq_len": int(
                        prefill_metadata.max_prefill_seq_len),
                    "topk_tokens": int(topk_tokens),
                    "positions": _pos.detach().cpu(),
                    "causal_indices": _buf.detach().cpu(),
                }, ["stage", "max_prefill_seq_len", "topk_tokens", "positions",
                    "causal_indices", "hidden_in"])

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks if not short_prefill else ():
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            from vllm.utils import glm5_prof_events as _pe

            _t = _pe.begin("read_prefill_gather")
            if not chunk.skip_kv_gather:
                from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
                    cp_gather_indexer_k_quant_cache_triton,
                )

                cp_gather_indexer_k_quant_cache_triton(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                    token_to_seq=chunk.token_to_seq,
                )
            _pe.end(_t)

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
                rocm_fp8_mqa_logits,
            )

            assert q_scale_slice is None
            _GH["cur"] = k_cache_prefix         # local: layer for the column-13 hash inside the logits
            logits = rocm_fp8_mqa_logits(
                q_slice_cast,
                (k_quant_cast, k_scale_cast),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            num_rows = logits.shape[0]

            # kpool: logits are pool-granular (compress_ratio == index_kpool),
            # so topk selects pools. We pick topk_tokens // kpool pools then
            # expand each pool back to its kpool constituent tokens.
            select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
            glm5_gh_scores(k_cache_prefix, logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, select_k)
            if index_kpool > 1:
                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=logits.device
                )
                topk_dst = pool_topk
            else:
                topk_dst = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]

            _t = _pe.begin("topk")
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_dst,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                select_k,
            )

            _pe.end(_t)
            _t = _pe.begin("ties")
            _glm5_topk_ties(topk_dst, logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, select_k)
            _pe.end(_t)
            _glm5_raw = (topk_dst.detach().clone()
                         if _glm5_dsa_on(k_cache_prefix) else None)
            _t = _pe.begin("sort_topk")
            _glm5_sort_pools(topk_dst)
            _pe.end(_t)
            glm5_gh_add(k_cache_prefix, 1, topk_dst)

            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:
                    # Fused expand-pools + append-tail into one Triton kernel
                    # (replaces ~25 elementwise ops). seq_len is token-granular
                    # (pos+1); the kernel derives pool_len internally.
                    q_seq = (
                        positions[chunk.token_start : chunk.token_end].to(torch.int32)
                        + 1
                    )
                    expanded = kpool_ops.expand_pools_and_append_tail(
                        pool_ids, q_seq, index_kpool
                    )
                else:
                    valid = pool_ids >= 0
                    expanded = kpool_ops.expand_pools_to_tokens(
                        pool_ids, valid, topk_tokens, index_kpool
                    )
                topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : expanded.shape[-1]
                ] = expanded
                if _glm5_dsa_on(k_cache_prefix):
                    _GLM5_DSA_RUN[0] += 1
                    _big = torch.iinfo(_glm5_raw.dtype).max
                    _srt = torch.where(_glm5_raw < 0, _big, _glm5_raw).sort(
                        dim=-1).values
                    _glm5_dsa_dump("idx", {
                        "stage": "scored_prefill",
                        "prefix": str(k_cache_prefix),
                        "tokens": int(chunk.token_end - chunk.token_start),
                        "select_k": int(select_k),
                        "topk_tokens": int(topk_tokens),
                        "q": q_slice_cast.detach().cpu(),
                        "hidden_in": hidden_states[
                            chunk.token_start : chunk.token_end].detach().cpu(),
                        "k_raw": (k[chunk.token_start : chunk.token_end].detach().cpu()
                                  if k is not None else None),
                        "k_quant": k_quant_cast.detach().cpu(),
                        "k_scale": k_scale_cast.detach().cpu(),
                        "weights": weights[
                            chunk.token_start : chunk.token_end].detach().cpu(),
                        "cu_seqlen_ks": chunk.cu_seqlen_ks.detach().cpu(),
                        "cu_seqlen_ke": chunk.cu_seqlen_ke.detach().cpu(),
                        "logits": logits.detach().cpu(),
                        "pools_raw": _glm5_raw.cpu(),
                        "pools_sorted": torch.where(_srt == _big, -1, _srt).cpu(),
                        "expanded": expanded.detach().cpu(),
                    }, ["stage", "select_k", "hidden_in", "k_raw", "q", "k_quant",
                        "k_scale", "weights",
                        "cu_seqlen_ks", "cu_seqlen_ke", "logits", "pools_raw",
                        "pools_sorted", "expanded"])

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim+4] for writes
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)

        # Update the tail before reading logits; completed pools are compressed
        # into the slot supplied by slot_mapping.
        # Spec verification groups tokens by request and preserves position
        # order so each token is stashed before the next completes its pool.
        # Positions must remain token-granular because the kernel derives the
        # pool phase and tail index from ``pos % kpool``.
        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
        ):
            num_requests = attn_metadata_narrowed.num_decodes
            # Kpool writes must recover the original request grouping after the
            # indexer's flattened decode path. Host metadata avoids a CUDA graph
            # sync when choosing the uniform or padded layout.
            per_req_lens = decode_metadata.per_req_decode_lens
            if per_req_lens is not None:
                use_uniform = (
                    decode_metadata.decode_is_uniform
                    and num_decode_tokens
                    == num_requests * decode_metadata.write_max_decode_len
                )
                group_lens = per_req_lens
                lmax = decode_metadata.write_max_decode_len
            else:
                # Legacy metadata without per-request lens: fall back to the
                # host-side requires_padding flag. Unreached now (per-request
                # lens is always populated for decode), kept defensive.
                use_uniform = not decode_metadata.requires_padding
                group_lens = decode_metadata.decode_lens
                lmax = int(decode_metadata.decode_lens.max().item())
            if not use_uniform:
                # Non-uniform decode_lens (mixed plain-decode + spec-verify, or
                # a variable MTP-verify batch): scatter actual tokens into a
                # padded [B, lmax] layout. int32 tensors can't go through
                # pack_seq_triton (float/uint8 only). The scatter indices are
                # shared by all five scatters below (and the tail slot one).
                scatter_idx = _build_decode_scatter_indices(
                    group_lens, num_requests, num_decode_tokens
                )
                dec_k = _scatter_decode_tokens_by_request(
                    k[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
                )
                dec_gate = _scatter_decode_tokens_by_request(
                    gate_score[:num_decode_tokens],
                    0,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_slot = _scatter_decode_tokens_by_request(
                    slot_mapping[:num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_pos = _scatter_decode_tokens_by_request(
                    positions[:num_decode_tokens].to(torch.int32),
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                next_n = num_decode_tokens // num_requests
                shape2 = (num_requests, next_n)
                dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
                dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
                dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
                dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape2)
            tail_meta = (
                attn_metadata.get(_resolve_layer_name(tail_prefix))
                if tail_prefix is not None
                else None
            )
            # Paged tail cache replaces the transient _DECODE_TAIL ring. Group
            # the tail group's token-granular slot_mapping per-request, mirroring
            # dec_slot / dec_pos, so the kernel gets each request's current-token
            # tail slot (block * kpool + pos % kpool).
            if tail_meta is not None:
                assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            if tail_meta is None or tail_kv_cache is None:
                dec_tail_slot = None
            elif not use_uniform:
                dec_tail_slot = _scatter_decode_tokens_by_request(
                    tail_meta.slot_mapping[:num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                dec_tail_slot = tail_meta.slot_mapping[:num_decode_tokens].view(shape2)
            # The compress kernel writes the raw fp8 cache (not the quant view);
            # pass the underlying kv_cache, not kv_cache_quant_view.
            if dec_tail_slot is not None:
                # Single batched launch over [num_requests, next_n] replaces the
                # per-token sequential loop. The kernel iterates each request's
                # tokens in position order internally, preserving the
                # pool-completion read-after-stash dependency that the loop
                # provided. Inputs are already grouped per request (uniform:
                # view; non-uniform: _scatter_decode_tokens_by_request padded to
                # [B, lmax]) — no per-token .contiguous() copies needed.
                kpool_ops.kpool_decode_update_and_maybe_write_cache_batched(
                    kv_cache_raw,
                    tail_kv_cache,
                    dec_tail_slot,
                    dec_k,
                    dec_gate,
                    compress_ape,
                    dec_slot,
                    dec_pos,
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
        if current_platform.is_cuda_alike() and _fill_short_decode_causal_indices(
            topk_indices_buffer,
            positions,
            num_decode_tokens,
            attn_metadata_narrowed.max_seq_len,
            topk_tokens,
        ):
            return topk_indices_buffer
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # Padding also covers short chunked prefills classified as decode.
            # MXFP4 uses zero-byte padding so padded slots dequantize to zero.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(-1, *weights.shape[1:])
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
            padded_weights = weights[:num_decode_tokens]
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            rocm_fp8_paged_mqa_logits,
        )

        assert padded_q_scale is None
        logits = rocm_fp8_paged_mqa_logits(
            padded_q_quant_cast,
            kv_cache,
            padded_weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_pool_len,
        )
        num_rows = logits.shape[0]
        # kpool: logits are pool-granular -> select topk_tokens//kpool pools,
        # then expand each pool back to its kpool tokens.
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        from vllm.utils import glm5_prof_events as _pe

        _t = _pe.begin("topk")
        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            seq_lens,
            topk_dst,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            select_k,
        )

        _pe.end(_t)
        from vllm.utils import glm5_prof_events as _pe

        _t = _pe.begin("ties")
        _glm5_topk_ties(topk_dst, logits, torch.zeros_like(seq_lens.reshape(-1)[:1]).expand(num_rows),
                        _glm5_decode_row_end(seq_lens, next_n, num_rows), select_k)
        _pe.end(_t)
        _t = _pe.begin("sort_topk")
        _glm5_sort_pools(topk_dst)
        _pe.end(_t)

        # Resolve to token-level indices in the output buffer.
        if index_kpool > 1:
            pool_ids = pool_topk.to(torch.int64)
            n = pool_topk.shape[0]
            # Decode seq_lens are pool-granular; recover token lengths from
            # positions using the padded [B, next_n] row layout when needed.
            if positions is not None:
                dec_seq = _decode_topk_seq_lens(
                    positions,
                    decode_lens,
                    num_decode_tokens,
                    batch_size,
                    next_n,
                    decode_metadata.requires_padding,
                )
            else:
                dec_seq = decode_metadata.seq_lens[:n]
                if dec_seq.ndim == 2:
                    dec_seq = dec_seq[:, -1]
                dec_seq = dec_seq.to(torch.int32)
            out = kpool_ops.expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            # Drop padded query rows introduced by the next_n padding above.
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    _glm5_topk_trace(k_cache_prefix, topk_indices_buffer, hidden_states.shape[0],
                     positions, has_prefill)
    return topk_indices_buffer


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_pool_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_pool_len = max_pool_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if index_kpool <= 1:
            if not rocm_aiter_ops.is_enabled():
                raise RuntimeError(
                    "Sparse attention indexer ROCm path is only supported on AITER. "
                    "Please enable aiter with VLLM_ROCM_USE_AITER=1"
                )
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_pool_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        # local: the kpool path is not aiter-specific (it is also what runs when aiter is enabled and
        # index_kpool > 1). aiter has no gfx1030 entry, so RDNA2 takes this route rather than hard-failing.
        return sparse_attn_indexer_kpool(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_quant,
            None,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_pool_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
        )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_hip(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )
