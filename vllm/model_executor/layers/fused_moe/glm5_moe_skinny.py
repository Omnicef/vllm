# SPDX-License-Identifier: Apache-2.0
"""GLM5_MOE_SKINNY=1 (local, NOT FOR UPSTREAM): W4A16 MoE decode skinny GEMV pair for <= 16 tokens on gfx1030.

The HIP source (csrc_glm5/glm5_moe_skinny.cu) is built once as a torch extension into
$GLM5_EXT_DIR (default /root/.cache/vllm/torch_ext, persistent on sp) and cached there; every later process only
loads the .so. Serving calls it from TritonWNA16Experts.apply for fp16, int4 group-quantized, symmetric experts
without EP; everything else (more tokens, other quant) stays on the Triton path.
"""
import os
import threading

import torch

_LOCK = threading.Lock()
_LOADED = [False]
MAX_TOKENS = 16


def _load() -> None:
    if _LOADED[0]:
        return
    with _LOCK:
        if _LOADED[0]:
            return
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc_glm5", "glm5_moe_skinny.cu")
        build = os.environ.get("GLM5_EXT_DIR", "/root/.cache/vllm/torch_ext")
        os.makedirs(build, exist_ok=True)
        load(name="glm5_moe_skinny", sources=[src], build_directory=build, is_python_module=False,
             extra_cuda_cflags=["-O3"], verbose=False)
        _LOADED[0] = True


def enabled() -> bool:
    return os.environ.get("GLM5_MOE_SKINNY") == "1"


def usable(hidden_states: torch.Tensor, quant_config, expert_map, apply_router_weight_on_input: bool) -> bool:
    return (enabled() and hidden_states.dtype == torch.float16 and 1 <= hidden_states.shape[0] <= MAX_TOKENS
            and getattr(quant_config, "use_int4_w4a16", False) and quant_config.w1_zp is None
            and quant_config.block_shape is not None and expert_map is None and not apply_router_weight_on_input)


def moe_skinny(hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, group_size: int,
               swiglu_limit: float | None, out: torch.Tensor | None = None) -> torch.Tensor:
    """Routed-expert output [M, K] fp16 (top-k weighted sum), same contract as the Triton WNA16 experts."""
    _load()
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    N = w1.shape[1] // 2
    act = torch.empty((M, topk, N), dtype=torch.float16, device=hidden_states.device)
    if out is None:
        out = torch.empty((M, K), dtype=torch.float16, device=hidden_states.device)
    torch.ops.glm5_skinny.moe_skinny_int4_decode(
        hidden_states.contiguous(), w1, w1_scale, w2, w2_scale, topk_weights.contiguous(), topk_ids.contiguous(),
        act, out, int(group_size), float(swiglu_limit or 0.0))
    return out
