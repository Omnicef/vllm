# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import ClassVar, Literal

import torch
from torch import nn

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mhc import (
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
    hc_contract,
    hc_expand,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import _get_moe_router_dtype
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from .attention import Glm5NextMLAAttention
from .kda import Glm5NextLinearAttention
from .multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
    Glm5NextVisionTransformer,
)

logger = init_logger(__name__)


# local (GLM5_PROF=1): name each block in torch-profiler traces so kernel time can be
# attributed per block (Phase 1). Forward hooks open/close a record_function around
# the whole layer ("L<i>"), its attention ("L<i>.kda" / "L<i>.mla"), the DSA indexer
# ("L<i>.indexer") and the MLP ("L<i>.moe" / "L<i>.mlp"); mHC is the layer minus its
# children. Eager mode only (a range cannot open inside a captured graph). Off: no hooks.
def _glm5_prof_attach(layer) -> None:
    import os as _os

    if _os.environ.get("GLM5_PROF_EVENTS") == "1":
        # graphs-safe variant: external timing events at the same boundaries
        from vllm.utils import glm5_prof_events as pe

        i = layer.layer_idx
        attn = layer.self_attn
        pe.attach(layer, f"L{i}")
        pe.attach(attn, f"L{i}.kda" if type(attn).__name__ == "Glm5NextLinearAttention" else f"L{i}.mla")
        if getattr(attn, "indexer", None) is not None:
            pe.attach(attn.indexer, f"L{i}.indexer")
        pe.attach(layer.mlp, f"L{i}.moe" if type(layer.mlp).__name__ == "Glm5NextMoE" else f"L{i}.mlp")
        return
    if _os.environ.get("GLM5_PROF") != "1":
        return

    def wrap(mod, name):
        def pre(_m, _args, _kwargs=None):
            rf = torch.profiler.record_function(name)
            rf.__enter__()
            _m._glm5_prof_rf = rf

        def post(_m, _args, _out):
            rf = getattr(_m, "_glm5_prof_rf", None)
            if rf is not None:
                rf.__exit__(None, None, None)
                _m._glm5_prof_rf = None

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

    i = layer.layer_idx
    wrap(layer, f"L{i}")
    attn = layer.self_attn
    wrap(attn, f"L{i}.kda" if type(attn).__name__ == "Glm5NextLinearAttention" else f"L{i}.mla")
    if getattr(attn, "indexer", None) is not None:
        wrap(attn.indexer, f"L{i}.indexer")
    wrap(layer.mlp, f"L{i}.moe" if type(layer.mlp).__name__ == "Glm5NextMoE" else f"L{i}.mlp")



# local: GLM5_TRACE=<decode_step> hashes hidden_states/residual after every decoder
# layer at exactly one decode step, so three runs can be diffed to find the FIRST
# layer at which they disagree. The input hash (layer=input) is taken after the
# embedding/PLE and before layer 0, so a divergence already present there is
# attributed to the input or sampler path rather than to a layer.
# Decode steps are counted as single-token forwards on rank 0, which for a
# single-sequence probe is the same as num_prefills == 0.
_GLM5_DECODE_STEP = [0]
_GLM5_REQ = [0]


def _glm5_sha(t):
    import hashlib
    if t is None:
        return "none"
    x = t.detach().contiguous()
    return hashlib.sha256(x.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]

def _glm5_moe_trace(moe, hidden_states, router_logits, out):
    """local: GLM5_TRACE_MOE=<layer idx> hashes the MoE stages of that layer's
    prefill forward, and GLM5_MOE_PROBE=<n> additionally calls self.experts n
    more times on the SAME inputs and hashes each result - an in-process
    determinism check of the expert path that needs no weight dump. The routing
    is recomputed locally from the logits as a cross-check: identical logits with
    a differing expert output means the non-determinism is below the router.
    """
    import os as _os

    want = _os.environ.get("GLM5_TRACE_MOE")
    if not want:
        return
    if (".layers.%s." % want) not in str(getattr(moe, "_glm5_prefix", "")):
        return
    if hidden_states.shape[0] <= 1:
        return
    # NOTE: self.experts contains collectives, so the repeat calls must run on
    # EVERY rank - a rank-0-only probe deadlocks the other seven (observed:
    # "No available shared memory broadcast block found in 60 seconds", forever).
    # Only the logging is rank-gated.
    try:
        _rk = (torch.distributed.get_rank()
               if torch.distributed.is_initialized() else 0)
    except Exception:
        _rk = 0
    tok = int(hidden_states.shape[0])
    lines = []
    if out is None:
        _GLM5_SUB_RUN[0] += 1
        if _rk == 0:
            lines.append(("hidden_in", _glm5_sha(hidden_states)))
            lines.append(("router_logits", _glm5_sha(router_logits)))
            _tk = torch.topk(router_logits.float(), 8, dim=-1)
            lines.append(("topk_ids_recomputed", _glm5_sha(_tk.indices)))
            lines.append(("topk_vals_recomputed", _glm5_sha(_tk.values)))
    else:
        if _rk == 0:
            lines.append(("moe_out", _glm5_sha(out)))
        _n = int(_os.environ.get("GLM5_MOE_PROBE", "0") or 0)
        for _i in range(_n):
            _again = moe.experts(hidden_states=hidden_states.clone(),
                                 router_logits=router_logits.clone())
            if _rk == 0:
                lines.append(("moe_out_repeat%d" % (_i + 1), _glm5_sha(_again)))
    if _rk != 0:
        return
    with open("/root/.cache/vllm/moe-l%s.log" % want, "a") as f:
        for k, h in lines:
            f.write("run=%d,tokens=%d,stage=%s,h=%s\n"
                    % (_GLM5_SUB_RUN[0], tok, k, h))


_GLM5_SUB_RUN = [0]


def _glm5_sub(layer_idx, tokens, **kw):
    """local: GLM5_TRACE_SUB=<layer idx> hashes the sub-stages of one decoder
    layer's prefill forward, so a divergence introduced inside the layer can be
    attributed to mHC pre / attention / mHC post / MLP rather than to "the
    layer".
    """
    import os as _os

    want = _os.environ.get("GLM5_TRACE_SUB")
    if want is None or tokens <= 1:
        return
    _want = [w for w in want.split(",") if w.strip()]
    if str(layer_idx) not in _want:
        return
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
    except Exception:
        pass
    with open("/root/.cache/vllm/sub-l%s.log" % layer_idx, "a") as f:
        for k, v in kw.items():
            if k == "_new":
                _GLM5_SUB_RUN[0] += 1
                continue
            f.write("run=%d,tokens=%d,stage=%s,h=%s\n"
                    % (_GLM5_SUB_RUN[0], tokens, k, _glm5_sha(v)))


def _glm5_state_sha(layer):
    """Hash the KDA conv + recurrent state slots a layer is about to read.

    The hidden-state trace alone cannot tell "this layer computed
    non-deterministically" from "this layer read a state that an earlier
    forward wrote differently", because a KDA layer's output depends on the
    recurrent state in the mamba cache as well as on its input. Only the slots
    this forward actually uses are hashed, taken from the layer's own
    GDNAttentionMetadata. Returns ("-", "-") for non-KDA layers.
    """
    if getattr(layer, "layer_kind", None) != "kda":
        return ("-", "-")
    try:
        from vllm.forward_context import get_forward_context

        attn = layer.self_attn
        cache = getattr(attn, "kv_cache", None)
        if cache is None or len(cache) != 2:
            return ("nocache", "nocache")
        conv, rec = cache
        md = get_forward_context().attn_metadata
        md = md.get(attn.prefix) if isinstance(md, dict) else None
        idx = getattr(md, "non_spec_state_indices_tensor", None)
        if idx is None:
            return ("nomd", "nomd")
        # slot 0 is NULL_BLOCK_ID
        sel = sorted({int(x) for x in idx.reshape(-1).tolist() if int(x) > 0})
        if not sel:
            return ("noslot", "noslot")
        # The profile run has dummy state indices but no allocated cache yet, and
        # indexing an empty cache with them is an out-of-bounds device read: that
        # faulted the load 5 times ("Memory access fault ... address (nil)")
        # before this guard.
        if conv.numel() == 0 or rec.numel() == 0 or sel[-1] >= min(
            conv.shape[0], rec.shape[0]
        ):
            return ("unalloc", "unalloc")
        return (_glm5_sha(conv[sel]), _glm5_sha(rec[sel]))
    except Exception as e:  # diagnostics must never break the forward
        return ("err:%s" % type(e).__name__, "-")


class Glm5NextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )

        self.swiglu_limit = swiglu_limit
        if self.swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit=self.swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextMoE(nn.Module):
    def __init__(
        self,
        config: Glm5NextConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self._glm5_prefix = prefix
        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        swiglu_limit = config.swiglu_limit
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=swiglu_limit,
            )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=None,
            router_logits_dtype=self.gate.out_dtype,
            swiglu_limit=swiglu_limit,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape

        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # MoERunner holds the gate (passed to FusedMoEFactory) and computes
        # the router logits itself, so nothing is precomputed here (matches
        # DeepseekV2MoE; `router_logits` is a placeholder).
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(num_tokens, hidden_dim)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        is_mtp_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts
        self.is_mtp_layer = is_mtp_layer
        self.mhc = config.mhc
        is_kda_layer = not is_mtp_layer and config.is_kda_layer(layer_idx)
        self.layer_kind = "kda" if is_kda_layer else "mla"
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if is_kda_layer:
            self.self_attn = Glm5NextLinearAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            # MLA layers require the latent head dims, which are guaranteed set
            # on MLA configs; narrow away the `int | None`.
            assert config.v_head_dim is not None
            assert config.kv_lora_rank is not None
            self.self_attn = Glm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                quant_config=None,  # MLA projections are BF16 in checkpoint
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                skip_rope=config.mla_nope,
            )

        # MTP layers sit past the base model's hidden layers (layer_idx >=
        # num_hidden_layers), so they're outside mlp_layer_types; default them
        # to the last base layer's MLP type (sparse/MoE for these checkpoints).
        mlp_layer_types = config.mlp_layer_types
        mlp_type = (
            mlp_layer_types[layer_idx]
            if layer_idx < len(mlp_layer_types)
            else (mlp_layer_types[-1] if mlp_layer_types else "sparse")
        )
        if self.is_moe and self.num_experts is not None and mlp_type == "sparse":
            self.mlp = Glm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Cached for the hot forward path (isinstance per layer per step).
        self._mlp_is_moe = isinstance(self.mlp, Glm5NextMoE)
        # In SP, the attention output projection leaves a partial sum; the
        # decoder-layer reduce_scatter after attention completes it (DSv4 pattern).
        # MTP layers use the non-mHC path which has no sp_reduce_scatter, so
        # their o_proj must still reduce normally.
        if self.is_sequence_parallel and not is_mtp_layer:
            self.self_attn.o_proj.reduce_results = False
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc and not is_mtp_layer:
            # mhc config
            self.mhc_num_residual_streams = config.mhc_num_residual_streams
            self.mhc_tau = config.mhc_tau
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            n = config.mhc_num_residual_streams
            d_model = n * self.hidden_size
            mix_hc = (2 + n) * n

            self.n = n

            # attn hc
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            # ffn hc
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            self.mhc_pre_op = MHCPreOp()
            self.mhc_post_op = MHCPostOp()
            self.mhc_fused_post_pre_op = MHCFusedPostPreOp()

            # local: only warm TileLang kernels that will actually run. On RDNA2
            # (gfx10xx) the TileLang MHC path is disabled (HAS_TILELANG_MHC is
            # False, the torch fallback runs), and compiling these warmups fails
            # with "'function' object has no attribute 'compile'".
            from vllm.model_executor.layers.mhc import HAS_TILELANG_MHC

            if vllm_config.kernel_config.enable_jit_warmup and HAS_TILELANG_MHC:
                from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                    _HC_PRENORM_GEMM_TILELANG_KERNEL,
                    _MHC_FUSED_TILELANG_KERNEL,
                    _MHC_POST_TILELANG_KERNEL,
                    _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
                )
                from vllm.utils.deep_gemm import is_deep_gemm_supported

                include_pre_gemm_splits = is_deep_gemm_supported()
                _MHC_PRE_BIG_FUSE_TILELANG_KERNEL.register_warmup(
                    vllm_config,
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                    use_norm_weight=True,
                    include_pre_gemm_splits=include_pre_gemm_splits,
                    include_broadcast_splits=False,
                    rms_eps=self.rms_norm_eps,
                    hc_pre_eps=self.hc_eps,
                    hc_sinkhorn_eps=self.hc_eps,
                    hc_post_mult_value=self.mhc_post_mult_value,
                    sinkhorn_repeat=self.mhc_sinkhorn_iterations,
                    norm_eps=(
                        self.input_layernorm.variance_epsilon,
                        self.post_attention_layernorm.variance_epsilon,
                    ),
                )
                if not include_pre_gemm_splits:
                    _HC_PRENORM_GEMM_TILELANG_KERNEL.register_warmup(
                        vllm_config,
                        hidden_size=self.hidden_size,
                        hc_mult=self.n,
                        n_out=self.n * (2 + self.n),
                    )
                _MHC_POST_TILELANG_KERNEL.register_warmup(
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                )
                _MHC_FUSED_TILELANG_KERNEL.register_warmup(
                    vllm_config,
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                )

        _glm5_prof_attach(self)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # 70B or MTP layers: KDA + MoE without HC.
        if not self.mhc or self.is_mtp_layer:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            attn_output = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
            hidden_states, residual = self.post_attention_layernorm(
                attn_output, residual=residual
            )
            hidden_states = self.mlp(hidden_states)
            if self.is_mtp_layer:
                # Return the unsummed pair: the MTP caller feeds it straight
                # into shared_head's fused_add_rms_norm (one kernel instead of
                # a separate residual-add + norm). The sum itself is unchanged
                # (fp32-accumulated inside the fused kernel).
                return hidden_states, residual, None, None
            hidden_states = residual + hidden_states
            return hidden_states, residual, None, None

        # mHC start. `post`/`comb` carry the previous layer's deferred
        # hc_post inputs (its ffn-pre outputs); when present, fuse that
        # hc_post with this layer's attn hc_pre into one kernel (inter-layer
        # fusion). Layer 0 has no incoming state -> standalone hc_pre.
        x = hidden_states
        _sub_n = x.shape[0]
        _glm5_sub(self.layer_idx, _sub_n, _new=1, a_in=x)
        if post is None:
            if self.layer_idx == 0:
                x = hc_expand(x, self.n)
            _glm5_sub(self.layer_idx, _sub_n, b_expand=x)
            residual = x
            post, comb, x = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
        else:
            residual, post, comb, x = self.hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )

        _glm5_sub(self.layer_idx, _sub_n, c_pre_x=x, c_pre_post=post,
                   c_pre_comb=comb, c_pre_residual=residual)

        # Attention needs the full token sequence; mHC above ran on the SP
        # shard. Gather for attention, scatter back afterward (DSv4 pattern).
        if self.is_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )

        if self.is_sequence_parallel:
            x = sp_reduce_scatter(x)

        _glm5_sub(self.layer_idx, _sub_n, d_attn=x)

        # Fuse post-attn hc_post + pre-FFN hc_pre (+ RMSNorm) into one kernel.
        residual, post, comb, x = self.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=self.post_attention_layernorm.weight.data,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )

        _glm5_sub(self.layer_idx, _sub_n, e_post_x=x, e_post_post=post,
                   e_post_comb=comb, e_post_residual=residual)

        # Fully Connected
        if self._mlp_is_moe:
            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)
        else:
            x = self.mlp(x)

        _glm5_sub(self.layer_idx, _sub_n, f_mlp=x)

        # mHC end. The last mHC layer materializes its final hc_post (nothing
        # to fuse with) then contracts; every other layer defers its hc_post to
        # the next layer's fused pre, returning the state.
        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = hc_contract(x, self.n)
            return x, None, None, None

        return x, residual, post, comb

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        post_mix, res_mix, layer_input = self.mhc_pre_op(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        return post_mix, res_mix, layer_input

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return self.mhc_post_op(x, residual, post, comb)

    def hc_fused_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        return self.mhc_fused_post_pre_op(
            x=x,
            residual=residual,
            post_layer_mix=post,
            comb_res_mix=comb,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            n_splits=1,
            tile_n=1,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )


class Glm5NextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type

        self.is_v32 = config.index_topk is not None
        if self.is_v32:
            topk_tokens = config.index_topk
            assert topk_tokens is not None
            # Reserve room for the incomplete pool tail.
            kpool = config.index_kpool
            assert kpool is not None
            buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
            # Sparse MLA tiles top-k in 128 columns; padded slots remain masked.
            sparse_topk_block_n = 128
            buffer_width = (
                (buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n
            ) * sparse_topk_block_n
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                buffer_width,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # Full-MLA config (no kpool sparse indexer): no topk buffer.
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # The active slice is fixed after construction; cache it so forward
        # doesn't rebuild the slice (a fresh list) every step.
        self._active_layers = self.layers[self.start_layer : self.end_layer]

        # local: say out loud how many syncs each GLM5_SYNC mode would place,
        # so a mode that silently matches nothing cannot be mistaken for a
        # configuration that is doing work (GLM5_SYNC=kda/dsa did exactly that).
        import os as _os

        _kinds = [getattr(_l, "layer_kind", "?") for _l in self._active_layers]
        _nk = _kinds.count("kda")
        logger.info(
            "GLM5_SYNC=%r would place: layer=%d kda=%d dsa=%d kda3=%d (of %d layers)",
            _os.environ.get("GLM5_SYNC", ""),
            len(_kinds), _nk, _kinds.count("mla"), _nk // 3, len(_kinds),
        )
        if _os.environ.get("GLM5_TORCH_DET") == "1":
            # warn_only: every non-deterministic op warns once per process
            # instead of raising, so one run enumerates all of them.
            torch.use_deterministic_algorithms(True, warn_only=True)
            logger.info("GLM5_TORCH_DET: use_deterministic_algorithms(warn_only)")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.is_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            post = None
            comb = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            # post/comb (deferred mHC hc_post state) are not propagated across
            # PP ranks; the receiving rank's first mHC layer uses standalone pre.
            post = None
            comb = None

        full_num_tokens = positions.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)

        import os as _os
        _sync_mode = _os.environ.get("GLM5_SYNC", "")
        _trace_at = _os.environ.get("GLM5_TRACE")
        _tf = None
        _tstep = None
        if _trace_at:
            try:
                _rk = (torch.distributed.get_rank()
                       if torch.distributed.is_initialized() else 0)
            except Exception:
                _rk = 0
            _want = [int(x) for x in _trace_at.split(",") if x.strip()]
            # never during graph capture: capture runs multi-token (MTP) decode forwards,
            # and the shas below copy to the host
            if _rk == 0 and full_num_tokens > 1 and not torch.cuda.is_current_stream_capturing():
                # a prefill means a new request: restart the per-request decode
                # counter so "step N" is the same logical point in every run.
                # Step -1 is the prefill forward itself, which is what writes
                # the state that decode step 0 reads.
                _GLM5_DECODE_STEP[0] = 0
                _GLM5_REQ[0] += 1
                if -1 in _want:
                    _tstep = -1
            if _rk == 0 and full_num_tokens == 1:
                _step = _GLM5_DECODE_STEP[0]
                _GLM5_DECODE_STEP[0] = _step + 1
                if _step in _want:
                    _tstep = _step
            if _tstep is not None:
                _tf = open("/root/.cache/vllm/trace-r%d-s%d.log"
                           % (_GLM5_REQ[0], _tstep), "a")
                _tf.write("req=%d,step=%d,layer=input,type=embed,tok=%d,"
                          "h=%s,r=%s,post=%s,comb=%s\n"
                          % (_GLM5_REQ[0], _tstep, full_num_tokens,
                             _glm5_sha(hidden_states), _glm5_sha(residual),
                             _glm5_sha(post), _glm5_sha(comb)))
        # local diagnostics (NOT FOR UPSTREAM): GLM5_SYNC_LOG=1 prints, on rank 0 for prefill forwards,
        # the step (position range) with row 0 of every block table / state-index tensor in the step's
        # attention metadata, then one line before each layer runs and one after its GLM5_SYNC sync.
        # With AMD_SERIALIZE_KERNEL=3 the last "pre" line before a GPU fault names the faulting layer.
        _sl = None
        if (_os.environ.get("GLM5_SYNC_LOG") and full_num_tokens > 1
                and not torch.cuda.is_current_stream_capturing()):
            try:
                _slr = (torch.distributed.get_rank()
                        if torch.distributed.is_initialized() else 0)
            except Exception:
                _slr = 0
            if _slr == 0:
                import sys as _sys
                _sl = _sys.stderr
                from vllm.forward_context import get_forward_context
                _md = get_forward_context().attn_metadata
                _md = _md[0] if isinstance(_md, list) else _md
                _rows = []
                if isinstance(_md, dict):
                    _seen = set()
                    for _nm, _m in _md.items():
                        if id(_m) in _seen:
                            continue
                        _seen.add(id(_m))
                        for _a in ("block_table", "block_table_tensor",
                                   "non_spec_state_indices_tensor",
                                   "state_indices_tensor"):
                            _t = getattr(_m, _a, None)
                            if isinstance(_t, torch.Tensor) and _t.numel():
                                _r0 = _t[0] if _t.dim() > 1 else _t[:8]
                                _rows.append("%s:%s.%s=%s" % (
                                    _nm.split(".")[-2] if "." in _nm else _nm,
                                    type(_m).__name__, _a,
                                    _r0[:64].tolist()))
                _sl.write("GLM5SL step tokens=%d pos=%d..%d %s\n" % (
                    full_num_tokens, int(positions.min()), int(positions.max()),
                    " | ".join(_rows)))
                _sl.flush()
        for _li, layer in enumerate(self._active_layers):
            if _sl is not None:
                _sl.write("GLM5SL L%d %s pre\n" % (_li, getattr(layer, "layer_kind", "?")))
                _sl.flush()
            if _tf is not None:
                _c, _r = _glm5_state_sha(layer)
                _tf.write("req=%d,step=%d,layer=%d,type=%s,pre_conv=%s,pre_ssm=%s\n"
                          % (_GLM5_REQ[0], _tstep, _li,
                             getattr(layer, "layer_kind", "?"), _c, _r))
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
            if _tf is not None:
                _tf.write("req=%d,step=%d,layer=%d,type=%s,h=%s,r=%s,"
                          "post=%s,comb=%s\n"
                          % (_GLM5_REQ[0], _tstep, _li,
                             getattr(layer, "layer_kind", "?"),
                             _glm5_sha(hidden_states), _glm5_sha(residual),
                             _glm5_sha(post), _glm5_sha(comb)))
            if _sync_mode:
                # layer_kind ("kda"/"mla") is set in Glm5NextDecoderLayer.__init__
                # from config.is_kda_layer(layer_idx). The earlier version of this
                # block read a "block_type" attribute that does not exist on these
                # layers, so GLM5_SYNC=kda/kda3/dsa never placed a single sync.
                _bt = getattr(layer, "layer_kind", "")
                # local: "kdaN" syncs after every Nth KDA layer, so the sync
                # COUNT can be matched to "dsa" (11 of 45) while the placement
                # stays on KDA boundaries. Comparing kda3 against dsa separates
                # "sync density matters" from "sync location matters"; plain
                # kda (34 syncs) vs dsa (11) confounds the two.
                _hit = (_sync_mode == "layer"
                        or (_sync_mode == "kda" and _bt == "kda")
                        or (_sync_mode == "dsa" and _bt == "mla"))
                if not _hit and _sync_mode.startswith("kda") and _bt == "kda":
                    _n = _sync_mode[3:]
                    if _n.isdigit():
                        _kda_seen = getattr(self, "_kda_every_count", 0) + 1
                        self._kda_every_count = _kda_seen
                        _hit = _kda_seen % int(_n) == 0
                if _hit and not torch.cuda.is_current_stream_capturing():
                    torch.cuda.synchronize()
                    if _sl is not None:
                        _sl.write("GLM5SL L%d synced\n" % _li)
                        _sl.flush()

        if _tf is not None:
            _tf.close()

        # local: GLM5_MEMSTATS=1 logs allocator counters on rank 0 after every
        # prefill forward. num_alloc_retries > 0 means the caching allocator had
        # to release cached blocks to satisfy a request, which is the "allocator
        # reuse under pressure" candidate for Bug B.
        if _os.environ.get("GLM5_MEMSTATS") == "1" and full_num_tokens > 1:
            try:
                _rk = (torch.distributed.get_rank()
                       if torch.distributed.is_initialized() else 0)
            except Exception:
                _rk = 0
            if _rk == 0:
                _ms = torch.cuda.memory_stats()
                print(
                    "GLM5_MEMSTATS tokens=%d retries=%s ooms=%s reserved=%s allocated=%s"
                    % (
                        full_num_tokens,
                        _ms.get("num_alloc_retries", -1),
                        _ms.get("num_ooms", -1),
                        _ms.get("reserved_bytes.all.current", -1),
                        _ms.get("allocated_bytes.all.current", -1),
                    ),
                    flush=True,
                )

        if not get_pp_group().is_last_rank:
            # PP is gated off for GLM-5.3-Flash (no make_empty_intermediate_tensors),
            # so this branch is not exercised. post/comb are the deferred
            # hc_post state of this rank's last mHC layer; a future PP path
            # would need to propagate them, but for now they are dropped (the
            # receiving rank's first layer would fall back to standalone pre).
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.is_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # MLA: fuse q_a_proj and kv_a_proj_with_mqa
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            # Indexer: fuse wk and weights_proj
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
            # KDA: merge q, k, v, b, f_a, g_a projections into one GEMM
            (".in_proj_qkvbfg_a", ".q_proj", 0),
            (".in_proj_qkvbfg_a", ".k_proj", 1),
            (".in_proj_qkvbfg_a", ".v_proj", 2),
            (".in_proj_qkvbfg_a", ".b_proj", 3),
            (".in_proj_qkvbfg_a", ".f_a_proj", 4),
            (".in_proj_qkvbfg_a", ".g_a_proj", 5),
        ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            # EPLB: the mapping enumerates physical experts, so it must cover
            # the redundant replicas or their slots are never loaded.
            num_redundant_experts = next(
                (
                    layer.mlp.n_redundant_experts
                    for layer in self.layers
                    if isinstance(layer, Glm5NextDecoderLayer)
                    and isinstance(layer.mlp, Glm5NextMoE)
                ),
                0,
            )
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
                num_redundant_experts=num_redundant_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # GLM-5.3-Flash NoPE checkpoints omit the RoPE rows from
        # ``kv_a_proj_with_mqa``; pad them with zeros for the model shape.
        kv_a_pad_size = 0
        if self.config.mla_nope and self.config.qk_rope_head_dim > 0:
            kv_a_pad_size = self.config.qk_rope_head_dim

        _pending_wk_fp8: dict = {}

        for args in weights:
            name, loaded_weight = args[:2]
            kwargs: dict = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            # Handle FP8 indexer WK: dequantize to BF16 for fusion with
            # weights_proj into wk_weights_proj.
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
            ):
                continue

            # FP8 checkpoint: dequantize BF16-kept MLA projections
            # (q_a_proj / kv_a_proj_with_mqa / o_proj) to BF16.
            if _try_load_fp8_attn_proj(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                kv_a_pad_size,
            ):
                continue

            # Pad kv_a_proj_with_mqa for NoPE models
            if kv_a_pad_size > 0 and ".kv_a_proj_with_mqa." in name:
                pad = torch.zeros(
                    kv_a_pad_size,
                    *loaded_weight.shape[1:],
                    dtype=loaded_weight.dtype,
                    device=loaded_weight.device,
                )
                loaded_weight = torch.cat([loaded_weight, pad], dim=0)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # QKV fusion: skip if fused module doesn't exist in model
                if param_name == ".fused_qkv_a_proj" and name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    # A checkpoint expert may map to several physical replicas
                    # under EPLB; keep `name` intact and try the next entry
                    # when this physical expert is not local to the rank.
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        return loaded_params


class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = Glm5NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=self.config.logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_num_heads,
            hf_config.linear_head_dim,
            conv_kernel_size=hf_config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[
        MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc
    ]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, MixtureOfExperts
):
    # The text model (KDA + dense-MLA + MoE) is a hybrid mamba model. The
    # multimodal wrapper must declare the same interfaces so vLLM treats it as
    # hybrid (auto-aligns mamba/attention block sizes, sizes the mamba state
    # cache); the mamba-state classmethods delegate to the text model.
    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True

    # NOTE: weight-prefix mapping is inherited from Glm4vForConditionalGeneration
    # (``model.visual.`` -> ``visual.``, ``model.language_model.`` ->
    # ``language_model.model.``, ``lm_head.`` -> ``language_model.lm_head.``),
    # matching the GLM-OCR / GLM-4V serialization convention. If the real
    # checkpoint's safetensors keys differ (e.g. ``language_model.model.`` with
    # no outer ``model.``), override ``hf_to_vllm_mapper`` accordingly.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm4vForConditionalGeneration, self).__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                # Read eps from the VISION sub-config, not the top-level
                # `config.rms_norm_eps`: Glm5NextConfig.__getattribute__ mirrors
                # the latter onto text_config (1e-5), silently ignoring the
                # vision tower's own (1e-6) rms_norm_eps.
                norm_eps=config.vision_config.rms_norm_eps,
                # Vision tower ships BF16 weights in this fp8 checkpoint (no
                # weight_scale_inv for visual.*), so it must NOT inherit the
                # global fp8 quant_config -- doing so incorrectly quantizes
                # the tower
                # and yields NaN image features. Mirrors the MLA/KDA proj
                # pattern (quant_config=None for BF16 submodules).
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )

        self.set_moe_parameters()

        # Glm5NextForCausalLM does not implement make_empty_intermediate_tensors,
        # so pipeline parallelism is gated off (consistent with the text-only
        # model) and we intentionally do not alias it here.

    def set_moe_parameters(self) -> None:
        self.moe_mlp_layers = [
            layer.mlp
            for layer in self.language_model.model.layers
            if isinstance(layer, Glm5NextDecoderLayer)
            and isinstance(layer.mlp, Glm5NextMoE)
        ]
        self.moe_layers = [moe.experts for moe in self.moe_mlp_layers]
        self.num_moe_layers = len(self.moe_layers)
        if not self.num_moe_layers:
            return
        example_moe = self.moe_mlp_layers[0]
        self.num_expert_groups = self.config.text_config.n_group
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if not self.num_moe_layers:
            return
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def get_encoder_cudagraph_config(self):
        # This vision tower does not produce the absolute position embedding
        # buffer used by GLM4V.
        config = super().get_encoder_cudagraph_config()
        config.buffer_keys = [k for k in config.buffer_keys if k != "pos_embeds"]
        return config


def get_spec_layer_idx_from_weight_name(
    config: Glm5NextConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(
                f"model.layers.{layer_idx + i}."
            ) or weight_name.startswith(f"layers.{layer_idx + i}."):
                return layer_idx + i
    return None


def _try_load_fp8_indexer_wk(name, tensor, buf, params_dict, loaded_params):
    if "indexer.wk." not in name or "wk_weights" in name:
        return False
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False
    layer_prefix = name.rsplit(".wk.", 1)[0]
    entry = buf.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    del buf[layer_prefix]
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = scaled_dequantize(
        weight_fp8,
        scale_inv,
        group_shape=GroupShape(block_size, block_size),
        out_dtype=torch.bfloat16,
    )

    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True


def _dequant_fp8_block(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize a block-FP8 (e4m3) weight with per-block scale to BF16.

    Unlike ``scaled_dequantize`` this tolerates a non-divisible (partial last
    block) shape by zero-padding to a multiple of ``block_size`` before the
    scale broadcast and trimming back afterwards (e.g. kv_a_proj_with_mqa is
    576 rows = 4*128 + 64).
    """
    out_dim, in_dim = weight_fp8.shape
    pad_out = (-out_dim) % block_size
    pad_in = (-in_dim) % block_size
    w = weight_fp8
    if pad_out or pad_in:
        w = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))
    # scale_inv is (ceil(out/block), ceil(in/block)); broadcast to (out, in).
    s = scale_inv.to(torch.float32)
    s_full = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    out = (w.to(torch.float32) * s_full).to(torch.bfloat16)
    return out[:out_dim, :in_dim].contiguous()


# FP8 checkpoint projections that the MODEL keeps in BF16, so the block-FP8
# (weight + weight_scale_inv) must be dequantized to BF16 on load.
# Maps checkpoint proj-suffix -> (buffer key, model target base, fused shard id
# or None for a direct projection, whether NoPE rope-padding applies).
_FP8_ATTN_PROJS = {
    ".q_a_proj.": ("q_a", "fused_qkv_a_proj", 0, False),
    ".kv_a_proj_with_mqa.": ("kv_a", "fused_qkv_a_proj", 1, True),
    ".q_b_proj.": ("q_b", "q_b_proj", None, False),
    ".o_proj.": ("o_proj", "o_proj", None, False),
}


def _try_load_fp8_attn_proj(
    name,
    tensor,
    buf,
    params_dict,
    loaded_params,
    kv_a_pad_size: int,
) -> bool:
    """Dequantize FP8 q_a_proj / kv_a_proj_with_mqa / o_proj to BF16 on load.

    The FP8 checkpoint stores these as block-FP8 (weight + weight_scale_inv),
    but the model holds them in BF16 (``fused_qkv_a_proj`` is always BF16 via
    DeepSeekV2FusedQkvAProjLinear; ``o_proj`` is excluded by
    modules_to_not_convert). When the model target is BF16 (no
    ``weight_scale_inv`` param) we dequantize; otherwise we return False so the
    normal stacked/direct path loads the FP8 tensor as-is.
    """
    matched = None
    for suffix, info in _FP8_ATTN_PROJS.items():
        if suffix in name:
            matched = (suffix, info)
            break
    if matched is None:
        return False
    suffix, (key, target_base, shard_id, is_kva) = matched
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False

    layer_prefix = name.rsplit(suffix, 1)[0]
    target_w = f"{layer_prefix}.{target_base}.weight"
    target_s = f"{layer_prefix}.{target_base}.weight_scale_inv"
    # If the model actually kept this projection in FP8, let the normal path
    # handle it (it has a weight_scale_inv param).
    if target_s in params_dict:
        return False

    entry = buf.setdefault(layer_prefix, {}).setdefault(key, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    buf[layer_prefix].pop(key, None)
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)
    # NoPE: pad kv_a rope portion (kv_lora_rank -> kv_lora_rank + qk_rope_head_dim).
    if is_kva and kv_a_pad_size > 0:
        pad = torch.zeros(
            kv_a_pad_size,
            weight_bf16.shape[1],
            dtype=weight_bf16.dtype,
            device=weight_bf16.device,
        )
        weight_bf16 = torch.cat([weight_bf16, pad], dim=0)

    param = params_dict[target_w]
    if shard_id is None:
        param.weight_loader(param, weight_bf16)
    else:
        param.weight_loader(param, weight_bf16, shard_id)
    loaded_params.add(target_w)
    return True
