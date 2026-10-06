"""GLM5_EMPTY_CACHE_AFTER_PROFILE plumbing on the ROCm indexer layer (no GPU work; needs the ROCm platform to import
glm5next/amd/sparse_indexer.py): the layer captures max_model_len at construction inside a config context, and
forward_hip passes it to sparse_attn_indexer_kpool when no config context is set (as in the profile run)."""
import os
import types

os.environ["GLM5_EMPTY_CACHE_AFTER_PROFILE"] = "1"
import torch

import vllm.models.glm5next.amd.sparse_indexer as m
from vllm.config import VllmConfig, get_current_vllm_config_or_none, set_current_vllm_config

# 1. construction: the real __init__ inside a config context; max_model_len from the config the layer sees
fake = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=262144))
real_get = m.get_current_vllm_config_or_none
m.get_current_vllm_config_or_none = lambda: fake
try:
    with set_current_vllm_config(VllmConfig()):              # CustomOp construction needs a config context
        kc = types.SimpleNamespace(prefix="model.layers.3.self_attn.indexer.k_cache", kv_cache=None)
        layer = m.SparseAttnIndexerKpool(kc, 128, "ue8m0", 2048, 128, 65536, 131076,
                                         torch.zeros(1, 2048, dtype=torch.int32))
finally:
    m.get_current_vllm_config_or_none = real_get
assert layer.max_model_len == 262144, layer.max_model_len

# 2. forward_hip outside any config context passes the captured value
seen = {}
real_fn = m.sparse_attn_indexer_kpool
m.sparse_attn_indexer_kpool = lambda *a, **k: seen.update(k) or "called"
try:
    assert get_current_vllm_config_or_none() is None
    out = layer.forward_hip(torch.zeros(4, 8), torch.zeros(4, 1, 8), torch.zeros(4, 8), torch.zeros(4, 1),
                            index_kpool=4)
finally:
    m.sparse_attn_indexer_kpool = real_fn
assert out == "called" and seen.get("max_model_len") == 262144, seen

# 3. no config at construction -> None (the window-2 failure mode, now logged as a warning at construction)
m.get_current_vllm_config_or_none = lambda: None
try:
    with set_current_vllm_config(VllmConfig()):
        layer2 = m.SparseAttnIndexerKpool(kc, 128, "ue8m0", 2048, 128, 65536, 131076,
                                          torch.zeros(1, 2048, dtype=torch.int32))
finally:
    m.get_current_vllm_config_or_none = real_get
assert layer2.max_model_len is None
print("PASS empty-cache plumbing: True")
