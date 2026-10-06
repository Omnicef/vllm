"""GLM5_PROF_EVENTS collective spans: with the flag off, coll_begin must return None before touching CUDA
(nothing can enter a captured graph); coll_end(None) is a no-op. CPU only."""
import os

os.environ.pop("GLM5_PROF_EVENTS", None)
import torch

from vllm.utils import glm5_prof_events as pe


def _boom(*a, **k):
    raise AssertionError("CUDA touched with GLM5_PROF_EVENTS off")


torch.cuda.Event = _boom
torch.cuda.is_current_stream_capturing = _boom
assert not pe.ENABLED
assert pe.coll_begin("all_reduce") is None
assert pe.coll_begin("all_gather", eager=True) is None
pe.coll_end(None)
print("coll spans off: PASS")
