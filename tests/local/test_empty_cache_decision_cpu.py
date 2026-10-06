"""GLM5_EMPTY_CACHE_AFTER_PROFILE decision (vllm/utils/glm5_empty_cache.py), CPU only: RAN at the production length,
SKIPPED by threshold / flag off / capturing / value missing, and the value used regardless of any config context."""
from vllm.config import get_current_vllm_config_or_none
from vllm.utils.glm5_empty_cache import empty_cache_after_profile as d

assert d("1", 262144, 131072, False) is True                  # production: RAN
assert d("1", 131072, 131072, False) is True                  # at the threshold
assert d("1", 32768, 131072, False) is False                  # SKIPPED by threshold (the 32k case #77 excluded)
assert d(None, 262144, 131072, False) is False                # flag unset
assert d("", 262144, 131072, False) is False                  # flag set empty (run-upstream GLM5_EMPTY_CACHE_AFTER_PROFILE=)
assert d("0", 262144, 131072, False) is False
assert d("1", 262144, 131072, True) is False                  # never inside a graph capture
assert d("1", None, 131072, False) is False                   # value missing (window 2's failure mode)
assert get_current_vllm_config_or_none() is None              # no config context here, and the decision does not need one
assert d("1", 262144, 131072, False) is True
print("PASS empty-cache decision: True")
