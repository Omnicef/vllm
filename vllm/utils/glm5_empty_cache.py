# SPDX-License-Identifier: Apache-2.0
"""GLM5_EMPTY_CACHE_AFTER_PROFILE decision (local, glm53-v031).

The indexer's profile-run branch can hand its ~1.07 GiB logits reservation back to the device right after freeing
it, so later profile-time allocations cannot pin the cached segment (phase 26/27: +0.51 GiB KV at 131k), but only
from GLM5_EMPTY_CACHE_MIN_LEN (default 131072): at 32k it cost 0.44 GiB KV (phase 27 S5 vs phase 28 R0).
max_model_len must come from the layer (captured at construction): the profile run executes outside any
set_current_vllm_config() context, so get_current_vllm_config_or_none() is None there (window 2, 2026-10-06).
"""


def empty_cache_after_profile(flag: str | None, max_model_len: int | None, min_len: int, capturing: bool) -> bool:
    return flag == "1" and max_model_len is not None and max_model_len >= min_len and not capturing
