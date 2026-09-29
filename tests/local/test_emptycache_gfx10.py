"""GLM5_EMPTY_CACHE_AFTER_PROFILE mechanism on one card: a freed 1.02 GiB reservation, then persistent 64 + 21 MiB
buffers, then the end-of-profile empty_cache. Off: the persistent buffers land in the cached segment and pin it
(reserved stays ~1.02 GiB). On (empty_cache right after the free): reserved ~ the persistent 85 MiB. Peak unchanged."""
import torch
G = 2 ** 30
def run(flag):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    r = torch.empty(int(1.02 * G), dtype=torch.uint8, device="cuda"); del r
    if flag:
        torch.cuda.empty_cache()
    keep = [torch.empty(64 << 20, dtype=torch.uint8, device="cuda"), torch.empty(21 << 20, dtype=torch.uint8, device="cuda")]
    torch.cuda.empty_cache()
    res, peak = torch.cuda.memory_reserved() / G, torch.cuda.max_memory_allocated() / G
    del keep; torch.cuda.empty_cache()
    return res, peak
off, on = run(False), run(True)
print(f"off: reserved {off[0]:.3f} GiB peak {off[1]:.3f} | on: reserved {on[0]:.3f} GiB peak {on[1]:.3f}")
ok = off[0] > 1.0 and on[0] < 0.2 and abs(off[1] - on[1]) < 1e-3
print("PASS empty-cache mechanism:", ok)
