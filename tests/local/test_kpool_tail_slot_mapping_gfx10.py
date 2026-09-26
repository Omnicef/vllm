#!/usr/bin/env python3
"""The kpool tail group through the generic V2 slot-mapping kernel (vllm#57317), one card.

Real code: vllm.v1.worker.gpu.block_table.BlockTables + _compute_slot_mappings_kernel. Groups as served for GLM-5.3:
attention (block 640), kpool tail (KpoolTailSpec, block = index_kpool = 4, one block per request, row width
get_block_table_width(1, 4, token_alignment=128) = 32), three KDA state groups (block 640, no slot mapping).
max_num_reqs 32, 512-token prefill steps for one request up to 32,768 tokens.
Stock v0.30.0 (KpoolTailSpec inherits uses_slot_mapping=True) vs the #57317 fix (uses_slot_mapping=False).
Rows 1.. of the tail table hold a sentinel block id (7777), so a tail slot >= 7777*4 proves the kernel read outside
row 0; reads past the whole table land in whatever is allocated next (a fault aborts the process).

  python3 test_kpool_tail_slot_mapping_gfx10.py stock|fix
"""
import sys
import torch
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.gpu.block_table import BlockTables

arm = sys.argv[1]
DEV = torch.device("cuda:0")
MAXLEN, STEP, REQS, SENT = 32768, 512, 32, 7777
sizes = [640, 4, 640, 640, 640]
widths = [get_block_table_width(-(-MAXLEN // 640), 640), get_block_table_width(1, 4), 52, 52, 52]
enabled = [True, arm == "stock", False, False, False]
bt = BlockTables(sizes, REQS, STEP, widths, DEV, kernel_block_sizes=sizes, slot_mapping_enabled=enabled)
tail = bt.block_tables[1].gpu
print(f"arm {arm}: tail table {tuple(tail.shape)} int32 = {tail.numel() * 4} bytes; attention width {widths[0]}", flush=True)
tail.fill_(SENT)                                         # other requests' rows: sentinel
bt.append_block_ids(0, ([i + 1 for i in range(-(-MAXLEN // 640))], [2], [3], [4], [5]), overwrite=True)
bt.apply_staged_writes()
torch.cuda.synchronize()
idx = torch.zeros(1, dtype=torch.int32, device=DEV)
qsl = torch.tensor([0, STEP], dtype=torch.int32, device=DEV)
first_out = None
for start in range(0, MAXLEN, STEP):
    pos = torch.arange(start, start + STEP, dtype=torch.int64, device=DEV)
    sm = bt.compute_slot_mappings(idx, qsl, pos, STEP)
    torch.cuda.synchronize()
    t = sm[1, :STEP]
    outside = int((t >= SENT * 4).sum()) if arm == "stock" else 0
    if outside and first_out is None:
        first_out = start
    if start % 4096 == 0 or (outside and start == first_out):
        print(f"  step at {start:5d}: tail read index up to {(start + STEP - 1) // 4:5d} "
              f"(byte offset {((start + STEP - 1) // 4) * 4:6d} from row 0 vs table {tail.numel() * 4}); "
              f"tail slots min {int(t.min())} max {int(t.max())}; reads outside row 0: {outside}", flush=True)
print(f"arm {arm}: survived all steps to {MAXLEN}; first step reading outside row 0: {first_out}", flush=True)
