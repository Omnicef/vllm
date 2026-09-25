# SPDX-License-Identifier: Apache-2.0
"""GLM5_PROF_EVENTS=1: per-block GPU time under CUDA/HIP graphs, with no tracer attached.

Tracers (rocprofv3 --kernel-trace, the torch profiler) hang graph replay on the gfx1030 box
(FINDINGS 09-24/25). Instead, each profiled module gets a pair of timing events recorded by
forward hooks; during graph capture those records become event nodes in the graph. Only
*external* events (hipEventRecordExternal, torch ``external=True``) can be read after a replay
on this ROCm; plain timing events fail with "invalid resource handle" (tests/local/
test_graph_events.py). After every graph replay, ``collect()`` synchronizes and adds each
pair's elapsed time to a per-block total; the totals go to
``$GLM5_PROF_EVENTS_OUT/glm5_prof_events.<pid>.json`` every 50 replays and at exit.
Profiling runs only: the per-replay synchronize removes host/GPU overlap.
"""
import atexit
import json
import os

import torch

ENABLED = os.environ.get("GLM5_PROF_EVENTS") == "1"
_pairs: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
_acc: dict[str, list[float]] = {}
_state = {"replays": 0}


def attach(module: torch.nn.Module, name: str) -> None:
    ev = (torch.cuda.Event(enable_timing=True, external=True),
          torch.cuda.Event(enable_timing=True, external=True))
    _pairs[name] = ev
    module.register_forward_pre_hook(lambda _m, _a: ev[0].record())
    module.register_forward_hook(lambda _m, _a, _o: ev[1].record())


def collect() -> None:
    """Call right after a graph replay: read every pair recorded by that replay."""
    torch.cuda.current_stream().synchronize()
    for name, (s, e) in _pairs.items():
        try:
            ms = s.elapsed_time(e)
        except Exception:
            continue                  # never recorded (e.g. a module outside this graph)
        a = _acc.setdefault(name, [0.0, 0])
        a[0] += ms
        a[1] += 1
    _state["replays"] += 1
    if _state["replays"] % 50 == 0:
        dump()


def dump() -> None:
    if not _acc:
        return
    out = os.environ.get("GLM5_PROF_EVENTS_OUT", "/tmp")
    with open(os.path.join(out, f"glm5_prof_events.{os.getpid()}.json"), "w") as f:
        json.dump({"replays": _state["replays"], "blocks_ms_sum_count": _acc}, f)


if ENABLED:
    atexit.register(dump)
