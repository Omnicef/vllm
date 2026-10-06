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


# Inline sub-op ranges (2026-09-27, indexer breakdown). begin(sub) / end(token) around a region records an
# external-event pair named "sub.<sub>.<seq>", seq counting spans within one forward (reset_seq() at the model
# forward), so the names a captured graph records are stable across replays and a loop body gets one name per
# iteration. Summed per sub-op by the table script. Events are recorded only under graph capture (decode). With GLM5_PROF=1 the same regions also get a torch-profiler
# range "idx.<sub>" (eager traces). Both off: begin() returns None and end() does nothing.
_PROF = os.environ.get("GLM5_PROF") == "1"
_seq = [0]


def reset_seq() -> None:
    _seq[0] = 0


def begin(sub: str):
    if not (ENABLED or _PROF):
        return None
    rf = None
    if _PROF:
        rf = torch.profiler.record_function("idx." + sub)
        rf.__enter__()
    ev = None
    # events only inside a graph capture (decode): collect() reads pairs after replays, and an eager (prefill)
    # record under the same name would be re-read as stale time on every replay
    if ENABLED and torch.cuda.is_current_stream_capturing():
        name = "sub.%s.%d" % (sub, _seq[0])
        _seq[0] += 1
        ev = _pairs.get(name)
        if ev is None:
            ev = _pairs[name] = (torch.cuda.Event(enable_timing=True, external=True),
                                 torch.cuda.Event(enable_timing=True, external=True))
        ev[0].record()
    return (rf, ev)


def end(tok) -> None:
    if tok is None:
        return
    rf, ev = tok
    if ev is not None:
        ev[1].record()
    if rf is not None:
        rf.__exit__(None, None, None)



# Collective spans (2026-10-06, v031 decode-step split: collectives vs kernels vs the rest). coll_begin(name) /
# coll_end(token) around a TP collective (distributed/parallel_state.py). Under graph capture: a "sub.coll.<name>.<n>"
# pair via begin(), read by collect() after each replay. Outside capture only when eager=True (the logits all_gather,
# which runs after the target graph replay): a plain event pair, synchronized at the end and summed into
# "eager.coll.<name>" (profiling only). Flag off: coll_begin returns None before any CUDA call, so nothing is
# recorded into a captured graph and the eager path is untouched.
def coll_begin(name: str, eager: bool = False):
    if not ENABLED:
        return None
    if torch.cuda.is_current_stream_capturing():
        return begin("coll." + name)
    if not eager:
        return None
    ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    ev[0].record()
    return ("eager", name, ev)


def coll_end(tok) -> None:
    if tok is None:
        return
    if tok[0] != "eager":
        end(tok)
        return
    _, name, ev = tok
    ev[1].record()
    ev[1].synchronize()
    a = _acc.setdefault("eager.coll." + name, [0.0, 0])
    a[0] += ev[0].elapsed_time(ev[1])
    a[1] += 1

# GLM5_STEP_TIMING=1 (2026-09-28): per FULL-graph replay, one plain timing-event pair around replay() on the stream
# (recorded outside any capture, so plain events work) plus host perf_counter times. Pairs are harvested once
# complete (query(), no stall). Per graph manager (target model vs drafter): GPU ms per replay, host ms inside the
# replay call, host ms between the end of one replay call and the start of the next. Dumped to
# $GLM5_PROF_EVENTS_OUT (default /root/.cache/vllm)/glm5_steptime.<pid>.json every 200 replays and at exit.
STEP = os.environ.get("GLM5_STEP_TIMING") == "1"
_st: dict = {}


def step_reset() -> None:
    _st.clear()


def step_begin(key):
    if not STEP:
        return None
    if not _st.get("_rpc"):
        # POST /collective_rpc {"method": "glm5_steptime_reset"} clears the samples (e.g. right before a benchmark)
        from vllm.v1.worker.gpu_worker import Worker

        Worker.glm5_steptime_reset = lambda self: step_reset()
        Worker.glm5_steptime_dump = lambda self: step_dump()
        _st["_rpc"] = True
    import time as _t
    t = _t.perf_counter()
    d = _st.setdefault(key, {"ring": [], "gpu": [], "call": [], "gap": [], "last_end": None, "n": 0})
    if d["last_end"] is not None:
        d["gap"].append((t - d["last_end"]) * 1e3)
    e0 = torch.cuda.Event(enable_timing=True)
    e0.record()
    return (key, e0, t)


def step_end(tok):
    if tok is None:
        return
    import time as _t
    key, e0, t = tok
    e1 = torch.cuda.Event(enable_timing=True)
    e1.record()
    d = _st[key]
    t1 = _t.perf_counter()
    d["call"].append((t1 - t) * 1e3); d["last_end"] = t1
    d["ring"].append((e0, e1)); d["n"] += 1
    while d["ring"] and (len(d["ring"]) > 4096 or d["ring"][0][1].query()):   # harvest completed pairs only
        a, b = d["ring"].pop(0)
        if not b.query():
            b.synchronize()                                   # only if 4096 replays are still queued
        d["gpu"].append(a.elapsed_time(b))
    for k in ("gpu", "call", "gap"):
        if len(d[k]) > 50000:
            del d[k][:10000]
    if d["n"] % 200 == 0:
        step_dump()


def step_dump():
    if not _st:
        return
    for d in [v for v in _st.values() if isinstance(v, dict)]:  # drain every completed pair
        while d["ring"] and d["ring"][0][1].query():
            a, b = d["ring"].pop(0)
            d["gpu"].append(a.elapsed_time(b))
    out = os.environ.get("GLM5_PROF_EVENTS_OUT", "/root/.cache/vllm")
    with open(os.path.join(out, f"glm5_steptime.{os.getpid()}.json"), "w") as f:
        json.dump({k: {"n": d["n"], "gpu": d["gpu"], "call": d["call"], "gap": d["gap"]} for k, d in _st.items()
                   if isinstance(d, dict)}, f)


if STEP:
    atexit.register(step_dump)

if ENABLED:
    atexit.register(dump)
