#!/usr/bin/env python3
"""Can timing events recorded inside a captured HIP graph be read after replay (this ROCm, gfx1030)?

Captures: e0 -> matmul A (small) -> e1 -> matmul B (large) -> e2, for two event kinds:
  plain   torch.cuda.Event(enable_timing=True)
  extern  torch.cuda.Event(enable_timing=True, external=True)   (hipEventRecordExternal)
Pass if, after replay: elapsed_time works, both segments are positive, B/A tracks the eager ratio
(within 2x), the values change when the graph is replayed again (fresh timestamps each replay),
and a 3-segment sum matches the replay's wall time measured by events outside the graph.
"""
import statistics, torch
DEV = "cuda:0"
a = torch.randn(512, 512, device=DEV); b = torch.randn(4096, 4096, device=DEV)


def eager_ratio():
    ts = {}
    for name, x in (("A", a), ("B", b)):
        for _ in range(5): x @ x
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); [x @ x for _ in range(4)]; e1.record(); e1.synchronize()
        ts[name] = e0.elapsed_time(e1) / 4
    return ts


def try_kind(kind):
    kw = {"enable_timing": True}
    if kind == "extern":
        kw["external"] = True
    ev = [torch.cuda.Event(**kw) for _ in range(3)]
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): a @ a; b @ b
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            ev[0].record(); ya = a @ a; ev[1].record(); yb = b @ b; ev[2].record()
    except Exception as e:
        return f"{kind}: capture FAILED: {type(e).__name__}: {str(e)[:160]}"
    rows = []
    for rep in range(4):
        o0, o1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        o0.record(); g.replay(); o1.record(); torch.cuda.synchronize()
        try:
            tA, tB = ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2])
        except Exception as e:
            return f"{kind}: capture ok, elapsed_time FAILED after replay: {type(e).__name__}: {str(e)[:160]}"
        rows.append((tA, tB, o0.elapsed_time(o1)))
    return kind, rows


eg = eager_ratio()
print(f"eager: A {eg['A']*1000:.1f} us, B {eg['B']*1000:.1f} us, B/A {eg['B']/eg['A']:.1f}")
for kind in ("plain", "extern"):
    r = try_kind(kind)
    if isinstance(r, str):
        print(r); continue
    kind, rows = r
    for i, (tA, tB, wall) in enumerate(rows):
        print(f"{kind} replay {i}: A {tA*1000:8.1f} us  B {tB*1000:8.1f} us  A+B {1000*(tA+tB):8.1f}  replay wall {wall*1000:8.1f} us")
    ok = (all(tA > 0 and tB > 0 for tA, tB, _ in rows)
          and 0.5 < (statistics.median(r[1] for r in rows) / statistics.median(r[0] for r in rows)) / (eg["B"] / eg["A"]) < 2
          and len({round(r[0], 6) for r in rows}) > 1
          and all(0.5 < (tA + tB) / wall <= 1.05 for tA, tB, wall in rows))
    print(f"{kind}: {'USABLE' if ok else 'NOT USABLE'}")
