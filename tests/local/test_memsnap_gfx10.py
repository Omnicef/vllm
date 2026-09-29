"""memsnap-diff.py on two tiny snapshots: A holds 64 MiB live + a 256 MiB transient, B 192 MiB live + 512 MiB."""
import pickle, subprocess, torch
def run(live_mb, tmp_mb, path):
    torch.cuda.memory._record_memory_history(max_entries=100000, stacks="python")
    keep = torch.empty(live_mb << 20, dtype=torch.uint8, device="cuda")
    t = torch.empty(tmp_mb << 20, dtype=torch.uint8, device="cuda"); del t
    pickle.dump(torch.cuda.memory._snapshot(), open(path, "wb"))
    torch.cuda.memory._record_memory_history(enabled=None)
    return keep
a = run(64, 256, "/tmp/A.pickle"); del a; torch.cuda.empty_cache()
b = run(192, 512, "/tmp/B.pickle")
out = subprocess.run(["python3", "/s/memsnap-diff.py", "/tmp/A.pickle", "/tmp/B.pickle"], capture_output=True, text=True)
print(out.stdout, out.stderr)
ok = "+0.125 GiB" in out.stdout and "+0.250 GiB" in out.stdout
print("PASS memsnap-diff:", ok)
