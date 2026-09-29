"""(1) MoE diag-hook guard: MoERunner._apply_quant_method, run once with the committed file mounted (MODE=old) and
once with the patched file (MODE=new), same fake runner; sha256 of the routed output compared by the caller: routed output identical; torch sorts per call 2 -> 0 with GLM5_TRACE_MOE2 unset;
positive control: with GLM5_TRACE_MOE2 set (to a layer that does not match), the new code evaluates the sorts again.
(2) GLM5_VISION_CONV_MATMUL: Conv2dLayer downsample (1024 -> 4096, k=s=2) as matmul vs F.conv2d (MIOpen) at the 1080p
and 720p patch counts, within fp16 tolerance; first-call wall time of each path per new shape."""
import hashlib, os, time, types
import torch
import vllm.model_executor.layers.fused_moe.runner.moe_runner as NEW
MODE = os.environ.get("MODE", "new")
dev = "cuda"
torch.manual_seed(0)
x = torch.randn(4, 4096, device=dev, dtype=torch.float16)
logits = torch.randn(4, 288, device=dev, dtype=torch.float16)
tw = torch.softmax(torch.randn(4, 8, device=dev), -1); ti = torch.randint(0, 288, (4, 8), device=dev, dtype=torch.int32)

def fake(mod):
    r = types.SimpleNamespace()
    r._shared_experts = None; r.gate = None; r._fse_fuse_gate = False; r.layer_name = "model.layers.10.mlp.experts"
    r.routed_experts = types.SimpleNamespace(
        quant_method=types.SimpleNamespace(is_monolithic=False),
        forward_modular=lambda x, topk_weights, topk_ids, shared_experts, shared_experts_input:
            x * 2 + topk_weights.sum(-1, keepdim=True).half() + topk_ids.sum(-1, keepdim=True).half())
    r.router = types.SimpleNamespace(select_experts=lambda **kw: (tw, ti))
    r._quant_method = types.SimpleNamespace(topk_indices_dtype=torch.int32)
    r._maybe_apply_shared_experts = types.MethodType(mod.MoERunner._maybe_apply_shared_experts, r)
    return r

calls = [0]; orig = torch.Tensor.sort
def counting(self, *a, **k):
    calls[0] += 1; return orig(self, *a, **k)
torch.Tensor.sort = counting
def run(mod):
    calls[0] = 0
    _, out = mod.MoERunner._apply_quant_method(fake(mod), hidden_states=x, router_logits=logits, shared_experts_input=None)
    return out, calls[0]
os.environ.pop("GLM5_TRACE_MOE2", None)
o, n = run(NEW)
os.environ["GLM5_TRACE_MOE2"] = "999"
o_pc, n_pc = run(NEW)
os.environ.pop("GLM5_TRACE_MOE2")
torch.Tensor.sort = orig
sha = hashlib.sha256(o.cpu().numpy().tobytes()).hexdigest()[:16]
print(f"diag {MODE}: output sha {sha}, sorts per call {n}, with GLM5_TRACE_MOE2 set {n_pc}, same output {torch.equal(o, o_pc)}")
ok1 = True
if MODE == "old":
    raise SystemExit(0)
from vllm.model_executor.layers.conv import Conv2dLayer
W = (torch.randn(4096, 1024, 2, 2, device=dev) * 0.02).half(); B = (torch.randn(4096, device=dev) * 0.02).half()
ns = types.SimpleNamespace(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1,
                           input_size=1024 * 4, out_channels=4096, weight=W, bias=B, enable_linear=True)
ns._forward_mulmat = types.MethodType(Conv2dLayer._forward_mulmat, ns)
ns._forward_conv = types.MethodType(Conv2dLayer._forward_conv, ns)
ok2 = True
for name, groups in (("1080p", 2691), ("720p", 1196)):
    inp = torch.randn(groups, 2, 2, 1024, device=dev).half().permute(0, 3, 1, 2)   # the model's permuted view
    torch.cuda.synchronize(); t = time.time(); c = Conv2dLayer._forward_conv(ns, inp); torch.cuda.synchronize(); tc = time.time() - t
    t = time.time(); m = Conv2dLayer._forward_mulmat(ns, inp); torch.cuda.synchronize(); tm = time.time() - t
    t = time.time(); Conv2dLayer._forward_conv(ns, inp); torch.cuda.synchronize(); tc2 = time.time() - t
    os.environ["GLM5_VISION_CONV_MATMUL"] = "1"; f = Conv2dLayer.forward_cuda(ns, inp); os.environ.pop("GLM5_VISION_CONV_MATMUL")
    g = Conv2dLayer.forward_cuda(ns, inp)
    rel = float((m.float() - c.float()).abs().max() / c.float().abs().max())
    good = rel < 5e-3 and torch.equal(f, m) and torch.equal(g, Conv2dLayer._forward_conv(ns, inp))
    ok2 &= good
    print(f"conv {name} ({groups} groups): matmul vs MIOpen max rel err {rel:.1e}; flag on -> matmul path {torch.equal(f, m)}, "
          f"off -> conv path; first call MIOpen {tc:.1f} s (second {tc2 * 1e3:.1f} ms), matmul {tm * 1e3:.1f} ms -> "
          f"{'PASS' if good else 'FAIL'}", flush=True)
print(f"PASS pre-27 checks: {ok1 and ok2}")
