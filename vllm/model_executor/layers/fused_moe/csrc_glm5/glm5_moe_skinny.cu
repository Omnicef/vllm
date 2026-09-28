// SPDX-License-Identifier: Apache-2.0
// GLM-5.3 on gfx1030 (local, NOT FOR UPSTREAM): W4A16 MoE decode skinny GEMV pair for <= 16 tokens.
//
// Ported from leapdragon's moe_skinny_int4_decode (Aron Hsiao, github.com/leapdragon/vllm-rdna2-qwen,
// csrc/rocm/skinny_gemms_int4.cu, 2026-09-06): one wave per output row, lanes stride along K (coalesced weight
// streaming), activations staged in LDS, the gated activation fused into the gate_up epilogue, the top-k weighted
// reduction inside the down kernel (no atomics: deterministic).
// Changes for GLM-5.3: the SwiGLU limit of the Triton path's silu_and_mul_with_clamp (gate = min(gate, limit),
// up = clamp(up, -limit, limit), out = silu(gate) * up, all on the fp16-rounded GEMM outputs, as the Triton path's
// fp16 intermediate cache holds them); limit <= 0 is plain SiLU. Expert-parallel map dropped (GLM runs without EP).
// Own op namespace (glm5_skinny), built as a torch extension.
//
// Layout (the Triton WNA16 path's buffers): w13 [E, 2N, K/2] bytes = [E, 2N, K/8] uint32 of k-sequential nibbles,
// value = (nibble - 8) * scale (symmetric uint4b8), gate rows [0, N), up rows [N, 2N); scales [E, rows, K/G] fp16.
#include <torch/all.h>
#include <torch/library.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <ATen/hip/HIPContext.h>

namespace {

__device__ __forceinline__ int expert_of(const void* ids, bool ids_i64, int idx) {
  return ids_i64 ? (int)reinterpret_cast<const int64_t*>(ids)[idx]
                 : reinterpret_cast<const int32_t*>(ids)[idx];
}
__device__ __forceinline__ float topk_w(const void* w, bool w_is_half, int idx) {
  return w_is_half ? __half2float(reinterpret_cast<const half*>(w)[idx])
                   : reinterpret_cast<const float*>(w)[idx];
}

template <int WAVES>
__global__ void w13_act_gemv(const half* __restrict__ input, const uint32_t* __restrict__ w13,
                             const half* __restrict__ s13, const void* __restrict__ topk_ids,
                             const bool ids_i64, half* __restrict__ act, const int K, const int N,
                             const int topk, const int group_size, const float limit) {
  const int m = blockIdx.z, s = blockIdx.y;
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  extern __shared__ half xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = input[m * K + i];
  __syncthreads();
  if (n >= N) return;
  const int expert = expert_of(topk_ids, ids_i64, m * topk + s);
  if (expert < 0) {
    if (lane == 0) act[((uint64_t)m * topk + s) * N + n] = __float2half(0.f);
    return;
  }
  const int K8 = K / 8, KG = K / group_size;
  const uint64_t base = (uint64_t)expert * 2 * N;
  const uint32_t* wg = w13 + (base + n) * K8;
  const uint32_t* wu = w13 + (base + N + n) * K8;
  const half* sg = s13 + (base + n) * KG;
  const half* su = s13 + (base + N + n) * KG;
  float accg = 0.f, accu = 0.f;
  for (int i = lane; i < K8; i += 32) {
    const uint32_t qg = wg[i], qu = wu[i];
    const int k0 = i * 8;
    float pg = 0.f, pu = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const float xv = __half2float(xs[k0 + j]);
      pg += (float)((int)((qg >> (4 * j)) & 0xF) - 8) * xv;
      pu += (float)((int)((qu >> (4 * j)) & 0xF) - 8) * xv;
    }
    const int g = k0 / group_size;
    accg += pg * __half2float(sg[g]);
    accu += pu * __half2float(su[g]);
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    accg += __shfl_xor(accg, off);
    accu += __shfl_xor(accu, off);
  }
  if (lane == 0) {
    float g = __half2float(__float2half(accg));   // the Triton path's fp16 intermediate
    float u = __half2float(__float2half(accu));
    if (limit > 0.f) {
      g = __half2float(__float2half(fminf(g, limit)));
      u = __half2float(__float2half(fmaxf(fminf(u, limit), -limit)));
    }
    act[((uint64_t)m * topk + s) * N + n] = __float2half(g / (1.0f + expf(-g)) * u);
  }
}

template <int WAVES>
__global__ void w2_gemv(const half* __restrict__ act, const uint32_t* __restrict__ w2,
                        const half* __restrict__ s2, const void* __restrict__ topk_ids,
                        const bool ids_i64, const void* __restrict__ tw, const bool w_is_half,
                        half* __restrict__ out, const int N, const int H, const int topk,
                        const int group_size) {
  const int m = blockIdx.z;
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int h = blockIdx.x * WAVES + wave;
  extern __shared__ half as[];
  for (int i = threadIdx.x; i < topk * N; i += blockDim.x) as[i] = act[(uint64_t)m * topk * N + i];
  __syncthreads();
  if (h >= H) return;
  const int N8 = N / 8, NG = N / group_size;
  float acc = 0.f;
  for (int s = 0; s < topk; s++) {
    const int expert = expert_of(topk_ids, ids_i64, m * topk + s);
    if (expert < 0) continue;
    const uint32_t* wrow = w2 + ((uint64_t)expert * H + h) * N8;
    const half* srow = s2 + ((uint64_t)expert * H + h) * NG;
    const half* xrow = as + s * N;
    float sacc = 0.f;
    for (int i = lane; i < N8; i += 32) {
      const uint32_t q = wrow[i];
      const int k0 = i * 8;
      float p = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++)
        p += (float)((int)((q >> (4 * j)) & 0xF) - 8) * __half2float(xrow[k0 + j]);
      sacc += p * __half2float(srow[k0 / group_size]);
    }
    acc += sacc * topk_w(tw, w_is_half, m * topk + s);
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) acc += __shfl_xor(acc, off);
  if (lane == 0) out[(uint64_t)m * H + h] = __float2half(acc);
}

}  // namespace

void moe_skinny_int4_decode(const at::Tensor& input, const at::Tensor& w13, const at::Tensor& w13_scale,
                            const at::Tensor& w2, const at::Tensor& w2_scale, const at::Tensor& topk_weights,
                            const at::Tensor& topk_ids, at::Tensor& act_buf, at::Tensor& output,
                            int64_t group_size, double swiglu_limit) {
  const int M = input.size(0), K = input.size(1), topk = topk_ids.size(1), N = act_buf.size(2);
  TORCH_CHECK(M >= 1 && M <= 16, "glm5 moe_skinny_int4_decode: M must be 1..16");
  TORCH_CHECK(K % 8 == 0 && N % 8 == 0 && K % group_size == 0 && N % group_size == 0, "shape");
  TORCH_CHECK(input.scalar_type() == at::kHalf && w13_scale.scalar_type() == at::kHalf &&
                  w2_scale.scalar_type() == at::kHalf && act_buf.scalar_type() == at::kHalf &&
                  output.scalar_type() == at::kHalf, "fp16 activations/scales only");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong, "topk_ids dtype");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat || topk_weights.scalar_type() == at::kHalf, "topk_w dtype");
  TORCH_CHECK(topk_ids.is_contiguous() && topk_weights.is_contiguous() && input.is_contiguous() &&
                  act_buf.is_contiguous() && output.is_contiguous(), "contiguous");
  const int64_t elem = w13.element_size();
  TORCH_CHECK(w13.numel() * elem == (int64_t)w13_scale.size(0) * 2 * N * K / 2, "w13 byte-size mismatch");
  TORCH_CHECK(w2.numel() * elem == (int64_t)w2_scale.size(0) * K * N / 2, "w2 byte-size mismatch");
  TORCH_CHECK(output.size(1) == K, "output width must equal hidden size");
  constexpr int WAVES = 8;
  const dim3 block(WAVES * 32), grid1((N + WAVES - 1) / WAVES, topk, M), grid2((K + WAVES - 1) / WAVES, 1, M);
  const hipStream_t stream = at::hip::getCurrentHIPStream();
  const bool ids_i64 = topk_ids.scalar_type() == at::kLong, w_half = topk_weights.scalar_type() == at::kHalf;
  w13_act_gemv<WAVES><<<grid1, block, K * 2, stream>>>(
      reinterpret_cast<const half*>(input.const_data_ptr()), reinterpret_cast<const uint32_t*>(w13.const_data_ptr()),
      reinterpret_cast<const half*>(w13_scale.const_data_ptr()), topk_ids.const_data_ptr(), ids_i64,
      reinterpret_cast<half*>(act_buf.mutable_data_ptr()), K, N, topk, (int)group_size, (float)swiglu_limit);
  w2_gemv<WAVES><<<grid2, block, topk * N * 2, stream>>>(
      reinterpret_cast<const half*>(act_buf.const_data_ptr()), reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
      reinterpret_cast<const half*>(w2_scale.const_data_ptr()), topk_ids.const_data_ptr(), ids_i64,
      topk_weights.const_data_ptr(), w_half, reinterpret_cast<half*>(output.mutable_data_ptr()), N, K, topk,
      (int)group_size);
}

TORCH_LIBRARY(glm5_skinny, m) {
  m.def("moe_skinny_int4_decode(Tensor input, Tensor w13, Tensor w13_scale, Tensor w2, Tensor w2_scale, "
        "Tensor topk_weights, Tensor topk_ids, Tensor(a!) act_buf, Tensor(b!) output, int group_size, "
        "float swiglu_limit) -> ()");
}
TORCH_LIBRARY_IMPL(glm5_skinny, CUDA, m) { m.impl("moe_skinny_int4_decode", &moe_skinny_int4_decode); }
