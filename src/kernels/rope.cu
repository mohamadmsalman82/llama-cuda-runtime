// Rotary position embedding.
//
// Llama rotates pairs split across the two halves of a head: element i is
// paired with element i + head_dim/2, not with i + 1. That is the layout the
// published weights were converted into, so the pairing has to match exactly or
// every attention score comes out wrong in a way that still looks like plausible
// text.
//
//   out[i]             = q[i] * cos(a) - q[i + half] * sin(a)
//   out[i + half]      = q[i + half] * cos(a) + q[i] * sin(a)
//   a                  = position * inv_freq[i]
//
// Both kernels fold a layout change into the same pass. The query kernel
// transposes into head-major order for the attention GEMM, and the key/value
// kernel writes straight into the cache. Neither costs an extra read of memory
// that was not being read anyway.
//
// The angle is formed as a float multiply, matching the reference
// implementation bit for bit in its rounding. sincosf rather than __sincosf:
// at a position of 100k the fast intrinsic's range reduction is not good
// enough, and this kernel is nowhere near the critical path.
#include <algorithm>

#include "common.h"
#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace lcr {
namespace {

constexpr int kBlock = 64;

__global__ void rope_q_kernel(elem_t* __restrict__ q_heads,
                              const elem_t* __restrict__ q_proj,
                              const float* __restrict__ inv_freq, int tokens,
                              int start_position, int num_heads, int head_dim,
                              int src_row_stride) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int half = head_dim / 2;
  const float position = static_cast<float>(start_position + token);

  const int64_t src = static_cast<int64_t>(token) * src_row_stride +
                      static_cast<int64_t>(head) * head_dim;
  const int64_t dst =
      (static_cast<int64_t>(head) * tokens + token) * head_dim;

  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    float sine, cosine;
    sincosf(position * inv_freq[i], &sine, &cosine);
    const float lo = elem_to_float(q_proj[src + i]);
    const float hi = elem_to_float(q_proj[src + i + half]);
    q_heads[dst + i] = float_to_elem(lo * cosine - hi * sine);
    q_heads[dst + i + half] = float_to_elem(hi * cosine + lo * sine);
  }
}

template <bool kPaged>
__global__ void rope_write_kv_kernel(KvCacheView view,
                                     const elem_t* __restrict__ k_proj,
                                     const elem_t* __restrict__ v_proj,
                                     const float* __restrict__ inv_freq,
                                     int tokens, int start_position,
                                     int src_row_stride) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int head_dim = view.head_dim;
  const int half = head_dim / 2;
  const int position = start_position + token;

  const int64_t src = static_cast<int64_t>(token) * src_row_stride +
                      static_cast<int64_t>(head) * head_dim;
  const size_t dst = view.offset<kPaged>(head, position);

  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    float sine, cosine;
    sincosf(static_cast<float>(position) * inv_freq[i], &sine, &cosine);
    const float lo = elem_to_float(k_proj[src + i]);
    const float hi = elem_to_float(k_proj[src + i + half]);
    view.keys[dst + i] = float_to_elem(lo * cosine - hi * sine);
    view.keys[dst + i + half] = float_to_elem(hi * cosine + lo * sine);
  }
  // Values are not rotated, only relocated.
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    view.values[dst + i] = v_proj[src + i];
  }
}

}  // namespace

void launch_rope_q(elem_t* q_heads, const elem_t* q_proj, const float* inv_freq,
                   int tokens, int start_position, int num_heads, int head_dim,
                   int src_row_stride, cudaStream_t stream) {
  if (tokens == 0) return;
  const dim3 grid(static_cast<unsigned>(tokens),
                  static_cast<unsigned>(num_heads));
  const int block = std::min(kBlock, ((head_dim / 2 + 31) / 32) * 32);
  rope_q_kernel<<<grid, block, 0, stream>>>(q_heads, q_proj, inv_freq, tokens,
                                            start_position, num_heads, head_dim,
                                            src_row_stride);
  CUDA_CHECK_LAUNCH();
}

void launch_rope_write_kv(const KvCacheView& view, const elem_t* k_proj,
                          const elem_t* v_proj, const float* inv_freq,
                          int tokens, int start_position, bool paged,
                          int src_row_stride, cudaStream_t stream) {
  if (tokens == 0) return;
  const dim3 grid(static_cast<unsigned>(tokens),
                  static_cast<unsigned>(view.num_kv_heads));
  const int block = std::min(kBlock, ((view.head_dim + 31) / 32) * 32);
  if (paged) {
    rope_write_kv_kernel<true><<<grid, block, 0, stream>>>(
        view, k_proj, v_proj, inv_freq, tokens, start_position, src_row_stride);
  } else {
    rope_write_kv_kernel<false><<<grid, block, 0, stream>>>(
        view, k_proj, v_proj, inv_freq, tokens, start_position, src_row_stride);
  }
  CUDA_CHECK_LAUNCH();
}

}  // namespace lcr
