// Root-mean-square normalization.
//
//   out[t] = x[t] * rsqrt(mean(x[t]^2) + eps) * weight
//
// One block per token. The row is read twice: once to accumulate the sum of
// squares, once to scale. The second read comes out of L2, which the row is
// still sitting in, so the extra pass costs far less than the shared memory it
// would take to avoid it, and it keeps the kernel correct for any hidden size.
#include <algorithm>

#include "common.h"
#include "cuda_utils.cuh"
#include "kernels.cuh"
#include "kernels/reduce.cuh"

namespace lcr {
namespace {

constexpr int kBlock = 256;

__global__ void rmsnorm_kernel(elem_t* __restrict__ out,
                               const elem_t* __restrict__ x,
                               const elem_t* __restrict__ weight, int dim,
                               float eps) {
  __shared__ float reduction[kBlock / 32];

  const int64_t row = static_cast<int64_t>(blockIdx.x) * dim;
  const elem_t* src = x + row;
  elem_t* dst = out + row;
  const int vecs = dim / 8;

  float sum_squares = 0.0f;
  for (int i = threadIdx.x; i < vecs; i += kBlock) {
    const ElemVec8 chunk = reinterpret_cast<const ElemVec8*>(src)[i];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = elem_to_float(chunk.v[j]);
      sum_squares += v * v;
    }
  }
  for (int i = vecs * 8 + threadIdx.x; i < dim; i += kBlock) {
    const float v = elem_to_float(src[i]);
    sum_squares += v * v;
  }

  const float total = block_reduce_sum(sum_squares, reduction);
  const float scale = rsqrtf(total / static_cast<float>(dim) + eps);

  for (int i = threadIdx.x; i < vecs; i += kBlock) {
    ElemVec8 chunk = reinterpret_cast<const ElemVec8*>(src)[i];
    const ElemVec8 gain = reinterpret_cast<const ElemVec8*>(weight)[i];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      chunk.v[j] = float_to_elem(elem_to_float(chunk.v[j]) * scale *
                                 elem_to_float(gain.v[j]));
    }
    reinterpret_cast<ElemVec8*>(dst)[i] = chunk;
  }
  for (int i = vecs * 8 + threadIdx.x; i < dim; i += kBlock) {
    dst[i] = float_to_elem(elem_to_float(src[i]) * scale *
                           elem_to_float(weight[i]));
  }
}

}  // namespace

void launch_rmsnorm(elem_t* out, const elem_t* x, const elem_t* weight,
                    int tokens, int dim, float eps, cudaStream_t stream) {
  if (tokens == 0) return;
  rmsnorm_kernel<<<tokens, kBlock, 0, stream>>>(out, x, weight, dim, eps);
  CUDA_CHECK_LAUNCH();
}

}  // namespace lcr
