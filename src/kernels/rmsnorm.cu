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

// The residual add folded into the norm. The sum is formed once, kept in
// registers for the reduction, and written to both the residual stream and, in
// normalized form, to the output.
__global__ void add_residual_rmsnorm_kernel(elem_t* __restrict__ out,
                                            elem_t* __restrict__ x,
                                            const elem_t* __restrict__ delta,
                                            const elem_t* __restrict__ weight,
                                            int dim, float eps) {
  __shared__ float reduction[kBlock / 32];

  const int64_t row = static_cast<int64_t>(blockIdx.x) * dim;
  elem_t* x_row = x + row;
  const elem_t* delta_row = delta + row;
  elem_t* dst = out + row;
  const int vecs = dim / 8;

  float sum_squares = 0.0f;
  for (int i = threadIdx.x; i < vecs; i += kBlock) {
    ElemVec8 a = reinterpret_cast<ElemVec8*>(x_row)[i];
    const ElemVec8 b = reinterpret_cast<const ElemVec8*>(delta_row)[i];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = elem_to_float(a.v[j]) + elem_to_float(b.v[j]);
      a.v[j] = float_to_elem(v);
      sum_squares += v * v;
    }
    // The updated residual stream is needed by the next layer, so it is stored
    // here rather than recomputed.
    reinterpret_cast<ElemVec8*>(x_row)[i] = a;
  }
  for (int i = vecs * 8 + threadIdx.x; i < dim; i += kBlock) {
    const float v = elem_to_float(x_row[i]) + elem_to_float(delta_row[i]);
    x_row[i] = float_to_elem(v);
    sum_squares += v * v;
  }

  const float total = block_reduce_sum(sum_squares, reduction);
  const float scale = rsqrtf(total / static_cast<float>(dim) + eps);

  // Second pass over the row, which comes out of L2 since it was just written.
  for (int i = threadIdx.x; i < vecs; i += kBlock) {
    ElemVec8 chunk = reinterpret_cast<const ElemVec8*>(x_row)[i];
    const ElemVec8 gain = reinterpret_cast<const ElemVec8*>(weight)[i];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      chunk.v[j] = float_to_elem(elem_to_float(chunk.v[j]) * scale *
                                 elem_to_float(gain.v[j]));
    }
    reinterpret_cast<ElemVec8*>(dst)[i] = chunk;
  }
  for (int i = vecs * 8 + threadIdx.x; i < dim; i += kBlock) {
    dst[i] = float_to_elem(elem_to_float(x_row[i]) * scale *
                           elem_to_float(weight[i]));
  }
}

}  // namespace

void launch_add_residual_rmsnorm(elem_t* out, elem_t* x, const elem_t* delta,
                                 const elem_t* weight, int tokens, int dim,
                                 float eps, cudaStream_t stream) {
  if (tokens == 0) return;
  add_residual_rmsnorm_kernel<<<tokens, kBlock, 0, stream>>>(out, x, delta,
                                                             weight, dim, eps);
  CUDA_CHECK_LAUNCH();
}

void launch_rmsnorm(elem_t* out, const elem_t* x, const elem_t* weight,
                    int tokens, int dim, float eps, cudaStream_t stream) {
  if (tokens == 0) return;
  rmsnorm_kernel<<<tokens, kBlock, 0, stream>>>(out, x, weight, dim, eps);
  CUDA_CHECK_LAUNCH();
}

}  // namespace lcr
