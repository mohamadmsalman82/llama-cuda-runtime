// Block-wide reductions used by the norm, softmax, and attention kernels.
#pragma once

#include <cuda_runtime.h>

#include <cfloat>

namespace lcr {

// The identity for an online softmax accumulator. Not -inf: two empty
// accumulators would then combine as (-inf) - (-inf), which is NaN, and a block
// with more threads than elements produces exactly that. -FLT_MAX subtracts
// from itself to zero and exponentiates to zero against any real score, so it
// behaves as an identity with no special case anywhere.
constexpr float kSoftmaxNegInf = -FLT_MAX;

constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(kFullMask, value, offset));
  }
  return value;
}

// Every thread in the block leaves with the total. `shared` must hold at least
// blockDim.x / 32 floats.
__device__ __forceinline__ float block_reduce_sum(float value, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = (blockDim.x + 31) >> 5;

  value = warp_reduce_sum(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();

  value = (threadIdx.x < warps) ? shared[threadIdx.x] : 0.0f;
  if (warp == 0) value = warp_reduce_sum(value);
  if (threadIdx.x == 0) shared[0] = value;
  __syncthreads();
  return shared[0];
}

// The online-softmax combine: two partial reductions, each with its own running
// maximum and exponential sum, merged onto the larger maximum. This is what
// lets attention accumulate over positions in one pass without ever holding the
// whole score row.
__device__ __forceinline__ void softmax_combine(float* m, float* l, float other_m,
                                                float other_l) {
  const float merged = fmaxf(*m, other_m);
  // exp(-inf - finite) is zero, which is exactly what an empty partial should
  // contribute, so the initial m = -inf needs no special case.
  *l = *l * __expf(*m - merged) + other_l * __expf(other_m - merged);
  *m = merged;
}

}  // namespace lcr
