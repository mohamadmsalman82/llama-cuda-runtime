// The element type used for weights and activations.
//
// The runtime computes in one 16-bit type chosen at build time, matching the
// checkpoint. bf16 is the default because that is what Llama 3.2 ships in;
// -DLCR_DTYPE=fp16 switches the whole pipeline over. Keeping it a compile-time
// choice means one code path rather than a templated model with a runtime
// dispatch on every kernel launch.
//
// Reductions never accumulate in 16 bits. Every kernel loads to float, does its
// arithmetic there, and narrows on the way out, which is also what the PyTorch
// reference does.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "common.h"

namespace lcr {

#if defined(LCR_ELEM_FP16)

using elem_t = __half;
constexpr DType kElemDType = DType::kF16;
constexpr const char* kElemName = "fp16";

__device__ __forceinline__ float elem_to_float(elem_t x) { return __half2float(x); }
__device__ __forceinline__ elem_t float_to_elem(float v) { return __float2half(v); }

#else

using elem_t = __nv_bfloat16;
constexpr DType kElemDType = DType::kBF16;
constexpr const char* kElemName = "bf16";

__device__ __forceinline__ float elem_to_float(elem_t x) {
  return __bfloat162float(x);
}
__device__ __forceinline__ elem_t float_to_elem(float v) {
  return __float2bfloat16(v);
}

#endif

// A 16-byte aligned block of eight elements. Every kernel here is bandwidth
// bound, so the streaming loops move data in 128-bit chunks rather than one
// 16-bit element at a time; that is the difference between saturating the
// memory bus and leaving half of it idle.
struct alignas(16) ElemVec8 {
  elem_t v[8];
};
static_assert(sizeof(ElemVec8) == 16, "ElemVec8 must be one 128-bit access");

// Reads kCount consecutive elements as one wide access and widens them to
// float. The attention kernel calls this once per position per thread, so the
// difference between one 32-bit load and two 16-bit loads shows up directly in
// achieved bandwidth.
template <int kCount>
__device__ __forceinline__ void load_elems(const elem_t* p, float* out) {
  if constexpr (kCount == 2) {
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(p);
    const elem_t* pair = reinterpret_cast<const elem_t*>(&packed);
    out[0] = elem_to_float(pair[0]);
    out[1] = elem_to_float(pair[1]);
  } else if constexpr (kCount == 4) {
    const uint2 packed = *reinterpret_cast<const uint2*>(p);
    const elem_t* quad = reinterpret_cast<const elem_t*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = elem_to_float(quad[i]);
  } else if constexpr (kCount == 8) {
    const uint4 packed = *reinterpret_cast<const uint4*>(p);
    const elem_t* oct = reinterpret_cast<const elem_t*>(&packed);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = elem_to_float(oct[i]);
  } else {
#pragma unroll
    for (int i = 0; i < kCount; ++i) out[i] = elem_to_float(p[i]);
  }
}

}  // namespace lcr
