// See cuda_runtime.h in this directory: compile-check stand-in, not a real bf16.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

struct alignas(2) __nv_bfloat16 {
  uint16_t __x = 0;
  __nv_bfloat16() = default;
  operator float() const { return 0.0f; }
};
using __nv_bfloat16_raw = __nv_bfloat16;

inline __nv_bfloat16 __float2bfloat16(float) { return __nv_bfloat16{}; }
inline float __bfloat162float(__nv_bfloat16) { return 0.0f; }
