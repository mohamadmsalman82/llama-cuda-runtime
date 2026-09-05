// See cuda_runtime.h in this directory: compile-check stand-in, not a real fp16.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

struct alignas(2) __half {
  uint16_t __x = 0;
  __half() = default;
  operator float() const { return 0.0f; }
};
using __half_raw = __half;

inline __half __float2half(float) { return __half{}; }
inline float __half2float(__half) { return 0.0f; }
