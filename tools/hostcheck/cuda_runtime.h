// Host-only stand-ins for the CUDA runtime and device intrinsics.
//
// This directory exists for one reason: the project is developed on a machine
// with no NVIDIA toolkit, and 2000 lines of kernel code that has never been
// through a compiler is 2000 lines of guesses. tools/hostcheck.sh strips the
// <<<...>>> launch syntax and compiles every .cu file as ordinary C++17
// against these headers, which type-checks every kernel body, every template
// instantiation, and every API call.
//
// It is a compile check and nothing more. The functions here return success
// without doing anything and the intrinsics return their input. Nothing in this
// directory is ever linked into the real build, and passing it says only that
// the code compiles, never that it computes the right answer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#define __global__
#define __device__
#define __host__
#define __constant__
#define __shared__
#define __forceinline__ inline
#define __launch_bounds__(...)

struct uint3 {
  unsigned x = 0, y = 0, z = 0;
};
struct uint2 {
  unsigned x = 0, y = 0;
};
struct alignas(16) uint4 {
  unsigned x = 0, y = 0, z = 0, w = 0;
};
struct dim3 {
  unsigned x = 1, y = 1, z = 1;
  dim3() = default;
  dim3(unsigned x_, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

extern uint3 threadIdx;
extern uint3 blockIdx;
extern dim3 blockDim;
extern dim3 gridDim;

inline void __syncthreads() {}
inline void __threadfence_block() {}

template <typename T>
T __shfl_xor_sync(unsigned mask, T value, int lane, int width = 32) {
  (void)mask;
  (void)lane;
  (void)width;
  return value;
}
template <typename T>
T __shfl_sync(unsigned mask, T value, int src, int width = 32) {
  (void)mask;
  (void)src;
  (void)width;
  return value;
}

inline float __expf(float v) { return v; }
inline float __logf(float v) { return v; }
inline float rsqrtf(float v) { return v; }
inline void sincosf(float angle, float* sine, float* cosine) {
  *sine = angle;
  *cosine = angle;
}
inline unsigned __float_as_uint(float v) {
  unsigned bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}
inline float __uint_as_float(unsigned bits) {
  float v;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

// CUDA makes these available unqualified in device code; the host compiler does
// not, so the overloads the kernels actually use are provided here.
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned long long max(unsigned long long a, unsigned long long b) {
  return a > b ? a : b;
}

inline int atomicAdd(int* address, int value) {
  const int old = *address;
  *address += value;
  return old;
}
inline unsigned long long atomicMax(unsigned long long* address,
                                    unsigned long long value) {
  const unsigned long long old = *address;
  if (value > old) *address = value;
  return old;
}

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;

enum cudaMemcpyKind {
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice,
  cudaMemcpyHostToHost,
};

struct CUstream_st;
struct CUevent_st;
using cudaStream_t = CUstream_st*;
using cudaEvent_t = CUevent_st*;

struct cudaDeviceProp {
  char name[256] = {};
  int major = 0;
  int minor = 0;
  int multiProcessorCount = 0;
  size_t totalGlobalMem = 0;
  int memoryBusWidth = 0;
  int memoryClockRate = 0;
  int l2CacheSize = 0;
};

inline const char* cudaGetErrorName(cudaError_t) { return "cudaSuccess"; }
inline const char* cudaGetErrorString(cudaError_t) { return "no error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* count) {
  *count = 1;
  return cudaSuccess;
}
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int) {
  return cudaSuccess;
}
inline cudaError_t cudaMalloc(void** pointer, size_t) {
  *pointer = nullptr;
  return cudaSuccess;
}
template <typename T>
cudaError_t cudaMalloc(T** pointer, size_t) {
  *pointer = nullptr;
  return cudaSuccess;
}
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) {
  return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind,
                                   cudaStream_t) {
  return cudaSuccess;
}
inline cudaError_t cudaMemset(void*, int, size_t) { return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t) {
  return cudaSuccess;
}
inline cudaError_t cudaStreamCreate(cudaStream_t*) { return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t*) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) {
  return cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
  *ms = 0.0f;
  return cudaSuccess;
}

enum cudaDataType_t {
  CUDA_R_16F,
  CUDA_R_16BF,
  CUDA_R_32F,
};
using cudaDataType = cudaDataType_t;
