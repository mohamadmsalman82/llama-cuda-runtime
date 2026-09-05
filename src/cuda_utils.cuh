// CUDA error checking, device properties, and timing.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "common.h"

namespace lcr {

namespace detail {
[[noreturn]] void throw_cuda_error(const char* file, int line, const char* expr,
                                   cudaError_t status);
[[noreturn]] void throw_cublas_error(const char* file, int line,
                                     const char* expr, cublasStatus_t status);
}  // namespace detail

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    const cudaError_t lcr_status_ = (expr);                                  \
    if (lcr_status_ != cudaSuccess) {                                        \
      ::lcr::detail::throw_cuda_error(__FILE__, __LINE__, #expr,             \
                                      lcr_status_);                          \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(expr)                                                   \
  do {                                                                       \
    const cublasStatus_t lcr_status_ = (expr);                               \
    if (lcr_status_ != CUBLAS_STATUS_SUCCESS) {                              \
      ::lcr::detail::throw_cublas_error(__FILE__, __LINE__, #expr,           \
                                        lcr_status_);                        \
    }                                                                        \
  } while (0)

// Catches launch configuration errors and, in debug builds, faults inside the
// kernel. Called after every launch.
#define CUDA_CHECK_LAUNCH()                                                  \
  do {                                                                       \
    CUDA_CHECK(cudaGetLastError());                                          \
    if (::lcr::synchronous_launches()) CUDA_CHECK(cudaDeviceSynchronize());  \
  } while (0)

// When true, every kernel launch is followed by a device synchronize so a fault
// is reported at the launch that caused it. Enabled by LCR_SYNC_KERNELS=1.
bool synchronous_launches();

struct DeviceInfo {
  int index = 0;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  int multiprocessors = 0;
  size_t total_memory = 0;
  int memory_bus_width_bits = 0;
  int memory_clock_khz = 0;
  int l2_cache_bytes = 0;

  // Theoretical peak HBM bandwidth in bytes per second, the denominator of the
  // headline decode number. Derived from the reported memory clock and bus
  // width, which the driver does not always report correctly on newer parts, so
  // the benchmark accepts an override.
  double peak_bandwidth_bytes_per_second() const;
  std::string summary() const;
};

DeviceInfo query_device(int index);

// A pair of CUDA events, for timing a region of the stream without stalling the
// host inside the region.
class GpuTimer {
 public:
  GpuTimer();
  ~GpuTimer();
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void start(cudaStream_t stream = nullptr);
  void stop(cudaStream_t stream = nullptr);
  // Blocks until the stop event has been reached.
  double elapsed_ms() const;

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

}  // namespace lcr
