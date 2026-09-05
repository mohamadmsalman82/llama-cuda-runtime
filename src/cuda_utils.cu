#include "cuda_utils.cuh"

#include <cstdlib>
#include <sstream>

namespace lcr {
namespace detail {

void throw_cuda_error(const char* file, int line, const char* expr,
                      cudaError_t status) {
  std::ostringstream oss;
  oss << file << ":" << line << ": CUDA call failed: " << expr << "\n  "
      << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
  throw Error(oss.str());
}

void throw_cublas_error(const char* file, int line, const char* expr,
                        cublasStatus_t status) {
  std::ostringstream oss;
  oss << file << ":" << line << ": cuBLAS call failed: " << expr << "\n  "
      << cublasGetStatusName(status) << ": " << cublasGetStatusString(status);
  throw Error(oss.str());
}

}  // namespace detail

bool synchronous_launches() {
  static const bool enabled = [] {
    const char* value = std::getenv("LCR_SYNC_KERNELS");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

double DeviceInfo::peak_bandwidth_bytes_per_second() const {
  // Memory clock in kHz, doubled for the double data rate, times the bus width
  // in bytes.
  return 2.0 * static_cast<double>(memory_clock_khz) * 1000.0 *
         (static_cast<double>(memory_bus_width_bits) / 8.0);
}

std::string DeviceInfo::summary() const {
  std::ostringstream oss;
  oss << name << " (sm_" << compute_major << compute_minor << ", "
      << multiprocessors << " SMs, " << format_bytes(total_memory) << ", "
      << memory_bus_width_bits << "-bit bus, "
      << peak_bandwidth_bytes_per_second() / 1e9 << " GB/s peak)";
  return oss.str();
}

DeviceInfo query_device(int index) {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  LCR_CHECK(count > 0, "no CUDA device is visible");
  LCR_CHECK(index >= 0 && index < count,
            "requested device " << index << " but only " << count
                                << " are visible");

  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, index));

  DeviceInfo info;
  info.index = index;
  info.name = props.name;
  info.compute_major = props.major;
  info.compute_minor = props.minor;
  info.multiprocessors = props.multiProcessorCount;
  info.total_memory = props.totalGlobalMem;
  info.memory_bus_width_bits = props.memoryBusWidth;
  info.memory_clock_khz = props.memoryClockRate;
  info.l2_cache_bytes = props.l2CacheSize;
  return info;
}

GpuTimer::GpuTimer() {
  CUDA_CHECK(cudaEventCreate(&start_));
  CUDA_CHECK(cudaEventCreate(&stop_));
}

GpuTimer::~GpuTimer() {
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}

void GpuTimer::start(cudaStream_t stream) {
  CUDA_CHECK(cudaEventRecord(start_, stream));
}

void GpuTimer::stop(cudaStream_t stream) {
  CUDA_CHECK(cudaEventRecord(stop_, stream));
}

double GpuTimer::elapsed_ms() const {
  CUDA_CHECK(cudaEventSynchronize(stop_));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
  return static_cast<double>(ms);
}

}  // namespace lcr
