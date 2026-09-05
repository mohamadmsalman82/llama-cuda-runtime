#include "arena.h"

#include <sstream>
#include <utility>

#include "cuda_utils.cuh"

namespace lcr {

DeviceArena::DeviceArena(size_t capacity_bytes) { reserve(capacity_bytes); }

DeviceArena::~DeviceArena() { free_block(); }

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_),
      capacity_(other.capacity_),
      used_(other.used_),
      peak_(other.peak_) {
  other.base_ = nullptr;
  other.capacity_ = 0;
  other.used_ = 0;
  other.peak_ = 0;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
  if (this != &other) {
    free_block();
    base_ = other.base_;
    capacity_ = other.capacity_;
    used_ = other.used_;
    peak_ = other.peak_;
    other.base_ = nullptr;
    other.capacity_ = 0;
    other.used_ = 0;
    other.peak_ = 0;
  }
  return *this;
}

void DeviceArena::free_block() {
  if (base_ != nullptr) {
    cudaFree(base_);
    base_ = nullptr;
  }
  capacity_ = 0;
  used_ = 0;
  peak_ = 0;
}

void DeviceArena::reserve(size_t capacity_bytes) {
  free_block();
  if (capacity_bytes == 0) return;
  void* block = nullptr;
  const cudaError_t status = cudaMalloc(&block, capacity_bytes);
  LCR_CHECK(status == cudaSuccess,
            "could not reserve an activation arena of "
                << format_bytes(capacity_bytes) << ": "
                << cudaGetErrorString(status));
  base_ = static_cast<char*>(block);
  capacity_ = capacity_bytes;
}

void* DeviceArena::allocate(size_t bytes, size_t alignment) {
  LCR_CHECK(alignment > 0 && (alignment & (alignment - 1)) == 0,
            "arena alignment must be a power of two, got " << alignment);
  const size_t offset = align_up(used_, alignment);
  LCR_CHECK(offset + bytes <= capacity_,
            "activation arena is full: need " << format_bytes(bytes)
                                              << " at offset "
                                              << format_bytes(offset)
                                              << " but capacity is "
                                              << format_bytes(capacity_));
  used_ = offset + bytes;
  if (used_ > peak_) peak_ = used_;
  return base_ + offset;
}

void DeviceArena::release(size_t mark) {
  LCR_CHECK(mark <= used_, "arena release to " << mark
                                               << " is past the current mark "
                                               << used_);
  used_ = mark;
}

std::string DeviceArena::usage_string() const {
  std::ostringstream oss;
  oss << format_bytes(peak_) << " peak of " << format_bytes(capacity_)
      << " reserved";
  return oss.str();
}

}  // namespace lcr
