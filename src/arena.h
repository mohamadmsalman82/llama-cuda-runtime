// Bump allocator for device activations.
//
// Every decode step touches the same activation shapes, so allocating them once
// and handing out offsets removes cudaMalloc from the hot loop entirely.
// cudaMalloc synchronizes the device, which at a few hundred microseconds a
// call would dominate a token that should take under a millisecond.
//
// The model carves its persistent buffers out of the arena at load time and
// then uses a scoped mark/release for the per-layer temporaries, so peak usage
// is the largest single layer rather than the sum of all of them.
#pragma once

#include <cstddef>
#include <string>

#include "common.h"

namespace lcr {

class DeviceArena {
 public:
  DeviceArena() = default;
  explicit DeviceArena(size_t capacity_bytes);
  ~DeviceArena();

  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;
  DeviceArena(DeviceArena&& other) noexcept;
  DeviceArena& operator=(DeviceArena&& other) noexcept;

  void reserve(size_t capacity_bytes);

  // Alignment defaults to 256 bytes: enough for the 128-bit vector loads the
  // kernels use and for cuBLAS to take its aligned paths.
  void* allocate(size_t bytes, size_t alignment = 256);

  template <typename T>
  T* alloc(size_t count) {
    return static_cast<T*>(allocate(count * sizeof(T), 256));
  }

  // Position in the arena, for scoping temporaries.
  size_t mark() const { return used_; }
  void release(size_t mark);
  void reset() { release(0); }

  size_t used() const { return used_; }
  size_t peak() const { return peak_; }
  size_t capacity() const { return capacity_; }
  std::string usage_string() const;

 private:
  void free_block();

  char* base_ = nullptr;
  size_t capacity_ = 0;
  size_t used_ = 0;
  size_t peak_ = 0;
};

// Releases back to the mark it was created at, so a block of temporaries costs
// one line to scope.
class ArenaScope {
 public:
  explicit ArenaScope(DeviceArena& arena) : arena_(arena), mark_(arena.mark()) {}
  ~ArenaScope() { arena_.release(mark_); }
  ArenaScope(const ArenaScope&) = delete;
  ArenaScope& operator=(const ArenaScope&) = delete;

 private:
  DeviceArena& arena_;
  size_t mark_;
};

}  // namespace lcr
