// Per-stage timing for a forward pass, built on CUDA events.
//
// Nsight Compute is the usual tool for this, but it needs GPU performance
// counters, which are unavailable inside most containers: the permission is set
// by a kernel module parameter on the host. Rather than depend on privileges
// the project may not have, the profile is built in, so anyone who clones the
// repo can reproduce it with a flag.
//
// The mechanism is a pair of CUDA events around each stage, accumulated by
// name across every layer. Events cost a microsecond or two to record, so the
// profile perturbs the very thing it measures; the totals it prints are
// therefore compared against an unprofiled run rather than trusted on their
// own. That gap is itself the interesting number, because it is dominated by
// per-launch overhead.
#pragma once

#include <cuda_runtime.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace lcr {

class Profiler {
 public:
  ~Profiler();

  void set_enabled(bool enabled) { enabled_ = enabled; }
  bool enabled() const { return enabled_; }

  // Marks the start and end of a named stage on `stream`. Nesting is not
  // supported; each begin must be followed by its end.
  void begin(const std::string& name, cudaStream_t stream);
  void end(cudaStream_t stream);

  // Blocks on every recorded event and folds the timings into the totals.
  // Called once per forward pass, after the stream has been synchronized.
  void collect();

  struct Entry {
    std::string name;
    double total_ms = 0.0;
    int launches = 0;
  };
  // Sorted by total time, descending.
  std::vector<Entry> entries() const;
  double total_ms() const;
  int total_launches() const;
  void reset();

 private:
  struct Span {
    std::string name;
    cudaEvent_t start;
    cudaEvent_t stop;
  };
  cudaEvent_t acquire_event();

  bool enabled_ = false;
  std::vector<Span> spans_;
  std::vector<cudaEvent_t> pool_;
  size_t pool_used_ = 0;
  std::unordered_map<std::string, Entry> totals_;
};

// Scoped stage marker, so a stage cannot be left open by an early return.
class ProfileScope {
 public:
  ProfileScope(Profiler* profiler, const char* name, cudaStream_t stream)
      : profiler_(profiler), stream_(stream) {
    if (profiler_ != nullptr && profiler_->enabled()) {
      profiler_->begin(name, stream_);
      active_ = true;
    }
  }
  ~ProfileScope() {
    if (active_) profiler_->end(stream_);
  }
  ProfileScope(const ProfileScope&) = delete;
  ProfileScope& operator=(const ProfileScope&) = delete;

 private:
  Profiler* profiler_;
  cudaStream_t stream_;
  bool active_ = false;
};

}  // namespace lcr
