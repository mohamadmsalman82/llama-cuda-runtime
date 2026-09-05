#include "profiler.h"

#include <algorithm>

#include "cuda_utils.cuh"

namespace lcr {

Profiler::~Profiler() {
  for (cudaEvent_t event : pool_) cudaEventDestroy(event);
}

cudaEvent_t Profiler::acquire_event() {
  // Events are reused across passes: creating one costs far more than
  // recording it, and a decode step needs a few hundred.
  if (pool_used_ == pool_.size()) {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    pool_.push_back(event);
  }
  return pool_[pool_used_++];
}

void Profiler::begin(const std::string& name, cudaStream_t stream) {
  Span span;
  span.name = name;
  span.start = acquire_event();
  span.stop = nullptr;
  CUDA_CHECK(cudaEventRecord(span.start, stream));
  spans_.push_back(std::move(span));
}

void Profiler::end(cudaStream_t stream) {
  LCR_CHECK(!spans_.empty() && spans_.back().stop == nullptr,
            "Profiler::end without a matching begin");
  spans_.back().stop = acquire_event();
  CUDA_CHECK(cudaEventRecord(spans_.back().stop, stream));
}

void Profiler::collect() {
  for (const Span& span : spans_) {
    if (span.stop == nullptr) continue;
    CUDA_CHECK(cudaEventSynchronize(span.stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, span.start, span.stop));
    Entry& entry = totals_[span.name];
    entry.name = span.name;
    entry.total_ms += static_cast<double>(ms);
    entry.launches += 1;
  }
  spans_.clear();
  pool_used_ = 0;
}

std::vector<Profiler::Entry> Profiler::entries() const {
  std::vector<Entry> out;
  out.reserve(totals_.size());
  for (const auto& [name, entry] : totals_) out.push_back(entry);
  std::sort(out.begin(), out.end(), [](const Entry& a, const Entry& b) {
    return a.total_ms > b.total_ms;
  });
  return out;
}

double Profiler::total_ms() const {
  double total = 0.0;
  for (const auto& [name, entry] : totals_) total += entry.total_ms;
  return total;
}

int Profiler::total_launches() const {
  int total = 0;
  for (const auto& [name, entry] : totals_) total += entry.launches;
  return total;
}

void Profiler::reset() {
  totals_.clear();
  spans_.clear();
  pool_used_ = 0;
}

}  // namespace lcr
