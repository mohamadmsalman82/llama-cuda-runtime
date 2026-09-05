#include "kv_cache.cuh"

#include <algorithm>
#include <sstream>

#include "cuda_utils.cuh"

namespace lcr {
namespace {

int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace

size_t KvCache::bytes_for(const ModelConfig& config, int64_t positions) {
  return static_cast<size_t>(positions) * config.kv_bytes_per_token(sizeof(elem_t));
}

size_t KvCache::layer_elements() const {
  return static_cast<size_t>(config_.num_kv_heads) *
         static_cast<size_t>(max_seq_) * static_cast<size_t>(config_.head_dim);
}

KvCache::KvCache(const ModelConfig& config, int max_seq, KvLayout layout,
                 int page_size)
    : config_(config), layout_(layout), max_seq_(max_seq) {
  LCR_CHECK(max_seq > 0, "KV cache needs a positive maximum sequence length");

  size_t key_elements = 0;
  if (layout_ == KvLayout::kContiguous) {
    page_size_ = max_seq_;
    key_elements = static_cast<size_t>(config_.num_layers) * layer_elements();
  } else {
    LCR_CHECK(page_size > 0, "paged KV cache needs a positive page size");
    page_size_ = page_size;
    max_pages_ = ceil_div(max_seq_, page_size_);
    // The pool is sized so a sequence of max_seq positions can never fail to
    // find a page. With one sequence in flight that is the same total as the
    // contiguous layout; the saving being measured is per sequence, in how much
    // of it is ever touched.
    pool_pages_ = max_pages_ * config_.num_layers;
    const size_t page_elements = static_cast<size_t>(config_.num_kv_heads) *
                                 static_cast<size_t>(page_size_) *
                                 static_cast<size_t>(config_.head_dim);
    key_elements = static_cast<size_t>(pool_pages_) * page_elements;

    page_tables_host_.assign(
        static_cast<size_t>(config_.num_layers) * max_pages_, -1);
    CUDA_CHECK(cudaMalloc(&page_tables_device_,
                          page_tables_host_.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(page_tables_device_, page_tables_host_.data(),
                          page_tables_host_.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
  }

  const size_t bytes = key_elements * sizeof(elem_t);
  CUDA_CHECK(cudaMalloc(&key_pool_, bytes));
  CUDA_CHECK(cudaMalloc(&value_pool_, bytes));
  // Zeroing is not needed for correctness, since attention only reads positions
  // that have been written, but it makes a stray read produce zeros rather than
  // whatever the last process left in that memory.
  CUDA_CHECK(cudaMemset(key_pool_, 0, bytes));
  CUDA_CHECK(cudaMemset(value_pool_, 0, bytes));
  allocated_bytes_ = 2 * bytes + page_tables_host_.size() * sizeof(int32_t);
}

KvCache::~KvCache() {
  cudaFree(key_pool_);
  cudaFree(value_pool_);
  cudaFree(page_tables_device_);
}

void KvCache::reserve_length(int length) {
  LCR_CHECK(length <= max_seq_,
            "sequence reached " << length
                                << " positions but the KV cache was built for "
                                << max_seq_);
  if (length <= length_) return;

  if (layout_ == KvLayout::kPaged) {
    const int have = ceil_div(length_, page_size_);
    const int need = ceil_div(length, page_size_);
    if (need > have) {
      for (int layer = 0; layer < config_.num_layers; ++layer) {
        int32_t* table = page_tables_host_.data() +
                         static_cast<size_t>(layer) * max_pages_;
        for (int page = have; page < need; ++page) {
          LCR_CHECK(pages_mapped_ < pool_pages_,
                    "paged KV cache pool is exhausted after "
                        << pages_mapped_ << " pages");
          table[page] = pages_mapped_++;
        }
        // Only the entries that just changed have to reach the device.
        CUDA_CHECK(cudaMemcpy(
            page_tables_device_ + static_cast<size_t>(layer) * max_pages_ + have,
            table + have, static_cast<size_t>(need - have) * sizeof(int32_t),
            cudaMemcpyHostToDevice));
      }
    }
  }
  length_ = length;
}

KvCacheView KvCache::view(int layer) const {
  LCR_CHECK(layer >= 0 && layer < config_.num_layers,
            "layer " << layer << " is out of range");
  KvCacheView v;
  v.num_kv_heads = config_.num_kv_heads;
  v.head_dim = config_.head_dim;
  v.page_size = page_size_;

  if (layout_ == KvLayout::kContiguous) {
    const size_t base = static_cast<size_t>(layer) * layer_elements();
    v.keys = key_pool_ + base;
    v.values = value_pool_ + base;
    v.page_table = nullptr;
    v.page_stride = 0;
    v.head_stride = max_seq_ * config_.head_dim;
  } else {
    v.keys = key_pool_;
    v.values = value_pool_;
    v.page_table = page_tables_device_ + static_cast<size_t>(layer) * max_pages_;
    v.page_stride = config_.num_kv_heads * page_size_ * config_.head_dim;
    v.head_stride = page_size_ * config_.head_dim;
  }
  return v;
}

size_t KvCache::ideal_bytes() const { return bytes_for(config_, length_); }

size_t KvCache::live_bytes() const {
  if (layout_ == KvLayout::kContiguous) {
    // A contiguous cache commits the whole window the moment it is created.
    return bytes_for(config_, max_seq_);
  }
  // Only the pages that have been handed out are touched. The tail of the last
  // page of each layer is the entire waste.
  const size_t page_bytes = static_cast<size_t>(config_.num_kv_heads) *
                            static_cast<size_t>(page_size_) *
                            static_cast<size_t>(config_.head_dim) *
                            sizeof(elem_t) * 2;
  return static_cast<size_t>(pages_mapped_) * page_bytes;
}

std::string KvCache::usage_string() const {
  std::ostringstream oss;
  const size_t live = live_bytes();
  const size_t ideal = ideal_bytes();
  oss << (layout_ == KvLayout::kContiguous ? "contiguous" : "paged");
  if (layout_ == KvLayout::kPaged) oss << " (page=" << page_size_ << ")";
  oss << ": " << length_ << " positions, " << format_bytes(live)
      << " resident, " << format_bytes(ideal) << " strictly needed";
  if (live > ideal) {
    oss << ", " << format_bytes(live - ideal) << " wasted ("
        << (100.0 * static_cast<double>(live - ideal) /
            static_cast<double>(live))
        << "%)";
  }
  return oss.str();
}

}  // namespace lcr
