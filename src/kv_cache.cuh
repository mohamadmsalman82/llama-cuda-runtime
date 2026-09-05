// Key/value cache, in a contiguous and a paged variant.
//
// Both store, per layer, the keys and values for every position generated so
// far, laid out head-major:
//
//     K[kv_head][position][head_dim]
//
// so that a decode step reading one head's entire history walks memory in a
// straight line. That is the single most bandwidth-sensitive read in the
// runtime after the weights themselves.
//
// The contiguous variant reserves the whole context window up front. For
// Llama-3.2-1B at its full 131072-token context that is 4.3 GB, twice the size
// of the model, almost all of it never touched. The paged variant cuts the
// context into fixed-size pages and maps them in as the sequence grows, so the
// only waste is the unused tail of the last page. Both are addressed by the
// same formula, with the page table folded away at compile time for the
// contiguous case.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "config.h"
#include "dtype.cuh"

namespace lcr {

enum class KvLayout { kContiguous, kPaged };

// The plain-old-data handle the kernels take. One layer's worth.
struct KvCacheView {
  // Non-const because the projection kernel writes through this same view; the
  // attention kernels only read.
  elem_t* keys = nullptr;
  elem_t* values = nullptr;
  // Logical page to physical page. Null for the contiguous layout, where the
  // whole layer is one page.
  const int32_t* page_table = nullptr;
  int page_size = 0;        // positions per page
  int page_stride = 0;      // elements between physical pages
  int head_stride = 0;      // elements between kv heads inside a page
  int num_kv_heads = 0;
  int head_dim = 0;

  // Offset of one position's vector for a given head. Identical arithmetic in
  // both layouts; the page table lookup is the only difference.
  template <bool kPaged>
  __device__ __forceinline__ size_t offset(int head, int position) const {
    const int page = kPaged ? page_table[position / page_size] : 0;
    const int slot = kPaged ? position % page_size : position;
    return static_cast<size_t>(page) * page_stride +
           static_cast<size_t>(head) * head_stride +
           static_cast<size_t>(slot) * head_dim;
  }
};

class KvCache {
 public:
  // `max_seq` is the longest sequence this cache will be asked to hold.
  // `page_size` is ignored for the contiguous layout.
  KvCache(const ModelConfig& config, int max_seq, KvLayout layout,
          int page_size = 16);
  ~KvCache();

  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;

  // Makes the cache able to hold `length` positions. Contiguous checks the
  // bound; paged maps in whatever new pages that needs.
  void reserve_length(int length);

  KvCacheView view(int layer) const;

  KvLayout layout() const { return layout_; }
  int page_size() const { return page_size_; }
  int max_seq() const { return max_seq_; }
  int length() const { return length_; }

  // Device memory this cache holds, whether or not it has been written.
  size_t allocated_bytes() const { return allocated_bytes_; }
  // Device memory backing positions the sequence has actually reached. The gap
  // between this and allocated_bytes is the waste the paged layout exists to
  // remove.
  size_t live_bytes() const;
  // What a perfect allocator would need for the current length, with no
  // reservation and no page rounding.
  size_t ideal_bytes() const;

  std::string usage_string() const;

  // Bytes a contiguous cache needs to hold `positions` positions, across every
  // layer and both K and V. Used by the report to price the same sequence under
  // the layout it is not using.
  static size_t bytes_for(const ModelConfig& config, int64_t positions);

 private:
  size_t layer_elements() const;

  ModelConfig config_;
  KvLayout layout_;
  int max_seq_ = 0;
  int page_size_ = 0;
  int length_ = 0;

  elem_t* key_pool_ = nullptr;
  elem_t* value_pool_ = nullptr;
  size_t allocated_bytes_ = 0;

  // Paged only. One table per layer, concatenated, with max_pages_ entries
  // each. Physical pages are handed out from a free counter.
  int32_t* page_tables_device_ = nullptr;
  std::vector<int32_t> page_tables_host_;
  int max_pages_ = 0;        // logical pages per layer
  int pool_pages_ = 0;       // physical pages in the pool
  int pages_mapped_ = 0;     // physical pages handed out so far
};

}  // namespace lcr
