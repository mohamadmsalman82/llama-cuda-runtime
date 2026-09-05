// Every hand-written kernel against a plain C++ reference of the same maths.
//
// The references here are deliberately slow and obvious: nested loops, float
// accumulators, no blocking. If the readable version and the fast version
// disagree, the fast one is wrong. Inputs are rounded through the storage type
// before the reference sees them, so the only difference left to measure is
// what the kernel does, not what the format costs.
//
// This binary needs a GPU. It is registered with CTest and is skipped by
// configurations built without CUDA.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <sstream>
#include <vector>

#include "arena.h"
#include "config.h"
#include "cuda_utils.cuh"
#include "kernels.cuh"
#include "kv_cache.cuh"
#include "test_util.h"

namespace {

using lcr::elem_t;
using lcr::elem_to_float;
using lcr::float_to_elem;

// Rounding to the storage type costs about one part in 256 for bf16. A kernel
// that accumulates in fp32 and stores once should land inside a few multiples
// of that; anything larger is a real disagreement, not a rounding artefact.
constexpr double kTolerance = 3e-2;

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    CUDA_CHECK(cudaMemset(data_, 0, count * sizeof(T)));
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() { return data_; }
  const T* get() const { return data_; }

  void upload(const std::vector<T>& host) {
    CUDA_CHECK(cudaMemcpy(data_, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> host(count_);
    CUDA_CHECK(cudaMemcpy(host.data(), data_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return host;
  }

 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

std::mt19937 rng(20260904);

std::vector<float> random_floats(size_t count, float scale = 1.0f) {
  std::normal_distribution<float> normal(0.0f, scale);
  std::vector<float> values(count);
  for (float& value : values) value = normal(rng);
  return values;
}

// Rounds a float vector through the storage type, so the reference and the
// kernel start from bit-identical inputs.
std::vector<elem_t> to_elems(const std::vector<float>& values) {
  std::vector<elem_t> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) out[i] = float_to_elem(values[i]);
  return out;
}

std::vector<float> to_floats(const std::vector<elem_t>& values) {
  std::vector<float> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) out[i] = elem_to_float(values[i]);
  return out;
}

// Compares against the magnitude of the expected signal rather than element by
// element, so a near-zero expected value does not create a spurious failure.
void expect_close(const char* what, const std::vector<float>& ours,
                  const std::vector<float>& expected,
                  double tolerance = kTolerance) {
  if (ours.size() != expected.size()) {
    std::ostringstream oss;
    oss << what << ": got " << ours.size() << " values, expected "
        << expected.size();
    ::test::report(false, __FILE__, __LINE__, oss.str());
    return;
  }
  double scale = 0.0;
  for (float value : expected) scale = std::max(scale, std::fabs((double)value));
  scale = std::max(scale, 1e-6);

  double worst = 0.0;
  size_t worst_index = 0;
  for (size_t i = 0; i < ours.size(); ++i) {
    const double error = std::fabs((double)ours[i] - (double)expected[i]) / scale;
    if (error > worst) {
      worst = error;
      worst_index = i;
    }
  }
  if (worst > tolerance) {
    std::ostringstream oss;
    oss << what << ": worst relative error " << worst << " at index "
        << worst_index << " (got " << ours[worst_index] << ", expected "
        << expected[worst_index] << ", tolerance " << tolerance << ")";
    ::test::report(false, __FILE__, __LINE__, oss.str());
  }
}

// ---------------------------------------------------------------------------

void test_rmsnorm() {
  const int tokens = 5;
  const int dim = 2048;
  const std::vector<elem_t> x = to_elems(random_floats(tokens * dim));
  const std::vector<elem_t> weight = to_elems(random_floats(dim, 0.5f));
  const std::vector<float> xf = to_floats(x);
  const std::vector<float> wf = to_floats(weight);
  const float eps = 1e-5f;

  std::vector<float> expected(xf.size());
  for (int t = 0; t < tokens; ++t) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double v = xf[t * dim + i];
      sum += v * v;
    }
    const float scale = 1.0f / std::sqrt(sum / dim + eps);
    for (int i = 0; i < dim; ++i) {
      expected[t * dim + i] = xf[t * dim + i] * scale * wf[i];
    }
  }

  DeviceBuffer<elem_t> dx(x.size()), dw(weight.size()), dout(x.size());
  dx.upload(x);
  dw.upload(weight);
  lcr::launch_rmsnorm(dout.get(), dx.get(), dw.get(), tokens, dim, eps, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  expect_close("rmsnorm", to_floats(dout.download()), expected);
}

void test_swiglu_and_residual() {
  const int64_t count = 8192 * 3 + 5;  // deliberately not a multiple of eight
  const std::vector<elem_t> gate = to_elems(random_floats(count));
  const std::vector<elem_t> up = to_elems(random_floats(count));
  const std::vector<float> gf = to_floats(gate);
  const std::vector<float> uf = to_floats(up);

  std::vector<float> expected(count);
  for (int64_t i = 0; i < count; ++i) {
    const float g = gf[i];
    expected[i] = (g / (1.0f + std::exp(-g))) * uf[i];
  }

  DeviceBuffer<elem_t> dg(count), du(count), dout(count);
  dg.upload(gate);
  du.upload(up);
  lcr::launch_swiglu(dout.get(), dg.get(), du.get(), count, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  expect_close("swiglu", to_floats(dout.download()), expected);

  // The residual add shares the vectorized-with-scalar-tail structure, so the
  // odd length matters here too.
  std::vector<float> sum(count);
  for (int64_t i = 0; i < count; ++i) sum[i] = gf[i] + uf[i];
  dg.upload(gate);
  lcr::launch_add_residual(dg.get(), du.get(), count, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  expect_close("add_residual", to_floats(dg.download()), sum);
}

void test_embedding_lookup() {
  const int vocab = 128;
  const int dim = 2048;
  const int tokens = 4;
  const std::vector<elem_t> table = to_elems(random_floats(vocab * dim));
  const std::vector<int32_t> ids{7, 0, 127, 63};

  std::vector<float> expected(tokens * dim);
  const std::vector<float> tf = to_floats(table);
  for (int t = 0; t < tokens; ++t) {
    for (int i = 0; i < dim; ++i) {
      expected[t * dim + i] = tf[static_cast<size_t>(ids[t]) * dim + i];
    }
  }

  DeviceBuffer<elem_t> dtable(table.size()), dout(tokens * dim);
  DeviceBuffer<int32_t> dids(ids.size());
  dtable.upload(table);
  dids.upload(ids);
  lcr::launch_embedding_lookup(dout.get(), dtable.get(), dids.get(), tokens, dim,
                               nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  expect_close("embedding_lookup", to_floats(dout.download()), expected, 0.0);
}

// The rotary pairing is the single easiest thing to get wrong in a Llama
// implementation, and getting it wrong still produces fluent text, so the
// reference here spells the convention out: element i pairs with i + head_dim/2,
// not with i + 1.
void test_rope_query() {
  const int tokens = 3;
  const int heads = 4;
  const int head_dim = 64;
  const int start = 11;
  const int half = head_dim / 2;

  const std::vector<elem_t> q = to_elems(random_floats(tokens * heads * head_dim));
  const std::vector<float> qf = to_floats(q);
  std::vector<float> inv_freq(half);
  for (int i = 0; i < half; ++i) {
    inv_freq[i] = std::pow(500000.0f, -2.0f * i / head_dim);
  }

  // Reference, in head-major order because the kernel transposes as it rotates.
  std::vector<float> expected(tokens * heads * head_dim);
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < heads; ++h) {
      const size_t src = (static_cast<size_t>(t) * heads + h) * head_dim;
      const size_t dst = (static_cast<size_t>(h) * tokens + t) * head_dim;
      for (int i = 0; i < half; ++i) {
        const float angle = static_cast<float>(start + t) * inv_freq[i];
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float lo = qf[src + i];
        const float hi = qf[src + i + half];
        expected[dst + i] = lo * c - hi * s;
        expected[dst + i + half] = hi * c + lo * s;
      }
    }
  }

  DeviceBuffer<elem_t> dq(q.size()), dout(q.size());
  DeviceBuffer<float> dfreq(inv_freq.size());
  dq.upload(q);
  dfreq.upload(inv_freq);
  lcr::launch_rope_q(dout.get(), dq.get(), dfreq.get(), tokens, start, heads,
                     head_dim, heads * head_dim, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  expect_close("rope_q", to_floats(dout.download()), expected);
}

void test_causal_softmax() {
  const int heads = 3;
  const int rows = 6;
  const int keys = 20;
  const int start = 8;  // rows are query positions 8 through 13

  const std::vector<float> scores = random_floats(heads * rows * keys, 4.0f);

  std::vector<float> expected(scores.size(), 0.0f);
  for (int h = 0; h < heads; ++h) {
    for (int r = 0; r < rows; ++r) {
      const size_t base = (static_cast<size_t>(h) * rows + r) * keys;
      const int limit = std::min(keys, start + r + 1);
      float max_score = -INFINITY;
      for (int j = 0; j < limit; ++j) max_score = std::max(max_score, scores[base + j]);
      float sum = 0.0f;
      for (int j = 0; j < limit; ++j) sum += std::exp(scores[base + j] - max_score);
      for (int j = 0; j < limit; ++j) {
        expected[base + j] = std::exp(scores[base + j] - max_score) / sum;
      }
      // Everything at or beyond the query's own position stays zero.
    }
  }

  DeviceBuffer<float> dscores(scores.size());
  DeviceBuffer<elem_t> dprobs(scores.size());
  dscores.upload(scores);
  lcr::launch_causal_softmax(dprobs.get(), dscores.get(), heads, rows, keys,
                             start, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> ours = to_floats(dprobs.download());
  expect_close("causal_softmax", ours, expected);

  // Each unmasked row must sum to one, and masked entries must be exactly zero
  // rather than merely small.
  for (int h = 0; h < heads; ++h) {
    for (int r = 0; r < rows; ++r) {
      const size_t base = (static_cast<size_t>(h) * rows + r) * keys;
      const int limit = std::min(keys, start + r + 1);
      double total = 0.0;
      for (int j = 0; j < keys; ++j) {
        if (j >= limit) CHECK_EQ(ours[base + j], 0.0f);
        total += ours[base + j];
      }
      CHECK_NEAR(total, 1.0, 1e-2);
    }
  }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

lcr::ModelConfig small_config(int num_heads, int num_kv_heads, int head_dim) {
  lcr::ModelConfig config;
  config.hidden_size = num_heads * head_dim;
  config.intermediate_size = 4 * config.hidden_size;
  config.num_layers = 1;
  config.num_heads = num_heads;
  config.num_kv_heads = num_kv_heads;
  config.head_dim = head_dim;
  config.vocab_size = 32;
  config.max_position_embeddings = 4096;
  return config;
}

// Fills a cache through the real write path, with the rotary frequencies set to
// zero so the rotation is the identity and the values land unchanged.
void fill_cache(lcr::KvCache* cache, const std::vector<elem_t>& keys,
                const std::vector<elem_t>& values, int positions, int kv_dim,
                bool paged) {
  DeviceBuffer<elem_t> dk(keys.size()), dv(values.size());
  DeviceBuffer<float> dfreq(static_cast<size_t>(kv_dim));
  dk.upload(keys);
  dv.upload(values);
  const std::vector<float> zeros(static_cast<size_t>(kv_dim), 0.0f);
  dfreq.upload(zeros);

  cache->reserve_length(positions);
  lcr::launch_rope_write_kv(cache->view(0), dk.get(), dv.get(), dfreq.get(),
                            positions, 0, paged, kv_dim, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<float> reference_attention(const std::vector<float>& q,
                                       const std::vector<float>& keys,
                                       const std::vector<float>& values,
                                       int num_heads, int num_kv_heads,
                                       int head_dim, int positions) {
  const int heads_per_kv = num_heads / num_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> out(static_cast<size_t>(num_heads) * head_dim, 0.0f);

  for (int h = 0; h < num_heads; ++h) {
    const int kv = h / heads_per_kv;
    std::vector<float> scores(positions);
    for (int p = 0; p < positions; ++p) {
      double dot = 0.0;
      for (int d = 0; d < head_dim; ++d) {
        // The cache is written from a [positions][num_kv_heads * head_dim]
        // projection, so that is the layout the reference indexes.
        dot += q[static_cast<size_t>(h) * head_dim + d] *
               keys[(static_cast<size_t>(p) * num_kv_heads + kv) * head_dim + d];
      }
      scores[p] = static_cast<float>(dot) * scale;
    }
    float max_score = -INFINITY;
    for (float s : scores) max_score = std::max(max_score, s);
    double sum = 0.0;
    for (float& s : scores) {
      s = std::exp(s - max_score);
      sum += s;
    }
    for (int d = 0; d < head_dim; ++d) {
      double acc = 0.0;
      for (int p = 0; p < positions; ++p) {
        acc += scores[p] *
               values[(static_cast<size_t>(p) * num_kv_heads + kv) * head_dim + d];
      }
      out[static_cast<size_t>(h) * head_dim + d] =
          static_cast<float>(acc / sum);
    }
  }
  return out;
}

void run_decode_attention_case(int num_heads, int num_kv_heads, int head_dim,
                               int positions, int splits, bool paged,
                               const char* label) {
  const lcr::ModelConfig config = small_config(num_heads, num_kv_heads, head_dim);
  const int kv_dim = num_kv_heads * head_dim;

  const std::vector<elem_t> keys = to_elems(random_floats(positions * kv_dim));
  const std::vector<elem_t> values = to_elems(random_floats(positions * kv_dim));
  const std::vector<elem_t> q =
      to_elems(random_floats(static_cast<size_t>(num_heads) * head_dim));

  lcr::KvCache cache(config, positions + 8,
                     paged ? lcr::KvLayout::kPaged : lcr::KvLayout::kContiguous,
                     16);
  fill_cache(&cache, keys, values, positions, kv_dim, paged);

  DeviceBuffer<elem_t> dq(q.size());
  DeviceBuffer<elem_t> dout(static_cast<size_t>(num_heads) * head_dim);
  DeviceBuffer<float> scratch(std::max<size_t>(
      2, lcr::decode_attention_scratch_floats(num_heads, head_dim, splits)));
  dq.upload(q);

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  lcr::launch_decode_attention(dout.get(), dq.get(), cache.view(0), num_heads,
                               num_heads / num_kv_heads, positions, scale, paged,
                               splits, scratch.get(), nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> expected =
      reference_attention(to_floats(q), to_floats(keys), to_floats(values),
                          num_heads, num_kv_heads, head_dim, positions);
  expect_close(label, to_floats(dout.download()), expected);
}

void test_decode_attention() {
  // Grouped-query, which is what the model uses and what the block-per-group
  // structure exists for.
  run_decode_attention_case(32, 8, 64, 200, 1, false, "attention gqa splits=1");
  run_decode_attention_case(32, 8, 64, 200, 8, false, "attention gqa splits=8");
  // A split count that does not divide the position count evenly, so the last
  // block gets a short range.
  run_decode_attention_case(32, 8, 64, 197, 7, false, "attention ragged splits");
  // More splits than positions, which the launcher has to clamp.
  run_decode_attention_case(32, 8, 64, 5, 64, false, "attention few positions");
  // Multi-head and multi-query, the two ends of the grouping range.
  run_decode_attention_case(8, 8, 64, 130, 4, false, "attention mha");
  run_decode_attention_case(8, 1, 64, 130, 4, false, "attention mqa");
  // The other compiled head dimension.
  run_decode_attention_case(8, 2, 128, 100, 2, false, "attention head_dim=128");
  // A single cached position, where the online softmax never has to rescale.
  run_decode_attention_case(32, 8, 64, 1, 1, false, "attention one position");

  // The paged layout has to produce the same answer through a page table.
  run_decode_attention_case(32, 8, 64, 200, 1, true, "attention paged splits=1");
  run_decode_attention_case(32, 8, 64, 200, 8, true, "attention paged splits=8");
  // A length that is not a whole number of pages.
  run_decode_attention_case(32, 8, 64, 201, 4, true, "attention paged partial page");
}

void test_kv_cache_accounting() {
  const lcr::ModelConfig config = small_config(32, 8, 64);
  {
    lcr::KvCache cache(config, 1024, lcr::KvLayout::kContiguous);
    cache.reserve_length(10);
    // A contiguous cache commits the whole window immediately, whatever the
    // sequence has reached.
    CHECK_EQ(cache.live_bytes(), lcr::KvCache::bytes_for(config, 1024));
    CHECK_EQ(cache.ideal_bytes(), lcr::KvCache::bytes_for(config, 10));
  }
  {
    lcr::KvCache cache(config, 1024, lcr::KvLayout::kPaged, 16);
    cache.reserve_length(10);
    // Ten positions is one page per layer, so the waste is the six unused slots
    // in that page and nothing else.
    CHECK_EQ(cache.live_bytes(), lcr::KvCache::bytes_for(config, 16));
    cache.reserve_length(33);
    CHECK_EQ(cache.live_bytes(), lcr::KvCache::bytes_for(config, 48));
    // Asking again for a length already covered must not map more pages.
    cache.reserve_length(40);
    CHECK_EQ(cache.live_bytes(), lcr::KvCache::bytes_for(config, 48));
  }
}

// ---------------------------------------------------------------------------

void test_sampling() {
  const int vocab = 4096;
  std::vector<float> logits(vocab, -20.0f);
  logits[1234] = 5.0f;   // the clear winner
  logits[99] = 4.0f;
  logits[7] = 3.0f;

  DeviceBuffer<float> dlogits(logits.size());
  DeviceBuffer<int32_t> dout(1);
  DeviceBuffer<float> scratch(4);
  dlogits.upload(logits);

  auto sample = [&](float temperature, int top_k, float top_p, float uniform) {
    lcr::launch_sample(dout.get(), dlogits.get(), vocab, temperature, top_k,
                       top_p, uniform, scratch.get(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    return dout.download()[0];
  };

  // Temperature zero is the argmax, whatever the random draw.
  CHECK_EQ(sample(0.0f, 0, 1.0f, 0.0f), 1234);
  CHECK_EQ(sample(0.0f, 0, 1.0f, 0.999f), 1234);
  // Keeping one candidate forces the argmax too.
  CHECK_EQ(sample(1.0f, 1, 1.0f, 0.5f), 1234);
  // A tight nucleus can only reach the top few.
  for (float u : {0.01f, 0.5f, 0.99f}) {
    const int32_t id = sample(1.0f, 0, 0.9f, u);
    CHECK(id == 1234 || id == 99);
  }

  // Two exactly equal logits and a flat draw: the boundary is at one half, so
  // the choice is decided entirely by the uniform, with no tie-break ambiguity.
  std::vector<float> pair(vocab, -60.0f);
  pair[3] = 1.0f;
  pair[9] = 1.0f;
  dlogits.upload(pair);
  CHECK_EQ(sample(1.0f, 0, 1.0f, 0.25f), 3);
  CHECK_EQ(sample(1.0f, 0, 1.0f, 0.75f), 9);
  // Greedy over a tie takes the lower id, which is what makes it reproducible.
  CHECK_EQ(sample(0.0f, 0, 1.0f, 0.5f), 3);
}

void test_arena() {
  lcr::DeviceArena arena(1 << 20);
  CHECK_EQ(arena.used(), static_cast<size_t>(0));

  float* first = arena.alloc<float>(100);
  CHECK(first != nullptr);
  // Allocations are aligned, so 400 bytes rounds up to the next boundary.
  CHECK_EQ(arena.used(), static_cast<size_t>(400));

  const size_t mark = arena.mark();
  float* second = arena.alloc<float>(64);
  CHECK(reinterpret_cast<uintptr_t>(second) % 256 == 0);
  CHECK(arena.used() > mark);
  {
    lcr::ArenaScope scope(arena);
    arena.alloc<float>(1000);
    CHECK(arena.used() > mark);
  }
  // The scope gave its allocation back, but the peak remembers it.
  CHECK(arena.peak() >= 400 + 1000 * sizeof(float));

  arena.release(mark);
  CHECK_EQ(arena.used(), mark);
  CHECK_THROWS("arena is full", { arena.alloc<char>(1 << 21); });
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::fprintf(stderr, "test_kernels: no CUDA device, skipping\n");
    return 0;
  }
  try {
    test_rmsnorm();
    test_swiglu_and_residual();
    test_embedding_lookup();
    test_rope_query();
    test_causal_softmax();
    test_decode_attention();
    test_kv_cache_accounting();
    test_sampling();
    test_arena();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_kernels: %s\n", e.what());
    return 1;
  }
  return test::finish("test_kernels");
}
