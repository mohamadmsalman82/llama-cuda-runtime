// Turning a row of logits into a token id.
//
// Greedy decoding is an argmax over the 128k-entry vocabulary, done with a
// packed 64-bit atomic so it is a single pass with no second kernel.
//
// Sampling is harder than it looks, because top-p is defined over the
// descending order of the whole distribution and sorting 128k floats per token
// would cost more than the attention it follows. The way out is that the
// nucleus is tiny: after temperature, everything more than a handful of nats
// below the maximum has no chance of being sampled. So the kernel builds a
// histogram of log-probabilities in one pass, reads off a cutoff that admits at
// most 1024 candidates, compacts those, and sorts only them. Three passes over
// the logits, all but the first out of L2, and an exact result whenever the
// nucleus fits in 1024 tokens, which for a 1B model it always does.
#include <algorithm>
#include <cfloat>

#include "common.h"
#include "cuda_utils.cuh"
#include "kernels.cuh"
#include "kernels/reduce.cuh"

namespace lcr {
namespace {

constexpr int kSampleBlock = 1024;
constexpr int kMaxCandidates = 1024;
// Log-probability range the histogram covers, relative to the maximum. A token
// 20 nats below the best one has a relative probability of 2e-9 and cannot
// matter.
constexpr float kLogRange = 20.0f;
constexpr int kHistogramBins = 256;

// Maps a float to an unsigned key that compares in the same order, so a single
// atomicMax over (key, index) finds the argmax. Positive floats already order
// correctly once the sign bit is set; negative ones need a full inversion.
__device__ __forceinline__ uint32_t order_preserving_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__global__ void argmax_kernel(unsigned long long* __restrict__ best,
                              const float* __restrict__ logits, int vocab) {
  const int stride = gridDim.x * blockDim.x;
  unsigned long long local = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab; i += stride) {
    // Index is stored complemented so that, among equal logits, the smaller
    // index produces the larger packed value and wins. That makes greedy
    // decoding reproducible rather than dependent on scheduling.
    const unsigned long long packed =
        (static_cast<unsigned long long>(order_preserving_key(logits[i])) << 32) |
        static_cast<unsigned long long>(0xFFFFFFFFu - static_cast<uint32_t>(i));
    local = max(local, packed);
  }
  __shared__ unsigned long long shared;
  if (threadIdx.x == 0) shared = 0;
  __syncthreads();
  atomicMax(&shared, local);
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(best, shared);
}

__global__ void argmax_unpack_kernel(int32_t* __restrict__ out_id,
                                     const unsigned long long* __restrict__ best) {
  *out_id = static_cast<int32_t>(
      0xFFFFFFFFu - static_cast<uint32_t>(*best & 0xFFFFFFFFull));
}

// Sorts `count` (value, index) pairs into descending value order. Bitonic sort
// over a power-of-two array with one element per thread; the tail is padded
// with -inf so it sinks to the bottom and never gets sampled.
__device__ void bitonic_sort_descending(float* values, int32_t* indices,
                                        int size) {
  for (int span = 2; span <= size; span <<= 1) {
    for (int step = span >> 1; step > 0; step >>= 1) {
      const int partner = threadIdx.x ^ step;
      if (partner > static_cast<int>(threadIdx.x)) {
        // Direction alternates per span so the halves end up sorted opposite
        // ways, which is what makes the next merge stage valid.
        const bool descending = (threadIdx.x & span) == 0;
        const bool swap = descending ? (values[threadIdx.x] < values[partner])
                                     : (values[threadIdx.x] > values[partner]);
        if (swap) {
          const float tv = values[threadIdx.x];
          values[threadIdx.x] = values[partner];
          values[partner] = tv;
          const int32_t ti = indices[threadIdx.x];
          indices[threadIdx.x] = indices[partner];
          indices[partner] = ti;
        }
      }
      __syncthreads();
    }
  }
}

__global__ void sample_kernel(int32_t* __restrict__ out_id,
                              const float* __restrict__ logits, int vocab,
                              float inv_temperature, int top_k, float top_p,
                              float uniform) {
  __shared__ float reduction[kSampleBlock / 32];
  __shared__ int histogram[kHistogramBins];
  __shared__ float shared_max;
  __shared__ float shared_sum;
  __shared__ int shared_cutoff_bin;
  __shared__ int shared_count;
  __shared__ float values[kMaxCandidates];
  __shared__ int32_t indices[kMaxCandidates];

  const int tid = threadIdx.x;

  // Pass one: the maximum logit, which anchors every exponential below.
  float local_max = -FLT_MAX;
  for (int i = tid; i < vocab; i += kSampleBlock) {
    local_max = fmaxf(local_max, logits[i]);
  }
  local_max = warp_reduce_max(local_max);
  if ((tid & 31) == 0) reduction[tid >> 5] = local_max;
  __syncthreads();
  if (tid == 0) {
    float m = reduction[0];
    for (int w = 1; w < kSampleBlock / 32; ++w) m = fmaxf(m, reduction[w]);
    shared_max = m;
  }
  __syncthreads();
  const float max_logit = shared_max;

  // Pass two: the exponential sum over the whole vocabulary, and a histogram of
  // how far below the maximum each token sits. The sum has to cover every token
  // because top-p is a fraction of the full distribution, not of the candidates.
  for (int b = tid; b < kHistogramBins; b += kSampleBlock) histogram[b] = 0;
  __syncthreads();

  float local_sum = 0.0f;
  for (int i = tid; i < vocab; i += kSampleBlock) {
    const float scaled = (logits[i] - max_logit) * inv_temperature;
    local_sum += __expf(scaled);
    // Bin 0 holds the tokens closest to the maximum.
    const int bin = min(kHistogramBins - 1,
                        static_cast<int>(-scaled * (kHistogramBins / kLogRange)));
    if (bin >= 0) atomicAdd(&histogram[bin], 1);
  }
  local_sum = block_reduce_sum(local_sum, reduction);
  if (tid == 0) shared_sum = local_sum;

  // The cutoff bin is the deepest one whose running total still fits in the
  // candidate buffer.
  if (tid == 0) {
    int running = 0;
    int cutoff = 0;
    for (int b = 0; b < kHistogramBins; ++b) {
      if (running + histogram[b] > kMaxCandidates) break;
      running += histogram[b];
      cutoff = b + 1;
    }
    // With a flat distribution even bin 0 can overflow. Taking bin 0 anyway and
    // letting the compaction clamp keeps the kernel total; the result is then a
    // sample from the 1024 best-scoring tokens.
    shared_cutoff_bin = max(cutoff, 1);
    shared_count = 0;
  }
  __syncthreads();

  const float total_sum = shared_sum;
  const int cutoff_bin = shared_cutoff_bin;

  // Pass three: compact the candidates.
  for (int i = tid; i < vocab; i += kSampleBlock) {
    const float scaled = (logits[i] - max_logit) * inv_temperature;
    const int bin = min(kHistogramBins - 1,
                        static_cast<int>(-scaled * (kHistogramBins / kLogRange)));
    if (bin < cutoff_bin) {
      const int slot = atomicAdd(&shared_count, 1);
      if (slot < kMaxCandidates) {
        values[slot] = scaled;
        indices[slot] = i;
      }
    }
  }
  __syncthreads();

  const int found = min(shared_count, kMaxCandidates);
  for (int i = tid; i < kMaxCandidates; i += kSampleBlock) {
    if (i >= found) {
      values[i] = -FLT_MAX;
      indices[i] = 0;
    }
  }
  __syncthreads();

  bitonic_sort_descending(values, indices, kMaxCandidates);

  // Walk the sorted candidates, accumulating probability against the full-vocab
  // denominator, and cut where top-k or top-p says to.
  if (tid == 0) {
    const int limit = top_k > 0 ? min(found, top_k) : found;
    const float inv_total = 1.0f / total_sum;

    int keep = 0;
    float kept_mass = 0.0f;
    for (int i = 0; i < limit; ++i) {
      kept_mass += __expf(values[i]) * inv_total;
      ++keep;
      // The token that crosses the threshold is kept, so the nucleus always
      // covers at least top_p of the distribution.
      if (top_p < 1.0f && kept_mass >= top_p) break;
    }
    if (keep == 0) keep = 1;

    // Renormalize over what survived and pick with the host's random draw.
    float target = uniform * kept_mass;
    int32_t picked = indices[0];
    float running = 0.0f;
    for (int i = 0; i < keep; ++i) {
      running += __expf(values[i]) * inv_total;
      if (running >= target) {
        picked = indices[i];
        break;
      }
      picked = indices[i];
    }
    *out_id = picked;
  }
}

}  // namespace

size_t sample_scratch_floats(int vocab, int top_k) {
  // Two floats, reinterpreted as the packed 64-bit argmax accumulator.
  (void)vocab;
  (void)top_k;
  return 2;
}

void launch_sample(int32_t* out_id, const float* logits, int vocab,
                   float temperature, int top_k, float top_p, float uniform,
                   float* scratch, cudaStream_t stream) {
  LCR_CHECK(vocab > 0, "cannot sample from an empty vocabulary");

  if (temperature <= 0.0f) {
    // Zero temperature is greedy decoding by definition.
    LCR_CHECK(scratch != nullptr, "argmax needs two floats of scratch space");
    auto* best = reinterpret_cast<unsigned long long*>(scratch);
    CUDA_CHECK(cudaMemsetAsync(best, 0, sizeof(unsigned long long), stream));
    const int blocks = std::min(1024, (vocab + 255) / 256);
    argmax_kernel<<<blocks, 256, 0, stream>>>(best, logits, vocab);
    CUDA_CHECK_LAUNCH();
    argmax_unpack_kernel<<<1, 1, 0, stream>>>(out_id, best);
    CUDA_CHECK_LAUNCH();
    return;
  }

  LCR_CHECK(top_p > 0.0f && top_p <= 1.0f,
            "top_p must be in (0, 1], got " << top_p);
  sample_kernel<<<1, kSampleBlock, 0, stream>>>(
      out_id, logits, vocab, 1.0f / temperature, top_k, top_p, uniform);
  CUDA_CHECK_LAUNCH();
}

}  // namespace lcr
