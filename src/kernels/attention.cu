// Attention: the causal softmax used during prefill, and the decode-phase
// kernel that this whole project exists to make fast.
//
// Prefill goes through cuBLAS. Q times K transpose and probabilities times V
// are both large matrix multiplies with real arithmetic intensity, and there is
// nothing to gain from writing them by hand. Only the softmax between them is a
// custom kernel.
//
// Decode is the opposite. There is exactly one query vector, so both matmuls
// degenerate into matrix-vector products: every byte read from the cache is
// used for a single multiply-add. The kernel is bandwidth bound from the first
// instruction, and its job is to read the cache once, in order, at the widest
// access the hardware supports.
//
// Two things make it faster than the obvious version. Grouped-query attention
// means four query heads share one key/value head, so one block covers a whole
// group and reads the cache once instead of four times. And the position range
// is split across blocks, because a group-per-block launch would put only eight
// blocks on a GPU with over a hundred multiprocessors; each split keeps a
// partial softmax that a second pass merges.
#include <algorithm>
#include <cfloat>

#include "common.h"
#include "cuda_utils.cuh"
#include "kernels.cuh"
#include "kernels/reduce.cuh"

namespace lcr {
namespace {

constexpr int kSoftmaxBlock = 256;
// Four warps per attention block. More would grow the shared-memory combine
// buffer, which is already the factor limiting how many blocks fit per
// multiprocessor; parallelism comes from splitting positions instead.
constexpr int kAttentionWarps = 4;

// ---------------------------------------------------------------------------
// Prefill softmax
// ---------------------------------------------------------------------------

__global__ void causal_softmax_kernel(elem_t* __restrict__ probs,
                                      const float* __restrict__ scores,
                                      int rows, int keys, int start_position) {
  __shared__ float shared_m[kSoftmaxBlock / 32];
  __shared__ float shared_l[kSoftmaxBlock / 32];
  __shared__ float row_max;
  __shared__ float row_sum;

  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int64_t base =
      (static_cast<int64_t>(head) * rows + row) * keys;
  // Query at position start_position + row can see keys 0 through its own
  // position and no further.
  const int limit = min(keys, start_position + row + 1);

  float m = kSoftmaxNegInf;
  float l = 0.0f;
  for (int j = threadIdx.x; j < limit; j += kSoftmaxBlock) {
    const float score = scores[base + j];
    const float merged = fmaxf(m, score);
    l = l * __expf(m - merged) + __expf(score - merged);
    m = merged;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_m = __shfl_xor_sync(kFullMask, m, offset);
    const float other_l = __shfl_xor_sync(kFullMask, l, offset);
    softmax_combine(&m, &l, other_m, other_l);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    shared_m[warp] = m;
    shared_l[warp] = l;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total_m = shared_m[0];
    float total_l = shared_l[0];
    for (int w = 1; w < kSoftmaxBlock / 32; ++w) {
      softmax_combine(&total_m, &total_l, shared_m[w], shared_l[w]);
    }
    row_max = total_m;
    row_sum = total_l;
  }
  __syncthreads();

  const float inv_sum = 1.0f / row_sum;
  for (int j = threadIdx.x; j < keys; j += kSoftmaxBlock) {
    // Masked positions are written as zero rather than left alone, because the
    // probability buffer is reused across chunks and would otherwise carry
    // stale values into the second matrix multiply.
    const float p =
        j < limit ? __expf(scores[base + j] - row_max) * inv_sum : 0.0f;
    probs[base + j] = float_to_elem(p);
  }
}

// ---------------------------------------------------------------------------
// Decode attention
// ---------------------------------------------------------------------------

// Layout of one split's partial result: the running maximum, the running
// exponential sum, then the unnormalized weighted value vector.
constexpr int kPartialHeader = 2;

template <bool kPaged, int kHeadDim, int kGroup, bool kSplit>
__global__ void decode_attention_kernel(elem_t* __restrict__ out,
                                        float* __restrict__ partials,
                                        const elem_t* __restrict__ q,
                                        KvCacheView view, int seq_len,
                                        float scale, int splits) {
  constexpr int kPerThread = kHeadDim / 32;
  constexpr int kWarps = kAttentionWarps;

  __shared__ float shared_m[kWarps][kGroup];
  __shared__ float shared_l[kWarps][kGroup];
  __shared__ float shared_acc[kWarps][kGroup][kHeadDim];

  const int kv_head = blockIdx.x;
  const int split = kSplit ? static_cast<int>(blockIdx.y) : 0;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int base_dim = lane * kPerThread;

  // Every block covers a contiguous run of positions. The last one is short
  // whenever the sequence length does not divide evenly.
  const int chunk = kSplit ? (seq_len + splits - 1) / splits : seq_len;
  const int begin = split * chunk;
  const int end = min(seq_len, begin + chunk);

  // The 1/sqrt(head_dim) scaling is folded into the query once, rather than
  // applied to every score.
  float query[kGroup][kPerThread];
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    const int head = kv_head * kGroup + g;
    load_elems<kPerThread>(q + static_cast<size_t>(head) * kHeadDim + base_dim,
                           query[g]);
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) query[g][i] *= scale;
  }

  float running_max[kGroup];
  float running_sum[kGroup];
  float acc[kGroup][kPerThread];
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    running_max[g] = kSoftmaxNegInf;
    running_sum[g] = 0.0f;
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) acc[g][i] = 0.0f;
  }

  for (int pos = begin + warp; pos < end; pos += kWarps) {
    const size_t offset = view.offset<kPaged>(kv_head, pos);

    // One key vector, read once and used by every query head in the group.
    float key[kPerThread];
    load_elems<kPerThread>(view.keys + offset + base_dim, key);

    float score[kGroup];
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      float dot = 0.0f;
#pragma unroll
      for (int i = 0; i < kPerThread; ++i) dot += query[g][i] * key[i];
      score[g] = dot;
    }
    // Each lane holds a slice of the head dimension, so the dot product is
    // finished across the warp. Every lane ends up with the full score, which
    // it needs for its own slice of the value accumulation.
#pragma unroll
    for (int offset_lane = 16; offset_lane > 0; offset_lane >>= 1) {
#pragma unroll
      for (int g = 0; g < kGroup; ++g) {
        score[g] += __shfl_xor_sync(kFullMask, score[g], offset_lane);
      }
    }

    float value[kPerThread];
    load_elems<kPerThread>(view.values + offset + base_dim, value);

#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      const float merged = fmaxf(running_max[g], score[g]);
      const float rescale = __expf(running_max[g] - merged);
      const float weight = __expf(score[g] - merged);
      running_sum[g] = running_sum[g] * rescale + weight;
#pragma unroll
      for (int i = 0; i < kPerThread; ++i) {
        acc[g][i] = acc[g][i] * rescale + weight * value[i];
      }
      running_max[g] = merged;
    }
  }

  // Merge the warps. Each holds a complete but partial softmax over its own
  // stripe of positions.
  if (lane == 0) {
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      shared_m[warp][g] = running_max[g];
      shared_l[warp][g] = running_sum[g];
    }
  }
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      shared_acc[warp][g][base_dim + i] = acc[g][i];
    }
  }
  __syncthreads();

  if (warp != 0) return;

#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    float total_m = shared_m[0][g];
    float total_l = shared_l[0][g];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) {
      softmax_combine(&total_m, &total_l, shared_m[w][g], shared_l[w][g]);
    }

    float merged[kPerThread];
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) merged[i] = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const float rescale = __expf(shared_m[w][g] - total_m);
#pragma unroll
      for (int i = 0; i < kPerThread; ++i) {
        merged[i] += rescale * shared_acc[w][g][base_dim + i];
      }
    }

    const int head = kv_head * kGroup + g;
    if constexpr (kSplit) {
      // Leave it unnormalized; the merge pass divides once, after combining
      // every split's maximum.
      float* slot = partials + (static_cast<size_t>(head) * splits + split) *
                                   (kPartialHeader + kHeadDim);
      if (lane == 0) {
        slot[0] = total_m;
        slot[1] = total_l;
      }
#pragma unroll
      for (int i = 0; i < kPerThread; ++i) {
        slot[kPartialHeader + base_dim + i] = merged[i];
      }
    } else {
      const float inv_sum = 1.0f / total_l;
#pragma unroll
      for (int i = 0; i < kPerThread; ++i) {
        out[static_cast<size_t>(head) * kHeadDim + base_dim + i] =
            float_to_elem(merged[i] * inv_sum);
      }
    }
  }
}

__global__ void merge_splits_kernel(elem_t* __restrict__ out,
                                    const float* __restrict__ partials,
                                    int splits, int head_dim) {
  __shared__ float total_m;
  __shared__ float total_l;

  const int head = blockIdx.x;
  const int stride = kPartialHeader + head_dim;
  const float* base =
      partials + static_cast<size_t>(head) * splits * stride;

  if (threadIdx.x == 0) {
    float m = kSoftmaxNegInf;
    float l = 0.0f;
    for (int s = 0; s < splits; ++s) {
      softmax_combine(&m, &l, base[s * stride], base[s * stride + 1]);
    }
    total_m = m;
    total_l = l;
  }
  __syncthreads();

  const float inv_sum = 1.0f / total_l;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float sum = 0.0f;
    for (int s = 0; s < splits; ++s) {
      const float* slot = base + s * stride;
      sum += __expf(slot[0] - total_m) * slot[kPartialHeader + i];
    }
    out[static_cast<size_t>(head) * head_dim + i] =
        float_to_elem(sum * inv_sum);
  }
}

// ---------------------------------------------------------------------------
// Dispatch
//
// The kernel is templated on the paging mode, the head dimension, the
// grouped-query group size, and whether splits are in play, so that every index
// calculation and register array is fixed at compile time. These three
// functions turn the runtime shape into the right instantiation.
// ---------------------------------------------------------------------------

struct AttentionLaunch {
  elem_t* out;
  float* partials;
  const elem_t* q;
  KvCacheView view;
  int num_heads;
  int seq_len;
  float scale;
  int splits;
  cudaStream_t stream;
};

template <bool kPaged, int kHeadDim, int kGroup>
void launch_typed(const AttentionLaunch& args) {
  const int num_kv_heads = args.num_heads / kGroup;
  const dim3 grid(static_cast<unsigned>(num_kv_heads),
                  static_cast<unsigned>(args.splits));
  const int block = kAttentionWarps * 32;

  if (args.splits > 1) {
    decode_attention_kernel<kPaged, kHeadDim, kGroup, true>
        <<<grid, block, 0, args.stream>>>(args.out, args.partials, args.q,
                                          args.view, args.seq_len, args.scale,
                                          args.splits);
    CUDA_CHECK_LAUNCH();
    merge_splits_kernel<<<args.num_heads, 128, 0, args.stream>>>(
        args.out, args.partials, args.splits, kHeadDim);
    CUDA_CHECK_LAUNCH();
  } else {
    decode_attention_kernel<kPaged, kHeadDim, kGroup, false>
        <<<grid, block, 0, args.stream>>>(args.out, nullptr, args.q, args.view,
                                          args.seq_len, args.scale, 1);
    CUDA_CHECK_LAUNCH();
  }
}

template <bool kPaged, int kHeadDim>
void launch_by_group(int heads_per_kv, const AttentionLaunch& args) {
  switch (heads_per_kv) {
    case 1: launch_typed<kPaged, kHeadDim, 1>(args); return;
    case 2: launch_typed<kPaged, kHeadDim, 2>(args); return;
    case 4: launch_typed<kPaged, kHeadDim, 4>(args); return;
    case 8: launch_typed<kPaged, kHeadDim, 8>(args); return;
    default:
      LCR_FAIL("decode attention is compiled for 1, 2, 4, or 8 query heads per "
               "key/value head, this model has " << heads_per_kv);
  }
}

template <bool kPaged>
void launch_by_head_dim(int head_dim, int heads_per_kv,
                        const AttentionLaunch& args) {
  switch (head_dim) {
    case 64: launch_by_group<kPaged, 64>(heads_per_kv, args); return;
    case 128: launch_by_group<kPaged, 128>(heads_per_kv, args); return;
    default:
      LCR_FAIL("decode attention is compiled for head_dim 64 or 128, this "
               "model has " << head_dim);
  }
}

}  // namespace

void launch_causal_softmax(elem_t* probs, const float* scores, int num_heads,
                           int rows, int keys, int start_position,
                           cudaStream_t stream) {
  if (rows == 0 || num_heads == 0) return;
  const dim3 grid(static_cast<unsigned>(rows),
                  static_cast<unsigned>(num_heads));
  causal_softmax_kernel<<<grid, kSoftmaxBlock, 0, stream>>>(
      probs, scores, rows, keys, start_position);
  CUDA_CHECK_LAUNCH();
}

size_t decode_attention_scratch_floats(int num_heads, int head_dim, int splits) {
  if (splits <= 1) return 0;
  return static_cast<size_t>(num_heads) * splits * (kPartialHeader + head_dim);
}

int choose_attention_splits(int num_kv_heads, int seq_len, int multiprocessors) {
  // Aim for two blocks per multiprocessor so the tail of one wave overlaps the
  // start of the next.
  const int wanted_blocks = std::max(1, 2 * multiprocessors);
  int splits = (wanted_blocks + num_kv_heads - 1) / num_kv_heads;

  // Splitting below a few hundred positions per block stops paying: the merge
  // pass and the launch itself start to cost more than the parallelism wins.
  constexpr int kMinPositionsPerSplit = 256;
  const int useful = std::max(1, seq_len / kMinPositionsPerSplit);
  splits = std::min(splits, useful);
  return std::max(1, std::min(splits, 64));
}

void launch_decode_attention(elem_t* out, const elem_t* q,
                             const KvCacheView& view, int num_heads,
                             int heads_per_kv, int seq_len, float scale,
                             bool paged, int splits, float* scratch,
                             cudaStream_t stream) {
  LCR_CHECK(seq_len > 0, "decode attention needs at least one cached position");
  LCR_CHECK(splits >= 1, "split count must be positive, got " << splits);
  LCR_CHECK(splits == 1 || scratch != nullptr,
            "a split decode attention needs scratch space for the partials");
  LCR_CHECK(view.head_dim % 32 == 0,
            "decode attention needs head_dim to be a multiple of the warp "
            "width, got " << view.head_dim);

  AttentionLaunch args;
  args.out = out;
  args.partials = scratch;
  args.q = q;
  args.view = view;
  args.num_heads = num_heads;
  args.seq_len = seq_len;
  args.scale = scale;
  // A split that would leave blocks with no positions at all is pointless.
  args.splits = std::min(splits, seq_len);
  args.stream = stream;

  if (paged) {
    launch_by_head_dim<true>(view.head_dim, heads_per_kv, args);
  } else {
    launch_by_head_dim<false>(view.head_dim, heads_per_kv, args);
  }
}

}  // namespace lcr
