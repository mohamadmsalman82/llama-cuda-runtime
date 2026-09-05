// Host-callable launchers for every hand-written kernel.
//
// Declarations live here so model.cu stays readable and the kernel tests can
// call the same entry points the model does.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "dtype.cuh"
#include "kv_cache.cuh"

namespace lcr {

// out[t, :] = table[ids[t], :]
void launch_embedding_lookup(elem_t* out, const elem_t* table,
                             const int32_t* ids, int tokens, int dim,
                             cudaStream_t stream);

// out[t, :] = x[t, :] * rsqrt(mean(x[t, :]^2) + eps) * weight
// The mean and the reciprocal square root are computed in fp32 regardless of
// the storage type, matching the reference implementation.
void launch_rmsnorm(elem_t* out, const elem_t* x, const elem_t* weight,
                    int tokens, int dim, float eps, cudaStream_t stream);

// x += delta, elementwise, for the two residual joins in each layer.
void launch_add_residual(elem_t* x, const elem_t* delta, int64_t count,
                         cudaStream_t stream);

// out = silu(gate) * up, with silu(v) = v * sigmoid(v).
void launch_swiglu(elem_t* out, const elem_t* gate, const elem_t* up,
                   int64_t count, cudaStream_t stream);

// Rotary position embedding on the query projection, fused with the transpose
// into head-major order.
//   in:  q_proj    [tokens][num_heads * head_dim]
//   out: q_heads   [num_heads][tokens][head_dim]
// Head-major is what the prefill attention GEMM wants, and doing the transpose
// here costs nothing because the kernel is already touching every element.
void launch_rope_q(elem_t* q_heads, const elem_t* q_proj, const float* inv_freq,
                   int tokens, int start_position, int num_heads, int head_dim,
                   cudaStream_t stream);

// Rotary embedding on the key projection and the copy of both key and value
// into the cache, in one pass.
//   k_proj, v_proj: [tokens][num_kv_heads * head_dim]
void launch_rope_write_kv(const KvCacheView& view, const elem_t* k_proj,
                          const elem_t* v_proj, const float* inv_freq,
                          int tokens, int start_position, bool paged,
                          cudaStream_t stream);

// Causal softmax over prefill attention scores, in fp32, writing bf16/fp16
// probabilities for the second GEMM.
//   scores: [num_heads][rows][keys] fp32, row r is query position
//           start_position + r
//   probs:  [num_heads][rows][keys] elem
void launch_causal_softmax(elem_t* probs, const float* scores, int num_heads,
                           int rows, int keys, int start_position,
                           cudaStream_t stream);

// Undoes the head-major layout so the output projection sees [tokens][hidden].
//   heads: [num_heads][tokens][head_dim] -> out: [tokens][num_heads * head_dim]
void launch_merge_heads(elem_t* out, const elem_t* heads, int tokens,
                        int num_heads, int head_dim, cudaStream_t stream);

// Copies a paged cache into a contiguous [num_kv_heads][positions][head_dim]
// buffer. Prefill attention goes through cuBLAS, which needs a uniform stride
// between positions; with the contiguous layout the cache already has one, and
// this is what gives the paged layout the same.
void launch_gather_kv(elem_t* out_keys, elem_t* out_values,
                      const KvCacheView& view, int positions,
                      cudaStream_t stream);

// Single-token attention against the whole cache. This is the kernel the
// decode-phase bandwidth number is about.
//
// One block covers one key/value head and every query head that shares it, so
// the cache is read once per group instead of once per query head. With
// grouped-query attention that is a four times reduction in the dominant read.
//
// `splits` cuts the position range into that many chunks, each handled by its
// own block, and a second pass merges the partial softmaxes. Without it the
// kernel launches only num_kv_heads blocks and leaves most of the GPU idle.
void launch_decode_attention(elem_t* out, const elem_t* q,
                             const KvCacheView& view, int num_heads,
                             int heads_per_kv, int seq_len, float scale,
                             bool paged, int splits, float* scratch,
                             cudaStream_t stream);

// Floats of scratch space launch_decode_attention needs for a given split
// count. Zero when splits is one.
size_t decode_attention_scratch_floats(int num_heads, int head_dim, int splits);

// A split count that keeps every SM busy without cutting the position range so
// finely that the merge pass dominates.
int choose_attention_splits(int num_kv_heads, int seq_len, int multiprocessors);

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

// Picks the next token id. A temperature of zero means greedy decoding, which
// takes a separate argmax path. `uniform` is a single random number in [0, 1)
// drawn on the host, so a given seed reproduces a given sequence exactly and
// the kernel needs no device RNG state.
void launch_sample(int32_t* out_id, const float* logits, int vocab,
                   float temperature, int top_k, float top_p, float uniform,
                   float* scratch, cudaStream_t stream);

size_t sample_scratch_floats(int vocab, int top_k);

}  // namespace lcr
