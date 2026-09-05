// Embedding lookup, residual add, SwiGLU, and the head-major merge.
//
// All four are pure streaming kernels with no reuse, so the only thing that
// matters is issuing the widest loads the alignment allows and keeping enough
// blocks in flight to cover the memory latency. Each moves in 128-bit chunks
// with a scalar tail for shapes that are not a multiple of eight.
#include <algorithm>

#include "common.h"
#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace lcr {
namespace {

constexpr int kBlock = 256;

// Grid-stride loops mean any block count is correct; this picks one that keeps
// the machine busy without launching millions of near-empty blocks.
int grid_for(int64_t elements) {
  const int64_t blocks = (elements + kBlock - 1) / kBlock;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(blocks, 8192)));
}

__global__ void embedding_lookup_kernel(elem_t* __restrict__ out,
                                        const elem_t* __restrict__ table,
                                        const int32_t* __restrict__ ids,
                                        int dim) {
  const int token = blockIdx.x;
  const int64_t row = static_cast<int64_t>(ids[token]) * dim;
  const int64_t dest = static_cast<int64_t>(token) * dim;
  const int vecs = dim / 8;

  for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
    reinterpret_cast<ElemVec8*>(out + dest)[i] =
        reinterpret_cast<const ElemVec8*>(table + row)[i];
  }
  for (int i = vecs * 8 + threadIdx.x; i < dim; i += blockDim.x) {
    out[dest + i] = table[row + i];
  }
}

__global__ void add_residual_kernel(elem_t* __restrict__ x,
                                    const elem_t* __restrict__ delta,
                                    int64_t count) {
  const int64_t vecs = count / 8;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = tid; i < vecs; i += stride) {
    ElemVec8 a = reinterpret_cast<ElemVec8*>(x)[i];
    const ElemVec8 b = reinterpret_cast<const ElemVec8*>(delta)[i];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      a.v[j] = float_to_elem(elem_to_float(a.v[j]) + elem_to_float(b.v[j]));
    }
    reinterpret_cast<ElemVec8*>(x)[i] = a;
  }
  for (int64_t i = vecs * 8 + tid; i < count; i += stride) {
    x[i] = float_to_elem(elem_to_float(x[i]) + elem_to_float(delta[i]));
  }
}

__device__ __forceinline__ float silu(float v) {
  // v * sigmoid(v), evaluated in fp32. __expf is the fast intrinsic; its
  // relative error is well under the 2^-8 that bf16 storage rounds away.
  return v / (1.0f + __expf(-v));
}

__global__ void swiglu_kernel(elem_t* __restrict__ out,
                              const elem_t* __restrict__ gate,
                              const elem_t* __restrict__ up, int64_t count) {
  const int64_t vecs = count / 8;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = tid; i < vecs; i += stride) {
    const ElemVec8 g = reinterpret_cast<const ElemVec8*>(gate)[i];
    const ElemVec8 u = reinterpret_cast<const ElemVec8*>(up)[i];
    ElemVec8 result;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      result.v[j] =
          float_to_elem(silu(elem_to_float(g.v[j])) * elem_to_float(u.v[j]));
    }
    reinterpret_cast<ElemVec8*>(out)[i] = result;
  }
  for (int64_t i = vecs * 8 + tid; i < count; i += stride) {
    out[i] = float_to_elem(silu(elem_to_float(gate[i])) * elem_to_float(up[i]));
  }
}

// heads[h][t][d] -> out[t][h * head_dim + d]
__global__ void merge_heads_kernel(elem_t* __restrict__ out,
                                   const elem_t* __restrict__ heads, int tokens,
                                   int num_heads, int head_dim) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int64_t src =
      (static_cast<int64_t>(head) * tokens + token) * head_dim;
  const int64_t dst =
      static_cast<int64_t>(token) * num_heads * head_dim +
      static_cast<int64_t>(head) * head_dim;
  const int vecs = head_dim / 8;

  for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
    reinterpret_cast<ElemVec8*>(out + dst)[i] =
        reinterpret_cast<const ElemVec8*>(heads + src)[i];
  }
  for (int i = vecs * 8 + threadIdx.x; i < head_dim; i += blockDim.x) {
    out[dst + i] = heads[src + i];
  }
}

}  // namespace

void launch_embedding_lookup(elem_t* out, const elem_t* table,
                             const int32_t* ids, int tokens, int dim,
                             cudaStream_t stream) {
  if (tokens == 0) return;
  embedding_lookup_kernel<<<tokens, kBlock, 0, stream>>>(out, table, ids, dim);
  CUDA_CHECK_LAUNCH();
}

void launch_add_residual(elem_t* x, const elem_t* delta, int64_t count,
                         cudaStream_t stream) {
  if (count == 0) return;
  add_residual_kernel<<<grid_for(count / 8 + 1), kBlock, 0, stream>>>(x, delta,
                                                                     count);
  CUDA_CHECK_LAUNCH();
}

void launch_swiglu(elem_t* out, const elem_t* gate, const elem_t* up,
                   int64_t count, cudaStream_t stream) {
  if (count == 0) return;
  swiglu_kernel<<<grid_for(count / 8 + 1), kBlock, 0, stream>>>(out, gate, up,
                                                                count);
  CUDA_CHECK_LAUNCH();
}

void launch_merge_heads(elem_t* out, const elem_t* heads, int tokens,
                        int num_heads, int head_dim, cudaStream_t stream) {
  if (tokens == 0) return;
  const dim3 grid(static_cast<unsigned>(tokens),
                  static_cast<unsigned>(num_heads));
  const int block = std::min(kBlock, ((head_dim + 31) / 32) * 32);
  merge_heads_kernel<<<grid, block, 0, stream>>>(out, heads, tokens, num_heads,
                                                 head_dim);
  CUDA_CHECK_LAUNCH();
}

}  // namespace lcr
