// The model: weights on the device, and the two forward passes.
//
// Prefill and decode run the same layers but hit completely different limits.
// Prefill has the whole prompt in flight, so every projection is a real matrix
// multiply with hundreds of rows and the GPU is compute bound. Decode has one
// token, so every projection is a matrix times a vector: each weight is read
// from HBM, used for exactly one multiply-add, and thrown away. Nothing about
// the arithmetic matters at that point, only how fast the weights can be
// streamed, which is why the interesting number is achieved bandwidth rather
// than FLOPs.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "arena.h"
#include "config.h"
#include "kv_cache.cuh"
#include "safetensors.h"

namespace lcr {

struct LayerWeights {
  const elem_t* input_norm = nullptr;          // [hidden]
  const elem_t* q_proj = nullptr;              // [num_heads * head_dim, hidden]
  const elem_t* k_proj = nullptr;              // [num_kv_heads * head_dim, hidden]
  const elem_t* v_proj = nullptr;              // [num_kv_heads * head_dim, hidden]
  const elem_t* o_proj = nullptr;              // [hidden, num_heads * head_dim]
  const elem_t* post_attention_norm = nullptr; // [hidden]
  const elem_t* gate_proj = nullptr;           // [intermediate, hidden]
  const elem_t* up_proj = nullptr;             // [intermediate, hidden]
  const elem_t* down_proj = nullptr;           // [hidden, intermediate]
};

// Byte counts and timings the benchmark reports.
struct ForwardStats {
  // Weight bytes that have to cross the memory bus for one decoded token.
  // Every projection plus the output head; the embedding table contributes one
  // row, not the whole matrix.
  size_t weight_bytes_per_token = 0;
  // Cache bytes read at a given sequence length, both keys and values, across
  // every layer.
  size_t kv_bytes_at(int seq_len) const;
  size_t kv_bytes_per_position = 0;

  double last_prefill_ms = 0.0;
  double last_decode_ms = 0.0;
};

class Model {
 public:
  struct Options {
    int device = 0;
    // Longest sequence the KV cache is built for. Not the model's declared
    // context window, which for Llama 3.2 would reserve 4.3 GB of cache.
    int max_seq = 4096;
    KvLayout kv_layout = KvLayout::kContiguous;
    int page_size = 16;
    // Query positions per prefill attention pass. The score buffer is
    // num_heads * chunk * max_seq floats, so this trades arena size against
    // how much of the prompt is in flight at once.
    int prefill_chunk = 256;
    // Blocks per key/value head in decode attention. Zero picks a value from
    // the sequence length and the multiprocessor count.
    int attention_splits = 0;
  };

  Model(const std::string& model_dir, const Options& options);
  ~Model();

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // Runs `tokens` through the network starting at the current position, and
  // returns device logits for the last one. The returned pointer stays valid
  // until the next call.
  const float* prefill(const std::vector<int>& tokens);
  // One token at the current position.
  const float* decode(int token);

  // Copies the logits to the host. Only for tests and the reference
  // comparison; generation samples on the device.
  std::vector<float> logits_to_host(const float* device_logits) const;

  void reset();
  int position() const { return position_; }

  const ModelConfig& config() const { return config_; }
  const KvCache& kv_cache() const { return *kv_cache_; }
  const DeviceArena& arena() const { return arena_; }
  const ForwardStats& stats() const { return stats_; }
  size_t weight_bytes() const { return weight_bytes_; }
  int multiprocessors() const { return multiprocessors_; }

  // Samples a token id from the logits produced by the last forward pass.
  // `uniform` must be in [0, 1); a temperature of zero ignores it and takes the
  // argmax.
  int sample(const float* device_logits, float temperature, int top_k,
             float top_p, float uniform);

  // Snapshots of the residual stream, in order, for the layer-by-layer
  // comparison against the PyTorch reference dump. Names match the ones
  // tools/dump_reference.py writes: "embed", "layer_0" through
  // "layer_{n-1}", and "final_norm".
  struct Tap {
    std::string name;
    int tokens = 0;
    int width = 0;
    std::vector<float> values;
  };
  void set_capture_activations(bool enabled) { capture_activations_ = enabled; }
  const std::vector<Tap>& activation_taps() const { return taps_; }

 private:
  void load_weights(const SafetensorsArchive& archive);
  void allocate_arena();
  void run_layers(int tokens, int start_position);
  void attention_prefill(int layer, int tokens, int start_position);
  void attention_decode(int layer, int position);
  void capture(const std::string& name, const elem_t* source, int tokens,
               int width);

  ModelConfig config_;
  Options options_;
  int multiprocessors_ = 0;

  cublasHandle_t cublas_ = nullptr;
  cudaStream_t stream_ = nullptr;

  // One allocation for every weight, so load time is one big copy per tensor
  // and the resident model is exactly its own size.
  elem_t* weight_block_ = nullptr;
  size_t weight_bytes_ = 0;
  const elem_t* embedding_ = nullptr;
  const elem_t* final_norm_ = nullptr;
  const elem_t* lm_head_ = nullptr;
  std::vector<LayerWeights> layers_;

  float* inv_freq_ = nullptr;

  DeviceArena arena_;
  std::unique_ptr<KvCache> kv_cache_;

  // Persistent activation buffers, carved out of the arena once.
  elem_t* x_ = nullptr;          // residual stream  [max_tokens][hidden]
  elem_t* normed_ = nullptr;     // [max_tokens][hidden]
  elem_t* q_proj_ = nullptr;     // [max_tokens][num_heads * head_dim]
  elem_t* k_proj_ = nullptr;     // [max_tokens][num_kv_heads * head_dim]
  elem_t* v_proj_ = nullptr;
  elem_t* q_heads_ = nullptr;    // [num_heads][max_tokens][head_dim]
  elem_t* attn_heads_ = nullptr; // [num_heads][max_tokens][head_dim]
  elem_t* attn_out_ = nullptr;   // [max_tokens][num_heads * head_dim]
  elem_t* proj_out_ = nullptr;   // [max_tokens][hidden]
  elem_t* gate_ = nullptr;       // [max_tokens][intermediate]
  elem_t* up_ = nullptr;
  elem_t* act_ = nullptr;
  float* scores_ = nullptr;      // [num_heads][chunk][max_seq]
  elem_t* probs_ = nullptr;
  float* logits_ = nullptr;      // [vocab]
  float* attention_scratch_ = nullptr;
  int32_t* token_ids_ = nullptr;
  int32_t* sampled_id_ = nullptr;
  // Contiguous keys and values for prefill attention. Points straight into the
  // cache when the layout is contiguous, and into gathered scratch when paged.
  elem_t* gathered_keys_ = nullptr;
  elem_t* gathered_values_ = nullptr;

  int position_ = 0;
  int max_tokens_ = 0;
  ForwardStats stats_;
  bool capture_activations_ = false;
  std::vector<Tap> taps_;
};

}  // namespace lcr
