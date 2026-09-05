// Model geometry, read from the checkpoint's config.json.
#pragma once

#include <string>
#include <vector>

#include "common.h"

namespace lcr {

// RoPE frequency rescaling. Llama 3.x stretches the rotary base so a model
// trained at 8k context extrapolates to 128k: low-frequency components are
// divided by `factor`, high-frequency components are left alone, and the band
// between the two is interpolated. `enabled == false` means plain RoPE.
struct RopeScaling {
  bool enabled = false;
  float factor = 32.0f;
  float low_freq_factor = 1.0f;
  float high_freq_factor = 4.0f;
  int original_max_position_embeddings = 8192;
};

struct ModelConfig {
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  int vocab_size = 0;
  int max_position_embeddings = 0;
  float rms_norm_eps = 1e-5f;
  float rope_theta = 500000.0f;
  bool tie_word_embeddings = true;
  RopeScaling rope_scaling;

  int bos_token_id = -1;
  std::vector<int> eos_token_ids;

  // Grouped-query attention: this many query heads share each key/value head.
  int heads_per_kv() const { return num_heads / num_kv_heads; }
  // Width of the packed K or V projection output, per token.
  int kv_dim() const { return num_kv_heads * head_dim; }
  // Width of the packed Q projection output, per token.
  int q_dim() const { return num_heads * head_dim; }
  // Bytes of KV cache one token occupies across every layer, both K and V.
  size_t kv_bytes_per_token(size_t elem_size) const {
    return static_cast<size_t>(num_layers) * 2 * kv_dim() * elem_size;
  }

  bool is_eos(int token_id) const;

  // Reads config.json, and generation_config.json when it is present, from a
  // checkpoint directory.
  static ModelConfig load(const std::string& model_dir);
  // Parses a config.json document held in memory. Split out so it can be
  // tested without a checkpoint on disk.
  static ModelConfig from_json_string(const std::string& text);

  void validate() const;
  std::string summary() const;
};

// One weight tensor the runtime needs: the name it has in the checkpoint and
// the shape it must have.
struct WeightSpec {
  std::string name;
  std::vector<int64_t> shape;
};

// Every tensor the forward pass reads, in the order it is read. The loader
// places them in a single device allocation in this order, so a decode step
// walks the weights forward through memory rather than jumping around it.
//
// The output head is not here: it is lm_head.weight when the checkpoint has
// one, and the embedding matrix again when the config ties them.
std::vector<WeightSpec> required_weights(const ModelConfig& config);

// Per-pair inverse frequencies for RoPE, length head_dim/2, with Llama 3
// frequency scaling applied when the config asks for it. Computed on the host
// once at load and uploaded, so the rotary kernel is a table lookup rather than
// a chain of transcendentals.
std::vector<float> compute_rope_inv_freq(const ModelConfig& config);

}  // namespace lcr
