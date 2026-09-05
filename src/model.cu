#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <unordered_map>

#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace lcr {
namespace {

#if defined(LCR_ELEM_FP16)
constexpr cudaDataType_t kCublasElemType = CUDA_R_16F;
#else
constexpr cudaDataType_t kCublasElemType = CUDA_R_16BF;
#endif

// The largest split count launch_decode_attention will ever choose, so the
// scratch buffer can be sized once.
constexpr int kMaxAttentionSplits = 64;

std::string layer_key(int layer, const char* suffix) {
  std::ostringstream oss;
  oss << "model.layers." << layer << "." << suffix;
  return oss.str();
}

// y[rows, out_features] = x[rows, in_features] * W[out_features, in_features]^T
//
// Everything in this runtime is row-major and cuBLAS is column-major, which
// costs nothing: a row-major [r, c] buffer already is a column-major [c, r]
// one. Reading the weight as column-major [in, out] and transposing it inside
// the GEMM produces exactly the product PyTorch's Linear computes, with no
// copy and no separate transpose kernel.
void gemm_nt(cublasHandle_t handle, void* y, cudaDataType_t y_type,
             const elem_t* x, const elem_t* w, int rows, int in_features,
             int out_features, float alpha = 1.0f, float beta = 0.0f) {
  CUBLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, rows, in_features, &alpha,
      w, kCublasElemType, in_features, x, kCublasElemType, in_features, &beta,
      y, y_type, out_features, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

}  // namespace

size_t ForwardStats::kv_bytes_at(int seq_len) const {
  return static_cast<size_t>(seq_len) * kv_bytes_per_position;
}

// ---------------------------------------------------------------------------
// Construction and weight loading
// ---------------------------------------------------------------------------

Model::Model(const std::string& model_dir, const Options& options)
    : options_(options) {
  CUDA_CHECK(cudaSetDevice(options.device));
  const DeviceInfo device = query_device(options.device);
  multiprocessors_ = device.multiprocessors;

  config_ = ModelConfig::load(model_dir);
  LCR_CHECK(options_.max_seq > 0, "max_seq must be positive");
  LCR_CHECK(options_.prefill_chunk > 0, "prefill chunk must be positive");
  max_tokens_ = std::min(options_.prefill_chunk, options_.max_seq);

  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasCreate(&cublas_));
  CUBLAS_CHECK(cublasSetStream(cublas_, stream_));

  const SafetensorsArchive archive(model_dir);
  load_weights(archive);

  kv_cache_ = std::make_unique<KvCache>(config_, options_.max_seq,
                                        options_.kv_layout, options_.page_size);

  const std::vector<float> inv_freq = compute_rope_inv_freq(config_);
  CUDA_CHECK(cudaMalloc(&inv_freq_, inv_freq.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(inv_freq_, inv_freq.data(),
                        inv_freq.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  allocate_arena();

  stats_.kv_bytes_per_position = config_.kv_bytes_per_token(sizeof(elem_t));
  // Everything the memory bus has to move to produce one token: every
  // projection in every layer, the two norms, the final norm, and the output
  // head. The embedding table is excluded because a decode step reads one row
  // of it, not the matrix.
  size_t per_token = 0;
  for (int layer = 0; layer < config_.num_layers; ++layer) {
    per_token += static_cast<size_t>(config_.hidden_size) * 2;  // both norms
    per_token += static_cast<size_t>(config_.q_dim()) * config_.hidden_size;
    per_token += static_cast<size_t>(config_.kv_dim()) * config_.hidden_size * 2;
    per_token += static_cast<size_t>(config_.hidden_size) * config_.q_dim();
    per_token +=
        static_cast<size_t>(config_.intermediate_size) * config_.hidden_size * 3;
  }
  per_token += config_.hidden_size;                                  // final norm
  per_token += static_cast<size_t>(config_.vocab_size) * config_.hidden_size;
  stats_.weight_bytes_per_token = per_token * sizeof(elem_t);
}

Model::~Model() {
  if (cublas_ != nullptr) cublasDestroy(cublas_);
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
  cudaFree(weight_block_);
  cudaFree(inv_freq_);
}

void Model::load_weights(const SafetensorsArchive& archive) {
  layers_.assign(static_cast<size_t>(config_.num_layers), LayerWeights{});

  // Where each named tensor ends up once it is on the device. The names and
  // shapes themselves come from required_weights(), so the loader and
  // lcr-inspect can never disagree about what a checkpoint has to contain.
  std::unordered_map<std::string, const elem_t**> destinations;
  destinations["model.embed_tokens.weight"] = &embedding_;
  destinations["model.norm.weight"] = &final_norm_;
  for (int i = 0; i < config_.num_layers; ++i) {
    LayerWeights& w = layers_[static_cast<size_t>(i)];
    destinations[layer_key(i, "input_layernorm.weight")] = &w.input_norm;
    destinations[layer_key(i, "self_attn.q_proj.weight")] = &w.q_proj;
    destinations[layer_key(i, "self_attn.k_proj.weight")] = &w.k_proj;
    destinations[layer_key(i, "self_attn.v_proj.weight")] = &w.v_proj;
    destinations[layer_key(i, "self_attn.o_proj.weight")] = &w.o_proj;
    destinations[layer_key(i, "post_attention_layernorm.weight")] =
        &w.post_attention_norm;
    destinations[layer_key(i, "mlp.gate_proj.weight")] = &w.gate_proj;
    destinations[layer_key(i, "mlp.up_proj.weight")] = &w.up_proj;
    destinations[layer_key(i, "mlp.down_proj.weight")] = &w.down_proj;
  }

  struct Placement {
    const TensorView* tensor;
    const elem_t** destination;
  };
  std::vector<Placement> plan;

  auto take = [&](const std::string& name, const std::vector<int64_t>& shape,
                  const elem_t** destination) {
    const TensorView& tensor = archive.get(name);
    LCR_CHECK(tensor.dtype == kElemDType,
              "tensor \"" << name << "\" is " << dtype_name(tensor.dtype)
                          << " but this build computes in " << kElemName
                          << "; rebuild with -DLCR_DTYPE="
                          << (kElemDType == DType::kBF16 ? "fp16" : "bf16"));
    tensor.expect_shape(shape);
    plan.push_back({&tensor, destination});
  };

  for (const WeightSpec& spec : required_weights(config_)) {
    take(spec.name, spec.shape, destinations.at(spec.name));
  }

  const bool has_lm_head = archive.has("lm_head.weight");
  if (has_lm_head) {
    take("lm_head.weight", {config_.vocab_size, config_.hidden_size}, &lm_head_);
  } else {
    LCR_CHECK(config_.tie_word_embeddings,
              "the checkpoint has no lm_head.weight and config.json does not "
              "say the embeddings are tied");
  }

  // One allocation for the whole model, laid out in the order above.
  size_t total = 0;
  for (const Placement& item : plan) {
    total = align_up(total, 256) + item.tensor->nbytes;
  }
  CUDA_CHECK(cudaMalloc(&weight_block_, total));
  weight_bytes_ = total;

  size_t offset = 0;
  for (const Placement& item : plan) {
    offset = align_up(offset, 256);
    char* destination = reinterpret_cast<char*>(weight_block_) + offset;
    CUDA_CHECK(cudaMemcpy(destination, item.tensor->data, item.tensor->nbytes,
                          cudaMemcpyHostToDevice));
    *item.destination = reinterpret_cast<const elem_t*>(destination);
    offset += item.tensor->nbytes;
  }

  // Tied embeddings: the output head is the embedding matrix, read a second
  // time rather than stored a second time. That is 525 MB not stored, and it is
  // also 525 MB the decode step still has to stream every token.
  if (!has_lm_head) lm_head_ = embedding_;
}

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

void Model::allocate_arena() {
  const int hidden = config_.hidden_size;
  const int ffn = config_.intermediate_size;
  const int heads = config_.num_heads;
  const int tokens = max_tokens_;
  const size_t e = sizeof(elem_t);

  // Sizes are named once and used both to reserve the block and to carve it
  // up, so the two can never drift apart.
  struct Plan {
    size_t residual, projections, q_dim_buffer, kv_buffer, ffn_buffer;
    size_t scores, probs, logits, attention_scratch, gathered, ids;
  } plan;
  plan.residual = static_cast<size_t>(tokens) * hidden * e;
  plan.projections = plan.residual;
  plan.q_dim_buffer = static_cast<size_t>(tokens) * config_.q_dim() * e;
  plan.kv_buffer = static_cast<size_t>(tokens) * config_.kv_dim() * e;
  plan.ffn_buffer = static_cast<size_t>(tokens) * ffn * e;
  plan.scores = static_cast<size_t>(heads) * tokens * options_.max_seq * sizeof(float);
  plan.probs = static_cast<size_t>(heads) * tokens * options_.max_seq * e;
  plan.logits = static_cast<size_t>(config_.vocab_size) * sizeof(float);
  plan.attention_scratch =
      decode_attention_scratch_floats(heads, config_.head_dim,
                                      kMaxAttentionSplits) *
      sizeof(float);
  plan.gathered = options_.kv_layout == KvLayout::kPaged
                      ? static_cast<size_t>(config_.num_kv_heads) *
                            options_.max_seq * config_.head_dim * e
                      : 0;
  plan.ids = static_cast<size_t>(tokens) * sizeof(int32_t);

  const size_t sizes[] = {
      plan.residual,          // x_
      plan.residual,          // normed_
      plan.projections,       // proj_out_
      plan.q_dim_buffer,      // q_proj_
      plan.kv_buffer,         // k_proj_
      plan.kv_buffer,         // v_proj_
      plan.q_dim_buffer,      // q_heads_
      plan.q_dim_buffer,      // attn_heads_
      plan.q_dim_buffer,      // attn_out_
      plan.ffn_buffer,        // gate_
      plan.ffn_buffer,        // up_
      plan.ffn_buffer,        // act_
      plan.scores,            // scores_
      plan.probs,             // probs_
      plan.logits,            // logits_
      plan.attention_scratch, // attention_scratch_
      plan.gathered,          // gathered_keys_
      plan.gathered,          // gathered_values_
      plan.ids,               // token_ids_
      sizeof(int32_t),        // sampled_id_
  };
  size_t total = 0;
  for (size_t size : sizes) total = align_up(total, 256) + size;
  arena_.reserve(total + 4096);

  x_ = arena_.alloc<elem_t>(plan.residual / e);
  normed_ = arena_.alloc<elem_t>(plan.residual / e);
  proj_out_ = arena_.alloc<elem_t>(plan.projections / e);
  q_proj_ = arena_.alloc<elem_t>(plan.q_dim_buffer / e);
  k_proj_ = arena_.alloc<elem_t>(plan.kv_buffer / e);
  v_proj_ = arena_.alloc<elem_t>(plan.kv_buffer / e);
  q_heads_ = arena_.alloc<elem_t>(plan.q_dim_buffer / e);
  attn_heads_ = arena_.alloc<elem_t>(plan.q_dim_buffer / e);
  attn_out_ = arena_.alloc<elem_t>(plan.q_dim_buffer / e);
  gate_ = arena_.alloc<elem_t>(plan.ffn_buffer / e);
  up_ = arena_.alloc<elem_t>(plan.ffn_buffer / e);
  act_ = arena_.alloc<elem_t>(plan.ffn_buffer / e);
  scores_ = arena_.alloc<float>(plan.scores / sizeof(float));
  probs_ = arena_.alloc<elem_t>(plan.probs / e);
  logits_ = arena_.alloc<float>(plan.logits / sizeof(float));
  attention_scratch_ =
      arena_.alloc<float>(std::max<size_t>(1, plan.attention_scratch / sizeof(float)));
  if (plan.gathered > 0) {
    gathered_keys_ = arena_.alloc<elem_t>(plan.gathered / e);
    gathered_values_ = arena_.alloc<elem_t>(plan.gathered / e);
  }
  token_ids_ = arena_.alloc<int32_t>(static_cast<size_t>(tokens));
  sampled_id_ = arena_.alloc<int32_t>(1);
}

// ---------------------------------------------------------------------------
// Forward passes
// ---------------------------------------------------------------------------

void Model::reset() {
  position_ = 0;
  taps_.clear();
}

const float* Model::prefill(const std::vector<int>& tokens) {
  LCR_CHECK(!tokens.empty(), "prefill needs at least one token");
  LCR_CHECK(position_ + static_cast<int>(tokens.size()) <= options_.max_seq,
            "prompt of " << tokens.size() << " tokens at position " << position_
                         << " exceeds the cache built for " << options_.max_seq);

  GpuTimer timer;
  timer.start(stream_);

  // The prompt is processed in chunks so the activation buffers, and above all
  // the attention score matrix, stay bounded by the chunk size rather than by
  // the prompt length. Each chunk attends to everything already in the cache,
  // which is exactly the causal mask the softmax applies anyway.
  size_t consumed = 0;
  while (consumed < tokens.size()) {
    const int chunk = static_cast<int>(
        std::min<size_t>(static_cast<size_t>(max_tokens_),
                         tokens.size() - consumed));
    std::vector<int32_t> ids(tokens.begin() + static_cast<long>(consumed),
                             tokens.begin() + static_cast<long>(consumed) + chunk);
    CUDA_CHECK(cudaMemcpyAsync(token_ids_, ids.data(),
                               ids.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_));

    kv_cache_->reserve_length(position_ + chunk);
    launch_embedding_lookup(x_, embedding_, token_ids_, chunk,
                            config_.hidden_size, stream_);
    run_layers(chunk, position_);
    position_ += chunk;
    consumed += static_cast<size_t>(chunk);
  }

  timer.stop(stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  stats_.last_prefill_ms = timer.elapsed_ms();
  if (profiler_.enabled()) profiler_.collect();
  return logits_;
}

const float* Model::decode(int token) {
  LCR_CHECK(position_ < options_.max_seq,
            "sequence reached " << position_
                                << " positions, the cache limit");
  GpuTimer timer;
  timer.start(stream_);

  const int32_t id = token;
  CUDA_CHECK(cudaMemcpyAsync(token_ids_, &id, sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_));
  kv_cache_->reserve_length(position_ + 1);
  launch_embedding_lookup(x_, embedding_, token_ids_, 1, config_.hidden_size,
                          stream_);
  run_layers(1, position_);
  position_ += 1;

  timer.stop(stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  stats_.last_decode_ms = timer.elapsed_ms();
  if (profiler_.enabled()) profiler_.collect();
  return logits_;
}

void Model::run_layers(int tokens, int start_position) {
  const int hidden = config_.hidden_size;
  const int ffn = config_.intermediate_size;
  const int64_t residual_elements = static_cast<int64_t>(tokens) * hidden;

  if (capture_activations_) {
    taps_.clear();
    taps_.reserve(static_cast<size_t>(config_.num_layers) + 2);
    capture("embed", x_, tokens, hidden);
  }

  for (int layer = 0; layer < config_.num_layers; ++layer) {
    const LayerWeights& w = layers_[static_cast<size_t>(layer)];

    {
      ProfileScope scope(&profiler_, "rmsnorm", stream_);
      launch_rmsnorm(normed_, x_, w.input_norm, tokens, hidden,
                     config_.rms_norm_eps, stream_);
    }
    {
      ProfileScope scope(&profiler_, "qkv_proj (3 GEMM)", stream_);
      gemm_nt(cublas_, q_proj_, kCublasElemType, normed_, w.q_proj, tokens,
              hidden, config_.q_dim());
      gemm_nt(cublas_, k_proj_, kCublasElemType, normed_, w.k_proj, tokens,
              hidden, config_.kv_dim());
      gemm_nt(cublas_, v_proj_, kCublasElemType, normed_, w.v_proj, tokens,
              hidden, config_.kv_dim());
    }

    const KvCacheView view = kv_cache_->view(layer);
    {
      ProfileScope scope(&profiler_, "rope + kv write", stream_);
      launch_rope_q(q_heads_, q_proj_, inv_freq_, tokens, start_position,
                    config_.num_heads, config_.head_dim, stream_);
      launch_rope_write_kv(view, k_proj_, v_proj_, inv_freq_, tokens,
                           start_position,
                           options_.kv_layout == KvLayout::kPaged, stream_);
    }
    {
      ProfileScope scope(&profiler_, "attention", stream_);
      if (tokens == 1) {
        attention_decode(layer, start_position);
      } else {
        attention_prefill(layer, tokens, start_position);
      }
    }
    {
      ProfileScope scope(&profiler_, "o_proj (GEMM)", stream_);
      gemm_nt(cublas_, proj_out_, kCublasElemType, attn_out_, w.o_proj, tokens,
              config_.q_dim(), hidden);
    }
    {
      ProfileScope scope(&profiler_, "residual add", stream_);
      launch_add_residual(x_, proj_out_, residual_elements, stream_);
    }
    {
      ProfileScope scope(&profiler_, "rmsnorm", stream_);
      launch_rmsnorm(normed_, x_, w.post_attention_norm, tokens, hidden,
                     config_.rms_norm_eps, stream_);
    }
    {
      ProfileScope scope(&profiler_, "mlp up/gate (2 GEMM)", stream_);
      gemm_nt(cublas_, gate_, kCublasElemType, normed_, w.gate_proj, tokens,
              hidden, ffn);
      gemm_nt(cublas_, up_, kCublasElemType, normed_, w.up_proj, tokens, hidden,
              ffn);
    }
    {
      ProfileScope scope(&profiler_, "swiglu", stream_);
      launch_swiglu(act_, gate_, up_, static_cast<int64_t>(tokens) * ffn,
                    stream_);
    }
    {
      ProfileScope scope(&profiler_, "mlp down (GEMM)", stream_);
      gemm_nt(cublas_, proj_out_, kCublasElemType, act_, w.down_proj, tokens,
              ffn, hidden);
    }
    {
      ProfileScope scope(&profiler_, "residual add", stream_);
      launch_add_residual(x_, proj_out_, residual_elements, stream_);
    }

    if (capture_activations_) {
      capture("layer_" + std::to_string(layer), x_, tokens, hidden);
    }
  }

  {
    ProfileScope scope(&profiler_, "final norm", stream_);
    launch_rmsnorm(normed_, x_, final_norm_, tokens, hidden,
                   config_.rms_norm_eps, stream_);
  }

  if (capture_activations_) capture("final_norm", normed_, tokens, hidden);

  // Only the last position's logits are ever used: during prefill to pick the
  // first generated token, during decode because there is only one position.
  // Projecting the whole prompt onto a 128k vocabulary would cost more than the
  // rest of prefill put together.
  const elem_t* last = normed_ + static_cast<int64_t>(tokens - 1) * hidden;
  {
    ProfileScope scope(&profiler_, "lm_head (GEMM)", stream_);
    gemm_nt(cublas_, logits_, CUDA_R_32F, last, lm_head_, 1, hidden,
            config_.vocab_size);
  }
}

void Model::capture(const std::string& name, const elem_t* source, int tokens,
                    int width) {
  const size_t count = static_cast<size_t>(tokens) * width;
  std::vector<elem_t> host(count);
  CUDA_CHECK(cudaMemcpyAsync(host.data(), source, count * sizeof(elem_t),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  Tap tap;
  tap.name = name;
  tap.tokens = tokens;
  tap.width = width;
  tap.values.resize(count);
  for (size_t i = 0; i < count; ++i) {
    tap.values[i] = static_cast<float>(host[i]);
  }
  taps_.push_back(std::move(tap));
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

void Model::attention_prefill(int layer, int tokens, int start_position) {
  const int head_dim = config_.head_dim;
  const int heads_per_kv = config_.heads_per_kv();
  const int num_kv_heads = config_.num_kv_heads;
  const int keys = start_position + tokens;
  const KvCacheView view = kv_cache_->view(layer);

  const elem_t* key_base = view.keys;
  const elem_t* value_base = view.values;
  long long key_head_stride = view.head_stride;
  if (options_.kv_layout == KvLayout::kPaged) {
    // cuBLAS needs one uniform stride between positions, which a page table
    // does not provide. Copying the live pages into a flat buffer costs a pass
    // over the cache; during prefill, which is compute bound, that is free.
    launch_gather_kv(gathered_keys_, gathered_values_, view, keys, stream_);
    key_base = gathered_keys_;
    value_base = gathered_values_;
    key_head_stride = static_cast<long long>(keys) * head_dim;
  }

  const int rows_per_group = heads_per_kv * tokens;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const float zero = 0.0f;
  const float one = 1.0f;

  // Scores. One batch entry per key/value head; the query heads that share it
  // are contiguous in the head-major layout, so a group is a single tall block
  // of rows and needs no gather of its own.
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      cublas_, CUBLAS_OP_T, CUBLAS_OP_N, keys, rows_per_group, head_dim, &scale,
      key_base, kCublasElemType, head_dim, key_head_stride, q_heads_,
      kCublasElemType, head_dim,
      static_cast<long long>(rows_per_group) * head_dim, &zero, scores_,
      CUDA_R_32F, keys, static_cast<long long>(rows_per_group) * keys,
      num_kv_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

  // Row r of head h is query position start_position + r, and may see keys 0
  // through that position.
  launch_causal_softmax(probs_, scores_, config_.num_heads, tokens, keys,
                        start_position, stream_);

  CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      cublas_, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, rows_per_group, keys, &one,
      value_base, kCublasElemType, head_dim, key_head_stride, probs_,
      kCublasElemType, keys, static_cast<long long>(rows_per_group) * keys,
      &zero, attn_heads_, kCublasElemType, head_dim,
      static_cast<long long>(rows_per_group) * head_dim, num_kv_heads,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

  launch_merge_heads(attn_out_, attn_heads_, tokens, config_.num_heads, head_dim,
                     stream_);
}

void Model::attention_decode(int layer, int position) {
  const int seq_len = position + 1;
  const int splits =
      options_.attention_splits > 0
          ? std::min(options_.attention_splits, kMaxAttentionSplits)
          : choose_attention_splits(config_.num_kv_heads, seq_len,
                                    multiprocessors_);
  const float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));

  // With a single token the head-major query buffer is already
  // [num_heads][head_dim], and the output lands directly in the layout the
  // output projection wants, so neither transpose kernel runs during decode.
  launch_decode_attention(attn_out_, q_heads_, kv_cache_->view(layer),
                          config_.num_heads, config_.heads_per_kv(), seq_len,
                          scale, options_.kv_layout == KvLayout::kPaged, splits,
                          attention_scratch_, stream_);
}

// ---------------------------------------------------------------------------
// Sampling and readback
// ---------------------------------------------------------------------------

int Model::sample(const float* device_logits, float temperature, int top_k,
                  float top_p, float uniform) {
  // Sampling runs after attention is done with the scratch buffer, and needs
  // only the two floats the packed argmax accumulator occupies.
  launch_sample(sampled_id_, device_logits, config_.vocab_size, temperature,
                top_k, top_p, uniform, attention_scratch_, stream_);
  int32_t id = 0;
  CUDA_CHECK(cudaMemcpyAsync(&id, sampled_id_, sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  LCR_CHECK(id >= 0 && id < config_.vocab_size,
            "sampling produced token id " << id << ", outside the vocabulary");
  return static_cast<int>(id);
}

std::vector<float> Model::logits_to_host(const float* device_logits) const {
  std::vector<float> host(static_cast<size_t>(config_.vocab_size));
  CUDA_CHECK(cudaMemcpy(host.data(), device_logits,
                        host.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return host;
}

}  // namespace lcr
