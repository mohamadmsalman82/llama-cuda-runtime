// Checks config.json parsing and the RoPE frequency table, including the
// Llama 3 frequency rescaling, against values recomputed independently here.
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "config.h"
#include "test_util.h"

namespace {

using lcr::ModelConfig;

// The geometry of Llama-3.2-1B-Instruct, trimmed to the fields the runtime
// reads.
const char* kLlama32_1B = R"({
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "bos_token_id": 128000,
  "eos_token_id": [128001, 128008, 128009],
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "intermediate_size": 8192,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 16,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "vocab_size": 128256
})";

void test_parses_llama_3_2_1b() {
  const ModelConfig c = ModelConfig::from_json_string(kLlama32_1B);
  CHECK_EQ(c.hidden_size, 2048);
  CHECK_EQ(c.intermediate_size, 8192);
  CHECK_EQ(c.num_layers, 16);
  CHECK_EQ(c.num_heads, 32);
  CHECK_EQ(c.num_kv_heads, 8);
  CHECK_EQ(c.head_dim, 64);
  CHECK_EQ(c.vocab_size, 128256);
  CHECK_EQ(c.bos_token_id, 128000);
  CHECK_EQ(c.eos_token_ids.size(), static_cast<size_t>(3));
  CHECK(c.is_eos(128009));
  CHECK(!c.is_eos(42));
  CHECK(c.tie_word_embeddings);
  CHECK_NEAR(c.rope_theta, 500000.0, 1e-3);
  CHECK_NEAR(c.rms_norm_eps, 1e-5, 1e-12);

  // Derived geometry. Four query heads share each KV head, so the KV cache is
  // a quarter the size a multi-head model of the same width would need.
  CHECK_EQ(c.heads_per_kv(), 4);
  CHECK_EQ(c.kv_dim(), 512);
  CHECK_EQ(c.q_dim(), 2048);
  // 16 layers, K and V, 512 lanes each, two bytes per bf16 element.
  CHECK_EQ(c.kv_bytes_per_token(2), static_cast<size_t>(16 * 2 * 512 * 2));

  CHECK(c.rope_scaling.enabled);
  CHECK_NEAR(c.rope_scaling.factor, 32.0, 1e-6);
  CHECK_EQ(c.rope_scaling.original_max_position_embeddings, 8192);
}

// head_dim is implied by hidden_size / num_attention_heads when absent, and
// num_key_value_heads defaults to full multi-head attention.
void test_fills_in_optional_fields() {
  const ModelConfig c = ModelConfig::from_json_string(R"({
    "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 22,
    "num_attention_heads": 32, "vocab_size": 32000,
    "max_position_embeddings": 2048, "rope_theta": 10000.0
  })");
  CHECK_EQ(c.head_dim, 64);
  CHECK_EQ(c.num_kv_heads, 32);
  CHECK_EQ(c.heads_per_kv(), 1);
  CHECK(!c.rope_scaling.enabled);
}

void test_rejects_unsupported_configs() {
  CHECK_THROWS("only implements the Llama architecture", {
    ModelConfig::from_json_string(R"({"model_type":"mistral",
      "hidden_size":8,"intermediate_size":8,"num_hidden_layers":1,
      "num_attention_heads":2,"vocab_size":8,"max_position_embeddings":8})");
  });
  CHECK_THROWS("must be a multiple of", {
    ModelConfig::from_json_string(R"({"hidden_size":64,"intermediate_size":64,
      "num_hidden_layers":1,"num_attention_heads":6,"num_key_value_heads":4,
      "vocab_size":8,"max_position_embeddings":8})");
  });
  CHECK_THROWS("only llama3 RoPE scaling", {
    ModelConfig::from_json_string(R"({"hidden_size":64,"intermediate_size":64,
      "num_hidden_layers":1,"num_attention_heads":2,"vocab_size":8,
      "max_position_embeddings":8,
      "rope_scaling":{"rope_type":"yarn","factor":2.0}})");
  });
  CHECK_THROWS("missing the integer field \"vocab_size\"", {
    ModelConfig::from_json_string(R"({"hidden_size":64,"intermediate_size":64,
      "num_hidden_layers":1,"num_attention_heads":2,
      "max_position_embeddings":8})");
  });
  CHECK_THROWS("only the SiLU/SwiGLU MLP", {
    ModelConfig::from_json_string(R"({"hidden_size":64,"intermediate_size":64,
      "num_hidden_layers":1,"num_attention_heads":2,"vocab_size":8,
      "max_position_embeddings":8,"hidden_act":"gelu"})");
  });
}

// Plain RoPE: inv_freq[i] == theta^(-2i/head_dim), untouched by any scaling.
void test_unscaled_rope_frequencies() {
  const ModelConfig c = ModelConfig::from_json_string(R"({
    "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 22,
    "num_attention_heads": 32, "vocab_size": 32000,
    "max_position_embeddings": 2048, "rope_theta": 10000.0
  })");
  const std::vector<float> freq = lcr::compute_rope_inv_freq(c);
  CHECK_EQ(freq.size(), static_cast<size_t>(32));
  for (int i = 0; i < 32; ++i) {
    const double want = std::pow(10000.0, -static_cast<double>(2 * i) / 64.0);
    CHECK_NEAR(freq[static_cast<size_t>(i)], want, want * 1e-6);
  }
}

// Llama 3 scaling: the fast pairs are untouched, the slow pairs are divided by
// the factor, and the band in between is a linear blend of the two.
void test_llama3_scaled_rope_frequencies() {
  const ModelConfig c = ModelConfig::from_json_string(kLlama32_1B);
  const std::vector<float> freq = lcr::compute_rope_inv_freq(c);
  CHECK_EQ(freq.size(), static_cast<size_t>(32));

  const double old_context = 8192.0;
  const double low_wavelen = old_context / 1.0;    // 8192
  const double high_wavelen = old_context / 4.0;   // 2048
  int untouched = 0, stretched = 0, blended = 0;

  for (int i = 0; i < 32; ++i) {
    const double base = std::pow(500000.0, -static_cast<double>(2 * i) / 64.0);
    const double wavelen = 2.0 * M_PI / base;
    double want;
    if (wavelen < high_wavelen) {
      want = base;
      ++untouched;
    } else if (wavelen > low_wavelen) {
      want = base / 32.0;
      ++stretched;
    } else {
      const double smooth = (old_context / wavelen - 1.0) / (4.0 - 1.0);
      want = (1.0 - smooth) * (base / 32.0) + smooth * base;
      ++blended;
    }
    CHECK_NEAR(freq[static_cast<size_t>(i)], want, std::fabs(want) * 1e-5);
  }

  // Every band has to be exercised, otherwise the test would pass on a
  // scaling implementation that only handles one of the three cases.
  CHECK(untouched > 0);
  CHECK(stretched > 0);
  CHECK(blended > 0);

  // The first pair rotates once per token and is far inside the trained
  // window, so scaling leaves it exactly alone.
  CHECK_NEAR(freq[0], 1.0, 1e-9);
  // The slowest pair is stretched by the full factor.
  const double slowest = std::pow(500000.0, -62.0 / 64.0);
  CHECK_NEAR(freq[31], slowest / 32.0, slowest * 1e-6);

  // Frequencies must stay strictly decreasing, otherwise two pairs would carry
  // the same rotation rate.
  for (size_t i = 1; i < freq.size(); ++i) CHECK(freq[i] < freq[i - 1]);
}

}  // namespace

int main() {
  test_parses_llama_3_2_1b();
  test_fills_in_optional_fields();
  test_rejects_unsupported_configs();
  test_unscaled_rope_frequencies();
  test_llama3_scaled_rope_frequencies();
  return test::finish("test_config");
}
