#include "config.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace lcr {
namespace {

using json = nlohmann::json;

std::string join_path(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  if (dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

// eos_token_id is a bare integer in older checkpoints and a list in Llama 3.x,
// which stops on end-of-text, end-of-message, or end-of-turn.
void collect_eos(const json& node, std::vector<int>* out) {
  if (node.is_number_integer()) {
    out->push_back(node.get<int>());
  } else if (node.is_array()) {
    for (const json& item : node) {
      if (item.is_number_integer()) out->push_back(item.get<int>());
    }
  }
}

}  // namespace

bool ModelConfig::is_eos(int token_id) const {
  return std::find(eos_token_ids.begin(), eos_token_ids.end(), token_id) !=
         eos_token_ids.end();
}

ModelConfig ModelConfig::from_json_string(const std::string& text) {
  json doc = json::parse(text, nullptr, false);
  LCR_CHECK(!doc.is_discarded(), "config.json is not valid JSON");
  LCR_CHECK(doc.is_object(), "config.json is not a JSON object");

  if (doc.contains("model_type") && doc["model_type"].is_string()) {
    const std::string type = doc["model_type"].get<std::string>();
    LCR_CHECK(type == "llama",
              "this runtime only implements the Llama architecture, config.json "
              "says model_type=\"" << type << "\"");
  }

  ModelConfig c;
  auto require_int = [&doc](const char* key) {
    LCR_CHECK(doc.contains(key) && doc[key].is_number(),
              "config.json is missing the integer field \"" << key << "\"");
    return doc[key].get<int>();
  };

  c.hidden_size = require_int("hidden_size");
  c.intermediate_size = require_int("intermediate_size");
  c.num_layers = require_int("num_hidden_layers");
  c.num_heads = require_int("num_attention_heads");
  c.vocab_size = require_int("vocab_size");
  c.max_position_embeddings = require_int("max_position_embeddings");

  // Multi-head attention checkpoints omit num_key_value_heads entirely.
  c.num_kv_heads = doc.contains("num_key_value_heads")
                       ? doc["num_key_value_heads"].get<int>()
                       : c.num_heads;
  // Llama 3.2 states head_dim explicitly; earlier checkpoints imply it.
  c.head_dim = doc.contains("head_dim") && doc["head_dim"].is_number()
                   ? doc["head_dim"].get<int>()
                   : c.hidden_size / c.num_heads;

  if (doc.contains("rms_norm_eps")) c.rms_norm_eps = doc["rms_norm_eps"].get<float>();
  if (doc.contains("rope_theta")) c.rope_theta = doc["rope_theta"].get<float>();
  if (doc.contains("tie_word_embeddings")) {
    c.tie_word_embeddings = doc["tie_word_embeddings"].get<bool>();
  }

  if (doc.contains("attention_bias") && doc["attention_bias"].is_boolean()) {
    LCR_CHECK(!doc["attention_bias"].get<bool>(),
              "attention projections with bias are not implemented");
  }
  if (doc.contains("mlp_bias") && doc["mlp_bias"].is_boolean()) {
    LCR_CHECK(!doc["mlp_bias"].get<bool>(),
              "MLP projections with bias are not implemented");
  }
  if (doc.contains("hidden_act") && doc["hidden_act"].is_string()) {
    const std::string act = doc["hidden_act"].get<std::string>();
    LCR_CHECK(act == "silu",
              "only the SiLU/SwiGLU MLP is implemented, config.json says \""
                  << act << "\"");
  }

  if (doc.contains("rope_scaling") && doc["rope_scaling"].is_object()) {
    const json& rs = doc["rope_scaling"];
    // The field is named rope_type in current checkpoints and type in older
    // ones. Either way, "default" means no rescaling.
    std::string type = "default";
    if (rs.contains("rope_type") && rs["rope_type"].is_string()) {
      type = rs["rope_type"].get<std::string>();
    } else if (rs.contains("type") && rs["type"].is_string()) {
      type = rs["type"].get<std::string>();
    }
    if (type != "default" && type != "none") {
      LCR_CHECK(type == "llama3",
                "only llama3 RoPE scaling is implemented, config.json says \""
                    << type << "\"");
      c.rope_scaling.enabled = true;
      if (rs.contains("factor")) {
        c.rope_scaling.factor = rs["factor"].get<float>();
      }
      if (rs.contains("low_freq_factor")) {
        c.rope_scaling.low_freq_factor = rs["low_freq_factor"].get<float>();
      }
      if (rs.contains("high_freq_factor")) {
        c.rope_scaling.high_freq_factor = rs["high_freq_factor"].get<float>();
      }
      if (rs.contains("original_max_position_embeddings")) {
        c.rope_scaling.original_max_position_embeddings =
            rs["original_max_position_embeddings"].get<int>();
      }
    }
  }

  if (doc.contains("bos_token_id") && doc["bos_token_id"].is_number_integer()) {
    c.bos_token_id = doc["bos_token_id"].get<int>();
  }
  if (doc.contains("eos_token_id")) collect_eos(doc["eos_token_id"], &c.eos_token_ids);

  c.validate();
  return c;
}

ModelConfig ModelConfig::load(const std::string& model_dir) {
  const std::string config_path = join_path(model_dir, "config.json");
  std::ifstream in(config_path);
  LCR_CHECK(in.good(), "cannot read " << config_path);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  ModelConfig c = from_json_string(buffer.str());

  // generation_config.json is the authority on stop tokens when it exists; the
  // copy in config.json is often stale.
  std::ifstream gen(join_path(model_dir, "generation_config.json"));
  if (gen.good()) {
    json doc = json::parse(gen, nullptr, false);
    if (!doc.is_discarded() && doc.is_object()) {
      std::vector<int> eos;
      if (doc.contains("eos_token_id")) collect_eos(doc["eos_token_id"], &eos);
      if (!eos.empty()) c.eos_token_ids = eos;
      if (doc.contains("bos_token_id") && doc["bos_token_id"].is_number_integer()) {
        c.bos_token_id = doc["bos_token_id"].get<int>();
      }
    }
  }
  return c;
}

void ModelConfig::validate() const {
  LCR_CHECK(hidden_size > 0 && intermediate_size > 0 && num_layers > 0,
            "config.json has a non-positive hidden_size, intermediate_size, or "
            "num_hidden_layers");
  LCR_CHECK(num_heads > 0 && num_kv_heads > 0,
            "config.json has a non-positive head count");
  LCR_CHECK(num_heads % num_kv_heads == 0,
            "num_attention_heads (" << num_heads
                                    << ") must be a multiple of "
                                       "num_key_value_heads ("
                                    << num_kv_heads << ")");
  LCR_CHECK(head_dim > 0 && head_dim % 2 == 0,
            "head_dim must be positive and even for RoPE, got " << head_dim);
  LCR_CHECK(vocab_size > 0, "vocab_size must be positive");
  LCR_CHECK(rms_norm_eps > 0.0f, "rms_norm_eps must be positive");
  LCR_CHECK(rope_theta > 1.0f, "rope_theta must be greater than one");
  if (rope_scaling.enabled) {
    LCR_CHECK(rope_scaling.factor > 0.0f, "RoPE scaling factor must be positive");
    LCR_CHECK(rope_scaling.high_freq_factor > rope_scaling.low_freq_factor,
              "RoPE high_freq_factor must exceed low_freq_factor");
    LCR_CHECK(rope_scaling.original_max_position_embeddings > 0,
              "RoPE original_max_position_embeddings must be positive");
  }
}

std::string ModelConfig::summary() const {
  std::ostringstream oss;
  oss << "layers=" << num_layers << " hidden=" << hidden_size
      << " ffn=" << intermediate_size << " heads=" << num_heads << "/"
      << num_kv_heads << " head_dim=" << head_dim << " vocab=" << vocab_size
      << " rope_theta=" << rope_theta
      << (rope_scaling.enabled ? " rope_scaling=llama3" : "")
      << (tie_word_embeddings ? " tied_embeddings" : "");
  return oss.str();
}

std::vector<float> compute_rope_inv_freq(const ModelConfig& config) {
  const int pairs = config.head_dim / 2;
  std::vector<float> inv_freq(static_cast<size_t>(pairs));
  for (int i = 0; i < pairs; ++i) {
    const double exponent = static_cast<double>(2 * i) / config.head_dim;
    inv_freq[static_cast<size_t>(i)] =
        static_cast<float>(1.0 / std::pow(static_cast<double>(config.rope_theta),
                                          exponent));
  }
  if (!config.rope_scaling.enabled) return inv_freq;

  const RopeScaling& rs = config.rope_scaling;
  const double old_context = rs.original_max_position_embeddings;
  const double low_wavelen = old_context / rs.low_freq_factor;
  const double high_wavelen = old_context / rs.high_freq_factor;
  const double two_pi = 2.0 * M_PI;

  for (int i = 0; i < pairs; ++i) {
    const double freq = inv_freq[static_cast<size_t>(i)];
    const double wavelen = two_pi / freq;
    double scaled;
    if (wavelen < high_wavelen) {
      // Short wavelength, rotates fast within the original window. Untouched.
      scaled = freq;
    } else if (wavelen > low_wavelen) {
      // Long wavelength, would wrap past the original window. Fully stretched.
      scaled = freq / rs.factor;
    } else {
      // In between, blend the stretched and unstretched frequency.
      const double smooth = (old_context / wavelen - rs.low_freq_factor) /
                            (rs.high_freq_factor - rs.low_freq_factor);
      scaled = (1.0 - smooth) * (freq / rs.factor) + smooth * freq;
    }
    inv_freq[static_cast<size_t>(i)] = static_cast<float>(scaled);
  }
  return inv_freq;
}

}  // namespace lcr
