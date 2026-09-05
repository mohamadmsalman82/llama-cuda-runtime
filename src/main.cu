// Command-line front end: generate text, or measure how fast it generates.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cuda_utils.cuh"
#include "model.h"
#include "tokenizer.h"

namespace {

using lcr::Model;

struct Args {
  std::string model_dir;
  std::string prompt = "The capital of France is";
  std::string system_prompt;
  bool chat = false;
  bool benchmark = false;
  int max_new_tokens = 128;
  float temperature = 0.0f;
  int top_k = 0;
  float top_p = 0.95f;
  uint64_t seed = 0;
  int max_seq = 4096;
  int prefill_chunk = 256;
  int page_size = 16;
  bool paged = false;
  int attention_splits = 0;
  int device = 0;
  // Overrides the peak bandwidth reported by the driver, which is wrong on
  // some parts. In GB/s.
  double peak_bandwidth_gb = 0.0;
  bool quiet = false;
};

[[noreturn]] void usage(int code) {
  std::fprintf(
      stderr,
      "usage: llama-run --model <dir> [options]\n"
      "\n"
      "  --model DIR            checkpoint directory (config.json, tokenizer.json,\n"
      "                         model.safetensors)\n"
      "  --prompt TEXT          prompt text (default: a short completion)\n"
      "  --system TEXT          system message, chat mode only\n"
      "  --chat                 wrap the prompt in the Llama 3 chat template\n"
      "  --max-tokens N         tokens to generate (default 128)\n"
      "  --temperature T        0 means greedy, which is the default\n"
      "  --top-k K              0 disables the top-k cut (default)\n"
      "  --top-p P              nucleus threshold (default 0.95)\n"
      "  --seed S               random seed for sampling\n"
      "  --max-seq N            KV cache capacity in positions (default 4096)\n"
      "  --prefill-chunk N      prompt positions per prefill pass (default 256)\n"
      "  --paged                use the paged KV cache\n"
      "  --page-size N          positions per page (default 16)\n"
      "  --splits N             decode attention blocks per KV head, 0 to choose\n"
      "  --device N             CUDA device index\n"
      "  --peak-bandwidth GB    override the theoretical peak used in the report\n"
      "  --bench                print timing and bandwidth instead of only text\n"
      "  --quiet                suppress the loading banner\n");
  std::exit(code);
}

std::string require_value(int argc, char** argv, int* i) {
  LCR_CHECK(*i + 1 < argc, argv[*i] << " needs a value");
  return argv[++(*i)];
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--model") args.model_dir = require_value(argc, argv, &i);
    else if (flag == "--prompt") args.prompt = require_value(argc, argv, &i);
    else if (flag == "--system") args.system_prompt = require_value(argc, argv, &i);
    else if (flag == "--chat") args.chat = true;
    else if (flag == "--bench") args.benchmark = true;
    else if (flag == "--quiet") args.quiet = true;
    else if (flag == "--paged") args.paged = true;
    else if (flag == "--max-tokens") args.max_new_tokens = std::stoi(require_value(argc, argv, &i));
    else if (flag == "--temperature") args.temperature = std::stof(require_value(argc, argv, &i));
    else if (flag == "--top-k") args.top_k = std::stoi(require_value(argc, argv, &i));
    else if (flag == "--top-p") args.top_p = std::stof(require_value(argc, argv, &i));
    else if (flag == "--seed") args.seed = std::stoull(require_value(argc, argv, &i));
    else if (flag == "--max-seq") args.max_seq = std::stoi(require_value(argc, argv, &i));
    else if (flag == "--prefill-chunk") args.prefill_chunk = std::stoi(require_value(argc, argv, &i));
    else if (flag == "--page-size") args.page_size = std::stoi(require_value(argc, argv, &i));
    else if (flag == "--splits") args.attention_splits = std::stoi(require_value(argc, argv, &i));
    else if (flag == "--device") args.device = std::stoi(require_value(argc, argv, &i));
    else if (flag == "--peak-bandwidth") args.peak_bandwidth_gb = std::stod(require_value(argc, argv, &i));
    else if (flag == "-h" || flag == "--help") usage(0);
    else {
      std::fprintf(stderr, "unknown option: %s\n", flag.c_str());
      usage(2);
    }
  }
  if (args.model_dir.empty()) {
    std::fprintf(stderr, "--model is required\n");
    usage(2);
  }
  return args;
}

// The Llama 3 chat format. Each message is a header, two newlines, the body,
// and an end-of-turn marker; the prompt ends with an empty assistant header so
// the model continues as the assistant.
std::string build_chat_prompt(const std::string& system,
                              const std::string& user) {
  std::string out = "<|begin_of_text|>";
  if (!system.empty()) {
    out += "<|start_header_id|>system<|end_header_id|>\n\n" + system +
           "<|eot_id|>";
  }
  out += "<|start_header_id|>user<|end_header_id|>\n\n" + user + "<|eot_id|>";
  out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  return out;
}

// A token can end in the middle of a multi-byte character, so output is held
// back until it forms complete UTF-8 rather than printing replacement
// characters mid-word.
class Utf8Streamer {
 public:
  void feed(const std::string& bytes) {
    pending_ += bytes;
    const size_t complete = complete_prefix_length(pending_);
    if (complete == 0) return;
    std::fwrite(pending_.data(), 1, complete, stdout);
    std::fflush(stdout);
    pending_.erase(0, complete);
  }
  void flush() {
    if (pending_.empty()) return;
    std::fwrite(pending_.data(), 1, pending_.size(), stdout);
    std::fflush(stdout);
    pending_.clear();
  }

 private:
  static size_t complete_prefix_length(const std::string& text) {
    size_t i = 0;
    while (i < text.size()) {
      const unsigned char lead = static_cast<unsigned char>(text[i]);
      size_t width = 1;
      if ((lead & 0x80) == 0x00) width = 1;
      else if ((lead & 0xE0) == 0xC0) width = 2;
      else if ((lead & 0xF0) == 0xE0) width = 3;
      else if ((lead & 0xF8) == 0xF0) width = 4;
      else width = 1;  // stray continuation byte, pass it through
      if (i + width > text.size()) break;
      i += width;
    }
    return i;
  }

  std::string pending_;
};

void print_report(const Args& args, const Model& model,
                  const lcr::DeviceInfo& device, double prefill_ms,
                  int prompt_tokens, double decode_ms, int decode_steps,
                  int final_length) {
  const lcr::ForwardStats& stats = model.stats();
  const double peak = args.peak_bandwidth_gb > 0.0
                          ? args.peak_bandwidth_gb * 1e9
                          : device.peak_bandwidth_bytes_per_second();

  // Bytes the memory bus has to move per decoded token: every weight, once,
  // plus the cache read, which grows by one position per step. Averaged over
  // the run the cache term sits at the mean sequence length.
  const double mean_length =
      prompt_tokens + (decode_steps + 1) / 2.0;
  const double bytes_per_token =
      static_cast<double>(stats.weight_bytes_per_token) +
      mean_length * static_cast<double>(stats.kv_bytes_per_position);
  const double seconds_per_token =
      decode_ms / 1000.0 / std::max(1, decode_steps);
  const double achieved = bytes_per_token / seconds_per_token;

  std::printf("\n");
  std::printf("device            %s\n", device.summary().c_str());
  std::printf("model             %s\n", model.config().summary().c_str());
  std::printf("weights resident  %s\n",
              lcr::format_bytes(model.weight_bytes()).c_str());
  std::printf("activation arena  %s\n", model.arena().usage_string().c_str());
  std::printf("kv cache          %s\n",
              model.kv_cache().usage_string().c_str());
  std::printf("\n");
  std::printf("prefill           %d tokens in %.2f ms (%.0f tokens/s)\n",
              prompt_tokens, prefill_ms,
              prompt_tokens / (prefill_ms / 1000.0));
  std::printf("decode            %d steps in %.2f ms (%.3f ms/token, "
              "%.1f tokens/s)\n",
              decode_steps, decode_ms,
              decode_ms / std::max(1, decode_steps),
              decode_steps / (decode_ms / 1000.0));
  std::printf("\n");
  std::printf("bytes per token   %s weights + %s cache at length %d\n",
              lcr::format_bytes(stats.weight_bytes_per_token).c_str(),
              lcr::format_bytes(stats.kv_bytes_at(final_length)).c_str(),
              final_length);
  std::printf("achieved          %.0f GB/s\n", achieved / 1e9);
  std::printf("theoretical peak  %.0f GB/s\n", peak / 1e9);
  std::printf("utilization       %.1f%% of peak\n", 100.0 * achieved / peak);
  std::printf("\n");
  std::printf("kv cache at this length, priced both ways:\n");
  std::printf("  contiguous, full %d-token context   %s\n",
              model.config().max_position_embeddings,
              lcr::format_bytes(
                  lcr::KvCache::bytes_for(model.config(),
                                          model.config().max_position_embeddings))
                  .c_str());
  std::printf("  contiguous, --max-seq %d            %s\n", args.max_seq,
              lcr::format_bytes(
                  lcr::KvCache::bytes_for(model.config(), args.max_seq))
                  .c_str());
  std::printf("  paged, page size %d                 %s\n", args.page_size,
              lcr::format_bytes(lcr::KvCache::bytes_for(
                                    model.config(),
                                    ((final_length + args.page_size - 1) /
                                     args.page_size) *
                                        args.page_size))
                  .c_str());
  std::printf("  strictly needed                     %s\n",
              lcr::format_bytes(
                  lcr::KvCache::bytes_for(model.config(), final_length))
                  .c_str());
}

int run(int argc, char** argv) {
  const Args args = parse_args(argc, argv);

  const lcr::DeviceInfo device = lcr::query_device(args.device);
  if (!args.quiet) {
    std::fprintf(stderr, "device: %s\n", device.summary().c_str());
    std::fprintf(stderr, "loading %s\n", args.model_dir.c_str());
  }

  const auto load_start = std::chrono::steady_clock::now();
  const lcr::Tokenizer tokenizer = lcr::Tokenizer::from_model_dir(args.model_dir);

  Model::Options options;
  options.device = args.device;
  options.max_seq = args.max_seq;
  options.prefill_chunk = args.prefill_chunk;
  options.page_size = args.page_size;
  options.attention_splits = args.attention_splits;
  options.kv_layout =
      args.paged ? lcr::KvLayout::kPaged : lcr::KvLayout::kContiguous;
  Model model(args.model_dir, options);
  const double load_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - load_start)
          .count();
  if (!args.quiet) {
    std::fprintf(stderr, "loaded %s of weights in %.1f s\n",
                 lcr::format_bytes(model.weight_bytes()).c_str(),
                 load_ms / 1000.0);
  }

  const std::string prompt_text =
      args.chat ? build_chat_prompt(args.system_prompt, args.prompt)
                : args.prompt;
  // A chat prompt carries its own begin-of-text marker; a raw completion needs
  // one prepended, which is what the checkpoint's post-processor does.
  const std::vector<int> prompt =
      tokenizer.encode(prompt_text, /*add_bos=*/!args.chat,
                       /*parse_special=*/true);
  LCR_CHECK(!prompt.empty(), "the prompt encoded to no tokens");
  LCR_CHECK(prompt.size() < static_cast<size_t>(args.max_seq),
            "prompt of " << prompt.size()
                         << " tokens does not fit in --max-seq "
                         << args.max_seq);

  Utf8Streamer streamer;
  if (!args.chat) streamer.feed(prompt_text);

  const float* logits = model.prefill(prompt);
  const double prefill_ms = model.stats().last_prefill_ms;

  std::mt19937_64 rng(args.seed);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

  double decode_ms = 0.0;
  int generated = 0;
  int decode_steps = 0;
  const int budget =
      std::min(args.max_new_tokens,
               args.max_seq - static_cast<int>(prompt.size()));

  // Prefill already produced the logits for the first generated token, so each
  // pass through this loop emits one token and then runs the decode step that
  // produces the next one. The final token needs no decode after it, which is
  // why the step count is one below the token count.
  for (int step = 0; step < budget; ++step) {
    const int token = model.sample(logits, args.temperature, args.top_k,
                                   args.top_p, uniform(rng));
    if (model.config().is_eos(token)) break;
    streamer.feed(tokenizer.decode_one(token, true));
    ++generated;
    if (generated >= budget) break;
    logits = model.decode(token);
    decode_ms += model.stats().last_decode_ms;
    ++decode_steps;
  }
  streamer.flush();
  std::printf("\n");

  if (args.benchmark) {
    print_report(args, model, device, prefill_ms, static_cast<int>(prompt.size()),
                 decode_ms, decode_steps, model.position());
  }
  (void)generated;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
