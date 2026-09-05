// Reports what a checkpoint contains and what running it will cost, without
// needing a GPU.
//
// Useful before the first run on a new machine: it says whether the directory
// has everything the runtime needs, in the right shapes and the right
// precision, and prints the byte counts that determine decode speed. If the
// tensor inventory here is clean, loading will not fail.
//
//   lcr-inspect --model models/Llama-3.2-1B-Instruct
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "common.h"
#include "config.h"
#include "safetensors.h"
#include "tokenizer.h"

namespace {

std::string join_path(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  return dir.back() == '/' ? dir + name : dir + "/" + name;
}

int run(int argc, char** argv) {
  std::string model_dir;
  bool list_tensors = false;
  int max_seq = 4096;
  int page_size = 16;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto value = [&]() {
      LCR_CHECK(i + 1 < argc, flag << " needs a value");
      return std::string(argv[++i]);
    };
    if (flag == "--model") model_dir = value();
    else if (flag == "--tensors") list_tensors = true;
    else if (flag == "--max-seq") max_seq = std::stoi(value());
    else if (flag == "--page-size") page_size = std::stoi(value());
    else {
      std::fprintf(stderr,
                   "usage: lcr-inspect --model DIR [--tensors] [--max-seq N] "
                   "[--page-size N]\n");
      return 2;
    }
  }
  LCR_CHECK(!model_dir.empty(), "--model is required");

  const lcr::ModelConfig config = lcr::ModelConfig::load(model_dir);
  std::printf("config\n  %s\n", config.summary().c_str());
  std::printf("  context window     %d positions\n",
              config.max_position_embeddings);
  std::printf("  stop tokens        ");
  for (size_t i = 0; i < config.eos_token_ids.size(); ++i) {
    std::printf("%s%d", i ? ", " : "", config.eos_token_ids[i]);
  }
  std::printf("\n  grouped-query      %d query heads per key/value head\n",
              config.heads_per_kv());

  const std::vector<lcr::WeightSpec> specs = lcr::required_weights(config);
  const lcr::SafetensorsArchive archive(model_dir);

  // Recomputed from the shapes the checkpoint actually declares rather than
  // from the config, so a mismatch shows up as a wrong number here.
  size_t required_bytes = 0;
  size_t element_size = 0;
  std::vector<std::string> problems;
  for (const lcr::WeightSpec& spec : specs) {
    const lcr::TensorView* tensor = archive.find(spec.name);
    if (tensor == nullptr) {
      problems.push_back("missing " + spec.name);
      continue;
    }
    try {
      tensor->expect_shape(spec.shape);
    } catch (const std::exception& e) {
      problems.push_back(e.what());
      continue;
    }
    required_bytes += tensor->nbytes;
    element_size = lcr::dtype_size(tensor->dtype);
  }

  const bool tied = !archive.has("lm_head.weight");
  std::printf("\ncheckpoint\n");
  std::printf("  tensors in file    %zu\n", archive.tensor_count());
  std::printf("  tensors required   %zu%s\n", specs.size(),
              tied ? " (output head tied to the embeddings)" : " plus lm_head");
  std::printf("  bytes on disk      %s\n",
              lcr::format_bytes(archive.total_bytes()).c_str());
  std::printf("  element type       %s\n",
              specs.empty() || archive.find(specs[0].name) == nullptr
                  ? "unknown"
                  : lcr::dtype_name(archive.get(specs[0].name).dtype));
  if (problems.empty()) {
    std::printf("  every required tensor is present with the expected shape\n");
  } else {
    std::printf("  %zu problem(s):\n", problems.size());
    for (const std::string& problem : problems) {
      std::printf("    %s\n", problem.c_str());
    }
  }

  // The number that sets the decode ceiling. Every weight in the forward pass
  // crosses the memory bus once per generated token; with tied embeddings the
  // output head is the embedding matrix, so that is the entire model.
  size_t per_token = 0;
  for (int layer = 0; layer < config.num_layers; ++layer) {
    per_token += static_cast<size_t>(config.hidden_size) * 2;
    per_token += static_cast<size_t>(config.q_dim()) * config.hidden_size;
    per_token += static_cast<size_t>(config.kv_dim()) * config.hidden_size * 2;
    per_token += static_cast<size_t>(config.hidden_size) * config.q_dim();
    per_token +=
        static_cast<size_t>(config.intermediate_size) * config.hidden_size * 3;
  }
  per_token += config.hidden_size;
  per_token += static_cast<size_t>(config.vocab_size) * config.hidden_size;
  per_token *= element_size ? element_size : 2;

  std::printf("\ndecode cost per token\n");
  std::printf("  weights streamed   %s\n",
              lcr::format_bytes(per_token).c_str());
  std::printf("  cache per position %s\n",
              lcr::format_bytes(config.kv_bytes_per_token(
                                    element_size ? element_size : 2))
                  .c_str());
  std::printf("  at 1000 positions  %s of cache on top of the weights\n",
              lcr::format_bytes(1000 * config.kv_bytes_per_token(
                                          element_size ? element_size : 2))
                  .c_str());
  std::printf("\n  A GPU with 1 TB/s of memory bandwidth therefore cannot\n"
              "  exceed %.0f tokens/s on this model, whatever the kernels do.\n",
              1e12 / static_cast<double>(per_token));

  std::printf("\nkv cache\n");
  std::printf("  full context, contiguous   %s\n",
              lcr::format_bytes(
                  static_cast<size_t>(config.max_position_embeddings) *
                  config.kv_bytes_per_token(element_size ? element_size : 2))
                  .c_str());
  std::printf("  --max-seq %d, contiguous  %s\n", max_seq,
              lcr::format_bytes(static_cast<size_t>(max_seq) *
                                config.kv_bytes_per_token(
                                    element_size ? element_size : 2))
                  .c_str());
  std::printf("  paged, page size %d, per page across all layers  %s\n",
              page_size,
              lcr::format_bytes(static_cast<size_t>(page_size) *
                                config.kv_bytes_per_token(
                                    element_size ? element_size : 2))
                  .c_str());

  const std::string tokenizer_path = join_path(model_dir, "tokenizer.json");
  const lcr::Tokenizer tokenizer = lcr::Tokenizer::from_file(tokenizer_path);
  std::printf("\ntokenizer\n  %d entries, begin-of-text is id %d\n",
              tokenizer.vocab_size(), tokenizer.bos_id());
  LCR_CHECK(tokenizer.vocab_size() == config.vocab_size,
            "tokenizer has " << tokenizer.vocab_size()
                             << " entries but config.json says "
                             << config.vocab_size);

  if (list_tensors) {
    std::printf("\ntensors\n");
    for (const std::string& name : archive.names()) {
      const lcr::TensorView& tensor = archive.get(name);
      std::printf("  %-52s %-6s %-16s %s\n", name.c_str(),
                  lcr::dtype_name(tensor.dtype), tensor.shape_string().c_str(),
                  lcr::format_bytes(tensor.nbytes).c_str());
    }
  }

  return problems.empty() ? 0 : 1;
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
