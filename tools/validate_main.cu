// Compares this runtime's activations against a PyTorch reference dump.
//
// Text that reads well is not evidence of a correct implementation. A rotary
// pairing that swaps the two halves, a norm that divides by the wrong count, a
// grouped-query mapping off by one: all of them still produce fluent English.
// The only way to know is to look at the numbers layer by layer, which is what
// this does. It runs the prompt from the dump, captures the residual stream
// after every layer, and reports how far each one has drifted.
//
//   python3 tools/dump_reference.py models/Llama-3.2-1B-Instruct ref/
//   ./build/lcr-validate --model models/Llama-3.2-1B-Instruct --reference ref/
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "model.h"
#include "tokenizer.h"

namespace {

using json = nlohmann::json;

struct ReferenceTensor {
  std::string name;
  int rows = 0;
  int width = 0;
  std::vector<float> values;
};

struct Reference {
  std::string dtype;
  std::string prompt;
  std::vector<int> input_ids;
  std::vector<ReferenceTensor> tensors;
  std::vector<int> greedy_continuation;

  const ReferenceTensor* find(const std::string& name) const {
    for (const ReferenceTensor& tensor : tensors) {
      if (tensor.name == name) return &tensor;
    }
    return nullptr;
  }
};

std::string join_path(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  return dir.back() == '/' ? dir + name : dir + "/" + name;
}

Reference load_reference(const std::string& dir) {
  const std::string manifest_path = join_path(dir, "manifest.json");
  std::ifstream in(manifest_path);
  LCR_CHECK(in.good(), "cannot read " << manifest_path);
  json doc = json::parse(in, nullptr, false);
  LCR_CHECK(!doc.is_discarded(), manifest_path << " is not valid JSON");

  Reference reference;
  reference.dtype = doc.value("dtype", "unknown");
  reference.prompt = doc.value("prompt", "");
  reference.input_ids = doc["input_ids"].get<std::vector<int>>();
  if (doc.contains("greedy_continuation")) {
    reference.greedy_continuation =
        doc["greedy_continuation"].get<std::vector<int>>();
  }

  for (const json& entry : doc["tensors"]) {
    ReferenceTensor tensor;
    tensor.name = entry["name"].get<std::string>();
    const auto shape = entry["shape"].get<std::vector<int>>();
    LCR_CHECK(shape.size() == 2,
              "tensor \"" << tensor.name << "\" in the dump is not 2-D");
    tensor.rows = shape[0];
    tensor.width = shape[1];

    const std::string path =
        join_path(dir, entry["file"].get<std::string>());
    std::ifstream file(path, std::ios::binary);
    LCR_CHECK(file.good(), "cannot read " << path);
    const size_t count = static_cast<size_t>(tensor.rows) * tensor.width;
    tensor.values.resize(count);
    file.read(reinterpret_cast<char*>(tensor.values.data()),
              static_cast<std::streamsize>(count * sizeof(float)));
    LCR_CHECK(static_cast<size_t>(file.gcount()) == count * sizeof(float),
              path << " is shorter than the shape in the manifest");
    reference.tensors.push_back(std::move(tensor));
  }
  return reference;
}

struct Comparison {
  double max_abs = 0.0;
  double rms_error = 0.0;
  double rms_reference = 0.0;
  double min_cosine = 1.0;

  // Error as a fraction of the signal, which is the number that matters: an
  // absolute error of 0.01 is nothing in a residual stream whose values run to
  // 30, and catastrophic in one that runs to 0.05.
  double relative_rms() const {
    return rms_reference > 0.0 ? rms_error / rms_reference : 0.0;
  }
};

Comparison compare(const std::vector<float>& ours,
                   const std::vector<float>& theirs, int rows, int width) {
  Comparison result;
  double squared_error = 0.0;
  double squared_reference = 0.0;

  for (int row = 0; row < rows; ++row) {
    double dot = 0.0, ours_norm = 0.0, theirs_norm = 0.0;
    for (int i = 0; i < width; ++i) {
      const size_t index = static_cast<size_t>(row) * width + i;
      const double a = ours[index];
      const double b = theirs[index];
      result.max_abs = std::max(result.max_abs, std::fabs(a - b));
      squared_error += (a - b) * (a - b);
      squared_reference += b * b;
      dot += a * b;
      ours_norm += a * a;
      theirs_norm += b * b;
    }
    if (ours_norm > 0.0 && theirs_norm > 0.0) {
      const double cosine = dot / std::sqrt(ours_norm * theirs_norm);
      result.min_cosine = std::min(result.min_cosine, cosine);
    }
  }
  const double count = static_cast<double>(rows) * width;
  result.rms_error = std::sqrt(squared_error / count);
  result.rms_reference = std::sqrt(squared_reference / count);
  return result;
}

std::vector<int> top_k_indices(const std::vector<float>& values, int k) {
  std::vector<int> order(values.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
  const int limit = std::min<int>(k, static_cast<int>(order.size()));
  std::partial_sort(order.begin(), order.begin() + limit, order.end(),
                    [&values](int a, int b) { return values[a] > values[b]; });
  order.resize(static_cast<size_t>(limit));
  return order;
}

int run(int argc, char** argv) {
  std::string model_dir;
  std::string reference_dir;
  double max_relative_rms = 0.02;
  double min_cosine = 0.999;
  int device = 0;
  int generate = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto value = [&]() {
      LCR_CHECK(i + 1 < argc, flag << " needs a value");
      return std::string(argv[++i]);
    };
    if (flag == "--model") model_dir = value();
    else if (flag == "--reference") reference_dir = value();
    else if (flag == "--max-relative-rms") max_relative_rms = std::stod(value());
    else if (flag == "--min-cosine") min_cosine = std::stod(value());
    else if (flag == "--device") device = std::stoi(value());
    else if (flag == "--generate") generate = std::stoi(value());
    else {
      std::fprintf(stderr,
                   "usage: lcr-validate --model DIR --reference DIR "
                   "[--max-relative-rms X] [--min-cosine X] [--generate N]\n");
      return 2;
    }
  }
  LCR_CHECK(!model_dir.empty() && !reference_dir.empty(),
            "--model and --reference are both required");

  const Reference reference = load_reference(reference_dir);
  std::printf("reference: %zu tokens, %s, prompt %s\n",
              reference.input_ids.size(), reference.dtype.c_str(),
              reference.prompt.c_str());

  lcr::Model::Options options;
  options.device = device;
  // One chunk, so every tap covers the whole prompt.
  options.prefill_chunk = static_cast<int>(reference.input_ids.size());
  options.max_seq = std::max<int>(
      options.prefill_chunk + generate + 1, options.prefill_chunk + 1);
  lcr::Model model(model_dir, options);
  model.set_capture_activations(true);

  const float* device_logits = model.prefill(reference.input_ids);
  const std::vector<float> logits = model.logits_to_host(device_logits);

  std::printf("\n%-14s %10s %12s %12s %10s\n", "tensor", "max abs", "rms error",
              "rel rms", "min cos");
  std::printf("%s\n", std::string(62, '-').c_str());

  bool ok = true;
  for (const lcr::Model::Tap& tap : model.activation_taps()) {
    const ReferenceTensor* expected = reference.find(tap.name);
    if (expected == nullptr) {
      std::printf("%-14s  (not in the reference dump)\n", tap.name.c_str());
      continue;
    }
    LCR_CHECK(expected->rows == tap.tokens && expected->width == tap.width,
              "shape mismatch for \"" << tap.name << "\": reference is "
                                      << expected->rows << "x"
                                      << expected->width << ", ours is "
                                      << tap.tokens << "x" << tap.width);

    const Comparison c =
        compare(tap.values, expected->values, tap.tokens, tap.width);
    const bool passed =
        c.relative_rms() <= max_relative_rms && c.min_cosine >= min_cosine;
    ok = ok && passed;
    std::printf("%-14s %10.5f %12.6f %12.2e %10.6f %s\n", tap.name.c_str(),
                c.max_abs, c.rms_error, c.relative_rms(), c.min_cosine,
                passed ? "" : "  FAIL");
  }

  // Logits are produced for the last position only, which is the one that
  // decides the next token.
  const ReferenceTensor* reference_logits = reference.find("logits");
  if (reference_logits != nullptr) {
    const size_t last = static_cast<size_t>(reference_logits->rows - 1) *
                        reference_logits->width;
    const std::vector<float> expected(
        reference_logits->values.begin() + static_cast<long>(last),
        reference_logits->values.end());
    const Comparison c = compare(logits, expected, 1, reference_logits->width);
    const bool passed =
        c.relative_rms() <= max_relative_rms && c.min_cosine >= min_cosine;
    ok = ok && passed;
    std::printf("%-14s %10.5f %12.6f %12.2e %10.6f %s\n", "logits", c.max_abs,
                c.rms_error, c.relative_rms(), c.min_cosine,
                passed ? "" : "  FAIL");

    const std::vector<int> ours_top = top_k_indices(logits, 5);
    const std::vector<int> theirs_top = top_k_indices(expected, 5);
    std::printf("\ntop-5 next tokens\n  ours   ");
    for (int id : ours_top) std::printf("%6d", id);
    std::printf("\n  theirs ");
    for (int id : theirs_top) std::printf("%6d", id);
    std::printf("\n");
    if (ours_top[0] != theirs_top[0]) {
      std::printf("  greedy token differs\n");
      ok = false;
    }
  }

  // Optionally continue greedily and compare the token sequence, which catches
  // anything that only shows up once the KV cache has history in it.
  if (generate > 0 && !reference.greedy_continuation.empty()) {
    const lcr::Tokenizer tokenizer = lcr::Tokenizer::from_model_dir(model_dir);
    std::vector<int> produced;
    const float* current = device_logits;
    const int steps = std::min<int>(
        generate, static_cast<int>(reference.greedy_continuation.size()));
    for (int step = 0; step < steps; ++step) {
      const int token = model.sample(current, 0.0f, 0, 1.0f, 0.0f);
      produced.push_back(token);
      if (model.config().is_eos(token)) break;
      current = model.decode(token);
    }
    std::printf("\ngreedy continuation\n  ours   %s\n",
                tokenizer.decode(produced, true).c_str());
    std::vector<int> theirs(
        reference.greedy_continuation.begin(),
        reference.greedy_continuation.begin() +
            static_cast<long>(std::min<size_t>(produced.size(),
                                               reference.greedy_continuation.size())));
    std::printf("  theirs %s\n", tokenizer.decode(theirs, true).c_str());
    size_t matched = 0;
    while (matched < produced.size() && matched < theirs.size() &&
           produced[matched] == theirs[matched]) {
      ++matched;
    }
    std::printf("  %zu of %zu tokens identical\n", matched, theirs.size());
  }

  std::printf("\n%s\n", ok ? "every layer is within tolerance"
                           : "one or more layers are outside tolerance");
  return ok ? 0 : 1;
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
