// Command-line front end for the tokenizer, used by tools/check_tokenizer.py to
// diff this implementation against the reference Hugging Face tokenizer.
//
//   lcr-tokenize <tokenizer.json> [--no-special] [--add-bos] < corpus.jsonl
//
// Each input line is one JSON-encoded string. For each line it prints the token
// ids separated by spaces. A line whose decode does not reproduce the input
// prints "ROUNDTRIP_FAILED" instead, so the checker sees it as a mismatch.
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tokenizer.h"

int main(int argc, char** argv) {
  std::string tokenizer_path;
  bool parse_special = true;
  bool add_bos = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--no-special") {
      parse_special = false;
    } else if (arg == "--add-bos") {
      add_bos = true;
    } else if (tokenizer_path.empty()) {
      tokenizer_path = arg;
    } else {
      std::fprintf(stderr, "unexpected argument: %s\n", arg.c_str());
      return 2;
    }
  }
  if (tokenizer_path.empty()) {
    std::fprintf(stderr,
                 "usage: lcr-tokenize <tokenizer.json> [--no-special] "
                 "[--add-bos] < corpus.jsonl\n");
    return 2;
  }

  try {
    const lcr::Tokenizer tokenizer = lcr::Tokenizer::from_file(tokenizer_path);
    std::ios::sync_with_stdio(false);

    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      nlohmann::json node = nlohmann::json::parse(line, nullptr, false);
      if (node.is_discarded() || !node.is_string()) {
        std::fprintf(stderr, "input line is not a JSON string: %s\n",
                     line.c_str());
        return 2;
      }
      const std::string text = node.get<std::string>();
      const std::vector<int> ids = tokenizer.encode(text, add_bos, parse_special);

      if (tokenizer.decode(ids) != text) {
        std::cout << "ROUNDTRIP_FAILED\n";
        continue;
      }
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << ' ';
        std::cout << ids[i];
      }
      std::cout << '\n';
    }
    std::cout.flush();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
