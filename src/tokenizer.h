// Byte-level BPE tokenizer for Llama 3 checkpoints, driven by tokenizer.json.
//
// Encoding runs four stages, matching what Hugging Face tokenizers does:
//   1. Added tokens (<|begin_of_text|>, <|eot_id|>, ...) are cut out of the
//      text by literal string match.
//   2. Each remaining span is split into pretokens by the Llama 3 regex, which
//      keeps words, digit runs, and punctuation from merging across each other.
//   3. Each pretoken's raw bytes are mapped through the GPT-2 byte-to-unicode
//      alphabet so BPE never has to deal with invalid UTF-8.
//   4. BPE merges are applied in rank order until no adjacent pair is
//      mergeable, and the surviving pieces are looked up in the vocabulary.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.h"

namespace lcr {

class Tokenizer {
 public:
  static Tokenizer from_model_dir(const std::string& model_dir);
  static Tokenizer from_file(const std::string& tokenizer_json_path);
  static Tokenizer from_json_string(const std::string& text);

  // Turns text into token ids. `parse_special` makes added tokens written out
  // in the text, such as <|eot_id|>, encode as themselves rather than as
  // literal angle brackets; the chat prompt builder depends on it. `add_bos`
  // prepends the beginning-of-text token, which is what the checkpoint's
  // post-processor does for a raw completion. A chat prompt already carries its
  // own <|begin_of_text|>, so it must not ask for another.
  std::vector<int> encode(const std::string& text, bool add_bos = false,
                          bool parse_special = true) const;

  std::string decode(const std::vector<int>& ids,
                     bool skip_special = false) const;
  // One token at a time, for streaming output. A single token can end mid
  // codepoint, so the caller has to concatenate before treating the result as
  // UTF-8.
  std::string decode_one(int id, bool skip_special = false) const;

  int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
  int bos_id() const { return bos_id_; }
  // Returns -1 when the token is not in the vocabulary.
  int token_to_id(const std::string& token) const;
  const std::string& id_to_token(int id) const;
  bool is_special(int id) const;

  // Stage 2 on its own, exposed so it can be tested against the reference
  // tokenizer without a vocabulary.
  static std::vector<std::string> split_pretokens(const std::string& text);
  // Stage 3 on its own: raw bytes to the byte-level alphabet.
  static std::string byte_level_encode(const std::string& raw);
  static std::string byte_level_decode(const std::string& encoded);

 private:
  void build_from_json(const std::string& text);
  // Runs stages 3 and 4 on one pretoken, appending its ids to `out`.
  void bpe_encode_pretoken(const std::string& pretoken,
                           std::vector<int>* out) const;

  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> token_to_id_;
  // Merge priority keyed by "left right". A raw space never occurs inside a
  // byte-level token, so that separator is unambiguous.
  std::unordered_map<std::string, int> merge_rank_;

  // Added tokens are matched literally against the input rather than going
  // through BPE, so they are held separately from the BPE vocabulary.
  std::unordered_map<std::string, int> added_by_content_;
  std::unordered_map<int, std::string> added_by_id_;
  std::unordered_set<int> special_ids_;
  // One bit per possible first byte of an added token, so the literal scan
  // rejects most input positions with a single array lookup.
  std::array<bool, 256> added_first_bytes_{};
  // Distinct added-token lengths, longest first, so the literal scan tries the
  // longest match at each position.
  std::vector<size_t> added_lengths_;

  // When the checkpoint sets ignore_merges, a pretoken that is already a whole
  // vocabulary entry is emitted directly instead of being rebuilt by merges.
  bool ignore_merges_ = false;
  int bos_id_ = -1;
};

}  // namespace lcr
