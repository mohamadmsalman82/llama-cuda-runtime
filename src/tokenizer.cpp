#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <queue>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

#include "unicode_data.h"

namespace lcr {
namespace {

using json = nlohmann::json;

// The exact pretokenizer pattern this file implements by hand. tokenizer.json
// carries the pattern as a string; if a checkpoint ships a different one the
// hand-written matcher below would silently produce the wrong split, so the
// loader compares and refuses instead.
constexpr const char* kLlama3SplitRegex =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::string join_path(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  if (dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level alphabet
//
// BPE is defined over characters, but the input is arbitrary bytes. The
// byte-level trick maps all 256 byte values onto printable non-space
// codepoints, so every byte sequence becomes a valid string and no vocabulary
// entry can contain a literal space or control character. Bytes that are
// already printable map to themselves; the remaining 68 land at U+0100 upward.
// ---------------------------------------------------------------------------

struct ByteAlphabet {
  std::array<uint32_t, 256> to_codepoint{};
  // Sparse reverse map. The largest codepoint used is 0x143, so a flat array
  // covering that range is both smaller and faster than a hash map.
  std::array<int16_t, 0x200> to_byte{};

  ByteAlphabet() {
    std::array<bool, 256> assigned{};
    auto identity = [&](int lo, int hi) {
      for (int b = lo; b <= hi; ++b) {
        to_codepoint[static_cast<size_t>(b)] = static_cast<uint32_t>(b);
        assigned[static_cast<size_t>(b)] = true;
      }
    };
    identity('!', '~');
    identity(0xA1, 0xAC);
    identity(0xAE, 0xFF);

    uint32_t next = 256;
    for (int b = 0; b < 256; ++b) {
      if (!assigned[static_cast<size_t>(b)]) {
        to_codepoint[static_cast<size_t>(b)] = next++;
      }
    }

    to_byte.fill(-1);
    for (int b = 0; b < 256; ++b) {
      to_byte[to_codepoint[static_cast<size_t>(b)]] = static_cast<int16_t>(b);
    }
  }
};

const ByteAlphabet& alphabet() {
  static const ByteAlphabet instance;
  return instance;
}

void append_utf8(uint32_t cp, std::string* out) {
  if (cp < 0x80) {
    out->push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// One decoded codepoint together with where it started, so a pretoken can be
// handed back as a byte substring of the original text.
struct Codepoint {
  uint32_t value;
  size_t offset;
};

// Decodes UTF-8 leniently: a malformed byte becomes a codepoint equal to that
// byte value. Such a codepoint is neither letter, number, nor whitespace, so
// the pretokenizer routes it through the punctuation branch and the byte-level
// stage turns it back into the exact original byte. Nothing is lost and no
// input can make the tokenizer throw.
std::vector<Codepoint> decode_utf8(const std::string& text) {
  std::vector<Codepoint> out;
  out.reserve(text.size());
  const auto* p = reinterpret_cast<const unsigned char*>(text.data());
  const size_t n = text.size();

  for (size_t i = 0; i < n;) {
    const unsigned char lead = p[i];
    size_t extra = 0;
    uint32_t cp = 0;
    if (lead < 0x80) {
      cp = lead;
    } else if ((lead & 0xE0) == 0xC0) {
      extra = 1;
      cp = lead & 0x1Fu;
    } else if ((lead & 0xF0) == 0xE0) {
      extra = 2;
      cp = lead & 0x0Fu;
    } else if ((lead & 0xF8) == 0xF0) {
      extra = 3;
      cp = lead & 0x07u;
    } else {
      out.push_back({lead, i});
      ++i;
      continue;
    }

    if (i + extra >= n) {
      // Truncated sequence at the end of the input.
      out.push_back({lead, i});
      ++i;
      continue;
    }
    bool ok = true;
    for (size_t k = 1; k <= extra; ++k) {
      if ((p[i + k] & 0xC0) != 0x80) {
        ok = false;
        break;
      }
      cp = (cp << 6) | (p[i + k] & 0x3Fu);
    }
    if (!ok) {
      out.push_back({lead, i});
      ++i;
      continue;
    }
    out.push_back({cp, i});
    i += extra + 1;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Pretokenizer
//
// A hand-written matcher for the Llama 3 split regex. Each function returns how
// many codepoints its alternative consumes at position `i`, or zero for no
// match. Regex alternation is leftmost-first, so the alternatives are tried in
// the order they appear in the pattern and the first non-zero result wins.
// ---------------------------------------------------------------------------

uint32_t ascii_lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// (?i:'s|'t|'re|'ve|'m|'ll|'d)
size_t match_contraction(const std::vector<Codepoint>& cps, size_t i) {
  if (cps[i].value != '\'') return 0;
  const size_t n = cps.size();
  auto at = [&](size_t k) -> uint32_t {
    return k < n ? ascii_lower(cps[k].value) : 0;
  };
  const uint32_t a = at(i + 1);
  if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
  const uint32_t b = at(i + 2);
  if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
      (a == 'l' && b == 'l')) {
    return 3;
  }
  return 0;
}

// [^\r\n\p{L}\p{N}]?\p{L}+
// The optional leading character is greedy, so the version that consumes it is
// tried first and only falls back when no letter follows. This is what keeps a
// leading space attached to the word after it.
size_t match_word(const std::vector<Codepoint>& cps, size_t i) {
  const size_t n = cps.size();
  auto allowed_prefix = [](uint32_t c) {
    return c != '\r' && c != '\n' && !is_letter(c) && !is_number(c);
  };
  for (int take_prefix = 1; take_prefix >= 0; --take_prefix) {
    size_t j = i;
    if (take_prefix) {
      if (!allowed_prefix(cps[j].value)) continue;
      ++j;
    }
    const size_t letters_start = j;
    while (j < n && is_letter(cps[j].value)) ++j;
    if (j > letters_start) return j - i;
  }
  return 0;
}

// \p{N}{1,3}
// Capped at three so long numbers split into groups rather than becoming one
// enormous rare token.
size_t match_digits(const std::vector<Codepoint>& cps, size_t i) {
  size_t j = i;
  while (j < cps.size() && j - i < 3 && is_number(cps[j].value)) ++j;
  return j - i;
}

//  ?[^\s\p{L}\p{N}]+[\r\n]*
size_t match_punctuation(const std::vector<Codepoint>& cps, size_t i) {
  const size_t n = cps.size();
  auto is_symbol = [](uint32_t c) {
    return !is_whitespace(c) && !is_letter(c) && !is_number(c);
  };
  for (int take_space = 1; take_space >= 0; --take_space) {
    size_t j = i;
    if (take_space) {
      if (cps[j].value != ' ') continue;
      ++j;
    }
    const size_t symbols_start = j;
    while (j < n && is_symbol(cps[j].value)) ++j;
    if (j == symbols_start) continue;
    while (j < n && (cps[j].value == '\r' || cps[j].value == '\n')) ++j;
    return j - i;
  }
  return 0;
}

// \s*[\r\n]+
// After backtracking this matches the whitespace run up to and including its
// last line break, so indentation preceding a newline stays with that newline.
size_t match_line_break(const std::vector<Codepoint>& cps, size_t i) {
  const size_t n = cps.size();
  size_t end = i;
  while (end < n && is_whitespace(cps[end].value)) ++end;
  size_t last = end;
  while (last > i && cps[last - 1].value != '\r' && cps[last - 1].value != '\n') {
    --last;
  }
  return last == i ? 0 : last - i;
}

// \s+(?!\S)
// A whitespace run that is not followed by anything printable. The lookahead
// forces the match to stop one character short when the run does end at
// printable text, which is how a trailing space gets handed to the word after
// it instead of the whitespace token before it.
size_t match_trailing_whitespace(const std::vector<Codepoint>& cps, size_t i) {
  const size_t n = cps.size();
  size_t end = i;
  while (end < n && is_whitespace(cps[end].value)) ++end;
  if (end == i) return 0;
  if (end == n) return end - i;
  return (end - i >= 2) ? (end - i - 1) : 0;
}

// \s+
size_t match_whitespace(const std::vector<Codepoint>& cps, size_t i) {
  size_t j = i;
  while (j < cps.size() && is_whitespace(cps[j].value)) ++j;
  return j - i;
}

}  // namespace

std::vector<std::string> Tokenizer::split_pretokens(const std::string& text) {
  const std::vector<Codepoint> cps = decode_utf8(text);
  std::vector<std::string> out;
  if (cps.empty()) return out;

  // Byte offset one past the last codepoint, needed to slice the final piece.
  auto byte_end = [&](size_t index) {
    return index < cps.size() ? cps[index].offset : text.size();
  };

  size_t i = 0;
  while (i < cps.size()) {
    size_t taken = match_contraction(cps, i);
    if (taken == 0) taken = match_word(cps, i);
    if (taken == 0) taken = match_digits(cps, i);
    if (taken == 0) taken = match_punctuation(cps, i);
    if (taken == 0) taken = match_line_break(cps, i);
    if (taken == 0) taken = match_trailing_whitespace(cps, i);
    if (taken == 0) taken = match_whitespace(cps, i);
    // Every codepoint falls into one of the alternatives above, but a split
    // that consumed nothing would spin forever, so never advance by zero.
    if (taken == 0) taken = 1;

    const size_t begin = cps[i].offset;
    const size_t end = byte_end(i + taken);
    out.emplace_back(text, begin, end - begin);
    i += taken;
  }
  return out;
}

std::string Tokenizer::byte_level_encode(const std::string& raw) {
  const ByteAlphabet& table = alphabet();
  std::string out;
  out.reserve(raw.size() * 2);
  for (unsigned char byte : raw) {
    append_utf8(table.to_codepoint[byte], &out);
  }
  return out;
}

std::string Tokenizer::byte_level_decode(const std::string& encoded) {
  const ByteAlphabet& table = alphabet();
  std::string out;
  out.reserve(encoded.size());
  for (const Codepoint& cp : decode_utf8(encoded)) {
    const int16_t byte =
        cp.value < table.to_byte.size() ? table.to_byte[cp.value] : -1;
    // A codepoint outside the byte-level alphabet cannot come from a token
    // this tokenizer produced. Emitting it as UTF-8 keeps decode total.
    if (byte < 0) {
      append_utf8(cp.value, &out);
    } else {
      out.push_back(static_cast<char>(static_cast<unsigned char>(byte)));
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// BPE
//
// The pieces of one pretoken are held in a doubly linked list over byte ranges
// of the encoded string, and a heap holds every adjacent pair that has a merge
// rank. Popping the heap always yields the lowest-ranked live pair, which is
// what the reference implementation's "merge the best pair, repeat" loop does,
// at O(n log n) instead of rescanning every pair each round.
// ---------------------------------------------------------------------------

namespace {

struct Symbol {
  size_t offset = 0;
  size_t length = 0;  // zero once the symbol has been absorbed by its left neighbour
  int prev = -1;
  int next = -1;
};

struct Bigram {
  int left = -1;
  int right = -1;
  int rank = 0;
  size_t merged_length = 0;  // used to spot entries invalidated by other merges
};

// std::priority_queue pops the greatest element, so "greater" has to mean
// "worse": a higher merge rank loses, and among equal ranks the leftmost pair
// wins so the merge order is deterministic.
struct BigramWorseThan {
  bool operator()(const Bigram& a, const Bigram& b) const {
    if (a.rank != b.rank) return a.rank > b.rank;
    return a.left > b.left;
  }
};

}  // namespace

void Tokenizer::bpe_encode_pretoken(const std::string& pretoken,
                                    std::vector<int>* out) const {
  if (pretoken.empty()) return;
  const std::string encoded = byte_level_encode(pretoken);

  // With ignore_merges set, a pretoken that is itself a vocabulary entry is
  // emitted as that entry. Llama 3 relies on this: many whole words exist in
  // the vocabulary but are not reachable by replaying the merge list.
  if (ignore_merges_) {
    auto it = token_to_id_.find(encoded);
    if (it != token_to_id_.end()) {
      out->push_back(it->second);
      return;
    }
  }

  // One symbol per byte-level character. Each is one to two UTF-8 bytes.
  std::vector<Symbol> symbols;
  symbols.reserve(encoded.size());
  for (const Codepoint& cp : decode_utf8(encoded)) {
    Symbol s;
    s.offset = cp.offset;
    s.prev = static_cast<int>(symbols.size()) - 1;
    symbols.push_back(s);
  }
  if (symbols.empty()) return;
  for (size_t k = 0; k < symbols.size(); ++k) {
    const size_t end =
        k + 1 < symbols.size() ? symbols[k + 1].offset : encoded.size();
    symbols[k].length = end - symbols[k].offset;
    symbols[k].next =
        k + 1 < symbols.size() ? static_cast<int>(k + 1) : -1;
  }

  std::priority_queue<Bigram, std::vector<Bigram>, BigramWorseThan> queue;
  std::string key;
  auto push_pair = [&](int left, int right) {
    if (left < 0 || right < 0) return;
    const Symbol& l = symbols[static_cast<size_t>(left)];
    const Symbol& r = symbols[static_cast<size_t>(right)];
    if (l.length == 0 || r.length == 0) return;
    key.assign(encoded, l.offset, l.length);
    key.push_back(' ');
    key.append(encoded, r.offset, r.length);
    auto it = merge_rank_.find(key);
    if (it == merge_rank_.end()) return;
    queue.push(Bigram{left, right, it->second, l.length + r.length});
  };

  for (size_t k = 0; k + 1 < symbols.size(); ++k) {
    push_pair(static_cast<int>(k), static_cast<int>(k + 1));
  }

  while (!queue.empty()) {
    const Bigram top = queue.top();
    queue.pop();
    Symbol& left = symbols[static_cast<size_t>(top.left)];
    Symbol& right = symbols[static_cast<size_t>(top.right)];
    // Skip entries that a previous merge invalidated: either endpoint absorbed,
    // no longer adjacent, or grown since the pair was queued.
    if (left.length == 0 || right.length == 0) continue;
    if (left.next != top.right) continue;
    if (left.length + right.length != top.merged_length) continue;

    left.length += right.length;
    left.next = right.next;
    if (right.next >= 0) symbols[static_cast<size_t>(right.next)].prev = top.left;
    right.length = 0;

    push_pair(left.prev, top.left);
    push_pair(top.left, left.next);
  }

  for (int index = 0; index >= 0; index = symbols[static_cast<size_t>(index)].next) {
    const Symbol& s = symbols[static_cast<size_t>(index)];
    key.assign(encoded, s.offset, s.length);
    auto it = token_to_id_.find(key);
    LCR_CHECK(it != token_to_id_.end(),
              "BPE produced the piece \"" << key
                                          << "\", which is not in the vocabulary");
    out->push_back(it->second);
  }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

namespace {

// Walks a pre_tokenizer node, which is either a single pretokenizer or a
// Sequence of them, and returns the Split pattern and the ByteLevel settings.
struct PreTokenizerSpec {
  bool found_split = false;
  std::string split_pattern;
  bool found_byte_level = false;
  bool add_prefix_space = false;
};

void inspect_pretokenizer(const json& node, PreTokenizerSpec* spec) {
  if (!node.is_object()) return;
  const std::string type =
      node.contains("type") && node["type"].is_string()
          ? node["type"].get<std::string>()
          : std::string();

  if (type == "Sequence" && node.contains("pretokenizers")) {
    for (const json& child : node["pretokenizers"]) {
      inspect_pretokenizer(child, spec);
    }
    return;
  }
  if (type == "Split" && node.contains("pattern") &&
      node["pattern"].is_object() && node["pattern"].contains("Regex")) {
    spec->found_split = true;
    spec->split_pattern = node["pattern"]["Regex"].get<std::string>();
    return;
  }
  if (type == "ByteLevel") {
    spec->found_byte_level = true;
    if (node.contains("add_prefix_space") &&
        node["add_prefix_space"].is_boolean()) {
      spec->add_prefix_space = node["add_prefix_space"].get<bool>();
    }
  }
}

// merges entries are "left right" in older files and ["left", "right"] in
// newer ones. Both mean the same pair.
bool parse_merge(const json& entry, std::string* key) {
  if (entry.is_string()) {
    const std::string text = entry.get<std::string>();
    // A byte-level token never contains a space, so the first space is the
    // separator and there is exactly one.
    const size_t split = text.find(' ');
    if (split == std::string::npos) return false;
    *key = text;
    return text.find(' ', split + 1) == std::string::npos;
  }
  if (entry.is_array() && entry.size() == 2 && entry[0].is_string() &&
      entry[1].is_string()) {
    *key = entry[0].get<std::string>() + " " + entry[1].get<std::string>();
    return true;
  }
  return false;
}

}  // namespace

void Tokenizer::build_from_json(const std::string& text) {
  json doc = json::parse(text, nullptr, false);
  LCR_CHECK(!doc.is_discarded(), "tokenizer.json is not valid JSON");
  LCR_CHECK(doc.contains("model") && doc["model"].is_object(),
            "tokenizer.json has no model object");

  const json& model = doc["model"];
  const std::string model_type =
      model.contains("type") && model["type"].is_string()
          ? model["type"].get<std::string>()
          : std::string();
  LCR_CHECK(model_type == "BPE",
            "this tokenizer only implements byte-level BPE, tokenizer.json says "
            "model.type=\"" << model_type << "\"");
  LCR_CHECK(!model.contains("byte_fallback") ||
                !model["byte_fallback"].get<bool>(),
            "byte_fallback tokenizers are not supported; the byte-level "
            "alphabet already covers every byte");

  if (model.contains("ignore_merges") && model["ignore_merges"].is_boolean()) {
    ignore_merges_ = model["ignore_merges"].get<bool>();
  }

  // The hand-written pretokenizer must match the pattern the checkpoint was
  // built with, or the splits silently diverge from the reference tokenizer.
  PreTokenizerSpec spec;
  if (doc.contains("pre_tokenizer")) inspect_pretokenizer(doc["pre_tokenizer"], &spec);
  LCR_CHECK(spec.found_split,
            "tokenizer.json has no Split pretokenizer, so it is not a Llama 3 "
            "style byte-level BPE tokenizer");
  LCR_CHECK(spec.split_pattern == kLlama3SplitRegex,
            "tokenizer.json uses a pretokenizer pattern this runtime does not "
            "implement.\n  expected: "
                << kLlama3SplitRegex << "\n  found:    " << spec.split_pattern);
  LCR_CHECK(!spec.add_prefix_space,
            "the ByteLevel pretokenizer asks for add_prefix_space, which this "
            "runtime does not implement");

  LCR_CHECK(model.contains("vocab") && model["vocab"].is_object(),
            "tokenizer.json has no model.vocab object");
  const json& vocab = model["vocab"];

  int max_id = -1;
  token_to_id_.reserve(vocab.size() * 2);
  for (const auto& entry : vocab.items()) {
    LCR_CHECK(entry.value().is_number_integer(),
              "vocabulary entry \"" << entry.key() << "\" has a non-integer id");
    const int id = entry.value().get<int>();
    LCR_CHECK(id >= 0, "vocabulary entry \"" << entry.key()
                                             << "\" has negative id " << id);
    token_to_id_.emplace(entry.key(), id);
    max_id = std::max(max_id, id);
  }

  std::vector<std::pair<std::string, int>> added;
  if (doc.contains("added_tokens") && doc["added_tokens"].is_array()) {
    for (const json& node : doc["added_tokens"]) {
      if (!node.is_object() || !node.contains("id") || !node.contains("content")) {
        continue;
      }
      const int id = node["id"].get<int>();
      const std::string content = node["content"].get<std::string>();
      const bool special = node.contains("special") &&
                           node["special"].is_boolean() &&
                           node["special"].get<bool>();
      added.emplace_back(content, id);
      added_by_content_.emplace(content, id);
      added_by_id_.emplace(id, content);
      max_id = std::max(max_id, id);
      if (special) special_ids_.insert(id);
    }
  }

  LCR_CHECK(max_id >= 0, "tokenizer.json defines no tokens");
  id_to_token_.assign(static_cast<size_t>(max_id) + 1, std::string());
  for (const auto& [token, id] : token_to_id_) {
    id_to_token_[static_cast<size_t>(id)] = token;
  }
  for (const auto& [content, id] : added) {
    // Added tokens are matched literally, so they deliberately do not go into
    // token_to_id_: text that happens to spell out "<|eot_id|>" must still
    // tokenize as ordinary characters when special parsing is off.
    id_to_token_[static_cast<size_t>(id)] = content;
  }

  std::set<size_t> lengths;
  for (const auto& [content, id] : added) {
    if (content.empty()) continue;
    lengths.insert(content.size());
    added_first_bytes_[static_cast<unsigned char>(content[0])] = true;
  }
  added_lengths_.assign(lengths.rbegin(), lengths.rend());

  LCR_CHECK(model.contains("merges") && model["merges"].is_array(),
            "tokenizer.json has no model.merges array");
  const json& merges = model["merges"];
  merge_rank_.reserve(merges.size() * 2);
  int rank = 0;
  std::string key;
  for (const json& entry : merges) {
    LCR_CHECK(parse_merge(entry, &key),
              "merge entry " << rank << " is not a pair of byte-level tokens");
    merge_rank_.emplace(key, rank);
    ++rank;
  }

  // The beginning-of-text token, used when encoding a raw completion prompt.
  for (const char* name : {"<|begin_of_text|>", "<s>"}) {
    auto it = added_by_content_.find(name);
    if (it != added_by_content_.end()) {
      bos_id_ = it->second;
      break;
    }
    auto vocab_it = token_to_id_.find(name);
    if (vocab_it != token_to_id_.end()) {
      bos_id_ = vocab_it->second;
      break;
    }
  }
}

Tokenizer Tokenizer::from_json_string(const std::string& text) {
  Tokenizer tokenizer;
  tokenizer.build_from_json(text);
  return tokenizer;
}

Tokenizer Tokenizer::from_file(const std::string& tokenizer_json_path) {
  std::ifstream in(tokenizer_json_path, std::ios::binary);
  LCR_CHECK(in.good(), "cannot read " << tokenizer_json_path);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return from_json_string(buffer.str());
}

Tokenizer Tokenizer::from_model_dir(const std::string& model_dir) {
  return from_file(join_path(model_dir, "tokenizer.json"));
}

// ---------------------------------------------------------------------------
// Public encode and decode
// ---------------------------------------------------------------------------

int Tokenizer::token_to_id(const std::string& token) const {
  auto it = token_to_id_.find(token);
  if (it != token_to_id_.end()) return it->second;
  auto added = added_by_content_.find(token);
  return added == added_by_content_.end() ? -1 : added->second;
}

const std::string& Tokenizer::id_to_token(int id) const {
  LCR_CHECK(id >= 0 && id < vocab_size(),
            "token id " << id << " is outside the vocabulary of size "
                        << vocab_size());
  return id_to_token_[static_cast<size_t>(id)];
}

bool Tokenizer::is_special(int id) const {
  return special_ids_.count(id) != 0;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos,
                                   bool parse_special) const {
  std::vector<int> out;
  if (add_bos) {
    LCR_CHECK(bos_id_ >= 0,
              "this tokenizer has no beginning-of-text token to prepend");
    out.push_back(bos_id_);
  }

  // Everything between added tokens goes through pretokenization and BPE.
  size_t chunk_start = 0;
  auto flush_through_bpe = [&](size_t end) {
    if (end <= chunk_start) return;
    const std::string chunk(text, chunk_start, end - chunk_start);
    for (const std::string& pretoken : split_pretokens(chunk)) {
      bpe_encode_pretoken(pretoken, &out);
    }
  };

  size_t pos = 0;
  std::string candidate;
  while (pos < text.size()) {
    // Cheap reject first: no added token can start with this byte.
    if (!parse_special ||
        !added_first_bytes_[static_cast<unsigned char>(text[pos])]) {
      ++pos;
      continue;
    }
    // Longest match wins, so the lengths are tried largest first.
    int matched_id = -1;
    size_t matched_length = 0;
    for (size_t length : added_lengths_) {
      if (pos + length > text.size()) continue;
      candidate.assign(text, pos, length);
      auto it = added_by_content_.find(candidate);
      if (it != added_by_content_.end()) {
        matched_id = it->second;
        matched_length = length;
        break;
      }
    }
    if (matched_id < 0) {
      ++pos;
      continue;
    }
    flush_through_bpe(pos);
    out.push_back(matched_id);
    pos += matched_length;
    chunk_start = pos;
  }
  flush_through_bpe(text.size());
  return out;
}

std::string Tokenizer::decode_one(int id, bool skip_special) const {
  auto added = added_by_id_.find(id);
  if (added != added_by_id_.end()) {
    // An added token's content is literal text, not byte-level encoded.
    if (skip_special && is_special(id)) return std::string();
    return added->second;
  }
  return byte_level_decode(id_to_token(id));
}

std::string Tokenizer::decode(const std::vector<int>& ids,
                              bool skip_special) const {
  std::string out;
  for (int id : ids) out += decode_one(id, skip_special);
  return out;
}

}  // namespace lcr
