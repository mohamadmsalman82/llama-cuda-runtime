// Exercises the three stages of the tokenizer separately: the Llama 3
// pretokenizer split, the byte-level alphabet, and BPE with added tokens. The
// BPE tests build a complete synthetic byte-level tokenizer.json in memory, so
// they run without the real 17 MB checkpoint file.
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "test_util.h"
#include "tokenizer.h"

namespace {

using lcr::Tokenizer;

// The pattern as it appears inside tokenizer.json, where each regex backslash
// is written as a JSON escaped backslash.
const char* kSplitPatternJson =
    R"RX((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)RX";

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

// A byte-level BPE tokenizer whose vocabulary covers all 256 single bytes, so
// any input encodes. Ids 0..255 are the raw bytes; the extras are merged or
// unreachable tokens the tests reason about.
std::string make_tokenizer_json(bool ignore_merges,
                                const std::string& split_pattern = kSplitPatternJson) {
  std::ostringstream j;
  j << R"({"version":"1.0","added_tokens":[)"
    << R"({"id":300,"content":"<|eot|>","special":true},)"
    << R"({"id":301,"content":"<|begin_of_text|>","special":true},)"
    << R"({"id":302,"content":"<|long_special_token|>","special":true}],)"
    << R"("pre_tokenizer":{"type":"Sequence","pretokenizers":[)"
    << R"({"type":"Split","pattern":{"Regex":")" << split_pattern
    << R"("},"behavior":"Isolated","invert":false},)"
    << R"({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},)"
    << R"("decoder":{"type":"ByteLevel"},)"
    << R"("model":{"type":"BPE","ignore_merges":)"
    << (ignore_merges ? "true" : "false") << R"(,"vocab":{)";

  for (int b = 0; b < 256; ++b) {
    const std::string ch =
        Tokenizer::byte_level_encode(std::string(1, static_cast<char>(b)));
    if (b) j << ",";
    j << "\"" << json_escape(ch) << "\":" << b;
  }
  const std::string space = Tokenizer::byte_level_encode(" ");
  // 256 "ab" and 257 " ab" are reachable by replaying the merges below.
  // 258 "xyz" is in the vocabulary but no merge sequence produces it, which is
  // exactly the case ignore_merges exists to handle.
  j << ",\"ab\":256"
    << ",\"" << json_escape(space) << "ab\":257"
    << ",\"xyz\":258";
  j << R"(},"merges":["a b",")" << json_escape(space) << R"( ab"]}})";
  return j.str();
}

void check_split(const std::string& input,
                 const std::vector<std::string>& expected) {
  const std::vector<std::string> got = Tokenizer::split_pretokens(input);
  if (got != expected) {
    std::ostringstream oss;
    oss << "split of \"" << input << "\" gave [";
    for (size_t i = 0; i < got.size(); ++i) {
      if (i) oss << ", ";
      oss << "\"" << got[i] << "\"";
    }
    oss << "], expected [";
    for (size_t i = 0; i < expected.size(); ++i) {
      if (i) oss << ", ";
      oss << "\"" << expected[i] << "\"";
    }
    oss << "]";
    ::test::report(false, __FILE__, __LINE__, oss.str());
  }
  // Whatever the split, gluing the pieces back together must reproduce the
  // input exactly, or bytes are being dropped or duplicated.
  std::string rejoined;
  for (const std::string& piece : got) rejoined += piece;
  CHECK_EQ(rejoined, input);
}

void test_pretokenizer_split() {
  check_split("", {});
  // A leading space attaches to the word after it, never the word before.
  check_split("Hello world", {"Hello", " world"});
  check_split("Hello, world!", {"Hello", ",", " world", "!"});
  // Contractions are their own pretoken and the match is case-insensitive.
  check_split("don't", {"don", "'t"});
  check_split("DON'T", {"DON", "'T"});
  check_split("they're", {"they", "'re"});
  check_split("I'll've", {"I", "'ll", "'ve"});
  // A run of digits is cut into groups of at most three.
  check_split("1234567", {"123", "456", "7"});
  check_split("v2.0", {"v", "2", ".", "0"});
  // Only the last space in a run joins the following word; the rest stand
  // alone. This is the \s+(?!\S) alternative doing its job.
  check_split("a  b", {"a", " ", " b"});
  check_split("  hello", {" ", " hello"});
  // Whitespace that ends in a line break is one pretoken up to and including
  // the last break, so indentation stays attached to its newline.
  check_split("  \n", {"  \n"});
  check_split("a\n\nb", {"a", "\n\n", "b"});
  check_split("a \n b", {"a", " \n", " b"});
  // Trailing whitespace with nothing after it is kept whole.
  check_split("a   ", {"a", "   "});
  // Punctuation runs stay together, with at most one leading space.
  check_split("()", {"()"});
  check_split(" (x)", {" (", "x", ")"});
  check_split("...", {"..."});
  // Non-ASCII letters are letters, and an emoji is punctuation.
  check_split("café", {"café"});
  check_split("日本語", {"日本語"});
  check_split("hi 🙂", {"hi", " 🙂"});
  // An apostrophe that is not a contraction suffix falls through to the
  // optional-prefix slot of the word alternative and stays glued to the word.
  check_split("'x", {"'x"});
  check_split(" 's", {" '", "s"});
}

void test_byte_level_alphabet() {
  // Every byte value must survive a round trip, including NUL, newline, and
  // bytes that are not valid UTF-8 on their own.
  std::string all_bytes;
  for (int b = 0; b < 256; ++b) all_bytes.push_back(static_cast<char>(b));
  const std::string encoded = Tokenizer::byte_level_encode(all_bytes);
  CHECK_EQ(Tokenizer::byte_level_decode(encoded), all_bytes);

  // The encoded form is printable and space-free, which is what lets a plain
  // space separate the two halves of a merge rule.
  CHECK(encoded.find(' ') == std::string::npos);
  // Printable ASCII maps to itself.
  CHECK_EQ(Tokenizer::byte_level_encode("Hello!"), std::string("Hello!"));
  // A space becomes U+0120 and a newline U+010A, the two markers that show up
  // all over a Llama vocabulary.
  CHECK_EQ(Tokenizer::byte_level_encode(" "), std::string("\xc4\xa0"));
  CHECK_EQ(Tokenizer::byte_level_encode("\n"), std::string("\xc4\x8a"));
}

void test_bpe_merges() {
  const Tokenizer tokenizer = Tokenizer::from_json_string(
      make_tokenizer_json(/*ignore_merges=*/false));
  CHECK_EQ(tokenizer.vocab_size(), 303);
  CHECK_EQ(tokenizer.bos_id(), 301);

  // "ab" merges through rule 0. " ab" needs rule 0 first, then rule 1, which
  // only happens if the queue re-examines the pair created by a merge.
  CHECK(tokenizer.encode("ab") == std::vector<int>({256}));
  CHECK(tokenizer.encode("ab ab") == std::vector<int>({256, 257}));

  // Nothing merges 'q' and 'r', so they stay as single-byte tokens.
  CHECK(tokenizer.encode("qr") == std::vector<int>({'q', 'r'}));

  // "xyz" is in the vocabulary but unreachable by merges, so without
  // ignore_merges it comes out as three separate bytes.
  CHECK(tokenizer.encode("xyz") == std::vector<int>({'x', 'y', 'z'}));

  const Tokenizer with_ignore = Tokenizer::from_json_string(
      make_tokenizer_json(/*ignore_merges=*/true));
  // Now the whole pretoken is looked up first and the merge list is skipped.
  CHECK(with_ignore.encode("xyz") == std::vector<int>({258}));
  // Merge-reachable tokens are unaffected.
  CHECK(with_ignore.encode("ab ab") == std::vector<int>({256, 257}));
  // A pretoken that is not a whole vocabulary entry still goes through merges.
  CHECK(with_ignore.encode("abq") == std::vector<int>({256, 'q'}));
}

void test_added_tokens() {
  const Tokenizer tokenizer = Tokenizer::from_json_string(
      make_tokenizer_json(/*ignore_merges=*/true));

  // With special parsing on, the literal text becomes the single added token.
  const std::vector<int> parsed = tokenizer.encode("hi<|eot|>", false, true);
  CHECK(parsed == std::vector<int>({'h', 'i', 300}));

  // With it off, the same text is just characters. Nothing in the BPE
  // vocabulary can produce id 300.
  const std::vector<int> literal = tokenizer.encode("hi<|eot|>", false, false);
  CHECK(literal.size() > 3);
  for (int id : literal) CHECK(id != 300);

  // Longest match wins when one added token is a prefix of the input at the
  // same position as a shorter one.
  CHECK(tokenizer.encode("<|long_special_token|>") == std::vector<int>({302}));

  // add_bos prepends the beginning-of-text token.
  const std::vector<int> with_bos = tokenizer.encode("ab", true, true);
  CHECK(with_bos == std::vector<int>({301, 256}));

  // Text between two added tokens still goes through the full pipeline.
  const std::vector<int> sandwiched =
      tokenizer.encode("<|eot|>ab ab<|eot|>", false, true);
  CHECK(sandwiched == std::vector<int>({300, 256, 257, 300}));
}

void test_decode_round_trip() {
  const Tokenizer tokenizer = Tokenizer::from_json_string(
      make_tokenizer_json(/*ignore_merges=*/true));

  const std::vector<std::string> samples = {
      "",
      "ab ab",
      "Hello, world!",
      "don't stop\n\n  indented",
      "café 日本語 🙂",
      "tabs\tand\rcarriage returns",
      std::string("nul\0inside", 10),
      "\xff\xfe raw bytes that are not utf-8",
  };
  for (const std::string& sample : samples) {
    const std::vector<int> ids = tokenizer.encode(sample);
    CHECK_EQ(tokenizer.decode(ids), sample);
    // Streaming one token at a time has to produce the same bytes.
    std::string streamed;
    for (int id : ids) streamed += tokenizer.decode_one(id);
    CHECK_EQ(streamed, sample);
  }

  // An added token decodes to its literal content, or to nothing when special
  // tokens are being skipped.
  CHECK_EQ(tokenizer.decode({300}), std::string("<|eot|>"));
  CHECK_EQ(tokenizer.decode({300}, true), std::string(""));
  CHECK(tokenizer.is_special(300));
  CHECK(!tokenizer.is_special('a'));
  CHECK_EQ(tokenizer.token_to_id("<|eot|>"), 300);
  CHECK_EQ(tokenizer.token_to_id("ab"), 256);
  CHECK_EQ(tokenizer.token_to_id("no such token"), -1);
}

void test_rejects_incompatible_tokenizers() {
  // A different pretokenizer pattern would split text differently, and the
  // hand-written matcher cannot tell. Refuse rather than encode wrongly.
  CHECK_THROWS("pretokenizer pattern this runtime does not implement", {
    Tokenizer::from_json_string(
        make_tokenizer_json(true, R"(\\w+|[^\\w\\s]+)"));
  });
  CHECK_THROWS("only implements byte-level BPE", {
    Tokenizer::from_json_string(
        R"({"model":{"type":"Unigram","vocab":[]}})");
  });
  CHECK_THROWS("no Split pretokenizer", {
    Tokenizer::from_json_string(
        R"({"model":{"type":"BPE","vocab":{"a":0},"merges":[]}})");
  });
}

}  // namespace

int main() {
  test_pretokenizer_split();
  test_byte_level_alphabet();
  test_bpe_merges();
  test_added_tokens();
  test_decode_round_trip();
  test_rejects_incompatible_tokenizers();
  return test::finish("test_tokenizer");
}
