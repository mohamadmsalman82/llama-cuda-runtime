#!/usr/bin/env python3
"""Diffs the C++ tokenizer against the reference Hugging Face tokenizer.

The C++ implementation replaces a regex engine and a Rust BPE library with
hand-written code, so the only honest way to trust it is to encode a large
adversarial corpus with both and compare ids position by position.

    pip install tokenizers
    python3 tools/check_tokenizer.py models/Llama-3.2-1B-Instruct

Exits non-zero on the first mismatch it cannot explain, after printing the
input, both id sequences, and the piece where they diverge.
"""

import argparse
import json
import random
import string
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def handwritten_cases():
    """Inputs chosen to hit each branch of the pretokenizer and BPE."""
    return [
        "",
        " ",
        "\n",
        "Hello world",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        # Contractions, the case-insensitive alternative.
        "don't they're I'll we've it's John's DON'T THEY'RE",
        "'x '9 ' s 'S 'll'll",
        # Digit grouping, capped at three per token.
        "0 1 12 123 1234 12345 1234567890 3.14159 1e-9 007",
        "$1,234.56 99% #42 (2026)",
        # Whitespace runs, the three alternatives that handle them.
        "a b", "a  b", "a   b", "a\tb", "a\n\nb", "a \n b", "trailing   ",
        "  leading", "\n\n\n", "  \n  \n  ", "\r\n\r\n", "mixed \t\n \t end",
        # Indentation, which is where the \s*[\r\n]+ alternative earns its keep.
        "def f():\n    return 1\n\n\nclass A:\n\tpass\n",
        # Punctuation runs with and without a leading space.
        "()[]{}", " (x) ", "...", " ... ", "a--b", "->", "=>", "!!!???",
        "https://example.com/path?a=1&b=2#frag",
        "/usr/local/bin", "C:\\Windows\\System32",
        # Non-ASCII letters, combining marks, and scripts without spaces.
        "café", "naïve résumé", "Ünicode", "ÅNGSTRÖM",
        "日本語のテキストです", "中文字符", "한국어 텍스트",
        "Здравствуйте, мир", "مرحبا بالعالم", "שלום עולם",
        "ελληνικά", "देवनागरी", "ไทย",
        # Emoji and other symbols, which are punctuation to this regex.
        "hi 🙂", "👨‍👩‍👧‍👦 family", "🇨🇦🇺🇸", "math: ∀x∈ℝ, x²≥0", "→ ← ↑ ↓",
        # Numerals that are \p{N} but not ASCII digits.
        "Ⅷ ½ ٣ ३ 一二三",
        # Control tokens written out as text.
        "<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>user<|end_header_id|>",
        "a<|eot_id|>b", "<|not_a_real_token|>",
        # Long single tokens and long repeats, which stress the merge queue.
        "a" * 300, "ab" * 200, " " * 100, "=" * 200,
        "supercalifragilisticexpialidocious",
        # Code, the other thing this model sees a lot of.
        "int main() { return 0; }",
        "SELECT * FROM t WHERE a = 'b' AND c > 1;",
        '{"key": [1, 2, {"nested": null}], "b": true}',
        "#include <cstdio>\n\nint main(int argc, char** argv) {\n  return 0;\n}\n",
    ]


def random_cases(count, seed=20260904):
    """Random text, biased toward the boundaries the regex cares about."""
    rng = random.Random(seed)
    alphabets = [
        string.ascii_letters,
        string.digits,
        string.punctuation,
        " \t\n\r",
        "  \n\n",
        "áéíóúàèìòùâêîôûäëïöüñçß",
        "日本語中文한국어",
        "🙂🎉🔥✨",
        string.ascii_letters + string.digits + string.punctuation + "  \n",
    ]
    out = []
    for _ in range(count):
        alphabet = rng.choice(alphabets)
        length = rng.randint(1, 120)
        out.append("".join(rng.choice(alphabet) for _ in range(length)))
    return out


def codepoint_sweep(step=97):
    """Walks the whole Unicode range so no category boundary goes untested."""
    out = []
    chunk = []
    for cp in range(0x20, 0x110000, step):
        ch = chr(cp)
        if unicodedata.category(ch) in ("Cs", "Cn", "Co"):
            continue
        chunk.append(ch)
        if len(chunk) == 40:
            out.append("x" + "".join(chunk) + "1 y")
            chunk = []
    if chunk:
        out.append("x" + "".join(chunk) + "1 y")
    return out


def source_file_cases():
    """Real text: this repository's own sources, a few lines at a time."""
    out = []
    for path in sorted(REPO_ROOT.glob("src/*.cpp")) + sorted(REPO_ROOT.glob("src/*.h")):
        if path.name == "unicode_data.cpp":
            continue  # generated, and 260 lines of hex adds nothing
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)
        for i in range(0, len(lines), 12):
            out.append("".join(lines[i:i + 12]))
    return out


def build_corpus():
    corpus = []
    corpus += handwritten_cases()
    corpus += codepoint_sweep()
    corpus += random_cases(600)
    corpus += source_file_cases()
    return corpus


def first_difference(mine, theirs):
    for i, (a, b) in enumerate(zip(mine, theirs)):
        if a != b:
            return i
    return min(len(mine), len(theirs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path,
                        help="directory holding tokenizer.json")
    parser.add_argument("--binary", type=Path,
                        default=REPO_ROOT / "build" / "lcr-tokenize")
    parser.add_argument("--max-report", type=int, default=5,
                        help="how many mismatches to print before stopping")
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer as ReferenceTokenizer
    except ImportError:
        print("this check needs the reference library: pip install tokenizers",
              file=sys.stderr)
        return 2

    tokenizer_json = args.model_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        print(f"no tokenizer.json in {args.model_dir}", file=sys.stderr)
        return 2
    if not args.binary.is_file():
        print(f"no lcr-tokenize at {args.binary}; build it first",
              file=sys.stderr)
        return 2

    corpus = build_corpus()
    payload = "".join(json.dumps(text) + "\n" for text in corpus)

    result = subprocess.run(
        [str(args.binary), str(tokenizer_json)],
        input=payload, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return 1

    ours = [
        None if line.strip() == "ROUNDTRIP_FAILED"
        else [int(t) for t in line.split()]
        for line in result.stdout.splitlines()
    ]
    if len(ours) != len(corpus):
        print(f"expected {len(corpus)} output lines, got {len(ours)}",
              file=sys.stderr)
        return 1

    reference = ReferenceTokenizer.from_file(str(tokenizer_json))
    mismatches = 0
    total_tokens = 0

    for text, mine in zip(corpus, ours):
        theirs = reference.encode(text, add_special_tokens=False).ids
        total_tokens += len(theirs)
        if mine == theirs:
            continue
        mismatches += 1
        if mismatches <= args.max_report:
            print(f"--- mismatch on {json.dumps(text)[:200]}")
            if mine is None:
                print("    decode did not reproduce the input")
                continue
            index = first_difference(mine, theirs)
            print(f"    diverges at token {index}")
            print(f"    ours   {mine[max(0, index - 3):index + 4]}")
            print(f"    theirs {theirs[max(0, index - 3):index + 4]}")
            if index < len(mine):
                print(f"    ours   piece {json.dumps(reference.decode([mine[index]]))}")
            if index < len(theirs):
                print(f"    theirs piece {json.dumps(reference.decode([theirs[index]]))}")

    print(f"{len(corpus)} inputs, {total_tokens} reference tokens, "
          f"{mismatches} mismatch(es)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
