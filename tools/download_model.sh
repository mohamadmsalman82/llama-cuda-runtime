#!/usr/bin/env bash
# Fetches a Llama-3.2-1B-Instruct checkpoint into models/.
#
# The official meta-llama repository is gated: accept the licence on the model
# page, export HF_TOKEN, and pass --official. The default is an ungated mirror
# of the same weights, which is what makes a fresh clone runnable without an
# account.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="unsloth/Llama-3.2-1B-Instruct"
DEST="$ROOT/models/Llama-3.2-1B-Instruct"

while [ $# -gt 0 ]; do
  case "$1" in
    --official) REPO="meta-llama/Llama-3.2-1B-Instruct" ;;
    --repo) REPO="$2"; shift ;;
    --dest) DEST="$2"; shift ;;
    -h|--help)
      echo "usage: $0 [--official] [--repo ORG/NAME] [--dest DIR]"
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$DEST"
BASE="https://huggingface.co/$REPO/resolve/main"

# The first three are what this runtime reads. The rest are kept so the
# reference tooling sees the same directory a transformers checkout would.
REQUIRED="config.json tokenizer.json model.safetensors"
OPTIONAL="generation_config.json tokenizer_config.json special_tokens_map.json"

fetch() {
  file="$1"
  required="$2"
  target="$DEST/$file"
  if [ -s "$target" ]; then
    echo "have $file"
    return 0
  fi
  echo "fetching $file"
  if [ -n "${HF_TOKEN:-}" ]; then
    code=$(curl -sS -L -H "Authorization: Bearer $HF_TOKEN" \
           -o "$target.part" -w '%{http_code}' "$BASE/$file")
  else
    code=$(curl -sS -L -o "$target.part" -w '%{http_code}' "$BASE/$file")
  fi
  if [ "$code" = "200" ]; then
    mv "$target.part" "$target"
    return 0
  fi
  rm -f "$target.part"
  if [ "$required" = "yes" ]; then
    echo "failed to fetch $file from $REPO (HTTP $code)" >&2
    echo "If the repository is gated, accept the licence at" >&2
    echo "https://huggingface.co/$REPO and export HF_TOKEN." >&2
    exit 1
  fi
  echo "  $file is not in this repository, skipping"
}

for file in $REQUIRED; do fetch "$file" yes; done
for file in $OPTIONAL; do fetch "$file" no; done

echo
echo "checkpoint ready in $DEST"
du -sh "$DEST"
