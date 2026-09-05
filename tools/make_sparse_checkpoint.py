#!/usr/bin/env python3
"""Builds a checkpoint directory whose weights are a sparse file of zeros.

A safetensors file is a JSON header followed by one flat data buffer, and the
header is the first few kilobytes. Everything the loader validates lives there:
tensor names, shapes, dtypes, and byte ranges. So the load path, the shape
checks, the tied-embedding detection and the memory accounting can all be
exercised against the real checkpoint's structure after downloading 200 KB
instead of 2.5 GB, by fetching the header and writing a sparse file that reports
the right length while occupying almost no disk.

    python3 tools/make_sparse_checkpoint.py /tmp/fake-llama
    ./build/lcr-inspect --model /tmp/fake-llama

The weights are zeros, so this is for testing the loader and the sizing, never
for generating text.
"""

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "unsloth/Llama-3.2-1B-Instruct"
SMALL_FILES = ["config.json", "generation_config.json", "tokenizer.json"]
# Comfortably more than any 1B checkpoint's header, and a rounding error next to
# the file it stands in for.
HEADER_PROBE_BYTES = 1 << 20


def fetch(url, byte_range=None):
    """Downloads with curl, which uses the system trust store.

    Some Python builds ship without a certificate bundle wired up, so urllib
    fails to verify huggingface.co where curl succeeds.
    """
    command = ["curl", "-sS", "-L", "--fail"]
    if byte_range is not None:
        command += ["-H", f"Range: bytes=0-{byte_range - 1}"]
    result = subprocess.run(command + [url], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dest", type=Path)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()

    base = f"https://huggingface.co/{args.repo}/resolve/main"
    args.dest.mkdir(parents=True, exist_ok=True)

    for name in SMALL_FILES:
        try:
            (args.dest / name).write_bytes(fetch(f"{base}/{name}"))
            print(f"fetched {name}", file=sys.stderr)
        except Exception as error:
            if name == "generation_config.json":
                print(f"  {name} not in this repository, skipping",
                      file=sys.stderr)
                continue
            raise SystemExit(f"could not fetch {name}: {error}")

    probe = fetch(f"{base}/model.safetensors", HEADER_PROBE_BYTES)
    header_length = struct.unpack("<Q", probe[:8])[0]
    if 8 + header_length > len(probe):
        raise SystemExit(f"header is {header_length} bytes, larger than the "
                         f"{HEADER_PROBE_BYTES} byte probe")
    header_bytes = probe[8:8 + header_length]
    header = json.loads(header_bytes)

    payload = max(entry["data_offsets"][1]
                  for name, entry in header.items() if name != "__metadata__")

    path = args.dest / "model.safetensors"
    with path.open("wb") as out:
        out.write(probe[:8])
        out.write(header_bytes)
        # Seeking past the end and writing one byte leaves a hole: the file
        # reports its full length and the filesystem stores nothing for the gap.
        out.seek(8 + header_length + payload - 1)
        out.write(b"\0")

    tensors = len(header) - (1 if "__metadata__" in header else 0)
    on_disk = path.stat().st_blocks * 512
    print(f"wrote {path}", file=sys.stderr)
    print(f"  {tensors} tensors, {path.stat().st_size / 1e9:.3f} GB reported, "
          f"{on_disk / 1e3:.0f} kB actually on disk", file=sys.stderr)
    print("  the weights are zeros: use this for the loader, not for text",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
