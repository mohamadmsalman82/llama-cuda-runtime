#!/usr/bin/env python3
"""Benchmarks this runtime against HuggingFace Transformers and llama.cpp.

Same model, same GPU, same prompt, same number of generated tokens. Prefill and
decode are timed separately because they are limited by different things:
prefill saturates the arithmetic units, decode saturates the memory bus. A
single tokens/sec number hides that, which is exactly the distinction this
project is about.

    python3 tools/benchmark.py --model models/Llama-3.2-1B-Instruct
    python3 tools/benchmark.py --model models/... \\
        --llama-bench ~/llama.cpp/build/bin/llama-bench \\
        --gguf models/Llama-3.2-1B-Instruct-f16.gguf

Writes a markdown table to stdout, and to docs/benchmarks.md with --write.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = (
    "Write a short paragraph explaining why generating one token at a time from "
    "a language model is limited by memory bandwidth rather than by arithmetic."
)


def run_ours(args):
    """Runs llama-run in JSON mode and returns its report."""
    command = [
        str(args.binary), "--model", str(args.model), "--json", "--quiet",
        "--prompt", args.prompt,
        "--max-tokens", str(args.tokens),
        "--max-seq", str(args.max_seq),
        "--prefill-chunk", str(args.prefill_chunk),
        "--temperature", "0",
    ]
    if args.paged:
        command += ["--paged", "--page-size", str(args.page_size)]
    if args.peak_bandwidth:
        command += ["--peak-bandwidth", str(args.peak_bandwidth)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"{args.binary} failed")
    # The generated text is printed before the JSON object.
    start = result.stdout.index("{")
    return json.loads(result.stdout[start:])


def run_transformers(args, bytes_per_token, peak_bandwidth):
    """Times HuggingFace Transformers on the same prompt and token count."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16).cuda().eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()
    prompt_tokens = input_ids.shape[1]

    def one_run():
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(input_ids, use_cache=True)
            torch.cuda.synchronize()
            prefill = time.perf_counter() - start

            past = outputs.past_key_values
            token = outputs.logits[:, -1:].argmax(-1)
            start = time.perf_counter()
            for _ in range(args.tokens - 1):
                outputs = model(token, past_key_values=past, use_cache=True)
                past = outputs.past_key_values
                token = outputs.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            decode = time.perf_counter() - start
        return prefill, decode

    one_run()  # warm up the kernels and the allocator
    runs = [one_run() for _ in range(args.repeats)]
    prefill = statistics.median(r[0] for r in runs)
    decode = statistics.median(r[1] for r in runs)
    steps = args.tokens - 1

    del model
    torch.cuda.empty_cache()

    return {
        "runtime": "HuggingFace Transformers",
        "prompt_tokens": prompt_tokens,
        "prefill_ms": prefill * 1000.0,
        "prefill_tokens_per_second": prompt_tokens / prefill,
        "decode_steps": steps,
        "decode_ms_per_token": decode * 1000.0 / steps,
        "decode_tokens_per_second": steps / decode,
        "achieved_bandwidth_bytes_per_second": bytes_per_token * steps / decode,
        "bandwidth_utilization":
            bytes_per_token * steps / decode / peak_bandwidth,
    }


def run_llama_cpp(args, bytes_per_token, peak_bandwidth, prompt_tokens):
    """Runs llama-bench and pulls out its prefill and decode numbers."""
    if not args.llama_bench or not args.gguf:
        return None
    command = [
        str(args.llama_bench), "-m", str(args.gguf),
        "-p", str(prompt_tokens), "-n", str(args.tokens),
        "-r", str(args.repeats), "-o", "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return None

    rows = json.loads(result.stdout)
    prefill_row = next((r for r in rows if r.get("n_prompt", 0) > 0), None)
    decode_row = next((r for r in rows if r.get("n_gen", 0) > 0), None)
    if prefill_row is None or decode_row is None:
        return None

    decode_rate = float(decode_row["avg_ts"])
    return {
        "runtime": f"llama.cpp ({decode_row.get('build_commit', 'unknown')})",
        "prompt_tokens": prompt_tokens,
        "prefill_ms": 1000.0 * prompt_tokens / float(prefill_row["avg_ts"]),
        "prefill_tokens_per_second": float(prefill_row["avg_ts"]),
        "decode_steps": args.tokens,
        "decode_ms_per_token": 1000.0 / decode_rate,
        "decode_tokens_per_second": decode_rate,
        "achieved_bandwidth_bytes_per_second": bytes_per_token * decode_rate,
        "bandwidth_utilization": bytes_per_token * decode_rate / peak_bandwidth,
    }


def format_table(rows):
    header = ("| runtime | prefill tok/s | decode tok/s | ms/token | "
              "achieved GB/s | % of peak |")
    divider = "|---|---:|---:|---:|---:|---:|"
    lines = [header, divider]
    for row in rows:
        lines.append(
            "| {runtime} | {prefill:.0f} | {decode:.1f} | {ms:.3f} | "
            "{bandwidth:.0f} | {utilization:.1f}% |".format(
                runtime=row["runtime"],
                prefill=row["prefill_tokens_per_second"],
                decode=row["decode_tokens_per_second"],
                ms=row["decode_ms_per_token"],
                bandwidth=row["achieved_bandwidth_bytes_per_second"] / 1e9,
                utilization=100.0 * row["bandwidth_utilization"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path,
                        default=REPO_ROOT / "build" / "llama-run")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--prefill-chunk", type=int, default=256)
    parser.add_argument("--paged", action="store_true")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--peak-bandwidth", type=float, default=0.0,
                        help="theoretical peak in GB/s, when the driver reports "
                             "it wrongly")
    parser.add_argument("--llama-bench", type=Path)
    parser.add_argument("--gguf", type=Path,
                        help="f16 GGUF of the same model; a quantized one moves "
                             "fewer bytes and is not the same comparison")
    parser.add_argument("--write", action="store_true",
                        help="also write docs/benchmarks.md")
    args = parser.parse_args()

    ours = run_ours(args)
    # Every runtime here reads the same weights in the same precision, so the
    # byte count this runtime measures is the right denominator for all of them.
    bytes_per_token = ours["bytes_per_token"]
    peak = ours["peak_bandwidth_bytes_per_second"]
    ours["runtime"] = "this runtime"

    rows = [ours]
    for candidate in (
        run_transformers(args, bytes_per_token, peak),
        run_llama_cpp(args, bytes_per_token, peak, ours["prompt_tokens"]),
    ):
        if candidate is not None:
            rows.append(candidate)

    report = [
        f"Device: {ours['device_summary']}",
        f"Model: {ours['model']} in {ours['dtype']}",
        f"Prompt: {ours['prompt_tokens']} tokens, generating {args.tokens}",
        f"Bytes moved per decoded token: {bytes_per_token / 1e9:.3f} GB "
        f"({ours['weight_bytes_per_token'] / 1e9:.3f} GB of weights plus the "
        f"cache at a mean length of "
        f"{ours['sequence_length'] - args.tokens // 2} positions)",
        "",
        format_table(rows),
        "",
        "KV cache for this sequence, priced under each layout:",
        "",
        "| layout | bytes |",
        "|---|---:|",
        f"| contiguous, full context | "
        f"{ours['kv_bytes_contiguous_full_context'] / 1e6:.0f} MB |",
        f"| contiguous, --max-seq {args.max_seq} | "
        f"{ours['kv_bytes_contiguous_max_seq'] / 1e6:.0f} MB |",
        f"| paged, page size {args.page_size} | "
        f"{ours['kv_bytes_paged'] / 1e6:.1f} MB |",
        f"| strictly needed | {ours['kv_bytes_ideal'] / 1e6:.1f} MB |",
    ]
    text = "\n".join(report)
    print(text)

    if args.write:
        out = REPO_ROOT / "docs" / "benchmarks.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text("# Benchmarks\n\nGenerated by tools/benchmark.py.\n\n"
                       + text + "\n")
        print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
