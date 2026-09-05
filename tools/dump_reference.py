#!/usr/bin/env python3
"""Dumps per-layer activations from the PyTorch reference implementation.

Nothing in this runtime gets optimized until it matches these numbers. A single
wrong sign in the rotary pairing, or a norm that divides by the wrong count,
produces text that still reads like English, so end-to-end output is not a
correctness test. Comparing every layer's residual stream against the reference
is.

    pip install torch transformers
    python3 tools/dump_reference.py models/Llama-3.2-1B-Instruct ref/
    ./build/lcr-validate --model models/Llama-3.2-1B-Instruct --reference ref/

Everything is written as little-endian float32 with a JSON manifest describing
the shapes, which is enough structure for the C++ side to read without a
dependency.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_PROMPT = "The capital of France is Paris, and the capital of Germany is"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="reference precision; bfloat16 matches what the "
                             "runtime computes in, float32 isolates whether a "
                             "difference is an implementation bug or rounding")
    parser.add_argument("--device", default="cpu",
                        help="where to run the reference; cpu works anywhere, "
                             "cuda is faster when a GPU is present")
    parser.add_argument("--generate", type=int, default=16,
                        help="greedy tokens to also record, as an end-to-end "
                             "check on top of the layer comparison")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        print(f"this script needs torch, transformers and numpy: {error}",
              file=sys.stderr)
        return 2

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device)
    model.eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(args.device)
    print(f"prompt is {input_ids.shape[1]} tokens", file=sys.stderr)

    # Hooks rather than output_hidden_states, because that option records the
    # input to each layer rather than its output and silently omits the last
    # layer's raw result.
    captured = {}

    def capture(name):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().to(torch.float32).cpu()
        return hook

    handles = [model.model.embed_tokens.register_forward_hook(capture("embed"))]
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(capture(f"layer_{index}")))
    handles.append(model.model.norm.register_forward_hook(capture("final_norm")))

    with torch.no_grad():
        outputs = model(input_ids)
    for handle in handles:
        handle.remove()

    captured["logits"] = outputs.logits.detach().to(torch.float32).cpu()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_dir": str(args.model_dir),
        "dtype": args.dtype,
        "prompt": args.prompt,
        "input_ids": input_ids[0].tolist(),
        "tensors": [],
    }
    for name in ["embed"] + [f"layer_{i}" for i in range(len(model.model.layers))] \
            + ["final_norm", "logits"]:
        tensor = captured[name]
        # Batch of one throughout; drop the leading axis so the C++ side sees
        # the [tokens, width] it works in.
        array = tensor[0].numpy().astype(np.float32)
        path = args.out_dir / f"{name}.bin"
        path.write_bytes(array.tobytes(order="C"))
        manifest["tensors"].append({
            "name": name,
            "shape": list(array.shape),
            "file": path.name,
        })
        print(f"  {name:16s} {list(array.shape)}", file=sys.stderr)

    if args.generate > 0:
        with torch.no_grad():
            generated = model.generate(
                input_ids, max_new_tokens=args.generate, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
        continuation = generated[0][input_ids.shape[1]:].tolist()
        manifest["greedy_continuation"] = continuation
        manifest["greedy_text"] = tokenizer.decode(continuation)
        print(f"greedy continuation: {manifest['greedy_text']!r}", file=sys.stderr)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out_dir}/manifest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
