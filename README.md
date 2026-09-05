# llama-cuda-runtime

A single-GPU inference runtime for Llama-3.2-1B, written from scratch in C++17 and
CUDA. No PyTorch, no libtorch, no ONNX Runtime, no existing inference library. The
only dependencies are the CUDA toolkit, cuBLAS, and a JSON parser.

The point of the project is the prefill/decode split. Prefill has the whole prompt in
flight, so every projection is a real matrix multiply and the GPU is compute bound.
Decode has one token, so every projection is a matrix times a vector: each of the 2.5
GB of weights is read out of HBM, used for a single multiply-add, and discarded.
Nothing about the arithmetic matters at that point, only how fast the weights can be
streamed. The headline number is therefore achieved decode bandwidth as a fraction of
theoretical peak, next to tokens/sec against llama.cpp and HuggingFace Transformers
on the same model and the same GPU.

## Scope

In: single GPU, batch size 1, bf16 or fp16 weights, one model family.
Out: batching, quantization, multi-GPU, speculative decoding, serving, custom GEMM.

Written by hand: safetensors loading, the BPE tokenizer, an activation arena, the KV
cache in both a contiguous and a paged variant, and CUDA kernels for RMSNorm, RoPE,
SwiGLU, softmax, decode-phase attention, and sampling. cuBLAS does the matmuls.

## Status

- [x] Repo skeleton, CMake, host/CUDA split so the host half builds without a GPU
- [x] safetensors loader and model config, with Llama 3 RoPE frequency scaling
- [x] BPE tokenizer, zero mismatches against the reference tokenizer over 54k tokens
- [x] Activation arena, contiguous and paged KV cache
- [x] All CUDA kernels, and a model forward pass for both phases
- [x] CLI, reference-dump validator, GPU kernel tests, benchmark harness
- [x] Runs on hardware. First `nvcc` build passed with no errors, every kernel
      test passes on device, and it generates coherent text.
- [x] Numerical validation against the PyTorch reference, layer by layer: every
      layer within tolerance, identical top-5 logits, 16 of 16 greedy tokens
      identical
- [x] Benchmark table, bandwidth analysis, prefill/decode breakdown, KV cache
      comparison. See [docs/benchmarks.md](docs/benchmarks.md).
- [x] Per-stage profiling and the kernel optimization pass. See
      [docs/optimization.md](docs/optimization.md).
- [ ] CUDA graphs for the decode step, the largest remaining win
- [ ] llama.cpp row in the benchmark table, which needs an f16 GGUF

## Results

RTX 4090, Llama-3.2-1B-Instruct in bf16, before any optimization:

| runtime | decode tok/s | ms/token | achieved GB/s | % of peak |
|---|---:|---:|---:|---:|
| this runtime | 322.3 | 3.103 | 798 | **79.1%** |
| HuggingFace Transformers | 72.0 | 13.894 | 178 | 17.7% |

Decoding one token means reading all 2.472 GB of weights out of HBM, so
tokens/sec and achieved bandwidth are one measurement in two units. The ceiling
on this card is 407 tokens/s. The HuggingFace figure is its default path, which
falls back to the legacy tuple KV cache; `StaticCache` with `torch.compile`
would raise it considerably. [docs/benchmarks.md](docs/benchmarks.md) has the
full table, the prefill split, and the KV cache comparison.

## Building

CMake 3.24+, a C++17 compiler, CUDA 12.x with cuBLAS.

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

Set `CMAKE_CUDA_ARCHITECTURES` to your card (80 for A100, 86 for A10/3090, 89 for
L4/L40S/4090, 90 for H100); leaving it unset builds all four and takes several times
as long. `-DLCR_DTYPE=fp16` switches weights and activations to fp16; the default is
bf16, which is what the checkpoint ships in.

Without nvcc, CMake drops the CUDA targets and still builds the loader, the config
parser, the tokenizer, and their tests. That is what makes the host half of the
project developable away from a GPU.

## Model

Llama-3.2-1B-Instruct. 16 layers, hidden size 2048, 32 query heads and 8 key/value
heads of dimension 64, SwiGLU MLP with intermediate size 8192, 128256-entry
vocabulary with tied input and output embeddings, RoPE base 500000 with llama3
frequency scaling.

```
./tools/download_model.sh
```

This pulls an ungated mirror of the weights so a fresh clone runs without an account.
`--official` uses `meta-llama/Llama-3.2-1B-Instruct`, which needs the licence accepted
on the model page and `HF_TOKEN` exported.

## Running

```
./build/llama-run --model models/Llama-3.2-1B-Instruct \
    --prompt "The capital of France is" --max-tokens 64

./build/llama-run --model models/Llama-3.2-1B-Instruct --chat \
    --prompt "Explain why decoding is memory bound." --temperature 0.7

./build/llama-run --model models/Llama-3.2-1B-Instruct --bench --max-tokens 256
```

`--bench` adds the timing and bandwidth report; `--json` emits the same numbers as
JSON. `--paged` switches the KV cache to the paged layout.

## Correctness

Text that reads well is not evidence of a correct implementation. A rotary pairing
that swaps the two halves of a head, a norm that divides by the wrong count, a
grouped-query mapping off by one: every one of those still produces fluent English.
So nothing gets optimized until the numbers match, at three levels.

**Kernels against a reference implementation of the same maths.** `tests/test_kernels.cu`
checks every kernel against a slow, obvious nested-loop version, with inputs rounded
through the storage type first so the only difference measured is the kernel's.
Attention is covered across grouped, multi-head and multi-query shapes, both compiled
head dimensions, split counts that do not divide the position range evenly, more
splits than positions, a single cached position, and the paged layout against the
contiguous one.

**Layers against PyTorch.**

```
pip install torch transformers
python3 tools/dump_reference.py models/Llama-3.2-1B-Instruct ref/
./build/lcr-validate --model models/Llama-3.2-1B-Instruct --reference ref/ --generate 16
```

The dump records the residual stream after every layer, through forward hooks rather
than `output_hidden_states`, which records layer inputs and omits the last layer's raw
output. The validator replays the same prompt and reports max absolute error, RMS
error relative to the signal, and worst per-token cosine similarity for each layer,
then the top-5 logits and the greedy continuation.

**The tokenizer against the reference tokenizer.**

```
python3 -m venv .venv && .venv/bin/pip install tokenizers
.venv/bin/python tools/check_tokenizer.py models/Llama-3.2-1B-Instruct
```

850 inputs: hand-picked cases for every branch of the pretokenizer regex, a sweep
across the Unicode range, 600 random boundary-heavy strings, and this repository's own
sources. Current result is 54,323 tokens with zero mismatches.

## Developing without a GPU

`tools/hostcheck.sh` compiles every `.cu` file as ordinary C++17 against stand-in
headers in `tools/hostcheck/`, after stripping the `<<<...>>>` launch syntax, which is
the only part of CUDA C++ a host compiler cannot parse. Every kernel body, every
template instantiation, and every cuBLAS call signature gets type-checked.

```
./tools/hostcheck.sh
```

It proves the code compiles. It proves nothing about what it computes; that is what
the tests above are for.

## Benchmarking

```
python3 tools/benchmark.py --model models/Llama-3.2-1B-Instruct --tokens 256 \
    --llama-bench ~/llama.cpp/build/bin/llama-bench \
    --gguf models/Llama-3.2-1B-Instruct-f16.gguf
```

Same model, same GPU, same prompt, same token count, with prefill and decode timed
separately. The GGUF has to be f16: a quantized one moves fewer bytes and is a
different measurement.

## Design notes

[docs/design.md](docs/design.md) has the arithmetic behind the project: why decode
is bandwidth bound and prefill is not, what actually crosses the memory bus per
token, the ceiling that implies on several GPUs, why the layouts are what they are,
and the questions the profiling pass exists to answer.

## Layout

```
src/
  safetensors.*      memory-mapped checkpoint reader
  config.*           config.json, and the RoPE frequency table
  tokenizer.*        byte-level BPE, Llama 3 pretokenizer by hand
  unicode_data.cpp   generated letter and number tables
  arena.*            bump allocator for activations
  kv_cache.cuh/.cu   contiguous and paged, one address formula
  kernels/           the hand-written kernels
  model.*            weights on the device, prefill and decode
  main.cu            command-line front end
tools/
  hostcheck/         stand-in CUDA headers for the host compile check
  dump_reference.py  PyTorch per-layer activation dump
  check_tokenizer.py differential test against the reference tokenizer
  benchmark.py       the comparison table
  validate_main.cu   layer-by-layer comparison against the dump
tests/
```
