# llama-cuda-runtime

A single-GPU inference runtime for Llama-3.2-1B, written from scratch in C++17 and CUDA.
No PyTorch, no libtorch, no ONNX Runtime, no existing inference library. The only
dependencies are the CUDA toolkit, cuBLAS, and a JSON parser.

The point of the project is the prefill/decode split. Prefill is compute-bound: one
matmul chews through every prompt token at once. Decode is memory-bandwidth-bound:
producing a single token requires streaming all 2.5 GB of weights out of HBM, so the
ceiling is the memory bus, not the SMs. The headline number is achieved decode
bandwidth as a fraction of theoretical peak, next to tokens/sec against llama.cpp and
HuggingFace Transformers on the same model and the same GPU.

## Scope

In: single GPU, batch size 1, bf16/fp16 weights, one model family.
Out: batching, quantization, multi-GPU, speculative decoding, serving, custom GEMM.

Written by hand: safetensors loading, BPE tokenizer, activation arena, KV cache
(contiguous and paged), and CUDA kernels for RMSNorm, RoPE, SwiGLU, softmax,
decode-phase attention, and sampling. cuBLAS does the matmuls.

## Status

- [x] Phase 0: repo skeleton, CMake, host/CUDA split
- [ ] Phase 1: safetensors loader and model config
- [ ] Phase 2: BPE tokenizer
- [ ] Phase 3: activation arena and KV cache
- [ ] Phase 4: CUDA kernels
- [ ] Phase 5: end-to-end forward pass, greedy decode
- [ ] Phase 6: numerical validation against a PyTorch reference dump
- [ ] Phase 7: paged KV cache
- [ ] Phase 8: profiling and kernel optimization
- [ ] Phase 9: benchmark table and bandwidth analysis

## Building

Needs CMake 3.24+, a C++17 compiler, and CUDA 12.x with cuBLAS.

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The host-side pieces (safetensors, tokenizer, config) build and test without a GPU, so
the loader and tokenizer can be developed on a machine with no NVIDIA hardware. CMake
detects nvcc and silently drops the CUDA targets when it is missing. Set the target
architecture explicitly for a faster build:

```
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

## Model

Llama-3.2-1B-Instruct. 16 layers, hidden size 2048, 32 query heads and 8 key/value
heads of dimension 64, SwiGLU MLP with intermediate size 8192, 128256-entry vocabulary
with tied input and output embeddings, RoPE base 500000 with llama3 frequency scaling.

```
./tools/download_model.sh
```

The weights are gated on Hugging Face. Accept the license on the model page and run
`huggingface-cli login` before the script will work.
