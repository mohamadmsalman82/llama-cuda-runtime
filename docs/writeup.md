# Building a Llama inference runtime from scratch in C++ and CUDA

A complete account of the project: what it is, how it was built, what broke,
what the numbers are, and what they mean.

Repository: https://github.com/mohamadmsalman82/llama-cuda-runtime
Built 2026-09-04, first run on hardware 2026-09-05.

---

## 1. What the project is

A standalone program that loads Llama-3.2-1B-Instruct and generates text, written
from nothing in C++17 and CUDA. No PyTorch, no libtorch, no ONNX Runtime, no
existing inference library of any kind. The only dependencies are the CUDA
toolkit, cuBLAS for matrix multiplication, and a JSON parser.

Written by hand: the safetensors checkpoint reader, the BPE tokenizer, a memory
arena for activations, the KV cache in two different layouts, and CUDA kernels
for RMSNorm, RoPE, SwiGLU, softmax, decode-phase attention, and sampling.
cuBLAS does the matrix multiplies, which is the one place where writing it
yourself gains nothing.

Deliberately out of scope: batching, quantization, multiple GPUs, speculative
decoding, serving infrastructure, and writing a custom GEMM.

### The question it answers

Generating text from a transformer runs the same layers under two completely
different conditions, and confusing them is the single most common mistake in
reasoning about inference performance.

**Prefill** processes the entire prompt at once. Every projection is a matrix
multiply with as many rows as there are prompt tokens, so each weight, once
fetched from memory, gets used for hundreds of multiply-accumulate operations.
The GPU's arithmetic units are the bottleneck.

**Decode** produces one token. Every projection collapses to a matrix times a
vector. Each of the 2.472 GB of weights is read from memory, used for exactly
one multiply-add, and discarded. The arithmetic units sit almost entirely idle,
waiting on memory.

Put numerically, for this model:

| | prefill, T tokens | decode, 1 token |
|---|---|---|
| arithmetic | 2 x 1.236e9 x T FLOP | 2.47 GFLOP |
| weight traffic | 2.472 GB | 2.472 GB |
| arithmetic intensity | T FLOP/byte | 1 FLOP/byte |

Machine balance, the arithmetic intensity above which a GPU stops being memory
bound, is roughly 150 FLOP/byte on an A100 and 300 on an H100. Prefill with a
few hundred tokens sits at or above that line. Decode sits two orders of
magnitude below it, and no amount of kernel cleverness moves it, because the
weights simply have to be read.

That gives decode a hard ceiling that follows directly from memory bandwidth:

| GPU | bandwidth | ceiling on this model |
|---|---:|---:|
| RTX 4090 | 1008 GB/s | 407 tokens/s |
| A100 80GB | 2039 GB/s | 825 tokens/s |
| H100 SXM | 3350 GB/s | 1355 tokens/s |

So the meaningful measurement is not tokens per second in isolation. It is what
fraction of that hardware ceiling the implementation actually reaches.

---

## 2. The unusual constraint

The entire project was written on an Apple M4 Pro MacBook. That machine has no
NVIDIA GPU, no CUDA toolkit, and no Docker. Nothing could be compiled, let alone
run.

Writing 2000 lines of CUDA that has never been near a compiler is a good way to
produce 2000 lines of plausible-looking nonsense. Every kernel launch
configuration, every template instantiation, every cuBLAS argument order would
meet a compiler for the first time on rented hardware, at hourly cost, all at
once.

### The workaround: a compile shim

CUDA C++ is ordinary C++17 with one syntactic exception: the `<<<grid, block>>>`
kernel launch, which no host compiler can parse. Everything else, including
`__global__` functions, `__device__` helpers, shared memory declarations and
template dispatch, is either valid C++ already or can be made so with a macro.

So `tools/hostcheck.sh` strips the launch syntax with a regular expression and
compiles every `.cu` file as plain C++17 against stand-in headers in
`tools/hostcheck/`. Those headers declare `threadIdx`, `blockIdx`, `uint4`,
`atomicMax`, `__shfl_xor_sync`, the CUDA runtime API, and the cuBLAS functions
with signatures matching the real ones exactly.

This type-checks every kernel body, every template instantiation, every argument
list, and every cuBLAS call. Stripping `<<<...>>>` leaves an ordinary function
call, so a kernel invoked with the wrong number of arguments still fails to
compile.

It proves the code compiles. It proves nothing about what it computes. The
distinction matters and is stated everywhere the tool appears.

To confirm it was not vacuous, I deliberately broke a kernel call by removing an
argument. The shim caught it. It also caught two genuine bugs during
development: `dtype.cuh` using `uint2` and `uint4` without including
`cuda_runtime.h`, and two missing includes in the test file.

### Testing the loader without the weights

A safetensors file is an 8-byte length, a JSON header describing every tensor,
and then one flat data buffer. Everything the loader validates lives in that
header, which is the first 17 kilobytes of a 2.5 GB file.

So `tools/make_sparse_checkpoint.py` fetches only the header with an HTTP range
request and writes a sparse file: seek past the end, write one byte, and the
filesystem reports the full 2.5 GB length while storing 41 kilobytes. The whole
load path, every shape check, the tied-embedding detection and all the memory
accounting can then be exercised after a 200 kB download.

That is how the checkpoint inventory below was verified before any GPU existed:
all 146 required tensors present, correct shapes, correct dtype.

---

## 3. How it was built

Twenty-one commits, in dependency order, each one tested before the next began.

### 3.1 Safetensors loader

Memory-maps the checkpoint. Opening a 2.5 GB file costs nothing until pages are
touched by the upload to the GPU. Handles both single-file and sharded
checkpoints via `model.safetensors.index.json`.

Every tensor's declared byte span is validated against its shape and dtype
before use. A mismatch would otherwise become an out-of-bounds read during the
GPU upload, which surfaces as a corrupted model rather than an error. Tests
build malformed checkpoints on disk and confirm each is rejected with a useful
message: wrong byte span, span past the buffer, header longer than the file,
non-JSON header, unknown dtype.

### 3.2 Model configuration and RoPE scaling

Parses `config.json` into the geometry the runtime needs, and computes the
rotary frequency table on the host once at load.

Llama 3.x rescales the rotary base so a model trained at 8k context extrapolates
to 128k. Low-frequency components are divided by a factor, high-frequency
components are untouched, and the band between is linearly interpolated. The
test recomputes all three cases independently and asserts every band is actually
exercised, so an implementation handling only one case cannot pass.

### 3.3 BPE tokenizer

The largest single component, and the one with the most ways to be subtly wrong.

Encoding runs four stages. Added tokens such as `<|eot_id|>` are cut out by
literal match. Each remaining span is split by the Llama 3 pretokenizer regex.
Each pretoken's bytes are mapped through the GPT-2 byte-level alphabet, so BPE
never sees invalid UTF-8. Then merges are applied in rank order.

The regex is written by hand as a matcher, because pulling in a regex engine
would break the from-scratch constraint. It needs Unicode letter and number
categories, which are generated from Python's `unicodedata` into sorted
codepoint ranges the tokenizer binary-searches.

Two details that matter. The loader compares the pattern in `tokenizer.json`
against the one implemented and refuses to run on a checkpoint that splits
differently, rather than silently encoding wrongly. And `ignore_merges` is
honoured, which Llama 3 requires: many whole words exist in its vocabulary but
are not reachable by replaying the merge list.

BPE itself uses a doubly linked list over byte ranges plus a priority queue of
ranked adjacent pairs, giving O(n log n) instead of rescanning every pair each
round.

**Validation.** `tools/check_tokenizer.py` encodes an 850-input corpus with both
this tokenizer and HuggingFace's and compares ids position by position. The
corpus is hand-picked cases for every branch of the regex, a sweep across the
entire Unicode range, 600 random strings from boundary-heavy alphabets, and the
repository's own source files.

Result: **54,323 tokens, zero mismatches.**

### 3.4 Memory: arena and KV cache

Activations come from a bump allocator sized once at load. `cudaMalloc`
synchronizes the device, and calling it inside a loop that should take one
millisecond would dominate the loop.

Weights go into a single allocation, laid out in the order the forward pass
reads them, so a decode step walks memory forward instead of jumping between 146
separate blocks.

The KV cache is head-major, `K[kv_head][position][head_dim]`. A decode step
reads one head's entire history, so that history must be contiguous. The obvious
alternative, position-major, makes the per-token write contiguous instead, which
optimizes the wrong thing: at position 4000 a step reads 4000 vectors and writes
one.

Two layouts share one address expression, with the page-table lookup templated
away for the contiguous case, so comparing them compares layouts rather than
kernels.

### 3.5 The CUDA kernels

RMSNorm, residual add, SwiGLU, embedding lookup, head merge, RoPE, causal
softmax, decode attention, sampling, and a paged-cache gather. All streaming
kernels move data in 128-bit chunks with a scalar tail for shapes that are not a
multiple of eight.

**Decode attention** is the kernel the project is about, and it has two
structural decisions.

*One block per grouped-query group.* This model has 32 query heads and 8
key/value heads, so four queries share each key and value. A block covers a
whole group and loads each cache vector once, using it for all four dot
products. Done per query head instead, the same cache lines would be read four
times. At 32k context that is the difference between 1 GB and 4 GB of cache
traffic per token.

*Splitting positions across blocks.* Eight groups means eight blocks, on a GPU
with 128 multiprocessors. Most of the machine would idle. Each block instead
takes a slice of the positions and keeps a partial softmax, which a second pass
merges.

The softmax is online throughout: a running maximum and exponential sum updated
per position, rescaled when a larger score appears. No score row is ever
materialized, which lets one block handle arbitrarily long history in fixed
registers.

One subtle detail. The identity element is `-FLT_MAX`, not `-inf`. Two empty
accumulators combining as `(-inf) - (-inf)` produces NaN, and a block with more
warps than positions produces exactly that. `-FLT_MAX` subtracts from itself to
zero and exponentiates to zero against any real score, behaving as a true
identity with no special case.

**Sampling** avoids an obvious trap. Top-p is defined over the descending order
of the whole distribution, and sorting 128,256 floats per token would cost more
than the attention preceding it. But the nucleus is tiny: after temperature,
anything more than a few nats below the maximum has no chance of selection. So
the kernel builds a histogram of log-probabilities in one pass, reads off a
cutoff admitting at most 1024 candidates, compacts those, and sorts only them.
Three passes over the vocabulary, all but the first from L2, and an exact result
whenever the nucleus fits in 1024 tokens, which for a 1B model it always does.

### 3.6 Forward pass

Prefill goes through cuBLAS batched GEMMs grouped by key/value head, which the
head-major layout makes free: the four query heads sharing a key/value head are
already one contiguous block of rows.

The rotary kernel transposes queries into head-major order on the way through,
costing nothing because it is already touching every element. During decode that
same layout is already what the attention kernel wants, so no transpose runs at
all.

The prompt is processed in chunks so the attention score matrix stays bounded by
chunk size rather than prompt length. Each chunk attends to everything already
cached, which is exactly the causal mask the softmax applies anyway.

Only the last position's logits are projected onto the vocabulary. Projecting a
whole prompt onto 128,256 columns would cost more than the rest of prefill
combined.

---

## 4. Getting onto hardware

### 4.1 Choosing the machine

The plan was an RTX 4090 on RunPod community cloud at $0.34/hour. The 4090 uses
the same Ada architecture as the local card available for later work, so fixes
transfer.

That failed. Community cloud requires a public IP for SSH, and no community 4090
host offers one. Secure cloud at $0.74/hour worked immediately. The first pod
also had to be recreated because it was created without exposing port 22.

### 4.2 Environment problems

Three, none of them interesting individually, all of them time sinks:

- Ubuntu 22.04 ships CMake 3.22; the project requires 3.24 because CUDA
  architecture 90 support landed there. Fixed with `pip install cmake`.
- transformers 5.x requires PyTorch 2.5+, and the pod image ships 2.4.1. It
  silently disabled PyTorch and would have produced a confusing failure much
  later. Pinned to transformers 4.46.3.
- Nsight Compute cannot collect counters inside a container. GPU performance
  counters are gated by a kernel module parameter on the host, so `ncu` fails
  with `ERR_NVGPUCTRPERM` and there is no fix from the guest. This forced
  building the profiler into the runtime, which turned out better anyway:
  anybody cloning the repository can reproduce the profile with a flag.

### 4.3 The first compile

The first real `nvcc` invocation produced **zero errors**, and all five binaries
linked. The compile shim had done its job.

---

## 5. Proving it is correct

This is the part that matters most, and the part most easily faked.

A language model with a bug still produces fluent English. A rotary pairing that
swaps the two halves of a head, a norm dividing by the wrong count, a
grouped-query mapping off by one: every one of those yields confident, readable,
wrong text. Reading the output tells you nothing. So correctness was established
at three levels, in order, before anything was optimized.

### 5.1 Kernels against a reference implementation

`tests/test_kernels.cu` checks every kernel against a deliberately slow,
obvious, nested-loop version of the same mathematics. Inputs are rounded through
the storage type first, so the only difference measured is the kernel's own.

Attention is covered across eleven shapes: grouped-query, multi-head and
multi-query; both compiled head dimensions; split counts that do not divide the
position range evenly; more splits than positions; a single cached position; and
the paged layout against the contiguous one.

All pass on the 4090. The suite was also confirmed to actually execute rather
than take its "no CUDA device, skipping" path, which would have passed silently.

### 5.2 Layers against PyTorch

`tools/dump_reference.py` records the residual stream after every layer from the
real PyTorch implementation. It uses forward hooks rather than
`output_hidden_states`, because that option records layer *inputs* and omits the
last layer's raw output.

The dump is taken in **float32 on purpose**. Comparing bf16 against bf16 would
confound two different things: an implementation bug, and ordinary rounding.
Against a float32 reference, the residual difference is attributable.

`lcr-validate` replays the same prompt and reports, per layer, maximum absolute
error, RMS error relative to signal magnitude, and worst per-token cosine
similarity.

Result:

| tensor | max abs | rel rms | min cosine |
|---|---:|---:|---:|
| embed | 0.00000 | 0.00e+00 | 1.000000 |
| layer_0 | 0.06867 | 3.16e-03 | 0.999978 |
| layer_7 | 4.38660 | 8.88e-03 | 0.999941 |
| layer_15 | 2.71747 | 1.34e-02 | 0.999898 |
| final_norm | 0.66541 | 1.04e-02 | 0.999883 |
| logits | 0.12246 | 1.11e-02 | 0.999938 |

Top-5 next tokens identical. Greedy continuation identical for **16 of 16
tokens**.

The residual ~1% error is bf16 storage, not a defect, and three independent
observations confirm that reading. The embedding lookup, which performs no
arithmetic at all, is exactly zero. The error is flat across depth and decreases
through the middle layers rather than compounding, which no wiring bug does. And
the large absolute figures sit on Llama's massive-activation dimensions, where
values reach the thousands and one bf16 step is about 8.

### 5.3 Tokenizer against the reference tokenizer

Covered above: 54,323 tokens, zero mismatches.

---

## 6. Results

RTX 4090, 128 SMs, 1008.1 GB/s theoretical peak. Llama-3.2-1B-Instruct in bf16.
27-token prompt, 256 tokens generated, greedy.

### Headline

| runtime | prefill tok/s | decode tok/s | ms/token | achieved GB/s | % of peak |
|---|---:|---:|---:|---:|---:|
| this runtime, optimized | 7151 | **322.3** | 3.103 | **798** | **79.1%** |
| this runtime, baseline | 7151 | 309.0 | 3.236 | 765 | 75.9% |
| HuggingFace Transformers | 1931 | 72.0 | 13.894 | 178 | 17.7% |

Each figure is the median of three runs agreeing to within 0.2 tokens/s.

**4.5x HuggingFace's throughput, at 79.1% of the hardware ceiling.**

**An honest caveat on that comparison.** This is Transformers in its standard
configuration, which is what somebody gets calling `from_pretrained` and looping
over `model(...)`. Its run logged a deprecation warning about the legacy tuple
KV cache, which reallocates and concatenates every step, and its per-token
Python dispatch is not free either. `StaticCache` with `torch.compile` would
raise it substantially, likely into the 150 to 200 tokens/s range. The honest
claim is 4.5x the default path, not 4.5x the best Transformers can do.

llama.cpp is absent because it needs an f16 GGUF of the same weights; a
quantized build moves fewer bytes and would not be the same measurement.

### Prefill against decode

| phase | per token | limited by |
|---|---:|---|
| prefill | 140 us | arithmetic |
| decode | 3103 us | memory bandwidth |

A 22x gap, and it widens with prompt length, because prefill amortizes each
weight read across the whole prompt while decode cannot amortize at all.

### KV cache, contiguous against paged

At 153 positions:

| layout | resident | waste |
|---|---:|---:|
| contiguous, full 131072-token context | 4295 MB | 99.9% |
| contiguous, capped at 4096 | 134 MB | 96.3% |
| paged, 16-position pages | 5.2 MB | 3.8% |
| strictly needed | 5.0 MB | 0 |

A contiguous cache commits its whole window at creation. Paging commits only
pages the sequence has reached, so waste is the unused tail of the last page.

With one sequence in flight the pool must still be sized for the worst case, so
this is per-sequence accounting rather than a claim about total allocation. The
reason paging wins in a real server is that the pool is shared across sequences,
which is out of scope here. Stating that is more useful than quoting the 96%
figure without it.

---

## 7. The optimization pass

### Where the time actually went

| stage | us/token | weight bytes | achieved | % of peak |
|---|---:|---:|---:|---:|
| lm_head | 559.8 | 525 MB | 938 GB/s | 93% |
| mlp up/gate (2 GEMM) | 1247.5 | 1073 MB | 860 GB/s | 85% |
| mlp down | 640.7 | 537 MB | 838 GB/s | 83% |
| o_proj | 204.0 | 134 MB | 658 GB/s | 65% |
| qkv_proj (3 GEMM) | 352.9 | 201 MB | 570 GB/s | 57% |
| everything else | 480.1 | activations only | | |

Two things are immediately visible. GEMMs are 86% of the token and the big ones
already run near peak, so there was never much to win there. And efficiency
falls with matrix size: the 525 MB output head reaches 93% while the 201 MB QKV
projection reaches 57%.

### A prediction that was wrong

The design document, written before any hardware existed, predicted that
per-launch overhead between kernels would dominate, on the reasoning that a
decode step issues roughly 250 launches at a few microseconds each.

The data refuted it. Only 5.8% of a token is unaccounted for between stages.

But the prediction was half right in a way the stage table hides. The small
kernels are not bandwidth-bound at all. RMSNorm moves 12 kB per call and takes
3.65 us, which is 3.3 GB/s against a 1008 GB/s bus. Residual add is 3.2 us,
RoPE and SwiGLU 5.0 us each. At those sizes the kernel is finished before the
memory system is doing anything interesting, so the time is launch and teardown
latency. Launch overhead was real. It appears *inside* kernel timings rather
than as gaps between them, which is why the naive measurement missed it.

Recording this rather than quietly deleting the prediction is the point. The
measurement changed the conclusion.

### Change 1: fuse QKV, and fuse gate with up

Q, K and V are separate matrices in the checkpoint but all multiply the same
input vector. Run separately, that vector is read three times and each GEMM is
too small for cuBLAS to work with. Stacked into one matrix, it is read once. The
same argument applies to the MLP's gate and up.

Concatenation is free at load time: weights are row-major
`[out_features, in_features]`, so stacking along the output dimension is a
contiguous append with no reordering.

| stage | before | after | achieved |
|---|---:|---:|---|
| qkv_proj | 352.9 us | 286.0 us | 570 → 705 GB/s |
| mlp gate/up | 1247.5 us | 1204.7 us | 860 → 890 GB/s |
| swiglu | 54.1 us | 80.7 us | regression |

QKV improved 19%. The MLP barely moved, already having little headroom.

**SwiGLU got 26 us worse.** Reading gate and up as two strided slices of one row
is less coalesced than reading two separate contiguous buffers. That is a real
regression partly offsetting the win. The net across the three stages is still
84 us in favour, so the change stays, but it is exactly the kind of cost easy to
leave unmeasured and unmentioned.

### Change 2: fuse each residual join into the following norm

Every residual join in a decode step is immediately followed by a norm. Since
these kernels are latency-bound rather than bandwidth-bound, the win is the
launch removed, not the traffic saved.

The MLP join at the end of a layer pairs with the *next* layer's input norm, or
the final norm after the last layer, so the fusion crosses the layer boundary.
That takes launches per token from 179 to 147.

| | before | after |
|---|---:|---:|
| rmsnorm + residual add, separate | 218.4 us | |
| residual+rmsnorm, fused | | 119.1 us |

99 us saved, 45%, from one fewer kernel start per join.

It also improved accuracy slightly. The sum stays in fp32 registers through the
reduction instead of being rounded to bf16, written, and read back. Per-layer
cosine against PyTorch went from 0.999870 to 0.999941 at layer 7.

### Cumulative

| | decode tok/s | achieved GB/s | % of peak |
|---|---:|---:|---:|
| baseline | 309.0 | 765 | 75.8% |
| + fused QKV and gate/up | 317.0 | 785 | 77.8% |
| + fused residual and norm | 322.3 | 798 | 79.1% |

4.3% overall. Correctness was revalidated against PyTorch after every change.

---

## 8. Mistakes made and caught

Recorded because the process of catching them is most of the actual work.

### The benchmark was measuring the wrong thing

Prefill initially reported 159 tokens/s, slower than decode, which is
impossible. Timing across three prompt lengths showed why:

| prompt | prefill time |
|---:|---:|
| 29 tokens | 169.45 ms |
| 202 tokens | 172.00 ms |
| 802 tokens | 187.19 ms |

The cost is fixed, not per token. Creating the cuBLAS handle, loading kernel
images and letting cuBLAS select algorithms costs about 168 ms on first use, and
all of it was landing in the prefill timer. Marginal cost is about 25 us per
token, so real prefill is near 39,000 tokens/s.

**The published figure would have been wrong by two orders of magnitude.**
`--bench` now discards a warmup pass.

### The validator was testing code that does not run

The first version of the residual/norm fusion kept a separate unfused path
whenever activation capture was enabled, because capture wants the residual
stream by itself.

That meant `lcr-validate` was checking a path that never executes in production
and reporting everything green for it. The test passed either way. Only the
question of *what* it tested distinguished the two situations.

The fused kernel already writes the updated residual stream to `x_` alongside
the normalized copy, so the tap sees the same tensor either way and the special
case was never needed. Removing it means the validator now exercises the
shipping path, which it does, still at 16 of 16 tokens identical.

### Two compile errors the shim caught

`dtype.cuh` used `uint2` and `uint4` without including `cuda_runtime.h`, and the
kernel test file was missing two includes. Both would have been trivial on a
GPU box and both were found for free before renting one.

---

## 9. What is left

**CUDA graphs for the decode step.** The largest remaining win and not done. The
small kernels still cost about 316 us per token, roughly 10%, almost entirely
launch latency. Capturing the whole decode step and replaying it as a single
launch would recover most of that, taking throughput to roughly 355 tokens/s.

It requires moving position and sequence length into device memory so a captured
graph stays valid as the sequence grows, and fixing the attention split count so
grid dimensions do not change between steps.

**The remaining GEMM headroom.** `mlp gate/up` is 36% of the token at 890 GB/s
and `mlp down` another 19% at 838. Between them, 55% of decode running at 83 to
88% of peak. Since the output head reaches 93% on the same hardware doing the
same kind of operation, a few points look available, but not a step change.
These are plain cuBLAS matrix-vector products, so the options are a hand-written
kernel tuned for m=1 or accepting cuBLAS is near the limit.

**llama.cpp in the benchmark table**, which needs an f16 GGUF conversion.

Ceiling check: at 2.472 GB per token, this GPU's 1008 GB/s allows 407 tokens/s.
322 is 79% of it. Removing all small-kernel latency gives roughly 355. The rest
is cuBLAS.

---

## 10. Summary

| deliverable | result |
|---|---|
| Working binary | generates correct text from the real checkpoint |
| Benchmark against baselines | 4.5x HuggingFace default path |
| Bandwidth utilization | 79.1% of the 4090's 1008 GB/s peak |
| Prefill vs decode breakdown | 140 us vs 3103 us per token, 22x |
| KV cache waste, contiguous vs paged | 96.3% → 3.8% |
| Per-kernel profile before and after | two fusions, both documented |

Roughly 3900 lines of C++ and CUDA. Twenty-one commits. Total GPU cost to build,
validate, benchmark and optimize: about 46 cents.

The result worth stating is not 322 tokens per second, which is a property of
the card. It is 79% of theoretical peak memory bandwidth, verified correct layer
by layer against PyTorch, with the measurement methodology stated precisely
enough that both the number and its limits can be checked.
