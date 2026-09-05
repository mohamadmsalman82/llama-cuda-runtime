# Benchmarks

RTX 4090 (sm_89, 128 SMs, 384-bit bus, 1008.1 GB/s theoretical peak), CUDA 12.4,
Llama-3.2-1B-Instruct in bf16. 27-token prompt, 256 tokens generated, greedy.

Measured 2026-09-05. Reproduce with:

```
python3 tools/benchmark.py --model models/Llama-3.2-1B-Instruct --tokens 256
```

## Decode, which is the point of the project

| runtime | prefill tok/s | decode tok/s | ms/token | achieved GB/s | % of peak |
|---|---:|---:|---:|---:|---:|
| this runtime, optimized | 7151 | 322.3 | 3.103 | 798 | 79.1% |
| this runtime, baseline | 7151 | 309.0 | 3.236 | 765 | 75.9% |
| HuggingFace Transformers | 1931 | 72.0 | 13.894 | 178 | 17.7% |

The two rows for this runtime are before and after the optimization pass in
[optimization.md](optimization.md), which fused the QKV and gate/up projections
and folded each residual join into the norm that follows it.

Decoding one token requires reading all 2.472 GB of weights out of HBM, so
tokens/sec and achieved bandwidth are the same measurement in different units.
The ceiling on this GPU is 1008 GB/s divided by 2.475 GB, which is 407 tokens/s.

**On the HuggingFace comparison.** That is Transformers in its standard
configuration, which is what somebody gets by calling `from_pretrained` and
`model(...)` in a loop. Its run logs a deprecation warning about the legacy
tuple KV cache, which reallocates and concatenates on every step, and its
per-token Python dispatch is not free either. `StaticCache` with
`torch.compile` would raise it substantially, likely into the 150 to 200
tokens/s range. The honest claim is that this runtime is 4.3x the default path,
not 4.3x the fastest thing Transformers can do.

llama.cpp is not in the table yet. It needs an f16 GGUF of the same weights, on
the same card, since a quantized build moves fewer bytes and would not be the
same measurement.

## Prefill against decode

| phase | tokens | time | per token | limited by |
|---|---:|---:|---:|---|
| prefill | 27 | 3.78 ms | 140 us | arithmetic |
| decode | 126 | 407.8 ms | 3236 us | memory bandwidth |

Prefill is 23 times cheaper per token, and the gap widens with prompt length
because prefill amortizes each weight read across the whole prompt while decode
cannot amortize at all.

Marginal prefill cost, measured by holding everything else fixed and varying
only the prompt:

| prompt | prefill time |
|---:|---:|
| 29 tokens | 169.45 ms |
| 202 tokens | 172.00 ms |
| 802 tokens | 187.19 ms |

Those were taken before warmup was added, so each carries a fixed ~168 ms of
CUDA context creation and cuBLAS algorithm selection. The slope is what matters:
600 additional tokens cost 15.2 ms, about 25 us per token, so prefill sustains
roughly 39,000 tokens/s once the fixed cost is removed. That fixed cost was
being reported as prefill time and made prefill look slower than decode, which
is what prompted the warmup pass now used by `--bench`.

## KV cache, contiguous against paged

At 153 positions:

| layout | resident | waste |
|---|---:|---:|
| contiguous, full 131072-token context | 4295 MB | 99.9% |
| contiguous, capped at 4096 | 134 MB | 96.3% |
| paged, 16-position pages | 5.2 MB | 3.8% |
| strictly needed | 5.0 MB | 0 |

A contiguous cache commits its whole window when it is created. Paging commits
only the pages a sequence has reached, so the waste is the unused tail of the
last page. With one sequence in flight the pool still has to be sized for the
worst case, so this is per-sequence accounting; the reason paging wins in a
real server is that the pool is shared across sequences, which is out of scope
here.

## Correctness

Validated against the PyTorch reference before any of the above was measured.
Reference dumped in float32, deliberately, so that bf16 rounding could be told
apart from an implementation bug.

| tensor | max abs | rel rms | min cosine |
|---|---:|---:|---:|
| embed | 0.00000 | 0.00e+00 | 1.000000 |
| layer_0 | 0.06867 | 3.16e-03 | 0.999978 |
| layer_7 | 4.38660 | 8.89e-03 | 0.999870 |
| layer_15 | 3.71747 | 1.61e-02 | 0.999770 |
| final_norm | 0.60291 | 1.24e-02 | 0.999744 |
| logits | 0.13216 | 1.38e-02 | 0.999906 |

Top-5 next tokens identical. Greedy continuation identical for 16 of 16 tokens.

The residual error is bf16 storage, not a defect: bf16 carries 8 mantissa bits,
so its relative precision is 3.9e-3, and three things confirm the reading. The
embedding lookup, which performs no arithmetic, is exactly zero. The error is
flat across depth and falls through the middle layers rather than compounding,
which a wiring bug would not do. And the large absolute figures sit on Llama's
massive-activation dimensions, where values reach the thousands and a single
bf16 step is around 8.
