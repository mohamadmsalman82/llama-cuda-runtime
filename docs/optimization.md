# Profiling and optimization

RTX 4090, Llama-3.2-1B-Instruct in bf16, batch 1, contiguous KV cache.
Reproduce any line here with `--profile` or `--bench`.

## How the profile was taken

Nsight Compute is the usual tool and it does not work here. GPU performance
counters are gated by a kernel module parameter on the host, so inside a
container `ncu` fails with `ERR_NVGPUCTRPERM` and there is no fix from the
guest side. Rather than depend on privileges the project may not have, the
profiler is built in: `--profile` wraps each stage in a pair of CUDA events and
accumulates by name.

That perturbs what it measures, since recording events costs time and forces
ordering, so the report prints the profiled wall clock, the sum of the stage
timings, and the gap between them instead of presenting stage times as free.
Comparing a profiled run against an unprofiled one puts the instrumentation
cost at roughly 450 us per token, which is why the optimization decisions below
were made on relative stage cost rather than on absolute numbers.

## Result

| | decode tok/s | ms/token | achieved GB/s | % of peak |
|---|---:|---:|---:|---:|
| baseline | 309.0 | 3.236 | 765 | 75.8% |
| fused QKV and gate/up | 317.0 | 3.154 | 785 | 77.8% |
| plus fused residual and norm | **322.3** | **3.103** | **798** | **79.1%** |

4.3% faster overall. Each figure is the median of three runs that agreed to
within 0.2 tok/s, and the layer-by-layer comparison against PyTorch was rerun
after every change: 16 of 16 greedy tokens identical throughout.

## Where the time goes

Before any optimization:

| stage | us/token | weight bytes | achieved | % of peak |
|---|---:|---:|---:|---:|
| lm_head | 559.8 | 525 MB | 938 GB/s | 93% |
| mlp up/gate (2 GEMM) | 1247.5 | 1073 MB | 860 GB/s | 85% |
| mlp down | 640.7 | 537 MB | 838 GB/s | 83% |
| o_proj | 204.0 | 134 MB | 658 GB/s | 65% |
| qkv_proj (3 GEMM) | 352.9 | 201 MB | 570 GB/s | 57% |
| everything else | 480.1 | activations only | | |

Two things are visible immediately. The GEMMs are 86% of the token, and the big
ones already run near peak, so there was never much to win there. And the
efficiency falls with matrix size: the 525 MB output head reaches 93% of peak
while the 201 MB QKV projection reaches 57%.

This refutes the guess recorded in `design.md`, which predicted that per-launch
overhead between kernels would dominate. It does not: only 5.8% of the token is
unaccounted for between stages.

But the prediction was half right, in a way the stage table hides. The small
kernels are not bandwidth-bound at all. RMSNorm moves 12 KB per call and takes
3.65 us, which works out to 3.3 GB/s against a 1008 GB/s bus. Residual add is
3.2 us, RoPE and SwiGLU 5.0 us each. At those sizes the kernel is over before
the memory system is doing anything interesting, so the time is launch and
teardown latency. Launch overhead was real, it just appears inside the kernel
timings rather than as gaps between them, which is why the naive measurement
missed it.

## Change 1: fuse QKV, and fuse gate with up

Q, K and V are three separate matrices in the checkpoint, but they all multiply
the same input vector. Run separately, that vector is read three times and each
GEMM is too small for cuBLAS to work with. Stacking them into one
`[q_dim + 2 * kv_dim, hidden]` matrix reads it once. The same argument applies
to the MLP's gate and up.

Concatenation is free at load time: the weights are row-major
`[out_features, in_features]`, so stacking along the output dimension is a
contiguous append with no reordering. The projection then writes Q, K and V
side by side in one row, which is why the RoPE kernels and SwiGLU take a source
row stride.

| stage | before | after | achieved |
|---|---:|---:|---|
| qkv_proj | 352.9 us | 286.0 us | 570 → 705 GB/s |
| mlp gate/up | 1247.5 us | 1204.7 us | 860 → 890 GB/s |
| swiglu | 54.1 us | 80.7 us | worse, see below |

QKV improved 19%. The MLP barely moved, because at 1073 MB it was already at
85% and had little headroom.

SwiGLU got 26 us *worse*. Reading gate and up as two strided slices of one row
is less coalesced than reading two separate contiguous buffers. That is a real
regression and it partly offsets the win; the net across the three stages is
still 84 us in favour of fusing, so it stays, but it is the kind of cost that
would be easy to leave unmeasured.

## Change 2: fuse each residual join into the following norm

Every residual join in a decode step is immediately followed by a norm. Since
these kernels are latency-bound rather than bandwidth-bound, the win is the
launch removed, not the traffic saved.

The MLP join at the end of a layer pairs with the *next* layer's input norm, or
with the final norm after the last layer, so the fusion crosses the layer
boundary. That removes 32 launches per token, taking the total from 179 to 147.

| stage | before | after |
|---|---:|---:|
| rmsnorm + residual add, separately | 218.4 us | |
| residual+rmsnorm, fused | | 119.1 us |

99 us saved, 45%, from what is fundamentally one fewer kernel start per join.

It also improved accuracy slightly. The sum stays in fp32 registers through the
reduction instead of being rounded to bf16, written, and read back; per-layer
cosine against the PyTorch reference went from 0.999870 to 0.999941 at layer 7.

## A correctness gap this nearly hid

The first version of the fusion kept a separate unfused path whenever activation
capture was on, because capture wants the residual stream by itself. That meant
`lcr-validate` was checking code that does not run in production, and it
reported everything green for a path nobody executes.

The fused kernel already writes the updated residual stream to `x_` alongside
the normalized copy, so the tap sees the same tensor either way and the special
case was never needed. Removing it means the validator now exercises the
shipping path. Worth recording because the test was passing either way: only
the question of *what* it was testing distinguished the two.

## What is left

`mlp gate/up` is 36% of the token at 890 GB/s, and `mlp down` another 19% at
838. Between them that is 55% of decode running at 83 to 88% of peak. Any
further large gain has to come from there, and since they are plain cuBLAS
matrix-vector products the options are a hand-written kernel tuned for m=1, or
accepting that cuBLAS is close to the hardware limit. Given that the output
head reaches 93% on the same hardware and the same kind of operation, a few
more points look available but not a step change.

The remaining small kernels cost 316 us per token, about 10%, almost all of it
launch latency. CUDA graphs would capture the whole decode step and replay it
as one launch, which is the natural fix and would recover most of that. It needs
the position and sequence length moved into device memory so the captured graph
stays valid as the sequence grows, and the attention split count fixed so grid
dimensions do not change between steps. That is the largest remaining win and
it is not done.

Ceiling check: at 2.472 GB per token, this GPU's 1008 GB/s allows 407 tok/s.
322 is 79% of that. Removing all the small-kernel latency would give roughly
355. The rest is cuBLAS.
