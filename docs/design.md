# Design notes

## The two phases are not the same problem

Generating text from a transformer runs the same layers twice under completely
different conditions.

**Prefill** processes the whole prompt at once. Every projection is a matrix
multiply with as many rows as there are prompt tokens, so each weight, once
loaded, is used for hundreds of multiply-accumulates. The GPU is doing
arithmetic.

**Decode** produces one token. Every projection is a matrix times a vector. Each
weight is read out of HBM, used for exactly one multiply-add, and thrown away.
The arithmetic units are almost entirely idle, waiting on memory.

The numbers for Llama-3.2-1B, which has 1.236 billion parameters and occupies
2.472 GB in bf16:

| | prefill, T tokens | decode, 1 token |
|---|---|---|
| arithmetic | 2 x 1.236e9 x T FLOP | 2.47 GFLOP |
| weight traffic | 2.472 GB | 2.472 GB |
| arithmetic intensity | T FLOP/byte | 1 FLOP/byte |

Machine balance, the intensity at which a GPU stops being memory bound, is
around 150 FLOP/byte on an A100 and around 300 on an H100. Prefill with a few
hundred tokens sits at or above that line. Decode sits two orders of magnitude
below it, and no amount of kernel work moves it: the weights have to be read.

So decode has a hard ceiling that follows straight from the memory bandwidth:

| GPU | bandwidth | ceiling on this model |
|---|---:|---:|
| L4 | 300 GB/s | 121 tokens/s |
| RTX 4090 | 1008 GB/s | 408 tokens/s |
| A100 80GB | 2039 GB/s | 825 tokens/s |
| H100 SXM | 3350 GB/s | 1355 tokens/s |

`lcr-inspect` prints the byte count these come from. The measurement this
project reports is what fraction of that ceiling the implementation reaches.

## What actually crosses the bus per token

Every weight in the forward pass, once: both norms and all seven projections in
each of the 16 layers, the final norm, and the output head. With tied
embeddings the output head is the embedding matrix, so per-token weight traffic
equals the entire checkpoint, 2.472 GB. The embedding table is read twice in a
sense and streamed once: the lookup touches a single 4 KB row, the output
projection touches all 525 MB of it.

On top of that, the KV cache: 32 KiB per cached position, across all layers,
both keys and values. That is small next to the weights early on and stops
being small later.

| sequence length | cache read per token | as a share of the total |
|---:|---:|---:|
| 512 | 16 MB | 0.7% |
| 4096 | 128 MB | 4.9% |
| 32768 | 1.0 GB | 29% |

Which is why the decode attention kernel matters more the longer the
conversation gets, and why the grouped-query saving below is worth having.

## Layout decisions

**Weights: one allocation, in read order.** Every tensor lands in a single
`cudaMalloc` in the order the forward pass touches it. A decode step then walks
the weights forward through memory instead of jumping between 146 separate
allocations, which is what the prefetchers and the memory controller want.

**Activations: a bump-allocated arena.** Decode shapes never change, so the
buffers are carved out once at load. `cudaMalloc` synchronizes the device;
calling it inside a loop that should take a millisecond would be the dominant
cost.

**KV cache: head-major, `K[kv_head][position][head_dim]`.** A decode step reads
one head's entire history, so that history should be contiguous. The obvious
alternative, `[position][kv_head][head_dim]`, makes the per-token write
contiguous instead, which is the wrong thing to optimize: at position 4000 the
step reads 4000 vectors and writes one.

**Queries: head-major too.** The rotary kernel is already touching every element
of the query projection, so it transposes into `[head][token][head_dim]` on the
way through, for free. That layout is what makes prefill attention a batched
GEMM with uniform strides, because the four query heads sharing a key/value head
become one contiguous block of rows. During decode the same layout is already
what the attention kernel wants, so nothing transposes at all.

## The decode attention kernel

Two structural decisions, both about reading the cache as few times as possible
and keeping the machine busy while doing it.

**One block per grouped-query group, not per query head.** This model has 32
query heads and 8 key/value heads, so four queries share each key and value. A
block covers a whole group and loads each key and value vector once, using it
for all four dot products. Per query head instead, the same cache lines would be
read four times. At 32k context that is the difference between 1 GB and 4 GB of
cache traffic per token.

**Splitting the position range across blocks.** Eight groups means eight blocks,
on a GPU with 108 or 132 multiprocessors. Most of the machine would sit idle.
Each block instead takes a slice of the positions and keeps a partial softmax,
its running maximum and exponential sum, which a second pass merges. The split
count targets two blocks per multiprocessor and stops splitting below a few
hundred positions per block, past which the merge costs more than the
parallelism is worth.

The softmax is online throughout: a running maximum and sum updated per
position, rescaled when a larger score appears. No score row is ever
materialized, which is what lets a single block handle an arbitrarily long
history in fixed registers. The identity element is `-FLT_MAX` rather than
`-inf`, because two empty accumulators combining as `(-inf) - (-inf)` is NaN,
and a block with more warps than positions produces exactly that.

## Contiguous against paged KV cache

A contiguous cache reserves the whole context window when it is created. For
this model at its declared 131072-token context that is 4.0 GB, which is larger
than the model, and for a 500-token conversation more than 99% of it is never
touched.

A paged cache cuts the window into fixed-size pages and maps them as the
sequence grows, so the only waste is the unused tail of the last page. Both
layouts are addressed by the same expression, with the page-table lookup
compiled away for the contiguous case, so the comparison is between two layouts
and not between two kernels.

For a 500-token sequence with 16-position pages:

| layout | resident |
|---|---:|
| contiguous, full context | 4.00 GiB |
| contiguous, capped at 4096 | 128 MiB |
| paged, 16-position pages | 15.6 MiB |
| strictly needed | 15.4 MiB |

With one sequence in flight the pool still has to be sized for the worst case,
so this is a per-sequence accounting rather than a claim about total allocation.
That is the honest version of the comparison; the reason paging wins in a real
server is that the pool is shared across sequences, which is out of scope here.

## What the profiling pass will ask

The optimization phase has not run yet. These are the questions it exists to
answer, in the order they are worth asking.

**Where does the time actually go?** Nsight Systems on a decode step, to get the
split between the seven GEMMs, the attention kernel, the elementwise kernels,
and the gaps between them.

**How much of the step is launch overhead?** A decode step issues roughly 250
kernel launches: about 15 per layer across 16 layers, plus the head and
sampling. At a couple of microseconds each that is a large fraction of a token
that should take barely more than a millisecond. If the gaps between kernels
turn out to dominate, CUDA graphs are the fix, and that is a bigger win than any
individual kernel.

**What does the attention kernel achieve against its own roofline?** It reads
the cache and nothing else, so its ceiling is exactly bandwidth. Nsight Compute
gives achieved DRAM throughput; the baseline for the comparison is the same
kernel with the split count forced to one, which is the version that leaves the
GPU mostly idle.

**Are the cuBLAS calls hitting the paths they should?** A matrix-vector product
is not what a GEMM library is tuned for, and it is worth confirming which kernel
gets selected for the m=1 case before concluding the ceiling is the hardware.

**What is left after that?** Fusing the residual add into the following norm,
which would remove a full read and write of the residual stream twice per layer.
Whether the L2 residency controls are worth using for the KV cache. Whether
padding pages to a cache-line boundary changes anything for the paged layout.

None of these are worth doing before the numbers in `lcr-validate` are clean.
An optimized kernel that computes the wrong thing is not progress.
