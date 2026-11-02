# Global-attention disaggregation with resident sliding KV

## Result and scope

This study adds a necessary steady-decode bound that couples graphics processing unit (GPU) execution latency to language processing unit (LP30) static random-access memory (SRAM) capacity.
The target remains Gemma 4 31B text with independent requests at context 101,000, a 102,000-token cache reservation, eight-bit floating-point (FP8) weights and separate FP8 key/value (KV) caches, and bfloat16 (BF16) attention.
Useful work is 129,261,109,248 matrix floating-point operations (FLOPs) per committed token.
Model FLOPs utilization (MFU) divides that useful rate by the combined dense FP8 peak of every allocated GPU and LP30.
These results are upper bounds under the declared analytical operator protocol and do not report attained throughput.
Prefill and complete-request MFU are outside this metric.

The [SubQuadratic Disaggregation paper](https://arxiv.org/abs/2609.13134), rechecked on November 3, 2026, motivates placing dense global attention on dynamic random-access memory devices and sliding-window attention plus feed-forward networks (FFNs) on SRAM devices.
The paper's GPU proxy and Rubin/LPX analytical results do not calibrate TokenSim's operator costs.
This study tests two narrower ownership hypotheses with the existing native operator provider.

| Ownership name | GPU work | LP30 work |
| --- | --- | --- |
| `global_attention` | All ten complete global-attention operator compositions, including normalization, query/key/value (QKV) projection, rotary position encoding, attention core and output projection | All sliding-window attention, all FFNs, residual paths and embedding/output head |
| `global_core` | The attention core in each of the ten global layers | All projections, normalization, sliding attention, FFNs, residual paths and embedding/output head |

The code uses native analytical GPU components with 0.2 microsecond graph kernel overhead and the catalog memory-efficiency factors.
GPU service occupies an exclusive worker for each native composition.
Collectives, activation transfers, LP30 buffer reservations, GPU memory admission, physical placement and compiler scheduling are relaxed.
Requests retain their complete sliding-window cache on LP30 throughout the decode cycle.
GPU context parallelism, alternate fused kernels, offload, cache sharing and speculative decoding remain outside the bound.
The native GEMM estimator charges matrix weights to high-bandwidth memory on every invocation.
Persistent GPU weight caching is an additional implementation hypothesis outside these native-cost comparisons.

## Why GPU bandwidth alone misses a capacity cost

Each request retains 50 sliding-window caches with 1,024 tokens, 16 KV heads and head dimension 256.
The native separate-FP8 ledger requires at least:

```text
S = 50 * 1024 * 16 * 256 * 2 = 419,430,400 bytes/request
```

The helper calls the native `kv_head_partitions` function with one unreplicated rank.
This lower bound omits FP8 scale metadata, cache-block padding, the allocator watermark, shard replication and transient append storage.
The native separate-cache partition currently charges payload bytes without per-head scale bytes.
The experimental shared-global-cache representation has its own explicit scale-byte accounting.
Adding separate-cache metadata would tighten the capacity bound.

For steady token rate `T`, resident requests `N`, total GPU path latency per request `s`, GPU head tensor parallelism `p`, independent batch width `b`, and LP30 capacity `C`, necessary conditions are:

```text
N >= T * s
GPU_count >= T * p * s / b
LP30_count * C >= N * S
```

The request's global attention layers execute sequentially, so their native rank-local service times add to `s` even when different layers use different GPU pipeline stages.
Little's law supplies the resident-request condition.
All sliding KV remains live during that GPU execution time.
More pipeline stages distribute the bytes across chips and preserve their aggregate amount.

Let `P_g` and `P_l` be per-device FP8 peaks and `D_l` be the independent LP30 arithmetic demand expressed as peak-seconds per useful token.
Combining GPU service, resident KV and LP30 arithmetic gives:

```text
GPU_peak_seconds_per_token = P_g * p * s / b
LP30_memory_peak_seconds_per_token = P_l * S * s / C
LP30_required_peak_seconds_per_token = max(LP30_memory_peak_seconds_per_token, D_l)

MFU <= useful_FLOPs_per_token
       / (GPU_peak_seconds_per_token + LP30_required_peak_seconds_per_token)
```

Storage and arithmetic constrain the same LP30 inventory, so the relaxation combines them with `max`.
It does not add both inventory requirements.
The LP30 arithmetic term retains general matrix multiplication (GEMM) efficiency 0.30, BF16 attention efficiency 0.50 and the catalog BF16/FP8 peak ratio.
The SRAM capacity term grants the complete 500,000,000 bytes of every LP30 to sliding KV.
Weights and every runtime reservation are free in this inventory-independent relaxation.

## Search coverage and unbounded-batch tail

The driver evaluates 2,304 explicit ownership/device/TP/batch cases.
It covers H100 SXM, B200 SXM and Rubin, all head tensor-parallel (TP) divisors of 32 query heads (`TP=1,2,4,8,16,32`), both ownership hypotheses and every integer batch width from 1 through 64.
Global grouped-query attention has four KV heads, so head TP above four replicates KV traffic.
The native provider retains that replication.

Each decode attention core's mandatory memory traffic is linear in independent batch width.
If its aggregate memory time per request is `a`, then `s(b) >= b*a`.
For every `b >= 65`:

```text
P_g * p * s(b) / b >= P_g * p * a
P_l * S * s(b) / C >= P_l * S * 65 * a / C
```

The driver records 36 tail bounds covering all larger integer batches.
Their largest MFU bound is 1.48758%, so no omitted larger batch improves the explicit winners under these hypotheses.
This tail argument uses the native attention work accounting, catalog bandwidth and catalog memory efficiency while removing GPU projections and launch costs.

| GPU | Complete global-attention bound | Winning TP / batch | Global-core-only bound | Winning TP / batch |
| --- | ---: | --- | ---: | --- |
| H100 SXM | 2.69076% | 4 / 2 | 3.72568% | 4 / 1 |
| B200 SXM | 3.08312% | 4 / 3 | 3.95295% | 4 / 1 |
| Rubin | 2.39722% | 4 / 7 | 2.75326% | 4 / 3 |

These bounds allow arbitrary GPU/LP30 inventory and remove the fixed matrix-weight footprint.
They constrain each listed native ownership protocol independently of total pipeline depth.
They do not exclude a lower-latency GPU implementation, context parallelism or sliding-KV offload.

## Finite inventory and matrix-weight capacity

The finite-inventory relaxation restores a lower bound `W` on LP30 FP8 matrix weights.
It includes tied embedding/output-head storage once and omits normalization weights, shard replication, scales and all runtime buffers.
`W` is 28,824,305,664 bytes for complete global-attention ownership and 30,696,013,824 bytes for core-only GPU ownership.

For each integer LP30 count `L` and GPU count `G` divisible by the selected head TP:

```text
resident_requests <= (L*C - W) / S
T <= G*b / (p*s)
T <= (L*C - W) / (S*s)
T <= L*P_l / D_l
```

Resident requests remain fractional in this relaxation.
The optimizer considers allocations using at most the requested inventory, including allocations that leave some permitted cards unused.
At fixed `L`, MFU increases with GPU groups until another throughput limit becomes active and then decreases.
Checking the two adjacent group counts and the inventory endpoint therefore covers the integer GPU allocation optimum for each `L`.
An independent exhaustive small-inventory test verifies this optimization.

| GPU / ownership | At most 256 cards | At most 512 cards | At most 1,024 cards | At most 2,048 cards |
| --- | ---: | ---: | ---: | ---: |
| H100 / complete global attention | 2.22492% | 2.47074% | 2.58252% | 2.63707% |
| B200 / complete global attention | 2.76826% | 2.93819% | 3.01392% | 3.04922% |
| Rubin / complete global attention | 2.29053% | 2.35010% | 2.37444% | 2.38630% |
| H100 / global core | 3.07591% | 3.41287% | 3.57210% | 3.64975% |
| B200 / global core | 3.63581% | 3.83178% | 3.91771% | 3.95295% |
| Rubin / global core | 2.70189% | 2.73949% | 2.74145% | 2.75151% |

Every entry counts GPUs and LP30s together in its inventory limit and includes their different FP8 peaks in the MFU denominator.
The 256-card B200 core-only relaxation selects 160 B200 GPUs and 95 LP30s, with GPU TP4 and batch one.
Its 235,398.85 TPS bound and 3.63581% MFU bound are optimistic ceilings with unverified placement and zero activation-transfer cost.
They do not describe a runnable serving configuration.

## Implications for the ongoing search

The complete-global-attention protocols fall below the 3.41960% experimental shared-cache decode incumbent at every inventory size under these native costs.
H100 and Rubin complete-global-attention bounds also fall below the 3.03914% existing-format incumbent at arbitrary inventory.
The B200 complete-global-attention relaxation falls below the existing-format incumbent through 1,024 total cards and remains narrowly above it at larger inventories.
Core-only B200 ownership remains a useful candidate for explicit capacity placement, communication and concurrent replay.
Core-only H100 remains open against the existing-format incumbent and becomes open against the experimental shared-cache point beyond the 512-card limit.
The shared-cache incumbent requires separate model-quality validation.
No global hardware optimum follows from comparisons with these protocol-specific upper bounds.

The [CUDA L2 cache-control guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html), checked on November 3, 2026, documents a configurable cache portion for persisting accesses and shared capacity constraints across concurrent kernels.
One global layer's FP8 QKV and output-projection matrices contain 46,792,704 bytes per GPU at head TP4.
Whether those weights can persist while global KV streams through the device requires measured device limits, a layer-specific placement and a priced cache-access path.
The CUDA mechanism supplies a reason to test weight reuse and does not prove that this working set fits or remains resident on the target GPUs.

## Reproduction and evidence

```bash
python scripts/bound_lpx_sqd_residency.py \
  --config experiments/lpx/fidelity-study.json \
  --maximum-batch 64 \
  --card-limits 256 512 1024 2048 \
  --output outputs/lpx-sqd-residency-bounds.json.gz
```

The compressed [portable result](../experiments/lpx/sqd-residency-bounds.json.gz) retains all explicit cases, unbounded-batch tail bounds, finite-inventory optima, native component witnesses, complete device catalogs, model geometry and source/driver identities.
Gzip output has a deterministic timestamp.
The eleven focused tests cover Little's-law algebra, joint arithmetic/storage requirements, exhaustive integer allocation, finite-weight convergence, the batch tail, native ownership, KV replication, empty inventory frontiers and invalid inputs.
The full suite runs 862 tests with 860 passes, one failure and one error from the unchanged macOS `/tmp` normalization and unavailable NVIDIA process facilities.
All 574 Python files pass the pinned Black check.
Two independent sweeps produce identical case rows, tail bounds, inventory winners and native component witnesses.
The validated code also reproduces the frozen 448-token paired-arena decode replay exactly, retaining 3.41006684% MFU and 32 measured completions for each of its 14 clients.
The broader validation record remains in the [maximum-MFU progress file](../experiments/lpx/maximum-mfu-search-progress.json).
