# LPX context-parallel decode specification

Specification date: 2026-10-21.
This document is Phase 0 of the [context-parallel implementation plan](lpx-context-parallel-plan.md).
It defines how TokenSim represents context parallelism (CP) for LPX decode and records expected results before the timing implementation exists.

## Evidence status

[NVIDIA's long-context article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) states that LPX employs "parallelism for both model weights and long-context KV cache".
The same article attributes small context-length variation to high tensor parallelism (TP).
It does not disclose a partition scheme, collective, rank count or KV placement for the Gemma endpoint.
[Helix parallelism](https://arxiv.org/abs/2507.07120), published by NVIDIA researchers, shards the key/value (KV) cache by sequence position inside the attention TP group and reuses the same ranks for the feed-forward network (FFN).
This specification follows Helix as a literature hypothesis.
The resulting simulations remain conditional on that hypothesis and on the uncalibrated LP30 parameters in the [fidelity study](lpx-fidelity-model.md).

## Notation

| Symbol | Meaning |
| --- | --- |
| `N` | Ranks in one pipeline stage of the attention domain, `tensor_parallel_size` |
| `KVP` | KV-parallel ranks that split one layer's sequence positions, `kv_parallel_size` |
| `TPA` | Attention TP ranks that split KV heads on a CP layer, `N / KVP` |
| `B` | Query tokens in one forward, one for serial decode |
| `H_q`, `H_kv` | Query heads and KV heads of the layer |
| `d` | Head dimension of the layer |
| `S` | KV context length of the forward, including the new token |
| `w` | Attention window of the layer, or none for full attention |
| `b` | Append block, `kv_append_block` |

## Configuration

`DomainParallelConfig` accepts three fields.

| Field | Default | Meaning |
| --- | --- | --- |
| `kv_parallel_size` | 1 | KVP for selected layers; it must divide `tensor_parallel_size` |
| `kv_parallel_layers` | `full_attention` | `full_attention` selects layers without a window; `all` selects every layer |
| `kv_append_block` | 16 | Consecutive token positions assigned to one KVP rank before ownership advances |

Only `af_attention` domains may declare `kv_parallel_size > 1`.
Unselected layers use `KVP = 1` and `TPA = N`, which is the existing TP behavior.
The defaults reproduce the existing configuration exactly.

## Rank coordinates

A stage-local TP rank `t` has attention coordinates `tpa_rank = t // KVP` and `kvp_rank = t % KVP`.
Each KVP group therefore occupies `KVP` consecutive TP ranks.
The native LPX configuration maps consecutive TP ranks to consecutive physical chips, so groups with `KVP <= 16` stay inside one 16-chip tray.
KV heads are assigned to `tpa_rank` with the existing head-partition rule, including replication when `TPA > H_kv`.

## Token ownership

Token position `p` on a selected layer belongs to KVP rank `floor(p / b) mod KVP`.
The rule applies to prompt tokens written by prefill and to generated tokens written by decode.
Let `m = b * KVP`.
The number of positions in `[0, x)` owned by rank `r` is:

```text
owned(x, r) = floor(x / m) * b + min(b, max(0, (x mod m) - r * b))
```

Positions in `[s, e)` give `owned(e, r) - owned(s, r)`.
A full-attention forward at context `S` reads `owned(S, r)` keys on rank `r`.
A windowed selected layer reads the owned positions in `[max(0, S - w), S)`.
Peak residency over a request with maximum context `M` is `owned(M, r)` for full attention.
For a windowed layer it is the maximum owned count over windows ending at positions up to `M`.
This maximum repeats with period `m` once the window is full, so a bounded scan computes it exactly.
A block size of `ceil(M / KVP)` assigns contiguous ranges and serves as the contiguous-ownership control.

## Decode execution on a CP layer

Each rank computes the pre-attention normalization, Q/K/V projection, Q/K normalization and rotary embedding for `H_q / TPA` query heads and its local KV heads.
Ranks in one KVP group compute identical projections, so the owning rank appends the new K/V entries without a transfer.
Each rank runs attention over its owned keys and produces a partial output and one log-sum-exp (LSE) value per token and query head.
The decode step finishes when the slowest rank in the group finishes, so the critical path uses `max_r owned(S, r)`.

An all-to-all inside each KVP group exchanges query-head slices.
Each rank keeps `H_q / N` heads and sends the other slices to their owners.
The bytes sent by one rank are:

```text
combine_send_bytes = (KVP - 1) * B * (H_q / N) * (d * partial_bytes + lse_bytes)
```

Partial outputs and LSE values use 32-bit floating point (FP32), so `partial_bytes = lse_bytes = 4`.
The volume does not depend on `S`.
The receiving rank merges `KVP` fragments per token and head with `L = max_i LSE_i`, `a_i = exp(LSE_i - L)` and `O = sum_i a_i * O_i / sum_i a_i`.
The LP30 vector unit charges `KVP` exponentials and approximately `KVP * (2 * d + 3)` scalar operations per token and retained head.
The output projection uses an input width of `(H_q / N) * d` on every rank, followed by the existing all-reduce over `N` ranks.
The FFN, residual and model-boundary operations retain the TP `N` behavior.
`H_q` must be divisible by `N` on a CP layer so every rank retains the same number of heads after the all-to-all.

## Memory accounting

KV bytes on rank `r` for a selected layer equal its local KV heads times `d` times two cached tensors times the KV dtype size times its owned token count.
Replicated K/V caches remain two tensors when `k_eq_v` shares the projection.
Q/K/V projection weights on a CP layer cover `H_q / TPA` query heads and the local KV heads, which replicates them across the `KVP` ranks of a group.
The output projection keeps its `(H_q / N) * d` row shard.
LP30 attention scratch uses the local key count.
The combine reserves a send buffer of `B * (H_q / TPA) * (d + 1) * 4` bytes and a receive buffer of `B * (H_q / N) * KVP * (d + 1) * 4` bytes.

## Scope and rejected combinations

The compiler accepts `kv_parallel_size > 1` only after the memory and execution phases land.
Until then, it raises `ConfigurationError` instead of ignoring the setting.
Prefill/decode KV handoff, external drafting, Rubin attention and non-colocated AF reject `KVP > 1` until their KV movement is specified.
Prefill reuses the decode combine step for each chunk with `B` equal to the chunk's query tokens; Phase 5 implements it.
Batch-wise overlap of the combine step is excluded because a serial client has one request per forward.

## Pre-registered expectations

The expectations below use the repository LP30 estimator with the bundled [fidelity configuration](../experiments/lpx/fidelity-study.json).
The baseline is the integrated TP16/PP8 result on 128 LP30 chips, with per-token times of 169.24, 179.26 and 276.92 microseconds at 1K, 10K and 100K.
The changes apply to the ten Gemma global layers with `N = 16`, four KV heads, 32 query heads and `d = 512`.
Unit services are evaluated at contexts 1,001, 10,750 and 101,000.

| Global-attention core per layer, microseconds | 1K | 10K | 100K |
| --- | ---: | ---: | ---: |
| `KVP = 1`, 2 query heads and 1 replicated KV head per rank | 0.377 | 1.433 | 11.198 |
| `KVP = 2`, 4 query heads per rank, half the keys | 0.348 | 0.998 | 7.006 |
| `KVP = 4`, 8 query heads per rank, a quarter of the keys | 0.352 | 1.000 | 7.010 |

These estimates include the estimator's default per-operator startup; the per-layer differences remove it.
Matrix work stays constant because each rank holds more query heads over fewer keys.
The saving comes from vector-unit KV conversion and static random-access memory (SRAM) reads, which scale with the local key count.
At 100K the matrix-unit service of 6.72 microseconds becomes the bottleneck at `KVP = 2`, so `KVP = 4` adds no further time saving.

The Q/K/V projection grows from 0.316 microseconds to 0.357 at `KVP = 2` and 0.438 at `KVP = 4`.
The combine is estimated at 0.5 microseconds per CP layer.
That value combines the 0.35-microsecond local fabric propagation, approximately 0.05 microseconds of wire time for 4,104-byte messages over six 14 GB/s lanes, and fill/drain residuals for the merge and the separate output projection.
The study's collective launch cost is zero.

| Configuration | 1K TPS | 10K TPS | 100K TPS | 100K / 10K |
| --- | ---: | ---: | ---: | ---: |
| Integrated baseline, `KVP = 1` | 5,908.78 | 5,578.41 | 3,611.17 | 0.647 |
| Expected `KVP = 2` on global layers | approximately 5,735 | approximately 5,546 | approximately 4,160 | approximately 0.750 |
| Expected `KVP = 4` on global layers | approximately 5,707 | approximately 5,520 | approximately 4,145 | approximately 0.751 |
| Published endpoint | none | 3,382 | 3,431 | 1.0145 |

The expectation predicts a large 100K improvement and a small loss at 1K and 10K.
It does not predict the published reversal.
The remaining modeled 10K-to-100K difference comes mainly from global-attention matrix work at the assumed 0.10 attention matrix efficiency, approximately 6.0 microseconds per layer and 60 microseconds per token.

| Memory per rank and global layer at 102,000 tokens | `KVP = 1` | `KVP = 2` | `KVP = 4` |
| --- | ---: | ---: | ---: |
| Resident KV bytes on the busiest rank | 104,448,000 | 52,232,192 | 26,116,096 |
| Q/K/V projection weight bytes | 8,257,536 | 13,762,560 | 24,772,608 |
| Net change relative to `KVP = 1` | 0 | −46,710,784 | −61,816,832 |

The busiest rank in the integrated layout holds 211,517,440 KV bytes.
Freeing tens of megabytes per global layer can admit layouts with fewer pipeline stages, so the minimum-chip search may fall below 128 chips under the same assumptions.
This specification records that possibility as a hypothesis without a predicted chip count.

## Acceptance checks

| Check | Requirement |
| --- | --- |
| Unchanged default | `KVP = 1` reproduces complete results and traces field by field, including 5,908.78, 5,578.41 and 3,611.17 TPS |
| Ownership | Closed-form counts match brute-force enumeration and sum to the context on every selected layer |
| Memory | Per-rank KV on selected layers sums to the unreplicated KV bytes, and replicated projection weights appear in memory reports |
| Communication | Combine messages match `combine_send_bytes` and stay independent of context length |
| Expectation comparison | Fixed-layout TPS differing from the table by more than 3% requires an explanation in the results record |

The expectation comparison checks the implementation against its own cost model.
Agreement does not validate Helix as the LPX mechanism or calibrate any LP30 parameter.
