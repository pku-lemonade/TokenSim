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

The compiler accepts `kv_parallel_size > 1` for colocated attention/FFN execution without multi-head latent attention (MLA).
It raises `ConfigurationError` for non-colocated AF, MLA and selected layers whose query heads do not divide `N`.
The native LPX configuration also rejects KV parallelism on non-LP30 attention devices and in cases with prefill/decode KV handoff or external drafting, because their KV movement is unspecified.
Prefill applies the decode combine step to each chunk with `B` equal to the chunk's query tokens.
Batch-wise overlap of the combine step is excluded because a serial client has one request per forward.

## Implementation

| Component | Module | Behavior |
| --- | --- | --- |
| Ownership and partitions | [`kv_sequence.py`](../TokenSim/af/runtime/kv_sequence.py) | Closed-form owned, visible and peak resident token counts; per-layer KVP selection; scope validation |
| KV partitions | [`layer_kv.py`](../TokenSim/af/runtime/layer_kv.py) | Shards KV heads over `TPA` and records `kv_parallel_size`, `kv_parallel_rank` and `kv_append_block` on selected layers |
| Admission | [`admission.py`](../TokenSim/af/runtime/admission.py) | Charges each rank's peak owned positions |
| Weights and scratch | [`memory.py`](../TokenSim/af/runtime/memory.py), [`lpx_memory.py`](../TokenSim/af/runtime/lpx_memory.py) | Replicates Q/K/V over KVP ranks, keeps `H_q / N` output-projection heads, sizes attention tiles from the local context and adds combine buffers |
| Decode graph | [`planning/context_parallel.py`](../TokenSim/af/planning/context_parallel.py) | Adds one `all_to_all` per KVP group and one `attention_output` node per rank before the attention all-reduce |
| Decode costs | [`providers/context_parallel.py`](../TokenSim/af/providers/context_parallel.py) | Prices attention over the rank's visible keys and prices the merge and output projection separately |
| LP30 merge work | [`lp30.py`](../TokenSim/operator_data/analytical/lp30.py) | `lse_merge` charges `2 * d + 3` vector operations and one exponential per fragment |
| Prefill | [`operator_table/kv_parallel.py`](../TokenSim/latency/operator_table/kv_parallel.py) | Applies the same heads, local contexts, merge and all-to-all to complete-model prefill backends |
| LPX refinement | [`streaming.py`](../scripts/lpx/streaming.py) | Prices the combine with the catalog topology and the routed LP30 fabric phase, and streams the all-reduce from `attention_output` |

Plan summaries, rank KV partitions and attention components emit the new fields only when `KVP > 1`.
The `KVP = 1` path keeps the previous graph, costs and serialized results.
The attention node's local shape carries the KVP fields, so operator estimates are cached separately per rank coordinate.
Contexts are supplied to provider queries at execution time, so the directed acyclic graph template cache needs no change.

A prefill chunk of `q` query tokens on rank `r` attends to `min(S, q + visible(S - q, r))` keys.
This bounds the chunk's own positions by `q` without tracking which rank owns them, which overstates prefill attention work on ranks that own none of the chunk.

The busiest rank's key count changes slope at every append-block boundary.
The LPX decode curve therefore adds knots at each `kv_append_block` boundary when `KVP > 1`, which keeps the 0.1-nanosecond interpolation tolerance without additional bisection.

### Command-line options

`Layout(a_tp=TPA, a_cp=KVP)` describes a KV-parallel layout with `N = a_tp * a_cp` ranks per stage.
`Layout.a_stage_ranks` returns `N`, and colocated layouts require `N = f_tp * f_ep`.

| Entry point | Option | Default | Meaning |
| --- | --- | --- | --- |
| `scripts/search_lpx_native.py` | `--kvp` | `1` | KVP sizes tried for every stage TP that they divide |
| `scripts/search_lpx_native.py` | `--kv-parallel-layers` | `full_attention` | Layer selection for KVP layouts |
| `scripts/search_lpx_native.py` | `--kv-append-block` | `16` | Ownership block for KVP layouts |
| `scripts/validate_lpx_fidelity.py` | `--kvp`, `--kv-parallel-layers`, `--kv-append-block` | `1`, `full_attention`, `16` | KV-parallel settings of the validated layout |
| `scripts/run_lpx_context_parallel.py` | `--cases`, `--minimum`, `--summary` | all cases | Fixed-layout experiments, memory-feasible stage search and portable summary |

The search records a `kv_parallel` settings entry, and the checkpoint parameters include it, only when an option differs from its default.
Existing checkpoint keys for searches without KV parallelism are unchanged.
The plan proposed fidelity-configuration fields for these settings; the implementation keeps them in the layout and command line because they describe a deployment, not an uncalibrated hardware rate.

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

## Results

Results date: 2026-10-22.
[`scripts/run_lpx_context_parallel.py`](../scripts/run_lpx_context_parallel.py) produced every value below, and the [portable summary](../experiments/lpx/context-parallel-results.json) records parameters, TPS and peak-rank memory for each case.
Every case uses the bundled fidelity configuration, compiled execution, a 0.35-microsecond chip-to-chip hop and the AA serial-client workloads at 1K, 10K and 100K input tokens.
Unless a row states otherwise, the layout is TP16/PP8 on 128 LP30 chips with layer counts 6, 8, 9, 7, 7, 10, 7 and 6 per stage.

### Fixed-layout comparison

| Case | 1K TPS | 10K TPS | 100K TPS | 100K / 10K | Change at 1K, 10K, 100K |
| --- | ---: | ---: | ---: | ---: | --- |
| `KVP = 1` | 5,908.78 | 5,578.41 | 3,611.17 | 0.647 | baseline |
| `KVP = 2` | 5,759.05 | 5,560.63 | 4,167.93 | 0.750 | −2.5%, −0.3%, +15.4% |
| `KVP = 4` | 5,730.77 | 5,533.30 | 4,152.44 | 0.750 | −3.0%, −0.8%, +15.0% |
| `KVP = 4`, 64-token append block | 5,728.79 | 5,531.72 | 4,151.61 | 0.751 | −3.0%, −0.8%, +15.0% |
| `KVP = 4`, contiguous ownership | 5,632.84 | 4,945.85 | 4,141.26 | 0.837 | −4.7%, −11.3%, +14.7% |
| Published endpoint | none | 3,382 | 3,431 | 1.0145 | none |

The `KVP = 1` case reproduces the integrated baseline to the reported precision.
The measured `KVP = 2` and `KVP = 4` values exceed the pre-registered expectations by 0.18% to 0.42%.
The per-token times are 0.4 to 0.8 microseconds below the expectation, which is smaller than the rounding of the 0.5-microsecond combine estimate over ten layers.
No comparison exceeds the 3% tolerance.

`KVP = 2` is faster than `KVP = 4` at every context.
At 100K the global-attention core is bound by matrix-unit service at both sizes, so `KVP = 4` saves no attention time and pays for the larger replicated Q/K/V projection.
The 64-token append block changes TPS by less than 0.04%.
Contiguous ownership assigns positions 0 to 25,499 to the first KVP rank.
At 10K that rank reads every key with four times the query heads of `KVP = 1`, which explains the 11.3% loss.
At 100K the busiest contiguous rank owns 25,500 positions, so contiguous and round-robin ownership differ by 0.27%.

The 100K/10K ratio rises from 0.647 to 0.750, which matches the expectation.
Helix-style KV parallelism in this model does not reproduce the published reversal, in which 100K is faster than 10K.
The remaining 10K-to-100K difference is dominated by global-attention matrix work at the configured 0.10 attention matrix efficiency.

### Communication sensitivity

Two controls raise the cost of every collective, including the combine step that KV parallelism adds on each global layer.
`launch 0.5` charges 0.5 microseconds per streamed collective launch.
`local 0.7` doubles the propagation of the local fabric tier from 0.35 to 0.7 microseconds.

| Case | 1K TPS | 10K TPS | 100K TPS | 100K / 10K | `KVP = 4` change at 1K, 10K, 100K |
| --- | ---: | ---: | ---: | ---: | --- |
| `KVP = 1`, launch 0.5 | 4,413.33 | 4,226.37 | 2,991.63 | 0.708 | baseline |
| `KVP = 4`, launch 0.5 | 4,222.20 | 4,114.03 | 3,298.49 | 0.802 | −4.3%, −2.7%, +10.3% |
| `KVP = 1`, local 0.7 | 3,932.53 | 3,783.40 | 2,762.67 | 0.702 | baseline |
| `KVP = 4`, local 0.7 | 3,801.61 | 3,713.69 | 3,036.08 | 0.818 | −3.3%, −1.8%, +9.9% |

Higher communication cost reduces the 100K gain from 15.0% to approximately 10% and enlarges the short-context loss.
The ratio stays below 0.82 in both controls.

### Layer selection

`kv_parallel_layers = all` also shards the 50 windowed layers.
Their replicated Q/K/V projections do not fit the baseline layer counts, where stage index 5 needs 30,834,688 KV bytes and has 19,533,824 available.
The case therefore uses the PP8 capacity partition with layer counts 6, 8, 8, 8, 8, 8, 8 and 6.
The serial decode TPS of the `full_attention` cases does not depend on that choice, because the validation partition below reproduces them to every recorded digit.

| Case | 1K TPS | 10K TPS | 100K TPS | 100K / 10K | Change from `KVP = 1` at 1K, 10K, 100K |
| --- | ---: | ---: | ---: | ---: | --- |
| `KVP = 4`, `full_attention` | 5,730.77 | 5,533.30 | 4,152.44 | 0.750 | −3.0%, −0.8%, +15.0% |
| `KVP = 4`, `all` | 4,913.07 | 4,767.24 | 3,705.58 | 0.777 | −16.9%, −14.5%, +2.6% |

Sharding the windowed layers adds 29.04 microseconds per token at every context, approximately 0.58 microseconds per windowed layer.
A windowed layer reads at most 1,024 keys, so splitting them saves little attention time, while each layer pays the combine, the merge and the larger Q/K/V projection.
The peak rank holds 382,604,288 weight bytes and 55,377,920 KV bytes, leaving 51,806,128 bytes of headroom.
These results support the `full_attention` default.

### Memory and minimum inventory

| Case | Peak-rank weights, bytes | Peak-rank KV, bytes | Peak-rank total, bytes | Minimum headroom, bytes |
| --- | ---: | ---: | ---: | ---: |
| `KVP = 1`, 128 chips | 220,549,632 | 211,517,440 | 440,169,038 | 59,830,962 |
| `KVP = 2`, 128 chips | 310,351,360 | 56,950,784 | 375,928,148 | 124,071,852 |
| `KVP = 4`, 128 chips | 321,361,408 | 30,834,688 | 361,347,412 | 138,652,588 |
| `KVP = 8`, TP16/PP6, 96 chips | 422,861,312 | 56,950,784 | 491,663,702 | 8,336,298 |

The peak rank moves from stage index 6, which holds seven layers, to stage index 5, which holds ten layers, once KV parallelism shrinks the resident KV.
The minimum-stage search uses capacity partitions at a 102,000-token maximum context with `full_attention` selection.

| Stage TP | `KVP = 1` | `KVP = 2` | `KVP = 4` | `KVP = 8` |
| --- | --- | --- | --- | --- |
| 8 | PP11, 88 chips | PP11, 88 chips | PP11, 88 chips | PP13, 104 chips |
| 16 | PP8, 128 chips | PP6, 96 chips | PP6, 96 chips | PP6, 96 chips |
| 32 | PP5, 160 chips | PP4, 128 chips | PP4, 128 chips | PP4, 128 chips |

At TP16 every KV-parallel size admits PP6.
The driver screens the smallest TP16 inventory and breaks the tie toward the largest KVP, so it ran `KVP = 8` on TP16/PP6.
That 96-chip layout reaches 5,716.56, 5,518.62 and 4,143.38 TPS, within 0.3% of `KVP = 4` on 128 chips.
Serial decode passes every layer regardless of the stage count, so fewer stages change TPS only through stage-boundary transfers and the per-stage layer mix.

### Complete prefill and decode validation

The fidelity validator ran the complete 100K prompt and the serial decode workloads with `KVP = 4`:

```bash
.venv/bin/python scripts/validate_lpx_fidelity.py \
  --kvp 4 \
  --output outputs/lpx-context-parallel/validation-kvp4
```

The capacity partition selected layer counts 6, 8, 9, 7, 8, 9, 7 and 6 on 128 chips, with a peak rank total of 330,845,522 bytes.
All 6,250 prefill chunks of 16 tokens executed and the KV lease reached 100,000 positions.
Simulated prefill took 6.756 seconds, compared with 6.398 seconds for the `KVP = 1` validation in the [fidelity results](../experiments/lpx/fidelity-results.json).
The 5.6% increase includes the per-chunk combine on every global layer and the prefill attention bound described under Implementation, which charges every rank for the chunk's own 16 positions.
Decode reproduced the fixed-layout `KVP = 4` TPS to every recorded digit and met the validator's 3,400 TPS and published-target thresholds.

### Interpretation

Under the Helix hypothesis and the current LP30 parameters, KV parallelism raises modeled 100K throughput by approximately 15%, costs up to 3% at 1K and lowers the minimum TP16 inventory from 128 to 96 chips.
It narrows the gap to the published 100K/10K ratio without closing it.
These results test the simulator against its own pre-registered cost model.
They do not show that LPX uses Helix-style KV parallelism, and they do not calibrate any LP30 parameter.
