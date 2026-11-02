# LP30 256-chip decode batching, pipeline concurrency and MFU

Record date: 2026-11-02.
The experiment compares batches of 1, 8, 9 and 11 independent requests with the same request counts served as single-request pipeline microbatches.
On the capacity partition, an 11-request batch produces 18,765.6 aggregate tokens per second (TPS) at 0.7701% model FLOP utilization (MFU).
Serving 11 requests with microbatch one produces approximately 32,267.3 aggregate TPS at 1.3241% MFU in the operator-DAG replay model.
These are analytical simulation results with uncalibrated LP30 operator and scheduling assumptions.
LP30 hardware performance and compiler support for the concurrent replay schedules remain unverified.

The [portable experiment results](../experiments/lpx/batch-concurrency-mfu-results.json) retain all 16 points, including two SRAM admission failures, exact metrics, configuration, measurement windows and source identity.

## Conditions

| Parameter | Value |
| --- | --- |
| Model | Gemma 4 31B text |
| Inventory | 256 LP30 chips, 500 MB static random-access memory (SRAM) per chip |
| Parallel layout | Tensor parallelism (TP) 32, pipeline parallelism (PP) 8, decode context parallelism (DCP) 8 |
| DCP implementation | `a_tp=4`, `a_cp=8`, `f_tp=32`; DCP applies to full-attention layers |
| Input and output lengths | 100,000 input tokens and at most 2,000 output tokens per request |
| Decode timing context | Fixed representative context of 101,000 tokens |
| Resident key-value (KV) reservation | 102,000 tokens per request, with independent prefixes |
| Precision | FP8 weights and KV cache; BF16 activations and attention matrix work |
| Operator efficiency inputs | GEMM 0.30; attention matrix work 0.50 |
| Other operator and memory settings | Bundled fidelity configuration, including vector work, KV conversion, tiled attention and scratch buffers |
| Chip-to-chip (C2C) injection | 96 links at 14 GB/s per direction, or 1.344 TB/s per chip |
| Collective settings | Ring collectives; zero collective launch cost; existing TP32 refinement behavior |
| Speculative decoding | Disabled; each request commits one token per forward |
| External-memory offload | Disabled |
| Timing exclusions | Prefill, token sampling/control latency and external I/O |

The configuration reuses [the article sweep's injection settings](../scripts/run_lpx_article_sweep.py), [the context-parallel environment](../scripts/run_lpx_context_parallel.py) and [the fidelity configuration](../experiments/lpx/fidelity-study.json).
The TP32 refinement retains the existing limitation that multi-tray collective/compute overlap is unmodeled.
The GEMM and attention efficiency inputs describe individual operator estimates; the reported MFU accounts for useful matrix work over elapsed time on all 256 chips.

## Request count and microbatch size

`N` denotes the total number of resident requests across the deployment.
`B` denotes the number of requests processed together in one forward.
In the single-cohort experiment, `N = B` and the cohort begins its next token after its preceding forward completes.
In the pipeline experiment, `B = 1` and `N` independent cohorts share the physical compute and communication resources.
Each cohort submits its next token only after its preceding token completes.

Eight PP stages provide eight stage positions for single-request microbatches, while every resident request retains its KV partitions across the stages.
Resident KV admission counts those requests once across the complete layout.
The request counts and aggregate TPS below already cover the entire 256-chip deployment and require no additional PP multiplier.

## Layer partitions and SRAM admission

| Partition | Layer counts across the eight PP stages | Admission at the tested request counts |
| --- | --- | --- |
| Default | `[8, 8, 8, 8, 7, 7, 7, 7]` | 1, 8 and 9 admitted; 11 rejected |
| Capacity | `[6, 7, 8, 8, 7, 11, 7, 6]` | 1, 8, 9 and 11 admitted |

Admission uses the existing weights, protocol workspace, live scratch, 5 MB per-chip reserve and KV watermark rules.
The capacity partition changes the placement of layers within the same TP32/PP8/DCP8 inventory.
The experiment checks the busiest chip rather than accepting a point solely from aggregate free SRAM.

For the single-cohort capacity partition, the physical SRAM ledger is:

| Requests and batch size | Peak chip occupancy (MB) | Minimum chip headroom (MB) |
| ---: | ---: | ---: |
| 1 | 229.295 | 270.705 |
| 8 | 404.783 | 95.217 |
| 9 | 434.580 | 65.420 |
| 11 | 494.174 | 5.826 |

The 11-request point passes the current admission policy with limited physical headroom.
These checks establish feasibility for the recorded points under the current memory model.
They do not establish an external-memory serving capacity or a concurrency limit at a target per-request TPS.

## Single cohort with batch size 1, 8, 9 or 11

For a forward duration `T`, each request produces one token, so per-request TPS is `1 / T` and aggregate TPS is `B / T`.
The capacity partition gives the following results:

| Batch size | Forward duration (us) | Per-request TPS | Aggregate TPS | MFU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 281.456 | 3,553.0 | 3,553.0 | 0.1458% |
| 8 | 493.889 | 2,024.7 | 16,198.0 | 0.6647% |
| 9 | 524.651 | 1,906.0 | 17,154.3 | 0.7039% |
| 11 | 586.180 | 1,706.0 | 18,765.6 | 0.7701% |

The default partition produces identical isolated forward durations for batches 1, 8 and 9.
Its batch-11 forward is rejected by resident KV admission before timing.
The batch sizes describe independent requests with separate KV caches; every request decodes one query token per forward.

## Pipeline execution with microbatch one

The scheduler replays native operator directed acyclic graphs (DAGs) concurrently on the same named physical resources.
Dependencies preserve each request's token order, and resource queues account for contention between requests.
The replay permits overlap between operators from different requests, including work within a PP stage.
Actual compiler support and buffer lifetime safety for these overlapping schedules require separate validation.

Per-request TPS is measured aggregate TPS divided by `N`.
Mean time per output token (TPOT) is measured from each request's completion intervals.
Their reciprocals can differ slightly because the finite measurement window cuts through requests in flight.

### Default layer partition

| Resident requests | Mean TPOT (us) | Per-request TPS | Aggregate TPS | MFU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 281.456 | 3,553.0 | 3,553.0 | 0.1458% |
| 8 | 362.666 | 2,756.8 | 22,054.8 | 0.9050% |
| 9 | 377.489 | 2,649.4 | 23,844.5 | 0.9785% |
| 11 | Infeasible | | | |

### Capacity layer partition

| Resident requests | Mean TPOT (us) | Per-request TPS | Aggregate TPS | MFU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 281.456 | 3,553.0 | 3,553.0 | 0.1458% |
| 8 | 311.668 | 3,206.8 | 25,654.2 | 1.0527% |
| 9 | 321.888 | 3,105.8 | 27,952.4 | 1.1470% |
| 11 | 340.898 | 2,933.4 | 32,267.3 | 1.3241% |

In the tested configurations, pipeline execution with microbatch one produces more aggregate output than putting every request in one batch.
The capacity partition also improves pipeline throughput relative to the default partition at 8 and 9 requests.
At eight requests, its approximately 3,207 TPS per request remains below a 3,400 TPS target under the current assumptions.
The 11-request pipeline point gives the highest aggregate throughput among these tested points while reducing per-request TPS to approximately 2,933.

## MFU accounting

Useful linear matrix work is 61.392099328 GFLOP per output token, summed using each layer's attention geometry.
Query-key and attention-value matrix work contributes another 67.869081600 GFLOP at the representative context.
The total is 129.261180928 GFLOP per output token.
The tables use the complete rack's 315 PFLOP/s FP8 peak:

```text
MFU = aggregate_TPS * 129.261180928e9 / 315e15
MFU_percent = 100 * MFU
```

This useful-work metric excludes replicated projection computation and does not count vector or exponential operations as matrix FLOPs.
The evidence also records linear-only MFU and MFU with a BF16 peak correction.
The latter uses the catalog assumption that BF16 peak is half the FP8 peak:

```text
MFU_with_BF16_peak_correction = aggregate_TPS * (61.392099328e9 + 2 * 67.869081600e9) / 315e15
```

For 11 requests, that correction gives 1.1744% for the single batch and 2.0193% for microbatch-one pipeline execution.
The BF16 peak ratio is an unmeasured assumption, so the main tables retain the same uniform FP8 denominator as the earlier calculation.

## Measurement windows and validation

The single-cohort measurements discard eight forwards and measure 64 subsequent forwards.
Nontrivial pipeline points use the following warm-up and measurement counts:

| Partition | Requests | Warm-up completions | Measured completions | Measured duration (ms) | Later-half TPS change relative to earlier half |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 8 | 256 | 2,048 | 92.86 | -0.53% |
| Default | 9 | 288 | 2,304 | 96.63 | +0.50% |
| Capacity | 8 | 256 | 2,048 | 79.83 | +0.96% |
| Capacity | 9 | 576 | 4,608 | 164.85 | +2.07% |
| Capacity | 11 | 704 | 5,632 | 174.54 | +2.11% |

The approximately 2% window variation at 9 and 11 requests supports reporting the pipeline results as estimates.
These comparisons measure simulation-window sensitivity and provide no statistical confidence interval for hardware throughput.
The context remains fixed at 101,000 tokens throughout each probe; the experiment does not execute the complete evolving-context 2,000-token generation.

Every captured profile passes `NativeProfile.verify()` against its modeled isolated duration.
Single-cohort replay also agrees with `B / T`.
The result collector checks token conservation, aggregate versus per-request TPS, positive physical SRAM headroom and the 281.456 us batch-one latency regression.

The probe reuses [native configuration](../scripts/lpx/native_config.py), [native profile capture and verification](../scripts/lpx/replay_profiles.py), [streaming settings](../scripts/lpx/native_search.py) and the [C++ replay scheduler](../scripts/lpx/replay_engine.cpp).
No simulator implementation change was required.
The portable record includes the operator, memory and fabric parameters plus the per-cohort completion counts needed to inspect these calculations.

Source commit: `4f3136c25455e4433ed9016824f6403688412ab7`.
Source fingerprint: `6a8f1dbc0155aceded7708d9e10f1144c5048eb20064b927de00f9502ee0a7bc`.
