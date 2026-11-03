# Pure LP30 partition search without a throughput floor

The confirmed hardware scope requires every accelerator to be an LP30.
The objective prioritizes useful model floating-point operation utilization (MFU), with no minimum aggregate or per-request throughput.
Inventories beyond 256 chips remain eligible.
All allocated chips contribute to the peak denominator, including transformer, tied embedding, output-head, assistant and communication roles.
Historical GPU-assisted experiments remain outside the eligible ranking.
Representative-context decode and complete-request serving retain separate numerators and measurement windows.

## Capacity-constrained layer cuts

This experiment uses tensor parallelism (TP) 16, decode context parallelism (DCP) 4 on full-attention layers, batch size one and the experimental `value_rope_fp8` global key-value (KV) representation.
It reserves cache through 102,000 tokens and measures decode at 101,000 tokens.
Weights and cache use eight-bit floating point (FP8), while activations and attention use bfloat16 (BF16).
Model-quality equivalence of this cache representation remains unverified.
The paired stage arenas reserve an additional 2,190,464 bytes per transformer chip while retaining the 5,000,000-byte safety reserve and native one-percent KV watermark.
The three-chip tied embedding and language-model head pool remains explicit.

A native 60-stage seed provides isolated per-layer body times with host submission removed from the partition objective.
A dynamic program minimizes the maximum sum of those times within any stage subject to the supplied native static random-access memory (SRAM) footprint.
The complete proxy frontier evaluates every admitted resident count N and pipeline parallelism (PP) depth from one through 60.
It contains 1,260 admitted `(PP, N)` combinations, reaches 26 resident requests and considers inventories through 963 LP30 chips.
The scalar and vectorized recurrences agree at all 68 supplied near-capacity points.
Nine additional comparisons enumerate 5,313 partitions and match the scalar recurrence exactly.

The proxy score relaxes boundary execution, pipeline handoffs and physical resource competition.
It selects native replay candidates and does not establish attained throughput or an MFU upper bound.
The nine highest-scoring combinations receive explicit native SRAM admission and paired-arena replays.

## Native replay results

The unchanged three-chip boundary pool also passes its explicit SRAM ledger.
Throughput is measured in tokens per second (TPS).
Every replay discards 4N warm-up completions and measures 16N subsequent completions across N clients.
Per-client completion counts and the throughput difference between measurement halves remain explicit in the record.
The control uses one issue interval, and each additional candidate uses three intervals derived from its isolated feedback duration.
The table reports the highest measured MFU within that supplied interval grid.
The directed acyclic graph (DAG) bound applies to each captured program and resident count.
It does not establish an attainable schedule or a bound over alternative partitions and programs.
The above-256-chip candidates retain an uncalibrated inter-rack propagation policy.

| LP30 chips | PP | Requests | Best aggregate TPS | Best decode MFU | Fixed-DAG MFU bound | Minimum SRAM headroom, bytes | Half-window TPS drift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 163 | 10 | 14 | 52,911.88 | 3.410067% | 3.975202% | 19,595,188 | -0.000% |
| 195 | 12 | 16 | 54,807.48 | 2.952586% | 3.766666% | 19,665,334 | +0.442% |
| 323 | 20 | 26 | 76,932.46 | 2.502098% | 3.572750% | 7,864,254 | +0.426% |
| 163 | 10 | 13 | 52,072.56 | 3.355974% | 3.691259% | 38,743,220 | -0.000% |
| 179 | 11 | 14 | 52,900.75 | 3.104603% | 3.605094% | 19,595,188 | +1.445% |
| 243 | 15 | 19 | 66,544.85 | 2.876772% | 3.546110% | 3,733,176 | -3.289% |
| 323 | 20 | 25 | 72,217.16 | 2.348741% | 3.435336% | 24,390,846 | +2.916% |
| 195 | 12 | 15 | 48,975.05 | 2.638382% | 3.531249% | 38,289,078 | -2.353% |
| 339 | 21 | 26 | 76,486.19 | 2.370176% | 3.390791% | 7,864,254 | -1.044% |

The PP10, fourteen-request control reproduces 3.410066841% MFU with explicitly charged paired stage arenas.
None of the additional partition candidates improves this control or the earlier 3.419600500% shared-cache timing record.
The existing-format decode record remains 3.039141553% MFU on 163 LP30 chips.
The finite complete-request result of 15.864688238% includes prompt work and retains its separate metric.
These short screens do not establish sustained throughput for the additional candidates.
All experimental shared-cache values remain conditional on model quality, physical buffer placement and compiler support.

## External-memory bandwidth necessities

The [NVIDIA architecture description](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) lists dynamic random-access memory (DRAM) capacities through fabric expansion and the host central processing unit (CPU).
The retrieved description does not specify sustained DRAM-to-LP30 read bandwidth or latency.
External storage with computation on LP30 remains within the accelerator constraint.
The bandwidth calculation below assumes independent request KV contents and the native separate or experimental shared representation.

The fixed-layout calculation retains the exact physical-state weights and paired buffers for TP16/DCP4, ten stages of six layers and three head chips.
For each transformer chip, the active KV bytes exceeding its remaining SRAM form a necessary external-read volume over a fair request cycle.
The calculation relaxes KV block rounding, watermark, transfer latency, write-back and route contention.

```text
missing_bytes_chip = max(0, N * active_KV_bytes_chip - available_KV_SRAM_chip)
minimum_offload_read_bytes_per_token = sum(missing_bytes_chip) / N
minimum_aggregate_read_bandwidth = aggregate_TPS * minimum_offload_read_bytes_per_token
```

A hypothetical 4% decode MFU on 163 chips requires 62,065.507 aggregate tokens per second.
The table divides the aggregate read floor by the ten transformer trays, each containing sixteen chips.
These values are necessary average read-bandwidth floors for this fixed placement and representation.
They are neither measured hardware bandwidth nor an achieved 4% serving result.

| Global KV representation | Requests | Minimum offload read per token, MB | Necessary average read per transformer tray, TB/s |
| --- | ---: | ---: | ---: |
| `separate` | 16 | 1674.194 | 10.391 |
| `separate` | 32 | 3115.292 | 19.335 |
| `value_rope_fp8` | 16 | 160.545 | 0.996 |
| `value_rope_fp8` | 32 | 1598.948 | 9.924 |

The shared-cache sixteen-request floor remains near 1 TB/s per transformer tray even under these relaxations.
Changing layer placement or cache precision requires a new ledger and bandwidth calculation.
Prefetch can overlap transfers with compute, while the aggregate byte conservation condition still applies.

## Evidence and remaining coverage

The [compressed experiment record](../experiments/lpx/pure-lp30-partition-search.json.gz) contains the supplied timing configuration, full proxy frontier, native admission reports, program fingerprints, per-client replay results, enumeration checks, offload ledgers and reproduction sources.
Native DAG bodies remain in the local output directory referenced by their fingerprints.
The [main search record](lpx-maximum-mfu-search.md) retains the broader workload and hardware assumptions.

This study covers one TP/DCP family and batch size one.
Other widths, larger batches, heterogeneous stage widths, different arena policies, request phases, speculative acceptance, fabric-memory prefetch and sustained complete-request serving remain open.
The highest observed values remain conditional model results, and the global maximum remains uncertified.
