# Within-head output channels for wider LP30 context parallelism

The native attention planner now supports equal output-channel fragments within an attention head when context parallelism supplies complete query heads on each rank.
This removes the previous whole-output-head restriction that prevented Gemma's global layers from using tensor parallelism (TP) 64, 128 or 256.
The completed experiment captures 67 native graphs from 84 configurations and evaluates 93,888 conditional throughput bounds.
The subsequent [boundary-pool and wider-pipeline study](lpx-boundary-pool-width-search.md) raises the attained decode record to approximately 3.827772% model floating-point operation utilization (MFU) on 445 chips.
It completes 88 wider compositions and measures a selected 884-chip native replay at approximately 3.357712% MFU.
The objective remains pure LP30 hardware, MFU priority and no minimum tokens per second (TPS).
The objective-wide maximum remains unproven.

## Attention and output ownership

For stage width N and decode context parallelism (DCP) K, attention uses `TPA = N/K` groups of query heads.
Each rank computes `H_q/TPA` complete query heads over its owned sequence positions.
The query/key dot product retains the original head dimension d and its original normalization.
Output ranks retain contiguous flattened channels of width `W_o = H_q * d / N` after the sequence-partition reduction.

When `H_q` is divisible by N, each output rank retains complete heads.
When N exceeds `H_q`, the implementation accepts `N/H_q` equal fragments per head if N is divisible by `H_q` and d is divisible by `N/H_q`.
Each output rank then retains one fragment of dimension `d_o = d / (N/H_q)`.
Query heads must remain divisible by TPA.
Unequal fragments crossing head boundaries remain unsupported.
Unselected local-window layers retain whole-head TP and reject inventories exceeding their query-head count.

The [Flash-Decoding authors](https://pytorch.org/blog/flash-decoding/) describe attention over sequence partitions with a log-sum-exp (LSE) scalar per query row and partition, followed by an LSE-weighted reduction.
For partition outputs O_i and scalars L_i, the reduction uses `m = max_i L_i`, `a_i = exp(L_i - m)` and `O = sum_i a_i * O_i / sum_i a_i`.
The scalar weight applies to every output channel in the same head.
Applying that formula to equal output-channel fragments is a mathematical inference from the reduction, tested here against complete attention and the complete output projection.

The numerical controls include grouped-query attention, two sequence lengths, empty sequence partitions and both whole-head and fragmented output ownership.
Every output channel has exactly one owner.
The assembled attention output and the sum of all projection shards match the complete calculation to the recorded floating-point tolerances.
These controls establish the partition algebra for the supplied tensors.
They do not validate the experimental shared FP8 cache against Gemma model quality.

## Work, communication and memory

Query/key/value (Q/K/V) projection widths and attention tiles retain complete heads.
The O projection stores and computes the nonzero input-channel shard `W_o` on every rank.
The existing attention all-reduce sums those projected partial results across all stage ranks.
The dense feed-forward network (FFN), residual and boundary operations retain their existing TP behavior.
Useful-work credit remains 129,261,109,248 floating-point operations per committed output token.
Repeated projections and LSE calculations receive no additional useful-token credit.

For H_o retained whole heads or fragments with channel dimension d_o, each rank reserves `B * H_o * K * (d_o + 1) * 4` bytes in each combine direction.
Partial outputs and LSE scalars use 32-bit floating point (FP32).
The send and receive buffers each include one complete LSE scalar for every retained head fragment and sequence partition.
Splitting a head between two output ranks duplicates its LSE scalar in the exchange.
The merge charges the corresponding repeated exponentials and vector work.

Gemma's global geometry has 32 query heads with dimension 512.
TP64/DCP16 computes eight complete query heads per rank and retains a 256-channel O shard.
Its per-query combine buffer is `16 * (256 + 1) * 4 = 16,448` bytes per direction.
The previous TP32/DCP8 case uses `8 * (512 + 1) * 4 = 16,416` bytes.
The wider case retains the extra scalar bytes.

The attention shape retains the complete `head_dim`.
Fragmented outputs add `output_head_dim`, `output_channel_start` and `output_channel_stop`.
The memory component records `o_head_dim` and `o_width` for fragmented O shards.
Whole-head shapes omit those additional fields.
Decode graphs and complete-model prefill use the same output partition and combine volume.
The [context-parallel specification](lpx-context-parallel.md) contains the detailed ownership equations.

## Native catalog

The catalog tests independent batches 1, 2 and 4.
It recaptures all sixty local configurations with one through five layers at TP2, TP4, TP8 or TP16.
It adds eighteen batch-one global configurations at TP64, TP128 and TP256, using every power-of-two DCP width from `TP/32` through TP.
Batch-two and batch-four global controls use DCP widths `TP/4`.
Sixty-seven configurations produce admitted native graphs, while seventeen fail native configuration or memory checks under the declared policy.
Every admitted raw capacity receives a native admission check and a rejection check for one additional request.
Residency is rounded down to complete independent batches.

Each transformer rank retains two complete native scratch, dense-residual and communication workspaces, together with two batched incoming hidden rows.
The 5,000,000-byte safety reserve, batch-scaled vocabulary backcharge, sixteen-token KV block rounding and one-percent KV watermark remain.
Weights and the experimental global key-value (KV) cache use eight-bit floating point (FP8), while activations and attention use bfloat16 (BF16).
The context remains 101,000 tokens for timing, with KV storage reserved through 102,000 tokens.
The global cache layout remains `value_rope_fp8` with the `split_dot` kernel.

| One global layer | Batch | Raw resident maximum | Complete-batch residents | Native body, microseconds |
| --- | ---: | ---: | ---: | ---: |
| TP64/DCP16 | 1 | 109 | 109 | 5.867708 |
| TP64/DCP16 | 2 | 109 | 108 | 7.240788 |
| TP64/DCP16 | 4 | 108 | 108 | 10.115172 |
| TP128/DCP32 | 1 | 220 | 220 | 5.681562 |
| TP128/DCP32 | 2 | 219 | 218 | 6.483239 |
| TP128/DCP32 | 4 | 218 | 216 | 8.200524 |
| TP256/DCP64 | 1 | 442 | 442 | 6.978221 |
| TP256/DCP64 | 2 | 440 | 440 | 7.484579 |
| TP256/DCP64 | 4 | 438 | 436 | 8.603778 |

These capacities describe isolated global-layer components.
A complete serving pipeline must also admit every local layer, retain all boundary weights and workspaces, satisfy token feedback and schedule every physical resource.
The component capacities do not multiply by the number of pipeline stages.

## Necessary bounds and remaining search

The bound screen enumerates all ordered partitions of five local layers using the admitted catalog, followed by one supplied global layer and repeated ten times.
Boundary pools contain three or four LP30 chips, subject to their full matrix and workspace ledgers.
Every allocated chip contributes to the MFU denominator.
The bound includes exact native segment paths and per-resource service, the complete embedding and vocabulary head, entry copies and one four-microsecond host submission per complete independent batch.
It charges only the minimum 0.35 microseconds of propagation at each cut.
Wire service, longer placement distances and shared-resource contention can lower attained throughput.

| Independent batch | Bound points | Points above 3.822824% | Surviving segment-multiset classes | Largest optimistic MFU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 75,168 | 0 | 0 | 3.129830% |
| 2 | 12,480 | 520 | 79 | 5.298249% |
| 4 | 6,240 | 11 | 6 | 4.917884% |

All 531 surviving ordered points use the supplied TP64/DCP16 global geometry.
The exclusions apply to the declared native graphs, batches, repeated-block family, head widths and memory policy.
Other batch widths, head pools, graph orders, memory placements and inventories remain open.
The optimistic bounds are not attained MFU measurements.

Wider physical placements require explicit port reconstruction.
The fabric assigns fewer lanes per peer to sufficiently wide dense collectives, so shifting only the chip identifiers can change the occupied physical port bundles.
Four native controls compare offsets 0, 64, 128 and 192 against reconstructed dense fabric phases.
Offsets 64 and 128 each expose four wire-resource set differences from the earlier translation rule.
Reconstruction preserves unit wire time, propagation time and single-program native duration while retaining the physical port sets at the new placement.
The subsequent multi-stage study uses that reconstruction.

## Validation and evidence

Six previously admitted whole-head components reproduce their complete native node arrays and native memory admission results bit for bit.
They include all three component types used by the existing 3.822824% record.
The full project suite runs 898 tests with 896 passing.
The two remaining platform failures are the unavailable `nvidia-smi` executable and macOS resolving `/tmp` to `/private/tmp`.
The suite uses the public native-replay cache helper to place the identical compiled scheduler in writable temporary storage.
Black formats and checks all 585 Python files in the required project paths.

The [compressed catalog record](../experiments/lpx/output-channel-fragment-catalog.json.gz) retains all 84 native receipts, 67 component bodies, all 93,888 bound points, exact whole-head regression controls, physical placement controls, affected source files, reproduction scripts and the full test log.
Its portable validator independently checks nonzero O weight shards, complete Q/K widths, graph costs, copied-frame reservations and every bound calculation.
The production simulation source identity is `9dfe67e50e9ab76edafd96f37033622e8ae62b2abb3a03f0d233a81920e2407e`.
The simulator retains the uncalibrated 1.15-microsecond transport fallback beyond 256 chips.
Physical compiler execution, SRAM banks, shared-cache quality and multi-rack timing remain unverified.
