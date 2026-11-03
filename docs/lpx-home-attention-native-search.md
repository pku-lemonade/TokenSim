# Native LP30 attention on SRAM storage chips

Progress date: 2026-11-04, Asia/Shanghai.
The attention-on-home design keeps additional caches in existing static random-access memory (SRAM).
It now has numerical controls, complete native layer graphs and closed-loop pipeline measurements.
The greatest short-screen result is approximately 3.130219% model floating-point operation utilization (MFU).
Its longer measurement reaches approximately 3.115374% MFU and does not improve the existing 3.827772117% experimental decode record.
The objective remains pure LP30 hardware, MFU priority and no minimum tokens per second (TPS).
The measurements use the existing analytical LP30 operator rates and native replay scheduler.
The objective-wide maximum remains unproven.

## Workload and admission

The workload retains Gemma 4 31B decode at a representative 101,000-token context, with memory reserved through 102,000 tokens.
The deployment allocates 440 transformer chips and five complete embedding and language-model head chips.
Every allocated chip remains in the peak denominator.
Useful credit remains 129,261,109,248 floating-point operations per committed output token.
Additional projection replicas, controller instructions, copies and partial-result merges receive no additional useful credit.
Complete-request serving retains its [separate workload and records](lpx-complete-serving-mfu.md).

The [preceding capacity screen](lpx-sram-kv-exchange-search.md#attention-computed-on-home-chips) reserves 54 permanent global key-value (KV) caches and four complete home caches on 125 existing local and head chips.
Local admission reserves all 58 request caches, and global admission retains the original 54 physical caches.
The 56-request cases use two home clients and retain the larger four-client memory allocation.
The additional home allocation preserves the 5 MB safety reserve, whole sixteen-position blocks and a separate one-percent KV watermark per home client and chip.
The minimum full home-chip headroom remains 14,904 bytes.

Each global layer distributes 6,375 blocks across the 125 homes.
The address plan assigns each block a consecutive range of sixteen positions on its recorded home chip.
Each position has one home, and each of the four clients has a separate watermarked cell.
The layout checks all 1,020,000 layer-position ownership assignments and the client/head addresses at each home boundary.

## Numerical attention and dirty appends

The numerical control uses the existing `value_rope_fp8` byte format and learned-key-scale query folding.
Each home computes a normalized partial output and a natural-log log-sum-exp (LSE) value for each query head.
The receiving rank merges those partials using their LSE weights.
Empty partitions return a zero partial and negative-infinity LSE.
The reference independently assembles virtual keys and computes complete-context softmax in 64-bit floating point (FP64).

Twenty-four cases cover contexts of 1, 15, 16, 17, 257 and 101,000 positions with all four KV heads and their eight-query-head groups.
The complete 101K case therefore exercises all 32 query heads and all 125 nonempty homes.
Each case replaces the newest slot with a dirty append and verifies that the older packed bytes remain unchanged.
Two result orders check the merge, including a deterministic random permutation.
The maximum absolute error in 32-bit floating point (FP32) at 101K is approximately `1.28034e-7`.
The FP64 partition-algebra error stays below `2e-12` across all cases.

A deliberately large-logit case reaches approximately `3.17912e-5` FP32 error against FP64.
The monolithic FP32 tiled control has the same error in that case, identifying arithmetic rounding independently of partitioning.
Its explicit FP32 tolerance is `1e-4`, while the other cases use `1e-5`.
An unweighted home average and a missing dirty append fail their negative controls.
These central processing unit (CPU) arithmetic checks do not establish bfloat16 (BF16) kernel compilation or model quality.

## Native computation and communication

The replacement starts from the captured 230-node native global component.
It preserves every query, key and value (Q/K/V) projection prefix, normalization, rotary operation and output projection shard.
It also preserves every dense feed-forward network (FFN) operation and output/FFN reduction.
The replacement changes the 32 attention cores, the four context-parallel combine operations with their routed refinements, and the LSE part of the 32 output operations.
The native output projection remains a nonzero eight-bit floating point (FP8) matrix operation on every output rank.

Four existing projection roots each supply eight query heads.
Each home receives a complete 32,768-byte query.
Only the newest position's owner receives the four 1,280-byte BF16 KV deltas.
Each home executes four serial single-KV-head attention kernels with eight query heads and a 128-key tile.
Their costs use the existing native rates and conservatively price all reserved positions and append/conversion work.
The numerical control masks unused positions and updates only the unique persistent owner.

Every home emits 65,664 bytes containing 32 FP32 partial outputs and LSE values.
Each of the 32 original output ranks receives one 2,052-byte fragment from every home.
The original eight-way merge becomes a 125-way native analytical LSE merge.
All source packing, destination stores, participant descriptor movement and explicit controller budgets retain worker/vector execution module (VXM) reservations.
Every transfer retains routed port, injection, wire and propagation costs.

| Workspace | Used bytes | Existing allocated budget |
| --- | ---: | ---: |
| Largest global rank, including query fanout, QKV, returned fragments and merge state | 1,301,196 | 2,179,712 per active frame |
| Home attention, auxiliary weights, query, partial output, KV message and control | 1,738,580 | 2,048,000 per home |

The global row-zero and row-one frames remain separate.
The home allocation includes the native conservative scratch for one attention head, and the four heads execute serially.
The source fanout copy materializes the charged outgoing payload inside the global frame.
Global active-frame ownership protects that fanout storage and the returned partials until the existing outgoing handoff completes.

The whole-buffer variant holds all 125 home workspaces under one lease until its result transfer finishes.
The grouped variant uses disjoint groups of at most sixteen homes.
Each group acquires its member workspaces atomically and releases them after that group's result transfer finishes.
All group results must be stored before output merging begins.
The grouped policy adds copy and transfer operations while permitting independent groups to overlap across pipeline visits.

Sixteen payload controls cover both policies, global stages zero and nine, and all four home clients.
They execute 64,000 returned head-fragment mappings and check complete query reconstruction, grouped-query head identity, unique KV-delta destinations and dirty FP8 writes.
Incorrect query-group routing and a missing returned fragment fail their negative controls.

## Closed-loop measurements

Every cohort contains two independent requests.
Home clients occupy row one, preserving the original row-zero attention gate and early handoff.
The driver permits one outstanding token job per cohort and waits for its completion before submitting that cohort's following token.
Local and boundary frames follow the existing two-frame assignment, and global frames retain their fixed row ownership.
Every submitted job is drained.

The supplied `best58` schedule uses the earlier 29-cohort ideal-order target stretched by 1.01, with a nominal cohort spacing of approximately 12.177311 microseconds.
The `incumbent` schedule uses the attained 54-request record's native schedule, with approximately 12.334490-microsecond spacing.
Readiness anchors remain explicit; these are selected supplied schedules and do not establish scheduling optimality.

| Schedule and home policy | Requests | Home cohorts | Nominal spacing, microseconds | Aggregate TPS | Short-screen MFU |
| --- | ---: | --- | ---: | ---: | ---: |
| `best58`, whole buffer | 58 | 0, 7, 14, 21 | 12.177311 | 121,499.664 | 2.868219% |
| `best58`, whole buffer | 56 | 0, 14 | 12.177311 | 130,635.606 | 3.083890% |
| `best58`, groups of sixteen | 58 | 0, 7, 14, 21 | 12.177311 | 124,617.963 | 2.941832% |
| `best58`, groups of sixteen | 56 | 0, 14 | 12.177311 | 132,598.149 | 3.130219% |
| `best58`, groups of sixteen | 56 | 0, 13 | 12.177311 | 130,834.922 | 3.088595% |
| `best58`, groups of sixteen | 56 | 0, 14 | 14.000000 | 130,936.305 | 3.090988% |
| `incumbent`, groups of sixteen | 56 | 0, 13 | 12.334490 | 132,332.348 | 3.123944% |
| `incumbent`, groups of sixteen | 58 | 0, 7, 14, 21 | 12.334490 | 129,388.680 | 3.054454% |

Each short screen discards two warm-up rounds and measures four rounds.
The 56-request and 58-request screens count 224 and 232 output tokens respectively.
Their relative half-window drift ranges from approximately -1.21% to +2.37%.
They remain finite screens.

The selected 56-request grouped case discards eight warm-up rounds and measures 32 rounds.
It counts 1,792 output tokens at approximately 131,969.306 TPS and 3.115374157% MFU.
Every cohort contributes 32 measured completions, and all 1,148 submitted cohort jobs drain.
Its relative half-window drift is approximately -0.1709%.
This finite measurement remains below the attained decode record and is not promoted to a new record.

A zero-home 54-request control reproduces approximately 162,146.954317 TPS with the same driver.
The independent validator can reconstruct and rerun that control directly from the archived native graph bodies.
All earlier decode, homogeneous-layout and complete-request records remain unchanged.

## Physical scope and remaining search

NVIDIA's [August 24 LPX technical article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) describes 96 chip-to-chip (C2C) links per LP30, pairwise connections, routing through other LP30 chips and 320-byte compute/communication overlap.
It does not disclose a fixed neighbor map or a compiled schedule for this 445-chip home-attention placement.
The current synthetic routed model charges logical peer collisions and propagation, while its dense logical phases still require a physical neighbor and relay mapping.
The new measurements preserve that existing sensitivity model and do not validate multi-rack wiring or physical banks.

The [portable experiment artifact](../experiments/lpx/home-attention-native-search.json.gz) retains both home policies, all ten global-layer placements, three pipeline captures, numerical and payload controls, finite replay traces and reproduction sources.
The source identity remains `9dfe67e50e9ab76edafd96f37033622e8ae62b2abb3a03f0d233a81920e2407e`.
Production simulation code remains unchanged.

Hierarchical query dissemination and home-side partial-result reduction remain unimplemented candidates.
Their attention and reduction scratch lifetimes must be separated explicitly to fit the existing home budget.
Other home groups, request counts, resource orders, operator placements and inventories remain open.
The subsequent [hierarchical attention and output-staging study](lpx-hierarchical-home-attention-search.md) implements the proposed hierarchical communication and selected complete pipeline replays.
Its supplied schedules remain below the attained decode record.
