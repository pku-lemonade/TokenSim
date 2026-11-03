# Independent batches and local-layer cuts for LP30 decode

An analytical native-graph replay reaches 3.820105133525% model floating-point operation utilization (MFU) on 444 LP30 chips with 54 resident requests.
The long replay measures 161,458.530493 aggregate tokens per second (TPS) and drains all 2,214 submitted forwards.
The layout uses tensor parallelism (TP) 4 for every local-window segment and TP32 with decode context parallelism (DCP) 8 for each global layer.
This improves the previous [heterogeneous pipeline record](lpx-heterogeneous-tp-search.md).
Multi-rack timing, heterogeneous compiler execution, physical memory banks and matching shared-cache model quality remain unverified.
The objective-wide maximum remains unproven.

## Workload and accounting

The objective remains pure LP30 hardware with no minimum TPS.
Every allocated transformer and boundary chip contributes to the MFU denominator.
The target remains Gemma 4 31B with fifty local-window layers and ten global-attention layers.
Decode uses a representative context of 101,000 tokens and reserves key-value (KV) storage through 102,000 tokens.
Weights and the experimental `value_rope_fp8` global cache use eight-bit floating point (FP8), while activations and attention use bfloat16 (BF16).
The kernel remains `split_dot`, and no speculative outputs receive credit.
Useful work remains 129,261,109,248 floating-point operations per committed output, against `315e15 / 256` FP8 operations per second per allocated chip.
Prompt-inclusive [complete-request serving](lpx-complete-serving-mfu.md) retains its separate metric and record.

An independent batch of B contains B distinct requests, each producing one output token.
N resident requests form N/B fixed feedback cohorts when N is divisible by B.
A cohort must complete its entire forward within `(N/B) * P` for issue period P.
Committed throughput is `B/P`, and the key-value cache remains charged per request.
Batching does not multiply the admitted request count or credit repeated work as additional tokens.

## Native segment catalog

The experiment evaluates 69 configurations at independent batch widths 1, 2 and 4.
Local segments contain one through five layers at TP2, TP4, TP8 or TP16.
Global segments use TP16/DCP4, TP32/DCP8 or TP32/DCP16.
Fifty-two configurations produce admitted native graphs.
Seventeen fail native configuration or capacity checks under the declared policy.
Every accepted maximum is checked with native admission, and an additional request beyond the raw capacity is rejected.
The selected resident count is rounded down to complete batches.

Every transformer chip retains its 5,000,000-byte safety reserve and batch-scaled vocabulary backcharge.
Two full native scratch, dense-residual and communication workspaces are reserved, together with two batched incoming hidden rows.
Native sixteen-token KV block rounding and the one-percent KV watermark remain.
The head pool retains the full tied embedding matrix, final normalization, vocabulary projection, softcap and greedy selection.
Its separate embedding and logits arenas conservatively reserve four complete query workspaces per independent request in a batch.
The batch-four, three-chip head fails this conservative memory ledger.
The four-chip head remains available for batch four.

| Segment | Batch | Raw resident maximum | Complete-batch residents | Native body, microseconds |
| --- | ---: | ---: | ---: | ---: |
| Local, five layers, TP16 | 1 | 129 | 129 | 12.883808 |
| Local, five layers, TP16 | 2 | 128 | 128 | 13.815914 |
| Local, five layers, TP16 | 4 | 128 | 128 | 16.289946 |
| Global, one layer, TP32/DCP8 | 1 | 54 | 54 | 7.033211 |
| Global, one layer, TP32/DCP8 | 2 | 53 | 52 | 9.543431 |
| Global, one layer, TP32/DCP8 | 4 | 53 | 52 | 14.719760 |

Larger batches amortize fixed operator work, while longer cohort latency and fewer resident cohorts constrain useful throughput.
The catalog supplies explicit timing and memory evidence for that tradeoff.

## Repeated-block bounds

The screen enumerates every ordered partition of five local layers using the admitted catalog segments, followed by one supplied global layer.
The same local/global block repeats ten times in the original logical layer order.
Each head width and batch size receives its own memory and timing ledger.
The necessary bound retains native segment paths, resource service, explicit batched embedding and logits, entry copies and one four-microsecond submission.
Every inter-segment cut costs only the minimum 0.35 microseconds of propagation.
Actual wire service, longer placement distances and shared-resource contention can lower throughput.

| Independent batch | Enumerated points | Points above 3.699683848029% | Distinct surviving segment multisets and head/global choices | Largest optimistic MFU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12,528 | 1,314 | 233 | 5.188687% |
| 2 | 12,480 | 74 | 36 | 4.477581% |
| 4 | 6,240 | 0 | 0 | 3.389481% |

The 31,248-point screen excludes batch four only within the supplied segment graphs, repeated-block layouts and conservative memory policy.
It does not exclude other frame policies, batches, block patterns or cache representations.
All 74 surviving ordered batch-two points are composed with actual physical handoffs.
A sample of 41 batch-one ordered points extends the local-layer cut search.
The resulting 115 full-program compositions retain every native non-host service and dependency.
Sixty-six batch-two compositions have necessary full-graph bounds below the new 3.820105133525% record.
The remaining eight receive 172 specified cyclic-order models with complete-cohort feedback.
Their best numerical result is 3.734480685717% MFU.
Other cyclic orders remain unexcluded.

## Exact resource equivalence for faster search

Several resource keys occur on exactly the same positive-occupancy nodes in a periodic template.
Those resources impose identical exclusion constraints for repeated copies of that template.
The search groups keys by their complete native node-incidence lists and keeps one canonical key per identical list.
Arena leases retain their original identities and release descendants.
Native programs and final replay retain every original physical resource key.

For the large batch-two control, 17,047 physical keys reduce to 1,173 equivalence classes.
Positive resource uses reduce from 7,351,389 to 379,332.
Eight recorded calendars reproduce every original node start and end time bit for bit.
An independent comparison also reproduces the complete linear-program constraint arrays bit for bit, including endpoint indices, period distances, lags, sinks and lower bounds.
The reduction changes the cost of constructing the search model and leaves its timing constraints intact.

## Period refinement and observed improvement

A separate batch-one refinement tests 124 calendars across three- and four-chip head pools.
It supplies 114 valid physical-and-arena calendars for fixed-order linear programs with complete-request feedback.
The four-chip-head variant reaches 3.755486345825% MFU at 151,577.504896 TPS over 1,728 measured outputs.
The long control drains all 2,214 submitted forwards.

The broader catalog then replaces the TP2 local segment with TP4 and tests the local cuts in different orders.
The selected block contains one local layer at TP4, two local layers at TP4, two local layers at TP4 and one global layer at TP32/DCP8.
Ten blocks allocate 120 local-stage chips and 320 global-stage chips.
Four additional head chips give 444 allocated chips and forty transformer stages.
The native global segment limits residency to 54 requests.
The selected calendar searches contribute 494 additional trials and 220 numerical cyclic-order models, including the batch-two cases.
Together with period refinement, this turn contributes 618 calendar trials and 334 numerical models.

Its optimized numerical issue period is approximately 6.193540204718 microseconds.
A relative pacing margin of `1e-7` produces the replay period of 6.193540824072 microseconds.
The complete forward spans 330.190636723483 microseconds.
Independent sweeps validate all original resource intervals, frame holds, release reachability and full-cohort feedback.
The construction preserves every native service and adds only readiness delays and concrete frame names.

| Warm-up rounds | Measured rounds | Measured outputs | Drained forwards | Aggregate TPS | MFU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 16 | 864 | 1,134 | 161,458.530493 | 3.820105133524% |
| 8 | 32 | 1,728 | 2,214 | 161,458.530493 | 3.820105133525% |

Each of the 54 requests completes 32 measured outputs in the long replay.
The measurement halves differ by approximately `6.48e-14` relative.
No submitted forward remains outstanding after drain.
Fresh native admission for every selected role retains 378,990 watermark-adjusted bytes on the limiting global chip.

## Coverage gaps and hardware evidence

A global-only TP64/DCP16 attempt is rejected because the native attention planner requires 32 query heads to be divisible by TP64.
The combine path assumes whole output heads per tensor rank.
This implementation restriction does not establish hardware infeasibility.
Output-channel fragments within a head require explicit geometry, combine-payload, projection-weight and memory validation before wider global groups can participate.

The directly opened [NVIDIA architecture description](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) reports sixteen LP30 chips per tray and distinguishes dynamic random-access memory (DRAM) through fabric expansion from DRAM through the host processor.
The page describes inter-rack chip-to-chip (C2C) connectivity and supplies no calibrated transport timing for this experiment.
The simulator continues to use its declared 1.15-microsecond fallback beyond 256 chips.
These multi-rack results remain conditional on that assumption.

The [MLSys SHIP paper](https://proceedings.mlsys.org/paper_files/paper/2026/hash/9c20f16b05f5e5e70fa07e2a4364b80e-Abstract-Conference.html) discusses adapting large pipelines to changing prefill/decode ratios and context lengths.
This experiment evaluates fixed-context decode and does not certify the complete-serving optimum.
Different resource orders, local/global block patterns, batch widths, output-channel partitions and physical compiler programs remain open.
Mixed per-stage batch widths require explicit joins, row ownership, role-specific frame periods and cohort-aware feedback before their throughput can be compared.

## Evidence

The [compressed experiment record](../experiments/lpx/heterogeneous-batch-and-phase-search.json.gz) retains all 69 catalog receipts, 52 native component bodies, all 31,248 screen points, 115 composition receipts, calendar and numerical optimization records, exact resource-equivalence controls and the selected full native replay witness.
Its portable checks reconstruct the selected constraint arrays from original physical keys and independently validate dependency timing, exclusive intervals, arena releases, memory blocks and drained per-client traces.
The simulation source identity remains `ae29c716bbf5693a8c9d70df1ee5c6435c3e93af93781bcc5aff57c2e8074986`.
The existing-format decode, homogeneous experimental decode and complete-request serving records remain unchanged.
