# Hierarchical LP30 home attention and global output staging

Progress date: 2026-11-04, Asia/Shanghai.
The selected hierarchical attention schedules do not improve the 3.827772117% experimental decode model floating-point operation utilization (MFU) record.
Their largest short-screen result is approximately 3.005298% MFU.
The corresponding longer measurement reaches approximately 2.953492% MFU.
A separate output-staging prototype fits inside an existing charged reserve, but the supplied fixed resource orders remain below the record.
The objective remains pure LP30 hardware, MFU priority and no minimum tokens per second (TPS).
All times and utilization values use the existing analytical operator rates and native replay scheduler.
The objective-wide maximum remains unproven.

## Preserved workload and memory

The experiment retains Gemma 4 31B decode at 101,000 positions, with storage reserved through 102,000 positions.
The 445 allocated LP30 chips include 440 transformer chips and five complete embedding and language-model head chips.
Every allocated chip remains in the peak denominator.
Useful credit remains 129,261,109,248 floating-point operations per committed output token.
Copies, duplicated projections, control budgets and additional merges receive no extra useful credit.
Complete-request serving retains its [separate workload and records](lpx-complete-serving-mfu.md).

The [preceding home-attention study](lpx-home-attention-native-search.md) admits 54 permanent global key-value (KV) caches and four additional complete caches on 125 existing local and head chips.
This experiment preserves that allocation, the separate 5 MB safety reserve and one-percent KV watermark.
Each home still reserves 2,048,000 bytes for attention, auxiliary weights, messages and control.
The 56-request cases use two home clients while retaining the larger four-client allocation.
The smallest full home-chip headroom remains 14,904 bytes.

## Hierarchical query and result movement

The experiment divides the 125 home chips into groups of at most four, eight or sixteen chips.
The lowest-index chip in each group acts as its leader.
The original global query, key and value (Q/K/V) projection prefixes, normalization, rotary operations, output projections and dense feed-forward network (FFN) operations remain.
Only the distributed attention cores and their partial-result communication and merging change.

The direct-query policy sends the original eight-query-head shards from four global roots to every home.
The tree-query policy sends those shards to group leaders, which forward complete queries to their followers.
Only the newest position's recorded owner receives the four KV deltas.
Every home executes the same four serial attention kernels with the existing experimental shared-cache format and a 128-key tile.
Their costs retain conservative reserved-position, conversion and append charges.

Each follower sends a complete 65,664-byte partial result to its leader.
The leader merges its group's normalized outputs and log-sum-exp (LSE) values in 32-bit floating point (FP32).
An additional explicit vector budget prices normalization and LSE emission.
The direct-result policy sends one 2,052-byte partial per group to each original output rank.
The root-result policy sends eight-head 16,416-byte fragments to each of four global roots, which redistribute the required heads locally.
The original output ranks then merge the group partials and execute their retained nonzero output projection shards.

The primary [FlashInfer recursive-attention tutorial](https://docs.flashinfer.ai/tutorials/recursive_attention.html) describes the associative composition of normalized attention outputs and LSE states.
The LP30 placement and timing are design inferences here, and the tutorial's GPU implementation does not calibrate LP30 execution.
All numerical controls in this experiment consistently use natural-log LSE values.

## Scratch reuse and controls

Every group acquires its member home workspaces atomically and retains them through the group's result transmission.
A leader's query relay completes transmission before the leader begins attention in the reused scratch region.
All group attention and source partial copies finish before the leader gathers inputs into its retired attention scratch.
The leader gathers and merges complete FP32 partials, with all copies and vector work charged on native worker/vector execution module (VXM) resources.

| Home phase | Maximum bytes |
| --- | ---: |
| Native conservative attention and existing messages | 1,738,580 |
| Sixteen-home query relay | 1,138,180 |
| Sixteen-home FP32 partial reduction | 1,762,836 |
| Existing allocation | 2,048,000 |

The global-root result policy also reuses global frame memory.
Root retention and redistribution begin only after all groups return, which implies that every global query send has completed.
The largest declared global phase across the twelve screened combinations uses 1,593,352 bytes within the existing 2,179,712-byte active frame.
Global row-zero and row-one frames remain separate.
These are explicit logical scratch lifetimes; compiler bank placement remains unverified.

Sixty numerical cases cover five context lengths, all four KV heads and all three group sizes.
The 101K cases collectively cover all 32 query heads.
They compare FP32 two-level merges against independent complete-context 64-bit floating-point (FP64) attention over the same eight-bit floating point (FP8) cache bytes.
The largest FP32 absolute error is approximately `1.10239e-6`, and the largest FP64 algebra error is approximately `2.44249e-15`.
Three all-empty controls retain zero outputs and negative-infinity LSE without producing not-a-number (NaN) results.
Discarding group LSE weights fails its negative control.

Forty-eight payload controls cover all twelve stage-zero combinations and all four home clients.
They check complete query reconstruction, unique dirty KV writes, all 125 home partials, all group partials and all 32 output heads.
The largest error against the FP64 merge of their tagged partials is approximately `1.47677e-6`.
Incorrect query-group routing and a missing gather fragment fail their negative controls.
The controls validate arithmetic and payload ownership; they do not establish model quality or compiled bfloat16 execution.

## Layer and pipeline measurements

The stage-zero screen captures twelve combinations of group size, query policy and result policy.
Their isolated completions include one four-microsecond host submission and range from approximately 16.710383 to 28.101103 microseconds.
Isolated latency does not establish pipeline throughput.
Four selected configurations capture all ten global-layer placements and complete pipeline graphs.
The portable artifact retains the twelve stage-zero captures and forty selected layer captures, including repeated stage-zero bodies where the two collections overlap.

Each cohort contains two independent requests, with home clients on row one.
The existing driver permits one outstanding token job per cohort and drains every submitted job.
All pipeline cases use the incumbent supplied schedule with approximately 12.334490-microsecond nominal cohort spacing.

| Group size | Query policy | Result policy | Requests | Aggregate TPS | Short-screen MFU |
| ---: | --- | --- | ---: | ---: | ---: |
| 16 | Direct | Global roots | 56 | 122,158.971 | 2.883783% |
| 16 | Direct | Global roots | 58 | 113,998.527 | 2.691141% |
| 16 | Direct | Direct | 56 | 127,306.402 | 3.005298% |
| 16 | Tree | Global roots | 56 | 119,815.750 | 2.828467% |
| 8 | Direct | Direct | 56 | 121,444.953 | 2.866928% |
| 8 | Direct | Direct | 58 | 114,924.811 | 2.713008% |

Each short screen discards two warm-up rounds and measures four rounds.
The 56-request and 58-request cases count 224 and 232 output tokens respectively.
Their relative half-window drift ranges from approximately -3.79% to +1.11%.
These remain finite supplied-schedule screens.

The longer sixteen-home direct-query/direct-result run discards eight warm-up rounds and measures 32 rounds.
It counts 1,792 output tokens at approximately 125,111.864 TPS and 2.953491826% MFU.
Every cohort contributes 32 measured completions, and all 1,148 submitted cohort jobs drain.
Its relative half-window drift is approximately -0.3971%.
The reduced number of returned partials does not yield a higher tested MFU after gather, merge, additional routing and contention are included.
Other group layouts, schedules and scratch policies remain open.

## Separate global output-staging prototype

The original global active frame remains leased through its outgoing pipeline handoff.
The new prototype copies the completed hidden output into a separate staging bank and releases active storage after that copy.
The staging bank remains leased until the original routed handoff completes.
The original compute durations, resource occupancies, input frames and routed transfer costs remain.
The prototype adds 640 native copies and forty zero-service control nodes to the 98,087-node parent.
It changes twenty active-frame release targets and adds copy-readiness dependencies to twenty outgoing transfer nodes.

The staging allocation reassigns part of the already charged 1,027,072-byte transformer vocabulary guard.
The transformer graph contains no vocabulary-head operators, and the complete head remains on its five separate chips.
The guard's full memory charge, both original native active frames, 5 MB safety and KV watermark remain.
Two banks per request row use 47,104 bytes per global chip including control.
Thirty-two banks per request row use 692,224 bytes per global chip including control.
Both fit within the charged guard.

The lifted incumbent optimized resource order gives approximately 3.818178115% conditional MFU with either bank count.
A further search lifts all seventy supplied incumbent calendar orders into the 32-bank program.
Thirty-two numerical models succeed, with a greatest fixed-order target of approximately 3.819043233% at a 12.362682-microsecond period.
Thirty-eight models are reported infeasible by the solver without independent Farkas certificates.
The artifact retains every lossless seed vector, every successful optimized vector and its primal/dual evidence.
The selected successful schedule also receives a full periodic resource and lease interval audit.
These are numerical fixed-order targets, and no attained native closed-loop MFU is assigned to output staging.
The tested successful orders remain below the decode record; other resource orders remain open.

## Separate projection-broadcast capacity proposal

The existing context-parallel graph repeats identical Q/K/V projections across each eight-rank global group.
A new capacity-only proposal keeps one projection root per group and broadcasts its results to the other seven ranks.
Removing the seven unused FP8 QKV matrices per group would release 6,936,330,240 weight bytes across the ten global stages.
All non-QKV weights and native buffer charges remain.
The proposal also reserves 155,648 bytes within the already charged vocabulary guard for two query/KV fanout frames, two incoming frames and control.
It keeps the separate 5 MB safety reserve.

The [conditional capacity receipt](../experiments/lpx/qkv-projection-broadcast-capacity.json) assigns nonuniform whole-sixteen-position KV ranges inside each head group.
It charges a separate one-percent KV watermark per request and rank.
The derived 56-request global allocation fits this declared policy, while 57 and 58 requests exceed its summed per-group block limits.
Existing conservative local 58-request native admissions remain sufficient for the unchanged local 56-request layout.
Fresh deduplicated-weight provenance, query broadcast routing, numerical append ownership, nonuniform attention timing and complete request-feedback replay remain unverified.
No TPS or MFU is assigned to this proposal.

## Evidence and remaining work

The [portable artifact](../experiments/lpx/hierarchical-home-attention-search.json.gz) retains the hierarchy graphs, arithmetic and payload controls, seven pipeline traces, output-staging prototypes, fixed-order evidence and reproduction sources.
Production simulation code remains unchanged at source identity `9dfe67e50e9ab76edafd96f37033622e8ae62b2abb3a03f0d233a81920e2407e`.
The current synthetic route model does not establish a fixed physical neighbor or relay mapping for this 445-chip deployment.
Compiler execution, physical banks, multi-rack calibration and matching experimental-cache quality remain unverified.

Fresh output-staging calendars and other cyclic resource orders remain candidates.
The release policy also needs native request-feedback replay before any attained performance claim.
The projection-broadcast proposal requires complete native computation, communication and memory validation.
Other transformer/head ratios, request counts, operator placements and inventories remain open.
