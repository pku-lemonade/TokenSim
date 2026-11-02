# Search for maximum MFU at 100K context

Progress date: 2026-11-03.
The objective is to maximize useful model FLOP utilization (MFU) when serving Gemma 4 31B with a 100,000-token input.
The search includes inventories beyond 256 LP30 chips.
Observed throughput, memory admission, model bounds and hardware evidence have separate meanings in this record.
The search remains open.
The existing-format non-speculative decode incumbent reaches 3.03914% MFU on 163 chips with ten resident requests.
It places transformer layers on 160 chips and the tied embedding and language-model (LM) head on three additional chips.
Assistant-aware draft and verification replays provide conditional acceptance scenarios and explicit transports of the repository's measured 100K GPU acceptance.
The measured GPU KV and assistant precision differ from some LP30 configurations.
An experimental shared global-cache format reaches 3.41960% conditional decode MFU with fourteen resident requests; matching model quality remains unverified.
The 5 MB per-chip safety reserve remains enabled.

## Metric and workload

The current timing probes measure decode at a representative fixed context of 101,000 tokens and reserve key-value (KV) cache for 102,000 tokens.
Each independent request produces one committed token per forward.
The probes use eight-bit floating-point (FP8) weights and KV and bfloat16 (BF16) attention and activations.
The fidelity inputs use general matrix multiplication (GEMM) efficiency 0.30 and attention efficiency 0.50.
Prefill and external token-control latency remain outside these decode probes.
The separate-head and assistant programs explicitly include distributed greedy token selection.
Stochastic sampling needs its own distribution and correction costs.
A complete serving result also needs the prefill workload, arrivals, admission policy and request completion behavior.

Useful work is measured in billions of floating-point operations (GFLOP).
Each output token requires 61.392027648 GFLOP of linear matrix work plus 67.869081600 GFLOP of attention matrix work at this context.
Throughput is measured in tokens per second (TPS).
The default MFU denominator includes the FP8 peak of every allocated LP30 chip:

```text
useful_GFLOP_per_token = 129.261109248
allocated_peak_FLOP_per_s = chip_count * 315e15 / 256
MFU_percent = 100 * aggregate_TPS * 129.261109248e9 / allocated_peak_FLOP_per_s
```

The current linear-work ledger excludes learned Q/K normalization scales from matrix parameters.
Historical raw records used 129.261180928 GFLOP per token, including 71,680 scalar-derived FLOPs.
Those records remain preserved under the historical convention in the portable record.
The correction changes MFU by approximately 0.00005545% relative and leaves the displayed candidate rankings unchanged.
The evidence also records a precision-specific correction using the catalog BF16 peak assumption.
Replicated projections, padding and speculative draft work receive no additional useful-token credit.
Candidates using graphics processing units (GPUs) must include all allocated GPUs in the hardware denominator under an explicitly stated precision convention.
The 3,400 TPS per-request reference remains a separate comparison; it is not imposed as a universal optimization constraint.

## Resident decode candidates with 16-row scratch

`TP` is tensor parallelism, `PP` is pipeline parallelism, and `DCP` is decode context parallelism on full-attention layers.
`N` counts resident requests across the complete deployment; microbatch size is one in this table.
Initial requests have evenly spaced phases, and subsequent token forwards obey both token feedback and a global issue interval.
The bounds use a captured directed acyclic graph (DAG) of model operators.

| Chips | TP / PP / DCP | N | Issue interval (us) | Aggregate TPS | TPS per request | MFU | Fixed-DAG MFU bound |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 176 | 16 / 11 / 4 | 9 | 23.2 | 43,103.4 | 4,789.3 | 2.5727% | 2.5852% |
| 208 | 16 / 13 / 4 | 11 | 19.05 | 52,493.4 | 4,772.1 | 2.65118% | 2.65122% |
| 256 | 16 / 16 / 4 | 12 | 17.69 | 56,529.1 | 4,710.8 | 2.3197% | 2.3208% |
| 320 | 32 / 10 / 8 | 20 | 14.0 | 63,925.8 | 3,196.3 | 2.0986% | 2.3150% |

The 208-chip point uses layer counts `[5, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2]`.
Its minimum physical static random-access memory (SRAM) headroom is 9,029,558 bytes per chip under the native reservation ledger.
Its isolated forward duration is 209.547104 us.
The 19.05 us issue interval yields an 11-request feedback period of 209.55 us.
The tighter interval leaves a 0.001382% relative throughput gap to this candidate's fixed-DAG bound.
An earlier 19.1 us interval produced 52,356.0 aggregate TPS, reproduced with an eight-round warm-up and 32 measured rounds.
The two measurement halves agree to floating-point precision for this periodic point.

The 176-, 256- and 320-chip screening runs use four warm-up rounds and 16 measured rounds.
The 208-chip table entry uses eight warm-up rounds and 32 measured rounds.
The 320-chip point has approximately 0.38% variation between measurement halves.
Its cross-rack propagation remains an uncalibrated legacy assumption.
These measurements do not certify LP30 compiler support or physical hardware throughput.

Phasing materially affects the replay scheduler.
An earlier TP16/PP16/DCP4 probe with all requests submitted at time zero reached approximately 16,270 aggregate TPS, while explicitly phased execution reached over 56,000 TPS on the same inventory.
Searches that fix all arrival phases to zero can miss useful schedules.
The [earlier batch comparison](lpx-batch-concurrency-mfu.md) retains its original TP32/PP8 settings and measurement windows.

## Decode-sized scratch reservation

Changing the memory policy's `prefill_chunk` from 16 to 1 sizes reusable scratch for one query row while preserving weights, cache precision, context reservation, operator rates and the 5 MB reserve.
The change applies to the current single-token decode workload.
Larger prefill chunks require a separate workspace policy or phase-aware allocation and admission.
These decode measurements do not include that prefill service.

The capacity search admits ten requests at TP16/PP11/DCP4 under this policy.
The selected partition is `[5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 3]`.
Native admission independently confirms the selected partition.

| Issue interval (us) | Aggregate TPS | TPS per request | MFU | Fixed-DAG MFU bound |
| ---: | ---: | ---: | ---: | ---: |
| 21.40 | 46,729.0 | 4,672.9 | 2.78914% | 2.87249% |
| 20.78 | 48,123.2 | 4,812.3 | 2.87236% | 2.87249% |

Both probes discard 80 completions and measure 320 subsequent completions.
Every cohort contributes 32 measured completions, and the two measurement halves agree to floating-point precision.
The isolated step remains 207.791104 us.
The tighter issue interval gives a 207.8 us feedback period for each of the ten cohorts.

The busiest chip holds 201,625,088 bytes of weights, 287,375,360 bytes of reserved KV and 7,961,548 bytes of buffers and auxiliary state.
It retains 3,038,004 bytes of physical headroom.
The buffer ledger includes 1,640,576 bytes of live scratch, 1,048,576 bytes of communication workspace and the unchanged 5,000,000-byte reserve.
The 16-row policy rejects the same ten-request partition through native KV admission: 287,375,360 required bytes exceed the compiled 286,511,104-byte KV budget.
That comparison includes the compiler watermark and distinguishes resident-lease admission from raw SRAM occupancy.

The tighter periodic placement has zero overlaps on declared occupied resources after folding every resource interval into its 20.78 us period.
Its longest local-stage span is 19.208267 us, so reusable compute scratch has one locally executing cohort at a time.
Including incoming and outgoing handoff propagation expands the longest stage envelope to 21.364267 us.
Two handoff cohorts can therefore coexist in the communication arena.
A logical intermediate-stage suballocation of four hidden activations and four full-head FP32 attention states uses at most 305,664 bytes, within the existing 1,048,576-byte workspace reservation.
Boundary-stage envelopes do not overlap.
These checks establish a conditional schedule and buffer arrangement in the declared resource model; physical bank placement and executable compiler support remain unverified.

Capacity-only sensitivity runs also use one-row scratch with 1 MB and zero additional reserve.
They remain separate assumptions and do not contribute the observed best value reported above.

## Capacity coverage and bounds

The capacity screen evaluates 36 combinations of tensor width, context width and attention-layer ownership mode.
Tensor widths are 1, 2, 4, 8, 16 and 32.
Context width divides tensor width, and ownership modes are full-attention-only or all attention layers.
Equivalent context-width-one ownership modes are evaluated once.
Pipeline depth ranges from one through 60, with a maximum inventory of 1,920 chips and a resident-request search limit of 256.
The screen finds 1,403 layouts admitted by its extracted footprint vectors.
None of the feasible frontiers reaches the resident-request search limit.

The partition search checks all contiguous cuts through a bitset dynamic program for every resident count.
It retains native KV block rounding and the compiler watermark.
It does not infer every larger PP depth from a feasible minimum depth.
The native footprint uses the maximum extracted protocol-buffer envelope across stages.
Capacity exactness applies to those supplied vectors; layouts with different per-stage buffers, standalone embedding or head stages, heterogeneous tensor widths or different precision remain outside this screen.
Every timed candidate also passes native resident-KV admission using its selected partition.

Timing bounds retain all DAG dependencies and relax physical resource contention.
Resource-demand bounds separately retain total occupied time on each named physical resource.

```text
latency_limited_TPS = resident_requests / DAG_critical_path_seconds
resource_limited_TPS = batch_size / max_resource_seconds_per_forward
aggregate_TPS_bound = min(latency_limited_TPS, resource_limited_TPS)
```

These are necessary steady-throughput bounds for one captured DAG and resident count.
They constrain changes to scheduling for that DAG, while changes to kernels, partitions, memory policy or model precision require new bounds.
A small gap to one candidate's bound does not establish a global optimum.
Finite-window results additionally allow for requests already in flight at the measurement boundary.

A separate batch captures 60 supplied cases with cohort widths 1, 2, 4 and 8 across selected TP/DCP/ownership families.
It chooses the highest resident-per-chip PP depth within each selected family and retains complete cohorts.
The supplied-case list records this heuristic selection explicitly.
These timing bounds do not cover every admitted PP depth or every feasible partition.

Eleven additional captures complete batch-one timing references for all 27 families with nonempty capacity frontiers.
The [partition projection driver](../scripts/bound_lpx_mfu_layouts.py) removes direct PP wire and propagation costs while retaining graph dependencies.
It projects the resulting lower latency across contiguous cuts under an explicit assumption that per-layer kernels and intra-stage collective costs are invariant under stage translation.
Its 71 family/batch groups project onto 3,661 PP points for the 16-row memory envelope.
The baseline batch-one references at PP11, PP13 and PP16 agree on a 198.611104 us zero-handoff critical path for TP16/DCP4.
Only three projected points exceed the earlier 2.65118% incumbent, and all three already have direct fixed-DAG bounds.
Projection exactness remains conditional on kernel invariance and the supplied footprint envelope.

For the one-row memory envelope, all admitted family/batch-width combinations number 321.
An additional 64 native profiles complete the missing batch-width references.
Together, 135 family/batch reference groups project onto 6,996 PP points.
A conservative arithmetic lower bound also uses useful linear work over the effective FP8 matrix rate plus attention work over the effective BF16 matrix rate, divided across TP chips and multiplied by cohort width.
Native model layers execute sequentially within each cohort under the current operator model.
All 321 widths have necessary bounds below the 3.03914% separate-head incumbent within the supplied footprint envelope.
The greatest zero-handoff relaxation is approximately 3.00525%.
Reserve sensitivities at 1 MB and zero additional reserve each admit 329 widths.
Four additional reference profiles close the missing sensitivity cases, and all 329 widths are bounded below the incumbent for each policy.
These sensitivity policies remain separate assumptions.
Mixed cohort widths, alternative kernels, other footprint envelopes and heterogeneous placements remain open.

An additional capacity attempt tests TP64, TP128 and TP256 with inventories through 15,360 chips.
The native planner rejects those tensor widths because its query-head partition requires TP to divide the model's 32 query heads.
Those failures identify software coverage gaps for head-feature sharding and heterogeneous operator widths.
They do not establish a hardware upper bound on wider deployments.

## Separate embedding and LM-head pool

The transformer proxy retains the original model geometry and replaces its vocabulary width with the transformer TP width.
The native transformer graph omits its embedding lookup and embedding collective, final normalization, LM head and logits gathering.
The external pool explicitly replaces those boundary operations using the full vocabulary.
Dummy boundary weights remain conservatively charged to the transformer chips.
A reserve backcharge of 1,027,072 bytes per transformer chip preserves the full model's 1,048,576-byte communication envelope for one query row.
The 5,000,000-byte safety reserve remains additional to that backcharge.

The selected transformer partition is TP16/PP10/DCP4 with six layers per stage and ten resident requests.
The three head chips store uneven vocabulary shards of 87,382, 87,381 and 87,381 columns.
Their tied embedding and output weights cover all 262,144 vocabulary entries once.
The program includes embedding broadcast, cross-pool transfers, final normalization, full-vocabulary logits, softcap and distributed greedy selection.
The memory ledger retains 3,021,876 bytes of minimum transformer headroom and 23,607,484 bytes of minimum head-pool headroom.

| Transformer chips | Head chips | Requests | Isolated step (us) | Aggregate TPS | TPS per request | MFU | Fixed-DAG MFU bound |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 160 | 3 | 10 | 212.052722 | 47,156.5 | 4,715.6 | 3.03914% | 3.03925% |
| 160 | 4 | 10 | 211.103234 | 46,901.2 | 4,690.1 | 3.00426% | 3.03430% |

The three-chip replay uses a 21.206 us issue interval, eight warm-up rounds and 32 measured rounds.
Both measurement halves agree to floating-point precision.
The four-chip control retains its wider default issue interval and needs no tighter replay because its necessary bound is below the three-chip observed result.
One- and two-chip head pools exceed SRAM capacity.
Direct fixed-component bounds cover head pools from three through nine chips.
Ten or more head chips cannot exceed the incumbent even when their execution cost is set to zero, because the transformer-only critical path and larger allocated peak already impose a lower MFU upper bound.
This establishes the preferred head count for that fixed transformer component under the declared model.

A broader separate-head capacity screen covers 36 supplied TP/DCP/ownership combinations.
Thirty-three have nonempty resident frontiers under the proxy footprint.
Six are excluded by a transformer arithmetic bound, and 27 receive native transformer-only timing references.
Those references project onto 1,419 PP points while removing every PP handoff and all external I/O execution time, and retaining at least three head chips.
Only TP16/PP10/DCP4 remains above the incumbent under that relaxation.
The 323-chip TP32/PP10/DCP8 configuration admits 23 requests but has a zero-handoff MFU bound of approximately 2.75492%.
The supplied footprint envelope and cut-invariant operator assumptions apply to these projections.
Other tensor widths, mixed-width cohorts and alternative operator placements remain outside this finite comparison.

Reusable head scratch uses an explicit arena resource.
An acquisition remains live until its designated completion node finishes after queueing and contention.
The replay kernel validates that every release is a descendant of its acquisition.
Tests include delayed completion, forward dependency references and snapshot reconstruction.
The old and extended kernels produce exactly equal JSON on a frozen native ten-request replay.
The program snapshots retain node-body and input hashes.
Physical SRAM bank placement and executable compiler scheduling remain unverified.
The scheduler now deduplicates changed-resource notifications using dense flags and a vector.
Candidate requests retain their existing total priority order.
A frozen three-program assistant replay gives exactly equal completion records and statistics for all 5,822 resources across nine completions.
That local comparison takes 57.6 seconds with the prior bookkeeping and 39.9 seconds with the new bookkeeping.
This is one host timing comparison, and does not change the modeled LP30 costs.

## Scratch and scheduling checks

A temporary timing instrumentation of the unchanged replay kernel inspected isolated per-stage execution spans.
For the 208-chip candidate, the longest local-resource span is 16.631505 us, below its 19.1 us issue interval.
The corresponding maxima for the 176-chip and 256-chip candidates are 19.208267 us and 14.054743 us.
These spans support examining a schedule with one locally executing cohort per PP stage.
The diagnostic excludes cross-stage handoff nodes and does not establish incoming/outgoing buffer lifetimes, SRAM bank assignments or compiler instruction legality.
Concurrent schedule verification must include those states before treating a replay candidate as deployable.

A dependency-based placement diagnostic also includes the direct handoff wire and propagation intervals in both participating stages.
Its longest stage envelope is 18.787505 us, below the tighter 19.05 us interval.
The isolated placement has no overlapping occupied intervals on any declared resource, and the remaining shared host service is 4 us per forward.
The 11-request feedback period also contains the complete isolated forward.
These conditions support a periodic placement within the declared resource model with one stage envelope active per cohort period.
They do not establish physical SRAM bank placement, an executable compiler program or omitted token-control costs.

## Hardware and algorithm avenues still open

The current [NVIDIA architecture description](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) distinguishes dynamic random-access memory (DRAM) through fabric expansion logic from DRAM through the host central processing unit (CPU).
The retrieved table lists up to 512 GB and 256 GB respectively per tray of 16 language processing units (LPUs).
It also describes chip-to-chip (C2C) connectivity across racks.
An assumption that every LP30 external-memory access must traverse host Peripheral Component Interconnect Express (PCIe) is insufficient to exclude fabric-memory prefetch.
The source does not supply a calibrated end-to-end bandwidth or latency model for that memory path.

The [SubQuadratic Disaggregation (SQD) paper](https://arxiv.org/abs/2609.13134) places dense attention on DRAM-based devices and sliding-window attention plus the feed-forward network (FFN) on SRAM-based devices.
That ownership pattern is directly relevant to Gemma's mixed attention layers.
The branch's global-attention disaggregation prototype provides a starting point, while concurrent GPU/LPU replay and a consistent mixed-device MFU denominator still require work.

The [pinned Gemma configuration](https://huggingface.co/google/gemma-4-31B-it/raw/842da3794eaa0b77d5f08bae87a17459d91ff475/config.json) specifies shared global K/V projections and a 0.25 partial rotary factor.
The [reference attention implementation](https://raw.githubusercontent.com/huggingface/transformers/e0eb2810ac1b2caf6d082f247d8d37162ecf67a0/src/transformers/models/gemma4/modeling_gemma4.py) applies different normalization and rotary operations to the two projected states.
The baseline therefore retains separate transformed K and V caches.
Storing a shared representation, or sharing only the unrotated dimensions, is an algorithm candidate whose compute cost and quantized-model accuracy remain unverified.

The [pinned Gemma 4 31B assistant configuration](https://huggingface.co/google/gemma-4-31B-it-assistant/raw/627c5ec1458b9086b841a91e0512fd31fd2fbbf1/config.json) specifies a four-layer, hidden-width-1024 assistant with all four layers sharing KV.
The [Transformers assistant documentation](https://huggingface.co/docs/transformers/model_doc/gemma4_assistant) describes reuse of target KV and target hidden states, with no independent assistant prefill.
The assistant parameter ledger totals 469,518,596 parameters and matches the official safetensors metadata.
The standalone native model planner does not represent cross-model KV sharing.
The new assistant composition explicitly runs attention at target KV owners and returns attention outputs to the assistant pool.
The [assistant search record](lpx-assistant-mfu-search.md) retains its parameter ledger, conditional MFU probes and remaining validation requirements.

The [Gemma technical report](https://arxiv.org/html/2607.02770v1) also describes a 37.5% global-cache reduction from tied projections and partial rotary position encoding.
A geometry-only screen retains a full value vector and the rotary quarter of the key vector, reducing global storage from 1,024 to 640 eight-bit elements per KV head and token.
It preserves native block rounding, the compiler watermark, full communication reservations and the 5 MB safety reserve.
It leaves reconstruction costs, additional codec buffers and native admission for the new representation unimplemented.

| Transformer TP / PP / DCP | Independent K/V requests | Value plus rotary-key quarter | Value-only storage |
| --- | ---: | ---: | ---: |
| 16 / 10 / 4 | 10 | 15 | 18 |
| 16 / 10 / 8 | 9 | 13 | 16 |
| 32 / 10 / 4 | 13 | 20 | 24 |
| 32 / 10 / 8 | 23 | 34 | 40 |

These are optimistic storage-geometry frontiers, not observed admitted deployments or MFU results.
A deterministic numerical check confirms the unquantized dot-product identity when the unrotated key scale is folded into the query.
The same check gives a counterexample to bitwise equivalence after independently quantizing K and V to FP8.
It uses random vectors, not Gemma weights or a quality benchmark.
A shared representation therefore needs a defined quantization policy, reconstruction work and measured quality before it can join the observed incumbent comparison.

The [experimental shared-KV study](lpx-shared-kv-mfu-search.md) charges physical scale bytes, native allocation and reconstruction work for a different global-cache representation.
Its split-dot candidate reaches 3.41960% conditional decode MFU on 163 chips with fourteen resident requests.
Matching-cache model quality remains unverified, so the existing-format incumbent stays separate.

The [Open-TQ-Metal paper](https://arxiv.org/abs/2604.16957) studies fused compressed-domain attention with Gemma 4 31B among its models.
Its Apple Silicon results motivate a separate KV representation study; they do not calibrate LP30 conversion costs or establish 100K quality equivalence for this workload.

| Avenue | Remaining evidence |
| --- | --- |
| Partitions and inventories | Optimize timing across custom cuts, embedding/head placement, heterogeneous tensor widths and replica packing |
| Scheduling and batching | Sweep phase offsets, mixed cohort widths, stage handoff buffers and legal periodic schedules |
| External memory | Model fabric-memory and host-memory paths separately, including shared ports, synchronization and repeated KV reads |
| GPU/LPU ownership | Replay concurrent global-attention or other layer-specific disaggregation with all-device accounting |
| Precision and KV representation | Include conversion work, quantization metadata and measured quality constraints |
| Speculative decoding | Calibrate 100K accepted-prefix distributions, extend draft length and slot sweeps, and verify concurrent target-stage lifetimes |
| Workload changes | Separate common-prefix sharing and cache reuse from independent-prefix serving |
| Complete serving | Include prefill, evolving context, arrivals, sampling/control and completion latency |
| Hardware calibration | Measure unit rates, physical routes, clock behavior, memory bank constraints and compiler support |

## Reproduce and extend

The search driver reuses native configuration, profile capture, SRAM admission, the C++ replay scheduler and checkpointed experiment receipts.
It builds an unchanged replay kernel in a versioned temporary directory.
The default configuration file is `experiments/lpx/fidelity-study.json`, with the article injection convention and attention efficiency 0.50 applied explicitly by the driver.

```bash
python scripts/search_lpx_mfu.py \
  --mode capacity \
  --max-chips 1920 \
  --request-limit 256 \
  --output outputs/lpx-maximum-mfu/capacity

python scripts/search_lpx_mfu.py \
  --tp 16 \
  --kvp 4 \
  --pp 13 \
  --resident-requests 11 \
  --initial-spacing-us 19.1 \
  --issue-interval-us 19.1 \
  --warmup-rounds 8 \
  --measured-rounds 32 \
  --output outputs/lpx-maximum-mfu/208-chip
```

`--batch` changes the cohort width, and `--layer-counts` supplies a custom contiguous partition together with an explicit resident count.
`--kv-parallel-layers all` extends DCP to sliding-window attention.
`--cases` accepts a nonempty JSON list of case objects with the `MFUCase` fields, and `--workers` runs independent cases in worker processes.
`--mode bound` captures and admits each case and records necessary bounds without a concurrent throughput replay.
Bounded cases receive status `bounded` and no observed MFU.
`--inter-rack-us` supplies an explicit propagation sensitivity, while all multi-rack results remain marked unvalidated.
`--max-chips` limits the declared inventory; it does not certify a physical inter-rack topology.
Results retain per-cohort completions, per-request rates, time-per-token percentiles and measurement-half convergence.

The decode-sized probe uses a configuration whose `memory.prefill_chunk` is 1 and whose `memory.reserve_bytes` remains 5000000.
The driver already accepts that configuration through `--config`; no operator-cost adjustment is required.
The portable record retains the exact memory policy and both observed issue intervals.

The separate-head probe replaces the native boundary operators and preserves the full communication reservation:

```bash
python scripts/search_lpx_io_pool.py \
  --tp 16 \
  --kvp 4 \
  --pp 10 \
  --pool-chips 3 \
  --resident-requests 10 \
  --issue-us 21.206 \
  --warmup-rounds 8 \
  --measured-rounds 32 \
  --output outputs/lpx-maximum-mfu/separate-head
```

The partition projection command reads cached, fingerprint-checked native profiles:

```bash
python scripts/bound_lpx_mfu_layouts.py \
  --progress experiments/lpx/maximum-mfu-search-progress.json \
  --additional-results outputs/lpx-maximum-mfu-20261102/bounds-extra/results.json \
  --profile-root outputs/lpx-maximum-mfu-20261102 \
  --output outputs/lpx-maximum-mfu/partition-projections.json
```

The [portable progress record](../experiments/lpx/maximum-mfu-search-progress.json) retains capacity rows, supplied timing-bound cases, observed probes, profile identities and schedule diagnostics.
Profile caches and full node timing traces remain under the ignored `outputs` directory.

## Validation status

Black 24.2.0 formatting and its check pass for all 558 Python files in the required project paths.
All 49 directly related shared-KV, MFU, fidelity, I/O-pool, assistant and context-parallel tests pass in the targeted run.
The latest full-suite run executes 827 tests, with an unchanged macOS path-resolution assertion failure and an unchanged GPU-runner test requiring Linux `/proc` and `nvidia-smi`.
The reporting dependency is installed, and all four reporting tests pass.
The two platform-dependent checks remain unresolved on this macOS host.
The run includes the generalized remote-attention workspace validation.
The failing GPU monitor/runner source and tests match the branch's committed versions.
An earlier code milestone's commit hook reported an advisory hardcoded-value scan timeout after two seconds.
The physical shared-KV code milestone commits successfully without that advisory message.
