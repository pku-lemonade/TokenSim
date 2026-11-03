# Integrated embedding and output-head placement on LP30

The pure-LP30 search evaluates placing tied embedding and language-model head weights on transformer pipeline chips.
This removes dedicated boundary-pool chips from the peak denominator while consuming additional transformer static random-access memory (SRAM).
The objective imposes no minimum throughput and allows inventories beyond 256 chips.
Representative-context decode remains separate from prompt-inclusive complete-request serving.

## Placement and capacity model

The screen considers three weight placements.
`duplicated_ends` stores a full sharded matrix at both pipeline ends, matching the default native boundary ownership.
`first_only` stores one matrix on the initial tensor-parallel group and sends final hidden activations there for output projection and greedy selection.
`last_only` stores one matrix on the final group and sends embedded input activations to the initial group.
Single-end tied-weight ownership remains a research projection without production compiler support.

Tensor parallelism (TP) uses widths 1, 2, 4, 8, 16 and 32.
Decode context parallelism (DCP) divides TP and applies either to full-attention layers or to all attention layers.
Pipeline parallelism (PP) ranges from one through 60 contiguous nonempty stages.
Weights and cache use eight-bit floating point (FP8), while attention and activations use bfloat16 (BF16).
The experimental `value_rope_fp8` cache format uses the three supplied reconstruction kernels and workspace policies.
The [NVIDIA GTC 2026 keynote](https://www.nvidia.com/en-us/on-demand/session/gtc26-s81595/) places native NVFP4 computation on the LP35 roadmap around timestamp 01:37:43.
That roadmap statement supplies no LP30 NVFP4 throughput assumption for this search.
Software packing of lower-precision KV would require separate conversion costs and model-quality evidence.
Timing uses 101,000 tokens and resident key-value (KV) capacity uses 102,000 tokens.
The native 5,000,000-byte safety reserve, block rounding and compiler watermark remain enabled.

The default duplicate-end seed cannot compile on some narrow groups because its boundary matrix already exceeds available capacity.
Footprint extraction temporarily raises the two device-capacity fields to 1 TB so those component vectors can be enumerated.
Every evaluated footprint restores exactly 500,000,000 bytes per chip before admission screening.
No evaluated deployment receives additional memory or bandwidth from this extraction step.
Eighty-four footprint comparisons match the original-capacity compiler output exactly.
Another 105 comparisons match native transformer weight, KV and block vectors against real-capacity proxy footprints and verify the full-vocabulary boundary-weight difference.

| Coverage | Count or range |
| --- | --- |
| TP/DCP/attention-ownership families | 36 |
| Capacity-policy and placement cases | 324 |
| Projected batch/PP deployment points | 79,387 |
| Points with independent batch size at least two | 75,005 |
| Largest independent batch in the relaxed capacity envelope | 54 |
| Largest considered inventory | 1,920 LP30 chips |
| New batch-one native layer references | 105 |
| Additional larger-batch native references | 6 |

## Necessary timing bounds

The [independent-batch study](lpx-pure-lp30-independent-batches.md) supplies the execution-unit latency relaxation and earlier native references.
The new batch-one references price one local layer and one global layer under each supplied kernel.
A full-model lower path sums fifty local layers and ten global layers with host submission and pipeline handoffs removed.
Projection assumes layer-local kernels and direct-route collective costs remain invariant under aligned stage translation and contiguous cuts.
Taking the smallest kernel latency and the largest supplied memory frontier can combine incompatible policies and enlarges the candidate set.
All boundary execution and boundary transfers are free in this envelope, while the transformer denominator includes every allocated chip.

The completed screen leaves 48 batch-one points above the experimental shared-cache decode reference.
Every larger-batch point has an upper bound at most 3.373683489%, below that reference.
The highest unresolved point uses `first_only`, TP16/DCP4, PP12 and an optimistic seventeen resident requests on 192 chips.
Its free-boundary, zero-handoff upper bound is 4.504748893% model floating-point operation utilization (MFU).
This is a necessary bound and does not demonstrate an executable schedule or matching model quality.

## Explicit paired-arena prototypes

Three prototypes retain every native transformer dependency and use two active and two incoming arenas per stage.
They reserve an additional 2,190,464 bytes per transformer chip for the additional scratch and dense residual.
The selected boundary group also reserves the complete FP8 weight shards, normalization weights, live head buffers and conservative communication workspace.
Proxy boundary weights remain charged in addition to the real tied weights.
The combined ledger passes the supplied native block and watermark arithmetic at every stage.
The native transformer proxy separately passes resident admission.
This combined ownership ledger still requires compiler implementation and physical bank placement.

The larger buffers reduce the optimistic seventeen-request PP12 point to sixteen admitted requests in this prototype.
Independent capacity checks reproduce all three prototype ledgers and reject one additional request at each requested PP depth.
Head execution uses the same worker and vector resources as transformer work on the selected physical group.
A separate exclusive head arena protects its live tensors, and explicit local copies or chip-to-chip (C2C) transfers move data into the next owned arena before its producer is released.
The C2C transfers and distributed greedy selection receive no additional useful-token credit.

The resource audit found missing vector execution module (VXM) bindings on native scalar kernels.
The [vector-resource correction](lpx-vector-resource-correction.md) fixes those bindings and preserves operator durations and dependencies.
The earlier three-interval screens remain recorded under their original resource model.
The table replays the highest-scoring earlier interval with corrected bindings, four warm-up completions per client on average and sixteen measured completions per client.
These finite screens do not exclude other issue intervals, client phases or programs.

| Placement | Chips | PP | Clients | Minimum projected SRAM headroom, bytes | Corrected MFU | Measurement-half throughput change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_only` | 176 | 11 | 14 | 16,877,428 | 2.933467056% | +1.321% |
| `first_only` | 192 | 12 | 16 | 16,625,558 | 3.079508240% | -1.758% |
| `last_only` | 160 | 10 | 12 | 2,625,718 | 3.270693263% | approximately zero |

No supplied prototype interval improves the displayed decode record.
Longer corrected controls measure 320 outputs for the existing KV format and 448 outputs for the experimental shared format.
Their displayed MFU values remain 3.039141553% and 3.419600500% respectively.
The resource correction adds scheduling exclusions, so earlier resource-relaxed necessary bounds remain optimistic.
The [prior independent-batch exclusion](lpx-pure-lp30-independent-batches.md) therefore retains its comparison against the revalidated shared-cache reference.

## Evidence and open work

The [compressed experiment record](../experiments/lpx/integrated-boundary-vector-resource-search.json.gz) contains capacity vectors, restoration checks, kernel references, all deployment bounds, unresolved points, combined memory ledgers, original screens, corrected controls, graph fingerprints, validation logs and reproduction sources.
The native graph bodies remain in the local output paths identified by their fingerprints.
The independent-batch artifact supplies the linked execution-unit estimates and earlier native-reference dependency.
The code correction passes 31 focused tests and 876 of 878 full-suite tests, with the same two macOS environment issues as the previous run.

Forty-eight batch-one points remain unresolved under the free-boundary envelope.
Open work includes finer operator scratch lifetimes, mixed cohort widths, alternative issue schedules, heterogeneous stage widths, offload and prefetch, and overlap across operator or layer boundaries.
The shared-cache representation also needs matching model-quality evidence.
Physical unit rates, SRAM banks, inter-rack routes and executable LP30 compiler schedules remain uncalibrated.
The global maximum remains uncertified.
