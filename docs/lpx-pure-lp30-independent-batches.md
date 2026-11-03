# Independent request batches in pure LP30 decode

The optimization scope requires pure LP30 accelerators, prioritizes useful model floating-point operation utilization (MFU) and imposes no throughput floor.
This study asks whether a homogeneous batch of at least two independent requests can improve the 3.419600500% experimental shared-cache decode record.
Every batch row owns its own request and key-value (KV) state.
The numerator credits one committed target token per row.
Speculative verification positions within one request have a different KV-reuse pattern and remain outside this study.

## Covered model class

Transformer tensor parallelism (TP) uses widths 1, 2, 4, 8, 16 and 32.
Decode context parallelism (DCP) divides TP and applies either to full-attention layers or to all attention layers.
Equivalent DCP-one ownership modes are evaluated once.
Pipeline parallelism (PP) ranges from one through 60 contiguous stages.
The transformer uses a separate tied embedding and language-model head pool with at least three LP30 chips.
All allocated chips contribute to the denominator.
The pool's 1,409,286,144 bytes of eight-bit floating-point (FP8) matrix weights require at least three 500,000,000-byte chips before other storage is considered.

Weights and cache use FP8, while attention and activations use bfloat16 (BF16).
The cache representation is the experimental `value_rope_fp8` format.
Its three supplied kernels are `assemble`, `split_dot` and `split_dot_packed`.
The screen retains the native 5,000,000-byte safety reserve, KV block rounding and compiler watermark in the capacity vectors.
It measures fixed decode at 101,000 tokens and reserves KV through 102,000 tokens.
The [main search record](lpx-maximum-mfu-search.md) defines the uncalibrated execution-unit rates and fabric assumptions.

| Coverage | Count or range |
| --- | --- |
| TP/DCP/ownership families | 36 |
| Current-source capacity policy cases | 108 |
| Nonempty capacity families | 35 |
| Family/batch pairs with batch at least two | 619 |
| Largest batch permitted by the relaxed capacity envelope | 54 |
| Admitted batch/PP envelope points with batch at least two | 28,502 |
| Largest considered inventory | 1,923 LP30 chips |
| Native two-layer kernel references | 78 |
| Full 60-layer projection controls | 4 |
| Native attention-key comparisons | 4,416 |
| Native workspace endpoint checks | 210 |

The TP-one family cannot fit a native single-layer stage before execution under these weights and buffers.
Thirty-two families admit at least one complete batch of two or more requests within the supplied capacity envelope.
The frontier search reaches native infeasibility before its 1,024-request guard in every supplied policy.

## Capacity and latency bounds

The [native workspace formulas](../TokenSim/af/runtime/lpx_memory.py) and [shared-cache workspace formulas](../TokenSim/config/kv_storage.py) contain nonnegative batch-dependent byte terms.
Weights, per-request KV vectors and KV block sizes remain fixed when independent batch width changes.
The batch-one footprint therefore supplies an optimistic resident-capacity envelope for larger batches within these policies.
The screen takes the greatest resident count from any of the three workspace policies at each PP depth.
It then rounds that count down to a whole number of cohorts of width B.
Batch widths exceeding the largest batch-one resident count cannot admit one complete cohort under these policies.
For each supplied batch and layout, the relaxed forward path does not depend on resident count.
Its throughput bound grows with resident count, so the maximum complete-cohort population also covers smaller populations.
Native compilation checks compare the weights, KV, block and boundary vectors and the buffer endpoints against the batch-one vectors.

The [LP30 analytical pipeline](../TokenSim/operator_data/analytical/lp30.py) prices each operator with execution-unit services, dependent query-prefix work and invocation overhead.
Replacing the tiled pipeline by its largest execution-unit service supplies a lower latency while preserving the prefix and overhead.
The initial envelope removes every boundary operation, collective and PP handoff.
It also chooses the smallest priced global-attention cost among the three supplied kernels independently of their capacity policy.
These timing and memory relaxations can combine incompatible policies and enlarge the candidate set.

N denotes resident requests, and B denotes independent requests per forward.

```text
N_upper = floor(maximum_batch_one_residents / B) * B
aggregate_TPS_upper = N_upper / lower_forward_latency
MFU_upper = 100 * aggregate_TPS_upper * useful_target_FLOPs_per_token / allocated_peak
```

Throughput is measured in tokens per second (TPS).
The N requests contain N/B cohorts, each generating B tokens per completed forward.
This accounts for batch width without multiplying resident count by B again.

## Native layer references and the remaining point

The initial bound leaves 26 family/batch shapes and 370 deployment points above the incumbent.
Seventy-eight native references evaluate those shapes under all three kernels.
Each reference contains one local-attention layer and one global-attention layer with the original target geometry and native output-vector communication.
The reference removes host submission and cross-stage handoff costs from its isolated layer bodies.
A full-model lower path sums fifty local-layer bodies and ten global-layer bodies.
The projection assumes layer-local kernels, collective costs and dependencies are invariant under contiguous cuts and aligned stage translation in the supplied direct-route fabric model.
Four complete 60-layer controls reproduce this sum after PP handoffs and host submission are removed.
The controls include batch two and three, TP16 and TP32, and both attention-ownership modes.

The native references reduce the unresolved set to TP16/DCP4, PP10, batch two and fourteen resident requests.
Its transformer-only, zero-handoff path is 263.644198 microseconds.
With three minimum boundary chips counted and all boundary work removed, that path permits a 3.422311236% MFU upper bound.
The `assemble` and `split_dot_packed` alternatives already have lower zero-handoff bounds at this point.

An exhaustive partition count admits exactly one PP10, fourteen-request cut in each relaxed capacity policy: ten stages of six layers.
Larger-batch workspace cannot add another cut to those envelopes.
Every smaller complete batch-two population has at most twelve residents and a zero-handoff MFU bound of at most 2.933409631%, so only the fourteen-request population needs the retained-handoff check.
A complete native transformer capture retains all PP handoffs and removes host submission and every external-pool operation.
Its critical path is 273.098198 microseconds, yielding a 3.303839087% MFU upper bound even when boundary execution and contention are free.
Every larger separate boundary pool increases the allocated peak, so this optimistic exclusion also covers those pool sizes.
The full three-chip-pool control admits fourteen requests and has a 3.144301642% fixed-DAG bound.
Its native admission report and program fingerprints remain in the experiment record.

## Result and remaining scope

All 28,502 points in this supplied independent-batch class have necessary bounds below the shared-cache decode incumbent.
The largest remaining bound is 3.406617750% at TP32/DCP8 on all attention layers, PP20, batch five and an optimistic fifty resident requests on 643 chips.
That value comes from the relaxed execution-unit envelope and is not an attained throughput measurement.
This batch study preserves its 3.419600500% timing comparison reference.
The existing-format 3.039141553% decode record and the separate prompt-inclusive serving metric also remain unchanged.

The [compressed evidence](../experiments/lpx/pure-lp30-independent-batch-bounds.json.gz) retains the current-source capacity vectors, operator keys and services, native reference fingerprints, complete projections, full-model controls, final exclusions, validation records and reproduction sources.
The reproduction inputs retain every capacity-policy configuration and the baseline reference configuration consumed by the source files.
Native DAG bodies remain in the local output paths identified by their input and body hashes.
The numerical checks do not calibrate physical execution units, static random-access memory (SRAM) banks, LP30 compiler support or inter-rack timing.
The shared-cache representation also requires matching model-quality evidence.

This exclusion requires homogeneous transformer widths, a separate boundary pool, full resident KV and the supplied sequential operator and output-vector communication model.
Colocated boundary weights, mixed cohort widths, speculative verification, alternative operators, overlap across model layers, heterogeneous placements and fabric/host-memory offload need separate bounds and implementations.
The global maximum remains uncertified.
The [periodic resource-calendar study](lpx-periodic-resource-calendar-search.md) revisits capacity after removing the transformer vocabulary backcharge and excludes 28,545 supplied independent-batch points.
It also reports a separate batch-one periodic decode result of 3.557996616% MFU.
