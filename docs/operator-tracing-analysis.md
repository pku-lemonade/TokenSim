# Analysis of profiler timeline differences

The existing experiments expose errors in the simulator's operator costs and limits in its execution model.
The large cumulative module-span differences also include host dispatch gaps and GPU kernels waiting for another rank.
Those effects require separate measurements before they can guide calibration.
This investigation changes no production estimator parameters or scheduling behavior.

## Experiment scope

The original captures use Qwen3-8B with tensor parallelism (TP) 1 and Qwen3-30B-A3B with TP 2 on A100 80 GB PCIe GPUs.
Both use BF16, one request, 32 input tokens, three output tokens, eager execution, disabled prefix caching and disabled chunked prefill.
The MoE capture has expert parallelism (EP) disabled.
GPUs 2 and 3 form an NV12 pair; `nvidia-smi nvlink -s` reports twelve active 25 GB/s links per GPU.
The simulator uses `analytical_only`, with no measured operator package and no fitted calibration.

The TokenSim source revision is `be872a598f41b0829fd03379d1c76998c975d644`.
The copied vLLM source revision is `9cf0c96c91c690ad2e3dfc8ddd81f20014565327`, running PyTorch 2.13.0+cu132.
The AISimulate reference checkout is revision `42e5bb8df6029650c4312db0b5cfbd963047a123`.
All new GPU experiments use the independent vLLM working copy and keep artifacts under `/home/youwei/bzh/project/TokenSim/profiler`.

## What the cumulative spans measure

For each matched module, the simulated span is its modeled end minus start.
The observed span extends from its earliest attributed GPU kernel start to its latest attributed GPU kernel end.
Its GPU busy time is the union of those attributed kernel and copy intervals.
The observed span therefore includes gaps between its kernels.
Summing spans across modules and ranks does not give request latency, elapsed execution time or useful arithmetic work.
Communication kernels can remain resident while waiting for a peer, so GPU busy time also requires interpretation.

The original reports have this exact decomposition, in milliseconds:

| Matched-unit quantity | Dense, TP 1 | MoE, TP 2, both ranks summed |
| --- | ---: | ---: |
| Simulated spans | 31.466 | 83.273 |
| Observed spans | 81.564 | 701.468 |
| Observed GPU busy intervals | 32.102 | 490.140 |
| Observed internal gaps: span minus busy | 49.462 | 211.328 |
| Observed busy minus simulated spans | 0.636 | 406.867 |

For Dense, internal gaps account for 49.462 ms of the 50.098 ms cumulative span difference.
The small remaining signed total conceals errors with opposite signs.
Summing each matched unit's absolute busy-time error and dividing by observed busy time gives 10.27%, compared with a signed aggregate difference of -1.98%.
These are diagnostics for one workload and the matched subset; neither quantity establishes end-to-end prediction accuracy.

## Causes independent of expert routing

### Instrumentation changes the observed execution

A new ablation loads each model once and measures repeated `generate` calls while switching module hooks, PyTorch Profiler and shape recording independently.
The measured interval excludes profiler startup, trace serialization and profiler shutdown.
Generated token IDs are identical across the conditions for each model.
There are seven observations in each unprofiled condition and two observations in each profiled condition.

| Condition | Dense median wall time, ms | MoE median wall time, ms |
| --- | ---: | ---: |
| No hooks or profiler, measured before the profiled conditions | 77.339 | 170.600 |
| Module hooks, no profiler | 92.138 | 194.910 |
| Profiler, no hooks, no shape recording | 122.987 | 240.988 |
| Profiler and hooks, no shape recording | 169.486 | 267.305 |
| Profiler, hooks and shape recording | 151.070 | 285.364 |
| No hooks or profiler, measured after the profiled conditions | 90.473 | 186.682 |

The profiled execution has substantial measurement overhead.
The ordering and sample counts do not establish a precise independent effect for shape recording; CPU frequency, scheduling and the before/after drift remain uncontrolled.
These observations must not supply a universal profiler correction factor.

Across the four new annotated captures per model, matched compute-kernel intervals total 31.941–31.950 ms for Dense and 33.152–33.176 ms for MoE.
The original totals are 31.957 ms and 33.203 ms respectively.
Kernel computation is much more repeatable here than module spans and communication-kernel residence.
For example, an original Dense attention invocation has a 577.637 microsecond gap between the cache-write and attention kernels; the attention kernel's correlated CUDA launch occurs late in that gap.
That example provides direct evidence of a host dispatch contribution.

### TP reduction kernels include peer-arrival waiting

The original MoE traces contain 291 native custom all-reduce kernels on each rank.
Their cumulative durations are 2.148 ms on rank 0 and 462.483 ms on rank 1.
Their median durations are 6.880 and 1776.594 microseconds respectively.
The paired calls use the same operation order and both trace files identify the same host and base timestamp.
On those raw clocks, the median start-time difference is 1767.333 microseconds, while the median end-time difference is 0.114 microseconds and the 95th percentile absolute end-time difference is 0.398 microseconds.

The inspected vLLM kernel calls a start barrier before reduction and an end barrier after reduction.
Its barrier implementation spins on peer signals.
The long residence of the earlier rank is therefore consistent with waiting for the later rank to arrive.
The new ablation captures also change which rank has the long reduction durations.
These observations do not support interpreting the original hundreds of milliseconds as network payload-transfer cost or as a permanent slow GPU.

The original run has EP disabled, and a separate observation of router scores gives identical expert histograms on both TP ranks.
Cross-rank expert-count imbalance does not explain this particular reduction asymmetry.
The exact division of host delay among profiler overhead, process scheduling and ordinary eager dispatch remains unmeasured.

### Eager and CUDA Graph collective timing have different boundaries

A separate probe invokes the copied vLLM's native TP all-reduce directly, without loading a model or installing module hooks.
It warms each shape and measures five CUDA-event samples, with 100 calls per eager sample or 100 graph replays containing 20 calls each.
The probe validates eager and captured outputs.
A trace recorded after timing confirms `CustomAllreduce` is enabled and dispatches `cross_device_reduce_1stage`.

| BF16 message | Simulator, microseconds | Eager medians from two launches, microseconds | CUDA Graph medians from two launches, microseconds |
| --- | ---: | ---: | ---: |
| 4096 bytes | 43.099 | 37.100–43.489 | 4.745–4.779 |
| 131072 bytes | 44.040 | 29.184–43.008 | 5.609–5.620 |

Eager CUDA-event measurements include intervals in which host code submits successive calls.
The graph measurements amortize submission across twenty captured calls and remove most of that dispatch exposure.
The existing formula is much closer to this eager measurement than to graph execution.
Its agreement with a small eager case does not validate the bandwidth formula independently.
Calibration must retain the execution regime and measured boundary; fitting one constant to all three of kernel residence, eager invocation and graph execution would mix different costs.

### Attention has a real operator-cost discrepancy

The Dense attention estimate totals 0.368 ms, while its observed module GPU busy intervals total 1.410 ms.
Excluding attributed device copies leaves 1.265 ms of compute kernels, which still exceeds the estimate by about 3.4 times.
The raw module executes separate cache-write and FlashAttention kernels, with additional work in some decode invocations.
The generic estimator includes KV traffic in its byte formula but prices the scope with one fixed 3 microsecond overhead.
It has no measured representation of the selected split-KV kernel's short-sequence efficiency or of all the separate launches and copies in this module.
The missing detail is independent of MoE routing.
The evidence establishes a scope and cost-model discrepancy; it does not yet assign the residual error quantitatively among cache writes, kernel occupancy, launch count and memory efficiency.

### Residual/Norm fusion retains two modeled overheads

The profiler matcher groups an observed fused residual/Norm kernel with its two corresponding simulated primitives.
Grouping preserves their summed modeled latency; it does not make the estimator price a fused kernel.
Both primitive queries still receive their own elementwise overhead.

| Matched fused residual/Norm scopes | Count | Simulated spans, ms | Observed GPU busy time, ms |
| --- | ---: | ---: | ---: |
| Dense | 216 | 1.371 | 0.931 |
| MoE, both ranks | 576 | 3.556 | 1.966 |

Every observed scope in this table has one kernel.
The Dense sum is about 47% high, and the MoE sum is about 81% high.
The discrepancy supports using the target backend's fused operator cost or an explicit fused analytical scope.
Reducing all normalization costs by one shared factor would also change the unfused normalization paths.

### Host dispatch and some physical operations are absent from the ordinary step model

The ordinary backend returns the sum of compute and communication costs, with the A100 PCIe step overhead configured as zero.
It has no explicit CPU launch queue or host-resource scheduling model for these eager forwards.
Ordinary TP traces repeat a representative estimate on each rank and label that assumption `symmetric_tp`.
They cannot reproduce the asymmetric barrier waiting observed above from an independently simulated host dispatch sequence.

Even without profiler or module hooks, measured request wall time remains 77–90 ms for Dense and 171–187 ms for MoE, compared with total simulated durations of 33.888 ms and 42.249 ms.
This comparison includes request handling and unmatched work on the measured side, so it does not quantify CPU overhead alone.
It does show that removing profiling does not resolve the end-to-end timing gap.

The original MoE traces also have one embedding all-reduce per forward, in addition to the 288 layer reductions per rank.
Those three additional calls per rank are unmatched.
The inspected vocabulary-parallel embedding implementation performs that all-reduce, while the ordinary simulated communication composition counts the layer output and feed-forward reductions only.
The ordinary composition also omits an explicit embedding-lookup cost.
LM-head attribution, sampling, metadata preparation and copies leave additional unmatched work.
The current matched subset therefore cannot establish a complete physical-operation inventory.

## Routing finding retained as a separate workload assumption

The original prompt contains 32 repetitions of token ID 42.
A separate gate-output observation finds exactly eight active experts in every prefill layer, with identical histograms on both TP ranks.
The analytical work formula assumes `min(local_experts, routed_pairs)` active experts: 128 for 32 tokens and top-8 routing.
Different token IDs generated with seed 123 touch an average of 54.5625 experts per prefill layer, ranging from 36 to 94.
This effect concerns the total expert-weight footprint, independently of uneven work across ranks.

For one rank's 48 prefill layers, replacing only the assumed active weight footprint with the observed counts changes the diagnostic analytical total from 19.243 ms to 1.686 ms.
The original measured compute total is 3.360 ms, so this substitution would leave a substantial error in the opposite direction.
The 96 decode-layer invocations already use eight active experts in the formula; their modeled compute total is 3.210 ms and their measured compute total is 4.720 ms.
No production cost or parameter is changed using this diagnostic substitution.

## What the AISimulate reference contributes

The inspected reference provides useful contracts and measurement practices:

- `collector/helper.py` records graph versus eager execution and uses warmup, repetition and device events.
- `collector/network/collect_all_reduce.py` collects separate native vLLM eager and graph measurements.
- `crates/core/src/perfmodel/operators/attention.rs` separates the attention table result from additional KV-write and optional normalization/rotary work.
- `crates/core/src/perfmodel/perf_database/attention.rs` retains kernel-source lanes and shape-specific measured tables.
- `crates/core/src/perfmodel/operators/op.rs` represents declared overlap with the maximum of the two branch totals.
- `docs/perf-model/methods/whole-forward.md` distinguishes exact-deployment forward measurements from scheduler-inclusive serving accuracy.

AISimulate's MoE speed-of-light path also does not use routing distribution in its mathematical formula.
Copying that analytical path would not remove the active-expert assumption identified here.
Its measured-data and execution-scope contracts are the relevant references for this investigation.
This analysis does not establish that an AISimulate prediction is accurate for this copied vLLM deployment.

## Correction order and remaining validation

The useful correction order is to establish comparable timing scopes, model backend fusion and missing physical operations, then collect the required shapes and execution regimes.
CPU dispatch and peer-arrival waiting require an execution-model decision before they can be assigned to operator or communication parameters.
An aggregate multiplier fitted to the original module spans would absorb profiler effects, duplicate rank waiting and conceal errors with opposite signs.

Multiple-request GPU validation remains outstanding.
The serial duration-injection experiment validates duration propagation and operation order only.
The available two-request simulation tests do not establish physical correctness of dynamic batching, mixed prefill/decode steps, backpressure or compute/communication overlap.
The observed-trace importer also requires explicit batch/request phase metadata before it can support those comparisons reliably.

Local evidence is retained under `outputs/operator_trace/investigation/` and remains ignored by Git.
`original-trace-audit.json` and its CSV preserve the original report decomposition and input hashes.
`ablation-audit.json`, the ablation manifests and raw traces preserve every observation and condition.
`routes/routes.json` retains router scores and histograms; `routes-audit.json` records the diagnostic active-weight substitution.
`collective-v2/` and `collective-v3/` retain per-rank collective timings; version 3 also retains selected-kernel evidence and output validation.
The standalone analysis and experiment scripts are retained alongside these artifacts.
