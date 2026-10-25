# Deferred work on operator timing and execution boundaries

The user deferred changes to host submission, collective rendezvous and operator calibration to separate pull requests.
The current tracing pull request adds visualization and retains the observations below without changing those execution or estimation paths.
Observed eager-runtime gaps and profiler overhead require an explicit measurement boundary before they can be assigned to a simulation component.
The exported timelines expose a substantial difference between modeled service and observed execution time.
Matching operator names and sequence does not establish elapsed-time accuracy.
All three simulations use analytical estimates without measured operator tables.
This analysis uses the comparison files recorded in `outputs/operator_trace/improvement/figures/manifest.json` and verifies them against all twelve retained GPU timing captures.
The retained `outputs/operator_trace/improvement/figures/audit_differences.py` reproduces the numerical audit and writes source hashes into `timing-difference-audit.json` in the same directory.
It does not change simulation parameters or modify the captured durations.
Raw captures, figures and audit scripts remain experiment artifacts excluded from Git.
The [analysis of the initial captures](operator-tracing-analysis.md) records the earlier investigation and implemented tracing corrections.

## Dense: missing dispatch gaps dominate elapsed time

The controlled four-request Dense case provides the clearest evidence because its selected profiled wall time, 86.632 ms, is close to the unprofiled median of 88.042 ms.
The unprofiled medians drift from 84.037 ms before collection to 94.126 ms afterward, so this ratio does not prove zero profiling overhead.
The simulation still takes only 34.878 ms.

| Quantity, four-request Dense | Duration, ms |
| --- | ---: |
| Simulated elapsed time | 34.878 |
| Observed device window, including unattributed work | 80.409 |
| Union of observed device activity intervals | 37.324 |
| Observed device gaps | 43.085 |
| Gaps before entry into the next activity's CUDA API | 34.481 |
| Gaps inside that API call | 6.160 |
| Gaps after the API returns and before device execution | 2.445 |

The arithmetic identity is `80.409 - 34.878 = 43.085 + (37.324 - 34.878)` milliseconds.
Device gaps account for about 94.6% of that signed elapsed-time difference.
This identity does not assign every gap to a particular software component.
Every device gap is paired with the correlated CUDA runtime or driver API for the following activity.
The 34.481 ms before API entry directly demonstrates late host submission in the captured run.
Separating Python dispatch, scheduler work, operating-system delays and residual profiler overhead requires additional host observations.
API residency and post-API gaps cannot be interpreted as pure launch cost without examining synchronization and dependency behavior.

The result is stable across the three timing repetitions.
Device activity totals 37.312, 37.313 and 37.324 ms; gaps total 41.790, 44.765 and 43.085 ms.
The small device-service variation does not account for the large pipeline stretching.

The ordinary backend sums compute and communication costs, and the selected A100 configuration has `step_overhead_us: 0.0`.
Its per-operator analytical launch floor does not represent an independently scheduled host submission queue.
The simulator therefore places dependent operators consecutively while eager vLLM repeatedly leaves the GPU without submitted work.
Measured device-service tables alone would leave these observed gaps unrepresented.

The matched Dense subset contains 37.102 ms of device activity, 16.787 ms of gaps inside module envelopes and 24.568 ms between envelopes, over a 78.457 ms window.
The remaining observed device activity is only 0.222 ms, so unmatched GPU work is too small to explain the 45.531 ms elapsed difference.
These subset quantities have different boundaries from the full device window and must not be added to it.

The prefill layer-0 image illustrates the same effect locally.
Its simulated layer span is 314.075 microseconds, the observed layer span is 1130.883 microseconds, and observed device activity totals 369.023 microseconds.

## Operator errors remain after removing gaps

Matched Dense service totals 34.878 ms simulated and 37.102 ms observed, giving a signed error of about -6.0%.
Summing each matched pair's absolute service error gives 15.1% of observed service.
Excluding Attention as requested leaves an absolute-error ratio of 11.4%.
Opposite signed errors cancel in the aggregate.

| Four-request Dense matched scope | Simulated service, ms | Observed device service, ms |
| --- | ---: | ---: |
| QKV projection | 3.900 | 4.277 |
| Attention output projection | 2.717 | 3.184 |
| FFN up projection | 14.551 | 13.944 |
| FFN down projection | 7.451 | 8.057 |
| Attention and FFN Norm scopes, residuals grouped where observed | 1.582 | 0.973 |

Residual/Norm matching groups the simulated primitives but retains their separately estimated overheads.
An observed fused kernel therefore remains overestimated even though the displayed module scope matches.
Token selection also remains an approximation: simulated argmax totals 0.012 ms while the attributed sampler device scope totals 0.116 ms.
These errors require backend-specific operation scopes and matching measurements.
Attention parameters remain unchanged under the user's requested scope.

## MoE: peer waiting and service errors have different effects

The MoE simulation takes 42.481 ms, while the observed device windows are 189.903 ms and 190.520 ms on the two ranks.
The unprofiled generation median remains 169.798 ms, so profiling alone cannot explain the elapsed difference.
The selected capture is still perturbed, with a profile/baseline ratio of 1.142.
Both ranks' service and wait sums must remain separate from elapsed latency.

The comparator attributes 108.718 ms of cumulative peer-arrival waiting in 582 inspected custom all-reduce calls across both ranks.
Rank 0 contributes 99.942 ms and rank 1 contributes 8.776 ms.
Much of this residence overlaps the other rank's host submission delays.
Subtracting the rank-summed wait from elapsed time would be invalid.
The original timelines retain these waits deliberately; the adjusted service panel removes the qualified waiting intervals.
Six NCCL all-gather calls retain unknown waiting and have no adjusted service estimate.

The simulator repeats an identical representative schedule on both TP ranks and labels it `symmetric_tp`.
It has no independently modeled host arrival sequence for those ranks.
The implemented wait correction establishes attribution in the comparison report; it does not implement asymmetric arrivals and barriers in the simulator.
The large rank asymmetry therefore remains a limitation in simulated pipeline behavior.

The communication service boundary also differs.
The simulated 582 custom all-reduce counterparts total 25.266 ms across ranks, about 43.4 microseconds per call.
Their observed after-arrival residence totals 4.176 ms, about 7.2 microseconds per call, before counting staging copies.
After-arrival residence still contains execution and barrier overhead and does not represent pure payload transfer time.
The generic topology formula and its software floor cannot be calibrated against raw waiting residence as though both measured network transfer.
The simulator configuration uses a DGX A100 topology approximation; the measured pair uses A100 PCIe GPUs connected by an NVLink bridge, so topology provenance also requires care.

MoE prefill has a separate active-expert assumption.
The current one-request case uses repeated token ID 42.
The earlier gate observation for this same input finds eight active experts in every prefill layer on both TP ranks; the formula assumes 128 active experts.
This concerns total expert work and weight traffic, independently of imbalance between ranks.
The new capture does not contain routing histograms, so those histograms remain evidence from the separate routing observation.

| Expert compute only, both TP ranks summed | Simulated, ms | Observed compute intervals, ms |
| --- | ---: | ---: |
| Prefill | 38.487 | 6.687 |
| Decode 1 | 3.210 | 4.672 |
| Decode 2 | 3.210 | 4.666 |

These rows exclude the MoE module's TP all-reduce.
Prefill overestimation and decode underestimation have opposite signs.
Routing assumptions do not explain the missing host gaps, fused Norm overestimation or communication timing-boundary mismatch.

## Scope and acceptance for follow-up pull requests

The trace comparison can associate the modeled operation sequence with observed modules, preserve real gaps, and identify qualified collective waiting.
The results do not establish accurate eager execution timelines.
The four-request case validates a controlled batch; it does not validate mixed-phase continuous batching or multi-request PD/AF execution.

Reducing the elapsed discrepancy requires an explicit eager-versus-graph execution contract and a host submission model that interacts with GPU work and collective arrivals.
Measured tables must retain backend fusion, execution regime and timing boundaries.
Collective rendezvous should create waiting from per-rank arrival times without charging it again as payload service.
Validation should compare per-forward elapsed time and dependencies in addition to cumulative module totals, using equal injected service durations to isolate scheduling behavior.
The existing analytical launch floor must be reconciled with any added host model to avoid charging submission overhead twice.

| Deferred work | Required evidence |
| --- | --- |
| Serving-runtime and profiler timing boundaries | Separate host submission, API residency and device execution; compare against unprofiled baselines and report drift. |
| Per-rank collective rendezvous | Preserve shared clocks, create waits from participant arrival times and avoid charging peer waiting as payload service. |
| Backend fusion and operation costs | Match the physical fused scope and retain graph/eager regimes in measured data. |
| MoE active-expert assumptions | Retain routing histograms or identify the synthetic distribution, and validate prefill and decode separately. |
| Complex serving pipelines | Exercise mixed continuous batching, PD and AF dependencies with equal injected device-service durations before fitting costs. |

The paired image exporter follows the layout of [Charon Figure 8](https://arxiv.org/pdf/2605.17164#page=10).
Its selection-relative alignment translates each side's selected origin while retaining the same time scale, original gaps and qualified cross-rank offsets.
The exported layout does not imply the prediction accuracy reported for Charon.
