# AgentX hardware validation and AISimulate comparison

The validation artifacts are retained in `results/agentx-accuracy-20261026/` and are excluded from Git because they contain full replay logs.
`manifest.json` freezes the trace hash, commands, reference identities, and code revisions; `comparison.json` retains errors and regressions for every available metric.
The [AgentX guide](agentx.md#comparison-with-the-inferencex-reference) presents one selected complete run per hardware.
The [committed result summary](data/agentx-selected-results.json) preserves their source values, result hashes, capacity policies, and the B300 capacity comparison for review without the full replay logs.
The results below cover the published DeepSeek-V4-Pro deployments and do not establish accuracy for other models or configurations.
Agreement with AISimulate is a differential check; acceptance of a change requires independent arithmetic invariants, target-framework evidence, or the stated hardware comparison.

The source reading covers the [SemiAnalysis article](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat), [agentic inference view](https://inferencex.semianalysis.com/inference?i_seq=agentic-traces), [InferenceX](https://inferencex.semianalysis.com), its benchmark APIs, and the retained launch and request artifacts.
The exact hardware points are B200 run 440239 and B300 run 439522.
The B200 point differs from run 439967 linked elsewhere in the article.
Pareto curves combine deployment configurations, so comparisons use individual runs and actual server launch arguments.

## Complete replay results

All four runs load the full 393-tree corpus and use live warmup, a 3,600-second closed-loop issue window, a 30-second grace period, SimpleCPU offload, and seeds 0 and 42.
The baseline is `89fa355ddd457f7bcebb685229ae5f930fa91f1d` and the corrected simulator is `acffa30`.
Each before/after pair uses identical simulation arguments except its result path.
The B200 concurrency is 196 and the B300 concurrency is 384.
Both deployments use eight GPUs, tensor parallelism 1, data/expert parallelism 8, and multi-token prediction (MTP) depth 3 with synthetic mean acceptance 2.49.
The source trace SHA-256 is `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`.
Every run completed with no unfinished requests.

Signed error is `100 * (prediction / measurement - 1)`.
ITL is the per-request mean inter-token latency after the first output token.

| Hardware and metric | Measured | Baseline | Corrected | Baseline error | Corrected error |
| --- | ---: | ---: | ---: | ---: | ---: |
| B200 output tokens/s | 3,599.26 | 2,111.51 | 2,076.03 | -41.33% | -42.32% |
| B200 TTFT p50, seconds | 3.3630 | 7.6166 | 7.8884 | +126.48% | +134.56% |
| B200 TTFT p90, seconds | 16.2065 | 46.9411 | 44.9659 | +189.64% | +177.46% |
| B200 ITL p50, seconds | 0.034248 | 0.071693 | 0.074760 | +109.34% | +118.29% |
| B200 ITL p90, seconds | 0.050246 | 0.110693 | 0.102778 | +120.30% | +104.55% |
| B300 output tokens/s | 6,360.26 | 5,307.11 | 5,453.81 | -16.56% | -14.25% |
| B300 TTFT p50, seconds | 4.2653 | 4.6493 | 4.6538 | +9.00% | +9.11% |
| B300 TTFT p90, seconds | 18.9158 | 20.0433 | 20.3670 | +5.96% | +7.67% |
| B300 ITL p50, seconds | 0.045212 | 0.050030 | 0.051119 | +10.66% | +13.07% |
| B300 ITL p90, seconds | 0.049485 | 0.069116 | 0.057215 | +39.67% | +15.62% |

The corrected implementation improves B300 throughput and tail ITL while worsening several median/TTFT metrics and B200 throughput.
These results do not support a claim of uniform accuracy improvement.
The comparison command's `--require-no-regression` gate rejects this matrix.
CPU unit tests establish arithmetic and execution invariants; this table evaluates serving predictions against archived hardware measurements.
The four runs used different pinned CPU cores concurrently, so their wall times do not establish simulator speedup.

## AISimulate implementation and executable checks

The source comparison uses the local AISimulate revision `dcadbc5087bc039bbe2672e7888440b792813720` and distinguishes it from the [v0.13.0 tag](https://github.com/ai-dynamo/aisimulate/releases/tag/v0.13.0), commit `289f8fb22599bfece5619c84753a3f8f65d33e38`.
The native Python differential probe uses the isolated installed package `aisimulate==0.13.0`.
The installed wheel SHA-256 is `3d3815659bd6d692d747f551d3bc3d71b9b2cb206bb33d783bdbe245c389bdb0`.
No reference repository source was modified.

| Concern | AISimulate implementation | TokenSim decision and validation |
| --- | --- | --- |
| Multi-axis extrapolation | [Joint-coordinate utilization transfer](https://github.com/ai-dynamo/aisimulate/blob/dcadbc5087bc039bbe2672e7888440b792813720/crates/core/src/perfmodel/perf_database/perf_interp.rs) evaluates complete collected coordinates and selects nearby anchors in joint log space | Bind every recursive outer coordinate; seven native-engine single-anchor comparisons agree with a shared analytical scaler |
| Batch timing | [Timing-provider contract](https://github.com/ai-dynamo/aisimulate/blob/dcadbc5087bc039bbe2672e7888440b792813720/crates/core/src/engine/timing.rs) carries exact prefill totals and decode active KV totals; providers may retain a mean-based fallback | Preserve context histograms and charge one ragged decode invocation; generic mixed/prefill fusion still requires physical-call evidence |
| Speculation | [Conditional draft sampler](https://github.com/ai-dynamo/aisimulate/blob/289f8fb22599bfece5619c84753a3f8f65d33e38/crates/core/src/engine/common/speculative.rs) draws sequentially and preserves acceptance before output clipping | Use explicit mean/conditional/empirical modes and committed-round deterministic draws; RNG streams differ, so distribution equivalence is the relevant contract |
| Runtime KV bytes | [Runtime memory resources](https://github.com/ai-dynamo/aisimulate/blob/289f8fb22599bfece5619c84753a3f8f65d33e38/crates/core/src/perfmodel/fpm/resources.rs) bind observed bytes to deployment settings and reject repeated graph deductions | Import deployment-bound per-rank bytes, keep analytical deductions as diagnostics, and test mismatches, rank coverage, exhaustion, and physical-slot allocation |
| Dependencies | [Weka typed edges](https://github.com/ai-dynamo/aisimulate/blob/dcadbc5087bc039bbe2672e7888440b792813720/crates/core/src/replay/loadgen/weka.rs) distinguish dispatch, completion, and joins | Audit against the benchmark's pinned AIPerf harness before changing TokenSim's completion-only branch release |
| Whole-forward measurements | [Forward models](https://github.com/ai-dynamo/aisimulate/blob/dcadbc5087bc039bbe2672e7888440b792813720/docs/perf-model/methods/whole-forward.md) support measured forward interpolation and fitted models | W5 remains unimplemented; request-level serving aggregates cannot substitute for scheduled-batch timing measurements |
| Latency reporting | [Request report](https://github.com/ai-dynamo/aisimulate/blob/dcadbc5087bc039bbe2672e7888440b792813720/crates/core/src/replay/report.rs) excludes singleton outputs from ITL and uses rounded-rank percentiles | Preserve the same per-request ITL denominator; retain InferenceX's linear percentile convention and inverse-latency interactivity |

The native differential probe creates one synthetic attention anchor using AISimulate's analytical speed-of-light (SOL) latency.
Both simulators use that same analytical function and the same anchor, removing hardware-coefficient differences from this consistency check.
AISimulate's generation-attention wrapper averages five integer contexts between 90% and 110% of the requested context.
The differential probe applies the identical five contexts to TokenSim before comparing results.
This smoothing is part of the probe contract and has not been introduced into TokenSim's scheduled-step timing.

| Heads, batch, context | AISimulate, microseconds | Bound TokenSim, microseconds | Unbound TokenSim, microseconds |
| --- | ---: | ---: | ---: |
| 32, 16, 1024 | 8.734161 | 8.734161 | 8.734161 |
| 128, 16, 1024 | 8.836295 | 8.836295 | 8.543779 |
| 128, 4, 64 | 0.166393 | 0.166393 | 0.187872 |
| 16, 1, 2048 | 1.089536 | 1.089536 | 1.096944 |

All seven tested queries agree within `1e-8` microseconds after binding.
The separable shapes can also agree without binding; the cross-head/context cases expose the defect.
This validates the anchor-coordinate arithmetic and does not measure physical GPU error.
TokenSim's recursive multi-anchor interpolation still differs from AISimulate's joint-neighbor algorithm, so the single-anchor result does not establish general interpolation equivalence.

The following local AISimulate Rust test filters passed: `perf_interp::tests` (44), `engine::common::speculative::tests` (7), `perfmodel::fpm::resources::tests` (14), `engine::common::perf_model::tests` (6), and `explicit_overlap_join_and_background_lower_to_typed_edges` (1).
These 72 tests execute the reference implementation and supplement TokenSim's 606 passing tests.

Executing the installed package's `_require_agentic_g2_scope` rejects DP8 with MTP and host offload because speculation must be disabled.
The same gate rejects DP8 without MTP because host offload requires attention DP1.
DP1 without speculation passes this early scope check, which does not constitute a successful serving replay.
AISimulate's ordinary HBM-only replay has broader support; the restriction applies to its agentic host-offload path.
An AISimulate throughput comparison for the exact DEP8/MTP/offload target is therefore unavailable from that supported path.
The retained `aisimulate-support-gate.json` records the actual rejection messages.

## Observed-capacity investigation

The B200 analytical budget provides 25.6010 GiB of KV capacity per rank.
The [hardware log search](https://inferencex.semianalysis.com/api/v1/server-log-search?id=440239&q=Available%20KV%20cache%20memory&limit=3) records 32.27 GiB on rank 0, and all eight ranks report the same 2,742,953 equivalent KV tokens.
The diagnostic assumes uniform rounded byte capacity across ranks and retains a 0.005 GiB rounding uncertainty.
It replaces available KV bytes without changing any timing parameters, hit ratios, workload settings, or scheduler policy.
This follows AISimulate's distinction between observed available bytes and an analytical non-KV allocation breakdown.

| B200 metric | Analytical capacity | Observed-capacity diagnostic | Hardware | Diagnostic error |
| --- | ---: | ---: | ---: | ---: |
| Output tokens/s | 2,076.03 | 3,347.55 | 3,599.26 | -6.99% |
| TTFT p50, seconds | 7.8884 | 3.4758 | 3.3630 | +3.35% |
| TTFT p90, seconds | 44.9659 | 17.0605 | 16.2065 | +5.27% |
| ITL p50, seconds | 0.074760 | 0.036523 | 0.034248 | +6.64% |
| ITL p90, seconds | 0.102778 | 0.055018 | 0.050246 | +9.50% |

This single-variable run identifies a material capacity error in the B200 configuration.
The imported observation conditions the simulator on a measured deployment and cannot establish the analytical memory model's accuracy or generalization to unmeasured hardware.
The production implementation is `d7ba4e2`, documented under [observed runtime KV capacity](agentx.md#observed-runtime-kv-capacity).
Its separate complete CLI replay was checked against the diagnostic's timing and replay events.
The replay completed successfully with 12,923 successful profiling requests and no unfinished requests.
The CLI replay and diagnostic have byte-identical `requests.jsonl`, `events.jsonl`, and `cache_windows.jsonl`, matching throughput, latency statistics, operator totals, and scheduling counters.
`runtime-memory-equivalence.json` retains these file hashes and `runtime-memory-comparison.json` reports the conditional hardware errors.
The replay used revision `d7ba4e2`; the subsequent positive-population metric fix is validated separately with the archived hardware samples.

## B300 capacity comparison

Both B300 capacity modes were rerun at revision `4574dd6dde620a1d6ce78ec2509a616eedbd289b` with identical model, data, seeds, scheduler, connector, and operator timing settings.
Both load the full 393-tree corpus and use live warmup and a 3,600-second issue window at concurrency 384, with no unfinished requests after cleanup.
The three analytical-baseline replay files match the frozen `acffa30` baseline byte-for-byte, confirming that the reporting and profile changes preserve its execution behavior.

The analytical budget is 104.4398 GiB per rank, or 91,260 physical KV slots.
The imported [B300 profile](../data/runtime_memory/agentx_b300_439522.json) uses 103.07 GiB per rank from reference run 439522, or 90,063 slots.
Rank 0 provides the available-byte observation; applying that rounded value to every rank is an explicit assumption with 0.005 GiB uncertainty.
The imported geometry yields 5,416,880 equivalent tokens per rank, within the server log's range of 5,416,820 to 5,416,940.

| Metric | Analytical capacity | Imported capacity | Hardware | Analytical error | Imported error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Output tokens/s | 5,453.81 | 5,441.33 | 6,360.26 | -14.25% | -14.45% |
| TTFT p50 (s) | 4.654 | 4.656 | 4.265 | +9.11% | +9.16% |
| TTFT p90 (s) | 20.367 | 20.288 | 18.916 | +7.67% | +7.26% |
| ITL p50 (ms) | 51.12 | 51.34 | 45.21 | +13.07% | +13.55% |
| ITL p90 (ms) | 57.21 | 57.03 | 49.49 | +15.62% | +15.24% |
| ITL p99 (ms) | 64.50 | 64.62 | 61.08 | +5.61% | +5.80% |
| End-to-end p90 (s) | 128.53 | 128.84 | 112.06 | +14.70% | +14.98% |

The selection rule was recorded before the runs completed: minimize the largest absolute percentage error across output throughput, TTFT p50/p90, and ITL p50/p90, while retaining every metric in the comparison.
The largest critical error is 15.62% with analytical capacity and 15.24% with imported capacity, so the main guide selects the complete imported-capacity run.
This choice improves TTFT p90 and ITL p90, with small regressions in throughput, median latencies, ITL p99, and end-to-end p90.
The two modes remain close and do not establish a general accuracy gain from the capacity change.

Both modes have zero preemptions and recomputed tokens.
The successful-request GPU cache ratio changes from 0.947164 to 0.946711, the CPU cache ratio from 0.007670 to 0.008115, and accumulated connector load wait from 1.2174 to 1.2378 seconds.
These changes are consistent with a small increase in CPU cache use at the lower GPU capacity; the counters alone do not identify the remaining timing-model error.
The selected imported-capacity run completes 21,963 successful profiling requests, compared with 22,001 under analytical capacity and 25,696 in the reference.

The local `b300-capacity-comparison/manifest.json`, `selection-policy.json`, and `comparison.json` retain commands, source identity, the selection rule, all errors, cache diagnostics, and baseline-equivalence hashes.
The [committed result summary](data/agentx-selected-results.json) retains both capacity modes and the selected complete run.

## Remaining sources of serving error

The [pinned AIPerf harness](https://github.com/SemiAnalysisAI/agentx-harness/blob/49634d1c6c2e6dcc18979d704a65e7c7064efc49/src/aiperf/timing/branch_orchestrator.py) dispatches overlapping SPAWN children from `on_credit_issued`; FORK children wait for the declaring response.
TokenSim currently releases child branches from completion and does not retain this mode in its minimal branch model.
The source audit finds 934 child-start overlap candidates among 9,450 child streams; each candidate still requires mode and parent-edge qualification.
Do not convert all overlaps to dispatch-triggered edges without that qualification.

Mapping B300's successful profiling records to source keys `(srcTrace, srcOuter, srcInner)` resolves all 25,696 records and finds no output-length mismatches.
The source-key audit rejects the interpretation that the dashboard's historical expected-output mismatch proves output truncation.
Actual tokenized inputs differ from trace lengths: median +33 tokens, mean +70.98, p90 +168, and p99 +651.05.
These inputs and prefix markers require benchmark-harness fidelity checks before further cache calibration.
Warmup and profile request populations remain separate.

The archived B300 successful records reconstruct output throughput as `23084003 / 3629.409517056 = 6360.2640847` tokens/s.
The denominator includes successful grace completions and exceeds the 3,600-second issue window.
The reconstructed slow-tail interactivity is `1 / p90(ITL) = 20.20794` tokens/s; taking `p90(1/ITL)` produces 25.98583 tokens/s from the same samples.
The metric fix changes reporting and does not change simulated execution.
The B300 timeline contains 25,687 successful multi-token requests, including two zero-ITL observations.
InferenceX excludes these zeros and retains 25,685 positive ITL samples; the production metric function applies the same population filter.
Hardware server cache counters include a different request population from TokenSim's successful-request first-admission counters, so their ratios remain diagnostic.

The reference serving image is vLLM `0.17.2rc1.dev3682+g426e59f62` with FlashInfer sparse attention and native FP4 indexing.
TokenSim's imported tables include vLLM 0.24.0 and SGLang 0.5.10 MegaMoE observations borrowed from GB200/GB300.
These framework, kernel, and hardware differences remain timing approximations.
A deployment-matched whole-forward dataset, held-out batch timing validation, typed dependency replay, and measured transfer behavior remain required for broader accuracy claims.
