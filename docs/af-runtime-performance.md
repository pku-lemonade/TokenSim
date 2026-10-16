# AF runtime performance

## Scope

The runtime optimizations preserve the model, query and provider interfaces, deterministic event scheduling, simulated durations, and complete exported traces.
The production optimizations use Python and the standard library and require no additional compiler or package.
The geometry and normalization measurements use `d8677a0` as their reference.
Later experiments identify their own reference revisions.

## Implementation

`TokenSim/af/caching/estimates.py` extracts service-key fields with a cached field layout, dictionary merge and `operator.itemgetter`.
Normalization builds each container key and its immutability flag in one pass and reuses tuples containing only strings, integers and `None`.
Booleans and floats keep type tags so invalid numeric inputs cannot reuse integer cache entries.
Mutable descendants prevent identity memoization even when their parent dataclass is frozen.
The identity memo remains bounded at 4,096 entries and the field-layout cache at 128 record types.

`TokenSim/af/providers/traffic.py` caches immutable wire schedules by the parameters its plan builders consume.
The key includes the model version, operation, payload bytes, dtype, mapped physical participants, statistical parameters and transport backend.
The provider pins placement and domain configuration for its lifetime.
Routing edges, tensor item identities, logical byte annotations and query topology remain available to callers but do not affect these wire-plan builders.
Participant and statistical validation still runs before cache lookup.
Every invocation creates fresh readiness events and resource reservations.
The existing service-key fields and trace records retain their provenance.

`TokenSim/af/planning/copies.py` reuses validated immutable endpoint records within one planner.
The endpoint key includes every endpoint field, and storage is bounded by the pinned runtime's endpoint geometry.
The completed-trace archive and its bounded readback behavior remain in use.

## Measurements

Measurements use an Apple M5 host, Python 3.12.12 and fresh local processes.
The DeepSeek cases use batch 32, attention tensor parallelism 4, feed-forward tensor parallelism 32, context 8,192, A100 SXM 80 GB, NVLink 3 and analytical-only estimates.
Full-model cases contain 61 layers.
The measurements in this section exclude the 100-step full-model workload measured separately below.
The one-step and cross-model results use the median of three fresh-process before/after pairs.
The full-model one-step pairs alternate their execution order.
The baseline probes load the changed modules from `git show d8677a0` without modifying the working checkout.
The four-step result is a single before/after pair and excludes final trace export while retaining trace collection, invariant validation and archival.

| Workload | Reference simulation | Optimized simulation | Speedup |
| --- | ---: | ---: | ---: |
| Full model, 1 step | 12.91 s | 7.97 s | 1.62x |
| Full model, 4 steps | 60.34 s | 39.59 s | 1.52x |
| One layer, 128 steps | 31.14 s | 19.34 s | 1.61x |
| Dense multi-head attention, 4 layers, 400 steps | 1.023 s | 0.857 s | 1.19x |
| Grouped-query attention with mixture-of-experts, 400 steps | 1.271 s | 0.944 s | 1.35x |

Including final trace export, the full-model one-step median falls from 15.18 seconds to 10.16 seconds, a 1.49x speedup.
The 128-step pair includes 150,784 events and uses 425.7 MiB and 409.4 MiB peak resident memory respectively.
The cross-model fixtures use batch 8 and context 128.

The four-step run records 270,380 events in both versions and retains at most 67,595 active events at once.
Each iteration leaves zero completed events in active memory after archival.
Peak resident memory in that pair fell from 1,089.8 MiB to 1,032.0 MiB.
All three full-model one-step traces, the complete 128-step trace, and their companion JSON files were byte-identical to the reference.
The four-step companion JSON files were also byte-identical.
Dense and mixture-of-experts traces and per-step simulation rows matched for all three repetitions.
The comparisons cover 11 before/after pairs and 41 file or per-step observation checks.
Service-query counts and service-cache hit counts are unchanged.
In the full-model one-step case, wire-plan cache hits increase from 720 to 61,845 across 62,013 lookups because invocation metadata no longer prevents geometric reuse.

A bounded reproduction of the full-model case uses the existing command-line interface.
The one-layer case uses the same arguments without `--full-model` and with `--steps 128`.

```bash
python scripts/run_deepseek_mla.py \
  --full-model \
  --a-tp 4 \
  --f-tp 32 \
  --batches 32 \
  --af-batch-size 32 \
  --context 8192 \
  --steps 1 \
  --device a100_sxm_80g \
  --network nvlink3 \
  --fallback analytical_only \
  --output outputs/af-performance-repro
```

## Scoped host collection and allocation reuse

The additional exploration uses `246cfb7` as its reference and keeps the Python executor and SimPy event loop.
The per-process memory budget is 4,000,000,000 bytes, and execution speed takes priority within that budget.
The screening covers scalar tuple memoization, query-key normalization, immutable traffic properties, dependency rewriting, activation-copy preparation, JSON encoder reuse, archive chunk sizes and garbage collection policy.
Scalar tuple memoization and inline scalar normalization did not show a useful gain and were excluded.
Increasing archive chunks from 256 to 1,024, 4,096 or 16,384 events kept one-step simulation plus export near 7.9 seconds while raising resident memory, so the existing chunk size is retained.

`TokenSim/af/runtime/host.py` supplies `run_environment(environment, until=None)` for exact simulation entry points.
During the call, the generation-zero allocation threshold is raised to 250,000 while automatic cyclic garbage collection remains enabled.
The remaining generation thresholds retain their caller values.
Disabled collection, a zero allocation threshold and higher caller thresholds are preserved.
The previous policy is restored after the last overlapping call, including exceptions and bounded stop/resume calls.
An explicit threshold change made during a call takes precedence over restoration.
The lock protects policy transitions and is released throughout event execution.

The fixed-batch decode factory, LPX native serving, concurrency probes and sampled-profile capture use this helper.
Direct callers of `env.run()` can use `run_environment(env)` to apply the same host policy.
The policy changes collection frequency and leaves scheduling, simulated latency, routing, communication occupancy and trace retention unchanged.
Python garbage collection settings are process-wide while a call is active.

The dependency rewrite shared by transfer coalescing and expert grouping reuses an immutable node when its dependencies stay equal.
Changed dependencies still construct and validate a replacement, preserving predecessor deduplication and endpoint order.
Activation-copy source geometry is computed once per planning call.
Each immutable traffic phase caches its resource identities, and trace export reuses one standard-library JSON encoder per file.
These changes introduce no model-specific cache key, native implementation or alternate execution semantics.

The threshold screen uses one full-model four-step run per candidate.
A threshold of 250,000 takes 25.82 seconds with 995 MiB peak resident memory, while 1,000,000 takes 25.67 seconds with 1,056 MiB.
The difference is below 1% in this single comparison and does not establish a stable benefit for the higher threshold.
Disabling automatic collection takes 24.01 seconds but raises the four-step peak to 1,831 MiB and leaves dead cycles pending until an explicit collection.
That diagnostic setting is excluded from production.

The full-model one-step and cross-model comparisons use the median of three alternating fresh-process pairs.
The four-step and one-layer 256-step comparisons use one pair each.
All cases retain complete traces and compare every simulation row and the full exported trace hash.
The reference loader executes the changed modules from `git show 246cfb7` while keeping their original module names and file paths.

| Workload | Reference simulation | Optimized simulation | Speedup | Reference / optimized peak resident memory |
| --- | ---: | ---: | ---: | ---: |
| Full model, 1 step | 8.376 s | 5.660 s | 1.48x | 462 / 441 MiB |
| Full model, 4 steps | 40.961 s | 26.630 s | 1.54x | 1,033 / 1,001 MiB |
| One layer, 256 steps | 36.187 s | 23.893 s | 1.51x | 411 / 358 MiB |
| Dense multi-head attention, 4 layers, 400 steps | 0.910 s | 0.868 s | 1.05x | 72 / 72 MiB |
| Grouped-query attention with mixture-of-experts, 400 steps | 0.973 s | 0.944 s | 1.03x | 75 / 75 MiB |

Including complete trace export, the one-step median falls from 10.668 seconds to 7.921 seconds, a 1.35x speedup.
The four-step totals are 50.322 seconds and 36.997 seconds, a 1.36x speedup.
The 256-step totals are 47.502 seconds and 35.296 seconds, a 1.35x speedup.
The longer single-layer pair contains 301,568 events and leaves zero completed events in active memory.
Service-query counts and service-cache hits match in these pairs.
These speedups are additional to the changes already present in the reference revision.
The small cross-model fixtures use batch 8 and context 128; their lower allocation pressure gives less benefit from collection tuning.

The concurrent validation keeps the original DeepSeek geometry and launches three batch-32 streams in one executor and process.
Each stream runs two consecutive iterations, giving six complete iterations and 405,570 trace events.
The original hardware layout has 64,701 resident KV blocks; four complete batch-32 streams would require 65,664 blocks, so three streams fit the configured capacity.
The reference simulation takes 68.97 seconds and the optimized simulation takes 40.67 seconds, a 1.70x speedup.
Including complete trace export, the totals are 89.43 seconds and 62.14 seconds, a 1.44x speedup.
Peak resident memory is 1,991 MiB and 1,928 MiB respectively, or about 2.09 GB and 2.02 GB per process.
Every iteration timing and the complete exported trace match exactly.
No completed events remain in active trace memory after these runs.
An extended optimized run keeps three streams active for four iterations each, producing 811,140 events across 12 iterations.
Its simulation takes 76.68 seconds and complete trace export takes 42.11 seconds.
Peak resident memory is 2,088 MiB, or 2.19 GB per process, with automatic collection observed during execution.
The completion samples rise from 1,470 MiB after iteration 0 to 1,948 MiB after iteration 3, 2,071 MiB after iteration 6 and 2,088 MiB after iteration 9.
The remaining completions do not increase the peak.
This extended case checks memory and successful complete export; it has no paired reference timing or full-trace equality claim.

The existing full-model one-step command also produces six byte-identical JSON files, including its complete trace.
Complete LPX result documents, including traces, match for the native Rubin, attention/feed-forward disaggregated, prefill/decode disaggregated and speculative-draft modes.
The comparison uses the small deterministic layouts from `tests/test_lpx_native.py`.
The measured memory budget applies to these configurations; the collection threshold does not impose a hard resident-memory cap.
The 513-test suite passes, including contracts for caller policy preservation, exceptions, stop/resume and overlapping host runs.
Black 24.2.0, the 500-line source limit and whitespace checks pass.

## Complete 100-step command

This comparison uses `2856d5c` as its reference and executes the original full-model command for 100 consecutive decode steps in each fresh local process.
The workload increments context and generation state normally and retains the complete trace.
The measured command includes configuration, simulation, invariant validation, archival, all six JSON outputs and archive cleanup.
Python imports and diagnostic file hashing are outside the timed interval in both versions.
The runs execute sequentially on the Apple M5 host with Python 3.12.12.
These measurements are a single complete pair, with shorter screening runs used to select candidates.

| Phase | Reference | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| Simulation | 682.788 s | 512.345 s | 1.33x |
| Complete trace export | 272.405 s | 214.962 s | 1.27x |
| Complete command | 955.568 s | 727.606 s | 1.31x |

The complete command uses 23.86% less time, and peak resident memory increases from 1.60 GB to 2.43 GB per process.
The trace contains 6,759,500 events and occupies 32,392,112,426 bytes in both versions.
Completed events remain archived, with zero completed events retained in the active trace collection at the end of the simulation.

The warm-step profile spreads execution time across structural normalization, template compilation and binding, communication execution, scheduling and trace handling.
The export profile shows repeated dataclass field copies and attribute lookups in addition to standard-library JSON encoding and archive reads.
The reference environment-run intervals total 607.212 seconds, leaving 75.576 seconds in setup and work between those intervals.
The reference restores the low collection threshold between steps, allowing subsequent routing submission to trigger frequent collection of large cached object graphs.

The existing host collection scope now covers the complete fixed-batch decode call and complete streaming trace export.
Automatic collection remains enabled, and nested or overlapping scopes restore the caller's policy after the last scope exits.
The template compiler and binder clear recursive closure references when each call returns, releasing their temporary memo dictionaries without waiting for cyclic collection.
Internal binding recipes use slots to reduce stored objects, and the bounded template cache holds up to 384 recurring shapes instead of 128.
The larger cache accommodates recurring routing shapes while retaining the existing probation and eviction policies.
No model-specific key fields or alternate scheduling path are introduced.
The complete optimized run observes 257 compiled templates and raises cache hits from 4,422 to 5,586 across the same 6,100 lookups.
Simulation time falls to 512.345 seconds, a 1.33x speedup.
Its environment-run intervals total 509.852 seconds, and setup plus work between those intervals falls to 2.493 seconds.
An ablation keeps the original 128-template capacity while applying the remaining optimizations.
Its simulation takes 573.729 seconds, export takes 212.973 seconds, and the complete command takes 787.000 seconds with a 1.53 GB peak.
The larger template cache saves another 59.394 seconds per complete command at an additional 0.89 GB of peak resident memory.
All six ablation JSON files are byte-identical to the original reference outputs.

Trace export reuses a dataclass's existing field dictionary when its attribute access, declared fields and dictionary order allow equivalent serialization.
Slots, descriptors, custom attribute access, omitted instance defaults and extra attributes use the generic field-by-field fallback.
The standard-library JSON encoder, event order, floating-point representation and atomic destination replacement remain in use.

All 100 iteration rows, service-query counts and service-cache hits match the reference.
The five companion JSON files are byte-identical, and every value and array position in the complete trace matches after JSON parsing.
Some object keys appear in a different order because the existing template binder inserts static fields before dynamic fields, and `RankLocalWork.trace_mapping()` preserves that dictionary order.
Increasing cache capacity changes which graphs use that binder, so the complete trace has a different byte hash.
Validation uses the additional 100-step ablation whose six outputs match the reference byte for byte, then compares both complete parsed traces with object keys sorted for diagnostic hashing.
This comparison includes all 6,759,500 events and file metadata without sorting arrays or rounding numeric values.
The production trace is retained as a losslessly compressed artifact whose decompressed SHA-256 matches the recorded uncompressed file.

A fresh concurrent comparison uses the same original geometry with three batch-32 streams and four consecutive iterations per stream.
Its 811,140 events and all iteration timings match after complete JSON-value comparison.
Simulation takes 74.192 seconds in the reference and 73.937 seconds with the optimizations, which does not establish a simulation speedup for this shorter concurrent case.
Complete trace export falls from 37.852 seconds to 28.545 seconds, reducing the combined time from 112.044 seconds to 102.483 seconds.
Peak resident memory is 2.16 GB and 2.21 GB respectively.
Extending the optimized case to 12 iterations per stream produces 2,433,420 events with a 3.10 GB peak and 278 cached templates.
That run takes 208.918 seconds for simulation and 93.395 seconds for complete export.
It records scalar iteration timings and releases completed iteration objects during execution.
This longer case checks memory and complete output generation without a paired reference throughput claim.
A saturation run extends each of the three streams to 24 iterations, giving 72 complete iterations and 4,866,840 trace events.
The cache reaches its 384-template limit by completion 54, and the run continues through completion 71.
Peak resident memory is 3,906,404,352 bytes, meeting the 4,000,000,000-byte per-process budget after the cache fills.
Simulation takes 416.880 seconds and complete export takes 198.917 seconds, with zero completed events retained in the active trace collection afterward.
Automatic collection remains enabled and performs collections during execution.
The measured budget applies to these configurations and does not impose a universal hard memory limit.
The dense multi-head attention and grouped-query attention with mixture-of-experts fixtures each preserve 400 iteration rows and complete exported trace bytes against `2856d5c`.
Complete LPX result documents match in the native Rubin, attention/feed-forward disaggregated, prefill/decode disaggregated and speculative-draft modes.
All 517 tests pass, including caller collection-policy restoration, export failures, dataclass attribute semantics, immediate binding-memo release and optional Rust contracts.
Black 24.2.0, the 500-line source limit and whitespace checks pass.

Increasing the structural fingerprint memo to 16,384 entries slows the four-step screen, and replacing its FIFO dictionary with an ordered dictionary provides no established gain.
Attribute-getter export and a custom pickle dispatch table also provide no useful complete-run improvement in the screening probes.
Disabling template caching reduces compilation cost in a short startup-heavy run but does not establish a benefit for the complete workload.
Those substitutions are excluded from production.

## Native code and parallel execution

A C++ prototype of recursive structural-key normalization reduced the initial geometry-interning candidate from 11.31 seconds to 10.08 seconds for the full-model one-step simulation.
That incremental gain did not meet the requested 2x threshold for adopting a native backend.
The retained Python implementation removes more redundant work without introducing a native build dependency.
An exploratory use of the third-party Rust JSON encoder `orjson` reduced trace export from about 2.2 seconds to 0.94 seconds, but the whole candidate still did not reach 2x and its JSON compatibility would require additional handling.
Neither prototype is part of the production implementation.
The separate [Rust kernel experiment](../scripts/af_rust/README.md) provides authored native implementations and a reproducible comparison entry point.

A separate concurrency probe ran two independent one-layer jobs of 600 steps each with attention and feed-forward tensor parallelism 2, batch 8, context 128 and trace retention disabled.
Threads and processes reproduced the sequential simulation rows exactly.

| Execution of two jobs | Wall time | Speedup over serial |
| --- | ---: | ---: |
| Serial | 12.42 s | 1.00x |
| Two threads | 14.82 s | 0.84x |
| Two spawned processes | 7.53 s | 1.65x |

Threads did not meet the requested near-linear scaling criterion and were not added to the executor.
The process measurement includes startup and configuration costs for these short jobs.
`scripts/lpx/runner.py` already runs independent experiment points in isolated processes.

## Further native-backend gate

The geometry and normalization changes provide roughly 1.5x to 1.6x simulation throughput relative to `d8677a0` for the tested DeepSeek cases.
The native probes do not demonstrate an additional 2x over the optimized Python runtime.
A deeper native backend would need to cover planning and scheduling together because the remaining time is spread across both paths.
The existing sampled C++ replay cannot retain complete execution traces and cannot substitute for exact dynamic execution.
A candidate replacement must preserve FIFO ties, failure and cancellation behavior, dynamic routing, communication phases, readiness and every exported event.
A native implementation should remain behind the existing Python interfaces and retain a reference path for validation.
It should be adopted only after representative complete runs demonstrate at least 2x throughput with equal result JSON and trace values.
Independent experiment processes remain the available concurrency path while that gate is unmet.

## Authored Rust kernel experiment

`scripts/af_rust` implements structural-key normalization, validated template binding and event dispatch in Rust with PyO3.
`scripts/profile_af_rust.py` selects Python or Rust inside a fresh diagnostic process and passes simulation arguments through the existing DeepSeek command-line interface.
No production model, provider or configuration module is changed by this experiment.
The build is explicit, dependency versions are locked, and artifacts are cached by Python version and Rust source digest.

The comparison uses the optimized Python revision `ba8c043` as its reference.
The default Rust probe selects key normalization and template binding.
The full-model one-step result uses the median of three alternating fresh-process pairs.
The longer cases use one pair each and include complete trace export and exact file comparisons.
All cases use the DeepSeek configuration described above.

| Workload | Optimized Python simulation | Rust keys and templates | Speedup |
| --- | ---: | ---: | ---: |
| Full model, 1 step | 8.313 s | 7.930 s | 1.048x |
| Full model, 4 steps | 41.215 s | 39.407 s | 1.046x |
| One layer, 128 steps | 20.049 s | 18.871 s | 1.062x |

Including trace export, the full-model one-step median is 10.604 seconds for Python and 10.227 seconds for Rust, a 1.037x speedup.
The full-model four-step totals are 51.006 seconds and 49.563 seconds, a 1.029x speedup.
Peak resident memory in that four-step pair is 1,051 MiB for Python and 1,147 MiB for Rust.
The additional compiled template representation adds storage while the Python recipes remain available.
All six result JSON files, including complete traces, match byte for byte in each of the five pairs.

The event-dispatch kernel continues to call Python generators and scheduling callbacks and does not release the global interpreter lock.
It preserves the Python event queue and its time, priority and insertion ordering.
The dispatch experiment has a separate selection because crossing this boundary alone did not improve complete-run performance.
In individual one-step full-model probes, Rust dispatch alone took 8.638 seconds and the three combined kernels took 8.388 seconds, compared with the 8.313-second Python median.
Both variants reproduce all six result JSON files byte for byte.
Per-event reference scopes prevent temporary Python references from accumulating across a run.

The 506-test suite passes with the default runtime and with all three Rust kernels installed in the test process.
Eight optional native contract tests cover dense and mixture-of-experts trace equivalence, mutable key descendants, bounded storage, event ordering, cancellation, callback growth, stop/resume, errors, temporary reference release and restoration of customized execution paths.
Black 24.2.0, Rust formatting, Clippy with warnings treated as errors, the 500-line source limit and whitespace checks pass.

The Rust kernels remain optional experiments because the measured whole-run gain is below the requested 2x adoption threshold.
A complete native engine would need typed planning and execution state, native resource queues and dependency tracking, and bounded trace conversion with fewer Python callbacks.
The existing Python model and provider interfaces could remain above that internal boundary.
The [Rust experiment guide](../scripts/af_rust/README.md) records the implemented scope, reproduction commands and remaining engine work.
