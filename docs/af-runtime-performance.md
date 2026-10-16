# AF runtime performance

## Scope

The runtime optimizations preserve the model, query and provider interfaces, deterministic event scheduling, simulated durations, and complete exported traces.
The production optimizations use Python and the standard library and require no additional compiler or package.
The reference revision for the measurements below is `d8677a0`.

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
No 100-step full-model workload was launched for this exploration.
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

The measured Python changes provide roughly 1.5x to 1.6x simulation throughput for the tested DeepSeek cases.
The requested additional 2x has not been demonstrated.
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
