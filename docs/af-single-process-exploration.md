# Single-process AF performance exploration

All measurements use production commit `7665583` as their reference.
That reference already contains the earlier runtime, cache, garbage-collection and bounded-trace improvements.
The prototype modules are opt-in diagnostic contexts under `scripts/`.
At the time of these measurements, production modules, configuration schemas and default execution paths remained unchanged.
The subsequent [integrated version](af-integrated-defaults.md) adopts finalized-prefix export with 1,024-event archive chunks for the DeepSeek command and adds configurable trace collection.

## Workload and measurement

The local host has an Apple M5, 10 logical processors and 32 GiB of memory.
The interpreter is Python 3.12.12 and the installed SimPy version is 4.1.2.
Every timed experiment runs one simulation process.
The optional trace-writing thread belongs to that same process.
Different measured experiments run sequentially.

The fixed-batch workload uses all 61 DeepSeek-V3 layers, attention tensor parallelism 4, feed-forward tensor parallelism 32, batch 32, attention/feed-forward batch size 32, context 8192, A100 SXM 80 GB, NVLink 3 and analytical-only estimates.
Short screens use four steps.
Longer checks use 16 steps, including both cold and warm template-cache behavior.
The timed interval includes configuration, simulation, validation, archival, all six public JSON outputs and archive cleanup.
CLI imports, prototype installation and diagnostic output hashing occur outside the timed interval.
Optional JSON dependencies imported lazily by their encoders remain inside the timed export call.
Peak resident memory is measured for the whole interpreter.
Memory values in the tables use decimal GB, and the per-process limit is 4,000,000,000 bytes.

The supplied plotting script consumes saved throughput, active-user and adjacent-token-latency summaries.
Its eight-point, request-rate-driven producer and source `results.json` were unavailable in this checkout.
The fixed-batch workload is a producer-cost proxy and does not establish a speedup for that complete figure scan.
It also does not reproduce the plotting script's 404K context, virtual memory inventory or prefill workload.

## Independent prototypes

| Branch | Kernels | Implemented experiment |
| --- | --- | --- |
| `codex/probe-simpy-loop` | `loop`, `conditions` | Inline the stock unbounded event-dispatch loop, or reuse condition callbacks |
| `codex/probe-trace-pipeline` | `pipeline`, `thread` | Encode finalized global trace prefixes before archival, optionally overlap byte writes |
| `codex/probe-json-encoder` | `json` | Use orjson with compatibility checks and reference fallbacks |
| `codex/probe-template-program` | `template` | Lower existing binding recipes into a postorder program |
| `codex/probe-trace-validation` | `validation` | Check single-capacity intervals with one ordered pass |
| `codex/probe-metrics-only` | `metrics` | Exercise the existing trace-disabled simulation path |
| `codex/probe-rapidjson` | `rapidjson` | Use python-rapidjson behind the existing exporter contract |
| `codex/probe-sparse-query` | `sparse` | Pre-normalize immutable defaults and normalize supplied query fields |
| `codex/probe-json-fragments` | `fragments` | Reuse bounded immutable dataclass JSON fragments |
| `codex/probe-query-front-cache` | `frontcache` | Reuse proven immutable query identities, full structural keys and their hashes |
| `codex/probe-archive-chunks` | `archive`, `archive4k` | Increase archive blocks from 256 to 1,024 or 4,096 events |

The aggregate branch is `codex/perf-single-process-exploration`.
Each independent branch includes a runnable benchmark, focused semantic tests and its implementation notes.
The aggregate branch contains all prototypes for direct combination experiments.
The optional JSON experiments require orjson or python-rapidjson only when selected.
The measured python-rapidjson version is 1.25.
The SimPy derivatives retain the upstream MIT license and notice.

## Four-step screens

Each prototype row is one fresh-process measurement.
Differences near one percent do not establish a repeatable gain.
The initial reference is 30.303 seconds.

| Kernel | Simulation | Export | Complete command | Speedup | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `loop` | 23.410 s | 6.839 s | 30.537 s | 0.992x | 0.871 GB |
| `conditions` | 23.146 s | 6.734 s | 30.167 s | 1.005x | 0.911 GB |
| `json` | 23.248 s | 10.661 s | 34.198 s | 0.886x | 0.879 GB |
| `template` | 23.513 s | 6.608 s | 30.452 s | 0.995x | 1.136 GB |
| `validation` | 23.495 s | 6.788 s | 30.579 s | 0.991x | 0.871 GB |
| `metrics` | 18.219 s | 0.000 s | 18.510 s | 1.637x | 0.554 GB |
| `rapidjson` | 23.478 s | 6.579 s | 30.350 s | 0.998x | 0.881 GB |
| `sparse` | 23.061 s | 6.711 s | 30.057 s | 1.008x | 0.881 GB |

The later screen is bracketed by 30.150-second and 29.960-second references.
Its speedup denominator uses their 30.055-second mean.

| Kernel | Simulation | Export | Complete command | Speedup | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fragments` | 23.106 s | 8.571 s | 31.966 s | 0.940x | 0.899 GB |
| `frontcache` | 25.833 s | 6.583 s | 32.712 s | 0.919x | 0.900 GB |
| `archive` | 23.034 s | 6.872 s | 30.194 s | 0.995x | 0.891 GB |
| `archive4k` | 23.068 s | 7.008 s | 30.371 s | 0.990x | 0.927 GB |
| `thread` | 28.435 s | 0.000 s | 28.724 s | 1.046x | 0.889 GB |

The current pipeline uses one ordered archive-range cursor per finalized prefix.
The earlier four-step pipeline and thread records predate this cursor improvement and remain in the evidence as initial screens.
The 16-step pipeline and the later thread measurement use the committed implementation.

## Longer fixed-batch checks

The three 16-step reference runs take 112.708, 112.648, 113.149 seconds.
The speedup denominator is their 112.835-second mean.
Each candidate row is one run, with the pipeline measured twice.
Small gains and differences between candidates remain sensitive to host variation.

| Case | Simulation | Export | Complete command | Speedup | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `template-16-a` | 84.655 s | 27.076 s | 112.272 s | 1.005x | 2.118 GB |
| `pipeline-16-a` | 108.021 s | 0.000 s | 108.449 s | 1.040x | 1.522 GB |
| `pipeline-16-b` | 112.448 s | 0.001 s | 112.883 s | 1.000x | 1.538 GB |
| `pipeline-archive-16-a` | 107.146 s | 0.000 s | 107.573 s | 1.049x | 1.514 GB |
| `thread-16-a` | 107.480 s | 0.001 s | 107.905 s | 1.046x | 1.507 GB |
| `combo-16-a` | 107.891 s | 0.000 s | 108.427 s | 1.041x | 2.150 GB |
| `metrics-16-a` | 63.989 s | 0.000 s | 64.403 s | 1.752x | 1.168 GB |
| `metrics-lean-16-a` | 62.794 s | 0.000 s | 63.211 s | 1.785x | 1.156 GB |
| `metrics-combo-16-a` | 62.214 s | 0.000 s | 62.735 s | 1.799x | 1.800 GB |

The full-trace combination selects `template,sparse,conditions,pipeline`.
The lean metrics combination selects `metrics,sparse,conditions`.
The larger metrics combination also selects `template`.
The archive combination selects `pipeline,archive`.
Combination speedups are measured directly and are not products of individual speedups.

## Concurrent iteration checks

Each case uses three simultaneous cohorts of batch 32 and context 8192, with four iterations per cohort.
All 811,140 ordered events, iteration rows and provider counters match the concurrent reference exactly.
Timing covers configuration through trace export and excludes archive cleanup and diagnostic hashing.

| Kernels | Simulation | Export | Total | Speedup | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `reference` | 69.687 s | 24.706 s | 94.393 s | 1.000x | 1.879 GB |
| `pipeline,archive` | 90.880 s | 0.000 s | 90.880 s | 1.039x | 1.891 GB |
| `thread,archive` | 91.884 s | 0.000 s | 91.885 s | 1.027x | 1.895 GB |

These runs satisfy the 4 GB limit for the measured twelve-iteration workload.
They do not establish a bound for arbitrary concurrency, template diversity or experiment length.

## Interpretation

The existing trace-disabled path gives the largest measured improvement when its output contract is sufficient.
The supplied figure needs throughput, active-user and adjacent-token summaries, so its producer can potentially avoid detailed event retention if it maintains those summaries online.
This requires checking the producer's metric sources before changing its trace setting.
The fixed-batch proxy establishes timing equivalence for its own summary and does not prove equivalence for the unavailable producer.

Finalized-prefix export avoids a later full archive read and encodes already-finalized live records.
Most JSON encoding work remains on the simulation thread.
The byte-writing worker overlaps file writes with simulation work.
Near-zero final export time means encoding was moved into simulation time, and complete-command measurements include that cost.

Postorder binding adds a second representation of each compiled recipe.
The 16-step prototype raises peak resident memory by about 0.63 GB while changing complete-command time by less than one percent.
The identity front cache adds Python field scanning and lookup overhead that outweighs saved normalization in this workload.
The fragment encoder adds immutability checks and marker substitution without reducing enough repeated encoding.
The compatible orjson path must check mutable descendants and exceptional JSON values before using its native encoder.
RapidJSON does not establish a complete-command improvement in the short screen.
These results support rejecting the measured implementations for default use.

The earlier instrumented warm-step profile in [the runtime performance notes](af-runtime-performance.md) assigns about 11.8 percent of exclusive time to SimPy and another 2.1 percent to heap operations.
That profile includes instrumentation overhead and an adjacent production revision, so its percentages are diagnostic context rather than a forecast for these runs.
Cumulative time in the environment's run and step functions includes model callbacks.
The implemented dispatch and condition prototypes do not support a SimPy fork as a high-return change.
The existing optional Rust results in those notes are earlier experiments and are excluded from the new prototype speedups.

## Recommendation

For a summary-producing experiment, prioritize the existing trace-disabled factory path after confirming that the producer maintains the required online measurements.
The simple metrics prototype has the largest clear improvement without adding a new execution backend.
The additional sparse-key and condition changes produce only a small measured difference and add maintenance cost.
The postorder-binding combination retains substantially more memory and does not establish a useful incremental return.

For experiments requiring complete event history, retain finalized-prefix export as a candidate for production evaluation.
The measured benefit is modest and varies across the longer repeats.
The writing thread does not improve the tested concurrent case over synchronous prefix export.
Larger archive blocks do not establish an independent gain.
The measurements do not support adopting the other prototypes or forking SimPy for this workload.
No preserving prototype establishes a twofold complete-command improvement with full traces.

All prototype implementations remain isolated and available for reproduction on their individual branches.
The aggregate branch provides direct combination measurements without changing production defaults.

## Correctness and compatibility

Preserving variants compare all six public output JSON files against the reference.
All iteration start times, completion times, makespans and provider cache counters are compared.
The orjson variant additionally hashes complete parsed trace values because equivalent floating-point JSON spellings can differ.
Event-array order is preserved in that comparison.
Diagnostic trace files are deleted after hashing to bound experiment disk use.

The pipeline retains archive readback, dependency patching and invariant validation.
It exports only the globally ordered finalized prefix, including archived fragments behind gaps created by concurrent iterations.
Explicit late event replacements invalidate the prefix and select the reference exporter.
Partial and repeated exports also select the reference path.
The threaded variant queues bounded immutable byte blocks after encoding.
A worker never reads mutable simulation records.
Tests cover interleaving, snapshots, late replacements, clear/reuse, failed writes and preserving existing destination files.

The metrics variant deliberately produces an empty trace.
Its summary JSON, iteration timing and provider query statistics match the fixed-batch reference.
It does not preserve event history or aggregates that depend on events.
The measured gain uses the factory's existing `trace=False` argument and must not be presented as a new event-engine speedup.
A request-rate producer must retain its active-user integral, completion timestamps and adjacent-token measurements independently of detailed trace retention before using this path.

The front cache keeps the reference validator and full structural service-cache lookup.
Unproven or mutable query fields fall back to structural normalization.
Its retained identities and entries are bounded.
The fragment encoder revisits mutable descendants and falls back on literal marker collisions.
The archive variants retain the reference transactions, explicit caller sizes, ordering and replacement semantics.

## Reproduction

Run any independent prototype with its listed kernel:

```bash
python scripts/profile_af_probe.py \
  --kernels pipeline \
  --steps 16 \
  --output outputs/probe-pipeline-16
```

Omit `--kernels` for the reference.
Use `--keep-trace` to retain the full trace or `--semantic-digest` to hash every parsed trace event.
The script writes `measurement.json` with elapsed times, peak resident memory, iteration rows, counters and public-output hashes.
Combination kernel names are comma-separated and applied in order.
The `frontcache` and `sparse` kernels replace the same wrapper and must be measured as alternatives.
Apply `pipeline` or `thread` before `fragments` so the fragment encoder also serves prefix export.

The concurrent diagnostic runs multiple interleaved batch-32 cohorts in one event environment:

```bash
python scripts/profile_af_concurrent_probe.py \
  --kernels pipeline \
  --cohorts 3 \
  --steps 4 \
  --output outputs/probe-concurrent-pipeline
```

It retains complete ordered traces and validates the physical key/value block inventory.
Its timing includes configuration, simulation and complete trace export.
Archive cleanup and trace hashing occur afterward.
It compares full trace bytes, iteration rows and provider counters.
It checks the 4 GB process limit during execution and after export.
Completed iteration graphs are released after each row is recorded.
This is simultaneous simulation work within one process.

## Evidence and validation

The [machine-readable evidence](../experiments/performance/single-process-20261017/results.json) contains 33 complete measured runs, branch commit identities, iteration rows, provider counters, peak memory and output hashes.
The aggregate suite completes 587 tests with 0 optional skips.
All eleven independent branches pass their full project suites.
Pinned Black formatting, its check, the source-file line limit and whitespace checks pass.
Optional native dependencies are isolated to their probes, and version-specific SimPy tests skip on other versions.
No production module or configuration schema is changed.
