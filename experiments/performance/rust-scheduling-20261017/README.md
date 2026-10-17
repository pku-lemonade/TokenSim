# Native resource scheduling on the integrated runtime

## Scope

This experiment evaluates a larger Rust boundary around communication resource queue state and atomic first-in-first-out (FIFO) arbitration.
The reference is the integrated Python runtime at `f7f74ff` on `codex/perf-integrated-defaults`.
The previous Rust timings used an older Python reference and do not describe the gains measured here.
Production model definitions, provider interfaces, planner nodes, trace configuration and default execution remain unchanged.
Rust is an optional diagnostic dependency.

The workload uses the original 61-layer DeepSeek model, batch 32, attention tensor parallelism 4, feed-forward tensor parallelism 32, context 8,192, A100 SXM 80 GB, NVLink 3, seed 42 and analytical-only estimates.
Each normal command runs in a fresh Python 3.12.12 process with checkpoint reuse disabled.
Every measurement retains full trace fields, invariant validation, bounded archival, prepared trace publication and all six public JSON files.
The timed interval includes configuration, simulation, trace preparation, publication and archive cleanup.
Imports, kernel installation and diagnostic hashing remain outside that interval.
Processes run sequentially on an Apple M5 host with 10 logical CPUs and 32 GiB of memory.
[results.json](results.json) records individual measurements, complete output hashes, service counters, source digests and profiler summaries.

## Native boundary

`scripts/af_rust/src/domains.rs` owns resource names, native priority queues, dirty-resource tracking, pending requests and ownership.
Queue scans, conflict checks, stable candidate ordering and cancellation cleanup operate without Python callbacks inside arbitration.
The native priority compares the simulated timestamp and the existing typed logical identity.
Integer identity components precede string components, matching the reference's type-name ordering.
Native containers retain requests until release and discard cancelled queue entries when they reach a resource front.
The Python adapter retains the existing request records, SimPy events, priority-2 arbitration turn and service generator.
Provider estimates, model operations and trace assembly continue in Python.

The prototype supports string resource names, signed 64-bit integer or Unicode string identity components and finite floating-point timestamps.
Integer timestamps must be within the inclusive range from `-2**53` to `2**53`.
Unsupported identity components and oversized integer timestamps fail explicitly before changing native queue state.
The diagnostic `owners` property returns a snapshot.
Custom subclasses that mutate the reference scheduler's private dictionaries require the reference implementation.
These limits are additional reasons to retain the experiment outside production defaults.

The existing `keys` and `templates` kernels can be combined with `domains`.
The separate `events` kernel retains Python callbacks and the Python event heap.
It measures whether moving dispatch across the language boundary adds value after the other native kernels are enabled.

## Measurements

The initial screening repeats each four-step variant twice in reverse order.
Times below use the median of each variant's two measurements.

| Four-step variant | Complete command | Speedup against integrated Python |
| --- | ---: | ---: |
| Integrated Python | 28.742 s | 1.000x |
| Native resource queues | 28.185 s | 1.020x |
| Native keys, templates and resource queues | 27.038 s | 1.063x |
| Native keys, templates, resource queues and dispatch | 27.837 s | 1.032x |

The resource queue gain includes Python validation, native argument conversion and SimPy event integration.
The combined gain includes the existing optional key and template kernels.
It must not be described as an additional gain from resource queues alone.
The event-dispatch variant reduces the benefit of that combination.

The final source comparison repeats the existing `keys,templates` selection and the `keys,templates,domains` selection twice in reverse order.
Their complete-command medians are 28.483 and 27.389 seconds, an incremental 1.040x speedup from adding the queue kernel to that optional selection.
Against the integrated Python screening median, the final combination measures 1.049x.
The difference between screening and final medians illustrates the limits of this short sample.
The final source additionally rejects oversized integer timestamps before floating-point rounding and exposes the existing fingerprint immutability result to other diagnostic probes.
These changes preserve the normal workload's queue ordering, and the evidence records both source digests.

A longer comparison uses the original model geometry for 16 steps.

| Sixteen-step variant | Complete command | Peak resident memory | Speedup |
| --- | ---: | ---: | ---: |
| Integrated Python | 109.480 s | 1,529,888,768 bytes | 1.000x |
| Native keys, templates and resource queues | 102.512 s | 1,751,302,144 bytes | 1.068x |

The native combination uses 6.36% less time in this single longer pair.
Both runs produce 1,081,520 trace events and identical complete outputs.
The native process uses more memory, and its measured peak remains below the 4,000,000,000-byte per-process target.
This observation does not establish peak memory for a larger concurrent workload.
The complete 100-step command is not repeated in this experiment.

The isolated scheduling comparison runs 262,144 positive-duration services across 32 resources in batches of 64 services.
Each service reserves one to three resources atomically.
Five repeats alternate backend order, and every ready/start/end interval has the same digest.
This workload includes SimPy integration and Python adapter costs but excludes model planning, providers and trace generation.
Its median times are approximately 1.360 seconds for Python and 1.084 seconds for Rust, a 1.255x speedup.
That local speedup must not be applied to the complete simulation.

## Profile interpretation

Fresh profiles instrument only the fourth environment-run interval of a four-step simulation.
The reference profile contains 13.437 seconds of function self time.
The table groups exclusive function time by module and avoids adding overlapping cumulative entries.

| Reference profile group | Self time | Share of instrumented self time |
| --- | ---: | ---: |
| Query cache and structural keys | 2.222 s | 16.5% |
| Python execution methods | 1.544 s | 11.5% |
| SimPy Python modules | 1.270 s | 9.4% |
| JSON encoder | 1.202 s | 8.9% |
| DAG template compilation and binding | 1.105 s | 8.2% |
| Python trace modules | 0.908 s | 6.8% |
| Python resource scheduler | 0.588 s | 4.4% |

The native queue profile records 0.344 seconds in its Python scheduler/adapter and 0.106 seconds in native queue calls.
Built-in heap operation time across the process falls from 0.263 to 0.154 seconds because the native queues own their heaps.
The event queue still uses Python's C-backed heap operations.
Other time includes built-ins, generated record initializers, planning and provider work outside the listed modules.
Generated dataclass initializer entries can be combined by the profiler when their generated filename, line and function name match.

Profiling overhead changes the apparent shares and makes these percentages diagnostic observations rather than precise normal wall-time fractions.
The measured normal complete-command times determine the adoption decision.
The profile does not identify a resource or dispatch kernel that dominates half of the complete run.
Replacing additional short scheduling functions alone does not demonstrate a route to a twofold complete-run gain.

## Validation and limits

The evidence contains 14 normal complete-command measurements and two separately instrumented runs.
Every run compares all six public JSON hashes, every iteration row, event counts, template counters and provider request/hit counters against a reference with the same step count.
The comparisons preserve complete trace bytes and ordering.
The optional native tests compare 40 arrival/cancellation histories against the existing reference scheduler across 20 seeded workloads with 80 services each.
They also cover mixed identity ordering, duplicate priority rejection, reused priorities after owner release, unsupported typed inputs, completed-request reclamation and installation cleanup after errors.
Existing native contracts exercise dense and mixture-of-experts traces, mutable query descendants, bounded caches, same-time event ties, interrupts, stop/resume and customized environment fallback.

All 635 tests pass with production defaults.
All 635 tests also pass with the four Rust kernels installed after test discovery.
Discovery precedes installation so explicit reference-oracle subclasses retain their Python implementation.
The explicit queue comparisons exercise native state independently of those oracles.
Black 24.2.0, Rust formatting, Clippy with warnings treated as errors, the 500-line source limit and whitespace checks pass.

The longer result is one paired measurement and does not characterize a timing distribution.
The original plot scan launcher is unavailable, so these measurements do not establish the speedup of a complete plot scan.
Full trace generation and invariant validation remain enabled throughout the comparisons.

## Development cost and decision

Model authors can retain the current Python definitions when a native backend consumes existing validated runtime data through an internal adapter.
Native resource state still adds Rust build requirements, ownership rules, boundary validation and cross-backend equivalence checks for backend maintainers.
Recent `afd` changes such as selected output normalization (`01a7070`) and independent prefill overlap (`3051bfe`) retain their Python implementation above this queue boundary.
Changes to queue arbitration such as dirty-resource tracking (`b85efb0`) affect the shared scheduling contract and require validation against both implementations.
The small boundary tested here provides a small complete-run improvement.
It does not meet the previously requested 2x complete-run adoption gate.
Production defaults retain the integrated Python implementation.

A larger implementation would need a generic execution representation for dependency counters, resource services, phase completion and event ordering.
Provider callbacks could remain in Python at explicit cache-miss boundaries.
Full trace records would need bounded conversion into the existing export schema.
New model operations would need supported generic instructions or a reference fallback.
Dynamic cancellation, same-time ordering and unsupported custom execution would require equivalent fallback and cleanup contracts.
That design could preserve model-facing APIs, but its performance and maintenance cost require a separate prototype.
This experiment does not establish that a complete native engine would double throughput.

## Reproduction

Build the pinned optional extension before selecting native kernels.

```bash
python scripts/profile_af_rust.py \
  --build \
  --offline

python scripts/profile_af_probe.py \
  --production-path \
  --steps 4 \
  --output outputs/native-queues-reference

TOKENSIM_RUST_PROBE_KERNELS=domains python scripts/profile_af_probe.py \
  --production-path \
  --kernels rustdomains \
  --steps 4 \
  --output outputs/native-queues-only

TOKENSIM_RUST_PROBE_KERNELS=keys,templates,domains python scripts/profile_af_probe.py \
  --production-path \
  --kernels rustdomains \
  --steps 16 \
  --output outputs/native-queues-combined

python scripts/profile_af_domains.py \
  --jobs 262144 \
  --width 64 \
  --repeats 5 \
  --output outputs/native-queues-isolated.json

python scripts/profile_af_probe.py \
  --production-path \
  --steps 4 \
  --profile-step 3 \
  --output outputs/native-queues-profile

python -m unittest \
  tests.af.test_rust_probe \
  tests.af.test_rust_domains
```

Omit `--offline` when the pinned Rust crates have not been downloaded.
`--profile-step` selects a zero-based environment-run interval and writes `runtime.prof` in the output directory.
Instrumented timings must be kept separate from normal command measurements.
The complete-run harness hashes each public output before deleting the large trace and checkpoint directory.
Use `--keep-trace` to retain those files for independent inspection.
