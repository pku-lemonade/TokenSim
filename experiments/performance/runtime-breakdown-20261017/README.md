# Integrated runtime time breakdown

This profile measures the integrated Python defaults at `ab31a79` on `codex/perf-integrated-defaults`.
The workload retains the original 61-layer DeepSeek model, batch 32, attention tensor parallelism 4, feed-forward tensor parallelism 32, context 8,192, A100 SXM 80 GB, NVLink 3, analytical-only estimates and seed 42.
Full trace fields, invariant validation, bounded archival and prepared JSON publication remain enabled.
Only the step count is reduced from 100 to 8 or 16.
No Rust kernel is selected.
The host is an Apple M5 with 10 logical CPUs and 32 GiB of memory, running Python 3.12.12.
All times describe the host simulator execution.

## Timing scope

Each measurement starts in a fresh interpreter with checkpoint reuse disabled.
The existing `profile_af_probe.py` harness times the complete command after imports, including configuration, simulation, trace preparation, publication and checkpoint work.
Diagnostic hashing and removal of large output files occur after the timed interval.
The profiler excludes imports and these post-run diagnostic operations.
The function profiler's statistics dump remains inside the harness command timer but outside the instrumented interval.

| Run | Steps | Complete command | Simulation | Trace events | Peak resident memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal defaults | 8 | 55.349 s | 54.919 s | 540,760 | 1,114,226,688 bytes |
| Function profiler | 8 | 123.106 s | 121.920 s | 540,760 | 1,107,410,944 bytes |
| Wall-clock sampler | 8 | 57.497 s | 57.027 s | 540,760 | 1,136,132,096 bytes |
| Wall-clock sampler | 16 | 109.856 s | 109.308 s | 1,081,520 | 1,514,307,584 bytes |

The eight-step normal run spends 54.729 seconds inside environment execution, 0.190 seconds elsewhere inside simulation and 0.430 seconds outside simulation.
These exclusive phases account for 98.9%, 0.3% and 0.8% of complete command time.
The final `write_profile` call takes approximately 0.00028 seconds because the prepared trace bytes have already been written during environment execution.
That publication time does not measure the cost of generating the trace.

## Wall-clock proportions

The table partitions sampled time into mutually exclusive categories.
The steady interval uses the final eight environment runs of the sixteen-step sample, excluding command setup and the earlier iterations.
Seconds in the complete column are sampled wall time and include the sampler's overhead.

| Category | Complete 16-step seconds | Complete share | Final eight steps share |
| --- | ---: | ---: | ---: |
| Trace JSON conversion, encoding and prefix output | 22.412 | 20.4% | 21.4% |
| Provider queries, structural keys and estimate caches | 21.982 | 20.0% | 20.8% |
| DAG construction, template compilation and binding | 13.041 | 11.9% | 10.7% |
| SimPy event engine and event objects | 10.529 | 9.6% | 10.0% |
| Python model and communication execution methods | 8.427 | 7.7% | 8.1% |
| Garbage collection | 7.582 | 6.9% | 6.1% |
| Trace archival and indexed history | 7.499 | 6.8% | 7.2% |
| Communication resource arbitration and lookup | 4.954 | 4.5% | 4.7% |
| Trace invariant checks and occupancy helpers | 3.991 | 3.6% | 3.9% |
| Trace event construction and capture | 3.746 | 3.4% | 3.6% |
| Other generated dataclass methods | 3.671 | 3.3% | 2.6% |
| Other project code and configuration | 2.020 | 1.8% | 1.1% |

Trace categories together account for 34.3% of the complete sample and 36.0% of the final eight steps.
The final eight environment runs take 52.931 seconds, averaging 6.616 seconds per step.
The initial two runs include approximately 3.218 seconds of DAG work, compared with approximately 1.398 seconds across the final two runs.
Initial planning costs therefore have a larger share in a short complete run.
The sixteen-step observation does not establish the exact distribution of the original 100-step command.

The communication resource category is 4.5% of sampled complete time.
Eliminating that category entirely would yield approximately `1 / (1 - 0.0451) = 1.047x` speedup under this partition and workload.
That arithmetic is an upper bound for removing this measured work, not a prediction of a Rust implementation.
The SimPy category is 9.6%, corresponding to an isolated elimination bound of approximately 1.106x.
Native code could cover work across multiple categories, but these measurements do not demonstrate such a combined implementation or its speedup.

## Profiling method and function evidence

`scripts/profile_af_breakdown.py --mode sample` uses a one-millisecond POSIX interval timer to inspect the interrupted Python stack.
The nearest recognized frame owns the sample, including work in native helpers called from that frame.
Elapsed intervals weight samples, and garbage-collection callbacks measure collection durations separately to avoid counting them again in the interrupted category.
Generated dataclass methods receive distinct diagnostic code filenames without function wrappers or altered execution bodies.
This identifies trace constructors separately from other records, and the original code objects are restored afterward.
Category definitions are in `scripts/profile_af_breakdown.py`.

The eight-step sampler takes 3.9% more complete time than the normal eight-step run in this sequence.
The independent sixteen-step sample produces similar major shares after the initial iterations.
Timer delivery can be deferred inside native code, and Python stack sampling does not provide exact native call stacks.
These proportions are approximate wall-clock observations, not per-function timing guarantees or confidence intervals.

`--mode calls` instruments the complete command using `cProfile`.
It records exclusive function time and assigns built-in and generic helper self time to their direct caller categories using `pstats` caller edges.
Function cumulative times overlap and are not added to form category proportions.
The function profile takes approximately 2.22 times the normal complete time, so its category shares are not used as the primary wall-clock percentages.
Its DAG share is 20.6%, compared with 13.0% in the eight-step wall sample, illustrating the effect of instrumentation on frequently called Python functions.

The eight-step function profile records 14,291,353 calls to `FingerprintMemo._visit`, 24,097,176 calls to `FingerprintMemo.__call__`, 9,654,631 calls to template `compile_value` and 12,132,119 calls to template `bind_value`.
It records 532,952 estimate-cache wrapper calls, 5,848,936 SimPy environment steps and 501,504 communication resource requests.
The cache receives 38,768 compute requests with 37,384 hits and 494,184 transfer requests with 493,944 hits.
These counts show that most provider work in this workload handles cached estimates and their keys.
[results.json](results.json) includes the largest 50 function entries from each profile, category totals, per-step wall samples, collection timings, complete output hashes and phase measurements.
The complete binary function profile remains at `outputs/runtime-breakdown-20261017/profile/runtime.prof` in this checkout.

## Equivalence and checks

All six public JSON hashes match between the normal, function-profiled and sampled eight-step runs.
The sixteen-step sample also matches all six public JSON hashes and every iteration row from the existing normal sixteen-step reference in `rust-scheduling-20261017`.
Event counts, provider request and hit counts, and template request and hit counts match the corresponding references.
The new scripts are diagnostic and do not change production imports, execution defaults or model-facing APIs.
Black formatting, the 500-line source limit, all 635 unit tests and whitespace checks pass.

## Reproduction

Run each command sequentially in a fresh output directory.

```bash
python scripts/profile_af_probe.py \
  --steps 8 \
  --output outputs/runtime-breakdown-normal
python scripts/profile_af_breakdown.py \
  --mode calls \
  --steps 8 \
  --output outputs/runtime-breakdown-functions
python scripts/profile_af_breakdown.py \
  --mode sample \
  --steps 8 \
  --output outputs/runtime-breakdown-samples-short
python scripts/profile_af_breakdown.py \
  --mode sample \
  --steps 16 \
  --output outputs/runtime-breakdown-samples-long
```

The wall sampler requires an unused POSIX `SIGALRM` timer in the main thread.
The diagnostic scripts write `measurement.json`, `progress.json` and `breakdown.json`.
The harness removes the large public trace and private checkpoint output after computing their output hashes.
The public JSON digests remain in `measurement.json`.
