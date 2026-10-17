# Integrated AF performance defaults

The `codex/perf-integrated-defaults` branch integrates the measured optimizations into the normal DeepSeek command.
The original command enables them without additional flags and retains complete validated traces.
The implementation uses Python and the standard library.

```bash
python scripts/run_deepseek_mla.py \
  --full-model \
  --a-tp 4 \
  --f-tp 32 \
  --batches 32 \
  --af-batch-size 32 \
  --context 8192 \
  --steps 100 \
  --device a100_sxm_80g \
  --network nvlink3 \
  --fallback analytical_only \
  --output outputs/v3-b32-c8192-s100
```

## Enabled behavior

| Area | Default implementation |
| --- | --- |
| Runtime planning | Bounded compiled template cache and binding reuse |
| Service estimates | Prepared queries, complete service keys and bounded immutable fingerprints |
| Communication planning | Immutable wire geometry and traffic resource reuse |
| Activation planning | Reuse validated endpoint and source-slice geometry |
| Python collection | Scoped allocation threshold with automatic collection and caller-policy restoration |
| Completed traces | Private disk archive with bounded read caching and complete ordered readback |
| Trace export | Encode finalized prefixes before archival and publish the complete file atomically |

The earlier runtime changes and their measurements are documented in [AF runtime performance](af-runtime-performance.md).
The integrated trace path uses 1,024-event archive chunks and one synchronous output buffer.
It removes the normal final archive-reading pass for JSON export.
It still performs JSON encoding, invariant validation and full archival during execution.
The shorter final export phase does not represent the whole-command speedup.

The command passes an intended trace path to `simulate_decode(..., trace_output=...)`.
The recorder prepares a temporary file in that path's parent directory.
`write_chrome_trace` publishes the complete file after execution supplies metadata.
Ordinary API calls without an output hint retain the existing archive and export behavior.
Both paths preserve event order, all event fields, simulated times and metadata.

Concurrent iterations can interleave event positions.
The recorder only encodes the contiguous prefix that has no outstanding active iteration.
Completed events behind an active gap remain readable from the archive until that prefix becomes available.
Explicit event replacement, archived event readback, partial exports, repeated exports, changed options and a different output directory use the complete archive for export.
Archived readback invalidates prepared bytes because nested event details can expose mutable values.
Failed publication preserves an existing destination and removes its temporary file.
Closing, clearing or abandoning the recorder releases its temporary resources.

## Experiment choices

Complete validated trace output remains the default.
Collection, validation, event types, layers, iterations and field detail are configurable through [AF trace configuration](af-trace-configuration.md).
Disabling trace collection changes the experiment's available outputs and remains an explicit choice.

The independent [single-process prototypes](af-single-process-exploration.md) remain available as diagnostics.
Native JSON encoders, JSON fragments, query identity caches and larger template programs are excluded from the default because the measurements showed regressions, additional memory or no established gain.
The SimPy loop and condition prototypes add implementation maintenance without a demonstrated whole-command benefit.
The optional writing thread showed no advantage over synchronous prefix export in the concurrent comparison and is excluded.
Sparse query normalization and specialized validation remain prototypes because their small screen gains do not establish a repeatable complete-command improvement.

## Measurements

The final source revision is `da6ac62`, and the fresh 16-step reference is `5119719`.
That reference already includes the earlier runtime, cache, collection and bounded-trace improvements and configurable trace output.
The host is an Apple M5 with 10 logical processors and 32 GiB of memory, running Python 3.12.12.
The reference and final 16-step measurements run sequentially in fresh interpreters with complete trace output.

| Workload | Reference total | Integrated total | Speedup | Integrated peak resident memory |
| --- | ---: | ---: | ---: | ---: |
| Original geometry, 16 steps | 115.265 s | 110.449 s | 1.044x | 1.488 GB |
| Original configuration, 100 steps | 688.741 s, historical | 639.529 s | 1.077x, historical comparison | 2.108 GB |

This fresh pair uses 4.18% less time.
It is one pair, so small differences remain sensitive to host variation.
All six output files are byte-identical, including the complete 1,081,520-event trace.
All iteration rows, provider request counts, cache hits and template counts match.
JSON preparation moves into the timed simulation, and the final publication call takes 0.000843 seconds.
The total interval includes both stages.
The [machine-readable evidence](../experiments/performance/integrated-defaults-20261017/results.json) records the full measurements and equality checks.

The 100-step reference is the previously recorded complete run described in [the runtime performance notes](af-runtime-performance.md).
It is a historical comparison, so the 49.212-second reduction does not establish a paired long-run speedup.
The new run completes all 6,759,500 events with 2,107,670,528 bytes of peak resident memory.
All 100 iteration rows, provider counts and all six JSON file hashes match that reference.
The matching uncompressed trace contains 32,392,112,426 bytes.
Its SHA-256 is `28f80d0b7a471c2e17bfb4feae77f5900e71c3912c72a17c820b930f5fa79b8a`.

The concurrent check runs three batch-32 streams with four iterations per stream in one process.
It completes in 92.794 seconds with 1,897,431,040 bytes of peak resident memory and no events remaining in active trace storage.
All 811,140 events produce the same complete trace hash as the recorded concurrent reference, and all iteration rows, provider counts and template observations match.
The hash is `af4384718c14133f639aafe2d6a654d930ec43b06f5aaee021db37ca089dfb9f`.
This is a short concurrent memory and output check and does not establish a paired concurrent speedup or memory bound for arbitrary concurrency.

All 604 unit tests pass, including trace selection, validation modes, interleaved completion, snapshot readback, late replacement, repeated export, output errors and temporary-file cleanup.
All edited source, test and script files satisfy the 500-line limit, and the pinned Black formatter and `git diff --check` pass.

## Reproduction

`scripts/profile_af_probe.py` with no `--kernels` selection measures the production command, including its default trace preparation.
A nonempty kernel selection measures the diagnostic implementation against the original archive/export path.
The harness measures configuration, simulation, validation, archival, all public JSON outputs and cleanup.
Imports and diagnostic output hashing are outside that timed interval.
The concurrent harness selects the production recorder with `--trace-pipeline`.

```bash
python scripts/profile_af_probe.py \
  --steps 16 \
  --output outputs/integrated-check
python scripts/profile_af_concurrent_probe.py \
  --trace-pipeline \
  --cohorts 3 \
  --steps 4 \
  --output outputs/integrated-concurrent-check
```

The per-process memory target is 4,000,000,000 bytes.
Measured memory applies to the tested workload and does not impose a universal resident-memory limit.
