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
