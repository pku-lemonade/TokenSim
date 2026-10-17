# Follow-up profiling of production archive mode

This experiment uses revision `2b08559` with all integrated production optimizations and the production indexed archive output.
The configuration retains the original full 61-layer DeepSeek model, batch 32, context 8,192, attention tensor parallelism 4, feed-forward tensor parallelism 32, seed 42, A100 SXM 80 GB, NVLink 3 and analytical-only service estimates.
The simulator runs locally on the Apple M5 with Python 3.12.12.
The A100 identifies simulated hardware.
Complete trace details and invariant validation remain enabled.
These measurements cover the original fixed-batch command and do not measure the unavailable complete plotting scan launcher.

## Profiling method

An eight-step wall-clock stack sample covers the complete command after imports.
Its complete time is 46.512 seconds, including 45.668 seconds of simulation and 0.361 seconds of snapshot publication.
The profiler samples at one-millisecond intervals and records cyclic garbage collection separately.
A separate eight-step process instruments only the last environment run with the Python call profiler.
Instrumented timings are excluded from candidate speed comparisons.
Benchmarks execute sequentially in fresh interpreters with checkpoint reuse disabled.
Conversion, diagnostic hashing and unit tests execute outside benchmark intervals.

| Sampled category | Share of complete command |
| --- | ---: |
| Provider queries, structural keys and caches | 23.68% |
| DAG planning, compilation and binding | 15.71% |
| SimPy event engine | 11.43% |
| Cyclic garbage collection | 9.53% |
| Execution methods | 9.01% |
| Internal archive and snapshot | 8.93% |
| Resource arbitration | 5.30% |
| Other generated record methods | 5.06% |
| Trace capture | 4.26% |
| Trace validation | 4.16% |
| Other project methods | 2.70% |
| JSON handling | 0.24% |

Stack samples can attribute time inside native operations to a nearby Python frame.
The pickler reducer receives 2.067 sampled seconds across eight steps, but the call profiler records only about 0.02 seconds in its Python callback in the warm step.
Replacing that callback provides no measured whole-command gain.
Category shares identify work to investigate and do not directly predict removable time.
The warm step performs 3,012,083 fingerprint calls, 1,782,630 recursive fingerprint visits and 731,117 SimPy event steps.

## Prototypes

Each prototype has an explicit diagnostic installation context and leaves production defaults unchanged during its screen.

- `spanindex` stores one index row for each contiguous range within a trace chunk and exposes the existing logical positions through a compatibility view.
- `pickledispatch` registers the immutable dictionary reducer in a per-pickler dispatch table.
- `picklereduce` gives the immutable dictionary its reduction method and uses the ordinary C pickler.
- `inlinescalars` normalizes scalar children inside structural traversal loops.
- `tupletypes` classifies scalar tuple contents through C iterator operations.
- `keyhash` precomputes hashes for memoized immutable structural tuples.
- `gc1m` raises the generation-zero allocation threshold to one million while preserving automatic collection.
- `namespacehash` precomputes the hash of the complete provider configuration namespace and recomputes it after namespace extensions.
- `combined` installs `spanindex`, `pickledispatch` and `inlinescalars` together.

## Reproduction

```bash
.venv/bin/python scripts/profile_af_breakdown.py \
  --steps 8 \
  --mode sample \
  --trace-format archive \
  --output outputs/archive-follow-up/profile

.venv/bin/python scripts/profile_af_probe.py \
  --steps 8 \
  --trace-format archive \
  --production-path \
  --kernels namespacehash \
  --keep-trace \
  --output outputs/archive-follow-up/namespacehash
```

An empty `--kernels` selection measures the production reference.
The diagnostic timer includes configuration, simulation, complete trace retention, validation, snapshot publication and checkpoint work.
The original 100-step archive result remains documented in [the complete format comparison](../trace-full-20261017/README.md).

## Initial complete-command screens

Each row has one eight-step measurement, with the same simulation geometry and complete trace retention.
The baseline takes 45.978 seconds.
Positive time reductions indicate shorter commands.

| Prototype | Complete command | Time reduction | Peak resident bytes |
| --- | ---: | ---: | ---: |
| Production reference | 45.978 s | Reference | 1,114,144,768 |
| Interval index | 45.632 s | 0.75% | 1,137,819,648 |
| Pickler dispatch table | 45.999 s | -0.05% | 1,105,985,536 |
| Dictionary reduction method | 45.832 s | 0.32% | 1,140,064,256 |
| Inline scalar children | 45.373 s | 1.32% | 1,100,562,432 |
| C tuple type classification | 45.428 s | 1.20% | 1,126,268,928 |
| Memoized tuple hashes | 46.282 s | -0.66% | 1,122,861,056 |
| One-million allocation threshold | 46.729 s | -1.63% | 1,110,179,840 |
| Interval index, dispatch and scalar combination | 45.318 s | 1.44% | 1,117,093,888 |

A longer fresh-process pair reverses the order and runs the combination before the reference for sixteen steps each.
The reference takes 87.464 seconds and the combination takes 85.962 seconds, using 1.72% less time.
Their simulation times are 86.177 and 84.734 seconds, and their peak resident memory is 1.545 and 1.535 gigabytes.
The range index reduces 540,760 per-event position rows to 536 spans in the eight-step sequential workload, but that reduction produces less than one percent command-level benefit in its isolated screen.
This combination remains diagnostic because its additional storage implementation has a small measured benefit.

## Adopted namespace hash cache

The namespace hash prototype takes 44.821 seconds for eight steps, using 2.52% less time than the 45.978-second reference.
A sixteen-step confirmation takes 85.871 seconds against the 87.464-second reference, using 1.82% less time.
Peak resident memory is 1,112,997,888 bytes for eight steps and 1,536,524,288 bytes for sixteen steps.
These short measurements establish a small benefit and do not establish a twofold speedup or an exact 100-step gain.

Production stores namespace values in a frozen internal record with a precomputed hash.
The existing namespace extension operator constructs a new record and recomputes its hash.
All configuration fields remain part of equality, so a hash collision cannot reuse an incompatible service estimate.
The cache continues to include complete query keys and the existing estimate signature.
The optimization is confined to the internal cache namespace representation.
The implementation adds one small immutable record per namespace and keeps the existing cache bounds.

`namespaceplain` substitutes the original tuple namespace constructor for a comparison against the integrated production implementation.
`namespacehash` installs the original prototype when a provider starts with a tuple namespace and leaves an existing cached namespace intact.
The other prototypes remain opt-in diagnostics.

```bash
.venv/bin/python scripts/profile_af_probe.py \
  --steps 8 \
  --trace-format archive \
  --production-path \
  --kernels namespaceplain \
  --keep-trace \
  --output outputs/archive-follow-up/plain

.venv/bin/python scripts/profile_af_probe.py \
  --steps 8 \
  --trace-format archive \
  --keep-trace \
  --output outputs/archive-follow-up/production
```

## Integrated implementation verification

A fresh pair runs the integrated production implementation before the tuple reference for eight archive steps each.
Production takes 45.177 seconds against 46.113 seconds for the reference, using 2.03% less time.
Peak resident memory is 1,137,606,656 and 1,138,868,224 bytes respectively.
All five companion JSON hashes, all iteration rows, template counts, service requests and cache hits match.

A separate two-step pair covers the ordinary JSON output path.
The tuple reference takes 16.373 seconds and production takes 15.941 seconds.
This short pair is a compatibility check and does not establish the benefit for a complete 100-step JSON run.
All six public JSON files match byte for byte, including the complete 135,190-event trace.

Complete sixteen-step conversion checks cover the tuple reference, the archive combination and the namespace hash prototype.
Each JSON file contains 1,081,520 ordered events and occupies 5,179,494,044 bytes.
All three files have SHA-256 `e084532f8e5429bd08e3ce63bea60a4edb92181a84b5c15b165a8b6f0c30c3d2`.
The integrated eight-step archive and its tuple reference have identical payload bytes in all 536 chunks, identical 540,760 position rows, identical iteration membership and identical export metadata.
These checks include every event and detail field.

All 672 tests pass, including namespace hash collisions, profile-extension cache invalidation, mutable query children, interleaved range indexing, failed archive transactions and saved snapshot conversion.
Black formatting, the 500-line Python source limit and whitespace checks pass.
[results.json](results.json) records every screen and confirmation, the sampled categories and the complete converted-trace hashes.
