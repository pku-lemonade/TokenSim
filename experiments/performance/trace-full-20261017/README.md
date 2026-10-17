# Complete 100-step trace persistence comparison

This experiment compares JSON, MessagePack and a snapshot of the existing indexed binary archive at revision `5d0ec44`.
Every run executes the original 100-step DeepSeek workload with all integrated production optimizations enabled.
The configuration uses 61 layers, batch 32, attention tensor parallelism 4, feed-forward tensor parallelism 32, context 8,192, seed 42, A100 SXM 80 GB, NVLink 3 and analytical-only latency estimates.
Every trace event and detail field remains enabled.
The host runs Python 3.12.12 on an Apple M5.
The simulated A100 hardware describes the latency model; the benchmark measures simulator execution on the Apple host.

The supplied plotting script reads saved `results.json` metrics and requires no trace conversion.
Its archived eight-point scan launcher and original input metrics are absent from this checkout.
These measurements cover the complete original fixed-batch command and do not measure the complete eight-point serving scan.

## Measurement method

The three simulation commands run sequentially in fresh Python interpreters with checkpoint reuse disabled.
The order is JSON, MessagePack and indexed archive snapshot.
The complete command timer starts after imports and includes configuration, simulation, trace preparation, export, checkpoint work and publication.
Reading files to compute diagnostic hashes occurs after the timed command.
Binary trace conversion runs in a separate process after its simulation command, and its time is reported separately.
Conversion timing includes writing the complete JSON file and excludes the later hash calculation.
Peak resident memory comes from each process's operating-system resource usage.
The benchmark does not run another simulation, conversion or test suite concurrently with a measured command.
Each format has one complete execution, so the observed differences have no repetition-based confidence interval.

## Complete command measurements

| Representation | Complete command | Time reduction versus JSON | Trace bytes | Peak resident memory |
| --- | ---: | ---: | ---: | ---: |
| JSON | 627.661 s | Reference | 32,392,112,426 | 2,137,341,952 bytes |
| MessagePack | 558.640 s | 11.0% | 26,412,772,131 | 2,119,057,408 bytes |
| Existing indexed archive snapshot | 495.089 s | 21.1% | 5,731,635,200 | 2,130,608,128 bytes |

MessagePack saves 69.022 seconds and provides 1.124x speedup relative to JSON.
Publishing the indexed archive saves 132.573 seconds and provides 1.268x speedup relative to JSON.
The MessagePack file is 18.5% smaller than JSON, and the archive snapshot is 82.3% smaller.
Every simulation process stays below the four-gigabyte memory target, with peaks between 2.119 and 2.138 gigabytes.
File sizes and memory in this report use decimal bytes; four gigabytes means 4,000,000,000 bytes.

The JSON simulation phase takes 626.821 seconds, and the MessagePack simulation phase takes 557.740 seconds.
Both formats prepare their public output during simulation, so their final export calls take approximately one millisecond.
The archive simulation phase takes 491.204 seconds, and its snapshot export takes 3.128 seconds.
The archive snapshot time is included in the reported complete command time of 495.089 seconds.

The archive keeps the existing event construction, validation, position indexes and internal pickle encoding.
It removes the additional traversal and encoding needed to prepare the public JSON or MessagePack stream.
The resulting file depends on compatible TokenSim record definitions when reopened.
MessagePack stores ordinary field mappings and preserves the existing private archive while preparing an additional public stream.
The measured speed and storage advantage favors the indexed archive when experiments can retain it and convert selected traces later.

## Conversion and equivalence

| Stored representation | Conversion to JSON | Conversion process peak resident memory | Command plus immediate conversion |
| --- | ---: | ---: | ---: |
| JSON | Already JSON | Included above | 627.661 s |
| MessagePack | 204.095 s | 91,193,344 bytes | 762.735 s |
| Indexed archive snapshot | 171.564 s | 148,307,968 bytes | 666.653 s |

Immediate JSON conversion makes the MessagePack workflow 21.5% slower than direct JSON and the archive workflow 6.2% slower.
The simulation command and conversion process run sequentially, so their resident memory peaks are separate.
The storage improvement remains available if the converted JSON is discarded after inspection.
The plotting script consumes metrics directly, so its normal use does not require these conversion costs.

Every run produces 6,759,500 events.
Both converted traces have exactly 32,392,112,426 bytes and SHA-256 `28f80d0b7a471c2e17bfb4feae77f5900e71c3912c72a17c820b930f5fa79b8a`, matching the direct JSON output byte for byte.
Every iteration row, compute and transfer cache counter, template cache counter and configuration JSON hash matches.
The summaries match after normalizing the output trace filename.
The [complete measurement record](results.json) contains all 100 per-step timings for each representation and both conversion measurements.
The [comparison figure](comparison.svg) shows complete command time, stored trace size and process memory.

## Reproduction

Run the commands sequentially with distinct output directories.
The diagnostic harness supplies the complete original model configuration and passes `--no-resume`.

```bash
python scripts/profile_af_probe.py \
  --steps 100 \
  --keep-trace \
  --output outputs/trace-full-20261017/json
python scripts/profile_af_probe.py \
  --kernels msgpack \
  --production-path \
  --steps 100 \
  --keep-trace \
  --output outputs/trace-full-20261017/msgpack
python scripts/profile_af_probe.py \
  --kernels archiveonly \
  --production-path \
  --steps 100 \
  --keep-trace \
  --output outputs/trace-full-20261017/archiveonly
```

Plain MessagePack uses the optional native `msgpack==1.1.1` package.
The archive snapshot uses the Python standard library and the current TokenSim record definitions.
The production default remains JSON; both binary outputs use the existing opt-in diagnostic implementations.

```bash
python -m scripts.af_probe_msgpack \
  outputs/trace-full-20261017/msgpack/decode-layers61-b32.trace.msgpack \
  outputs/trace-full-20261017/msgpack/converted.trace.json
python -m scripts.af_probe_archiveonly \
  outputs/trace-full-20261017/archiveonly/decode-layers61-b32.trace.sqlite3 \
  outputs/trace-full-20261017/archiveonly/converted.trace.json
```

The complete public JSON trace, configuration files, iteration rows and cache counters are compared across representations.
Summary comparison normalizes the trace filename extension.
Large public JSON, MessagePack, converted JSON and checkpoint trace links are removed after recording their byte counts and hashes to bound disk use.
The indexed archive snapshot remains available locally, and the checked-in measurement record preserves timing rows, simulation values, file sizes and complete trace hashes.
