# Binary trace format probe

This diagnostic compares JSON with a MessagePack stream on the integrated Python runtime at `c7df462`.
The workload retains the original 61-layer DeepSeek model, batch 32, attention tensor parallelism 4, feed-forward tensor parallelism 32, context 8,192, A100 SXM 80 GB, NVLink 3, analytical-only estimates and seed 42.
The step count is reduced from 100 to 8.
All event fields, trace validation, event order, simulation logic and the indexed history archive remain enabled.
The host is an Apple M5 running Python 3.12.12 with MessagePack 1.1.1 and its native C encoder.
Times measure execution of the host simulator.

## Format and scope

The prototype replaces the prepared prefix writer through `scripts/af_probe_msgpack.py`.
It reuses the existing event-to-Chrome mapping and dataclass field extraction.
The output contains a versioned header, consecutive complete Chrome event mappings and a footer with metadata and an event count.
MessagePack stores floating-point values as binary64, preserving the timestamp and duration values used by the existing JSON exporter.
The native implementation supports streaming records and custom object conversion without a schema for every model-specific detail field.
See the [MessagePack Python implementation](https://github.com/msgpack/msgpack-python) and the [format specification](https://github.com/msgpack/msgpack/blob/master/spec.md).

The writer uses a one-mebibyte staging buffer and atomically publishes the completed stream.
The converter reads one record at a time, validates the header, footer, event count and complete input consumption, and atomically publishes the JSON destination.
Malformed or incomplete input preserves an existing destination.
The converter has a 16-mebibyte MessagePack buffer limit.
The prototype uses the format's signed or unsigned 64-bit integer range; larger Python integers require an extension before general adoption.
This remains an optional diagnostic dependency and introduces no production format option or default change.
Fallback exports after prepared-history invalidation continue to use the production JSON exporter.
A production format option would need consistent fallback handling, checkpoint identity and reader contracts.

## Measurements

Each run starts in a fresh interpreter with checkpoint reuse disabled.
The command timer includes configuration, simulation, trace encoding, publication and checkpoint work after imports.
Hashing and optional format conversion run after that timer.
Runs execute sequentially without another benchmark or test suite running beside them.
The exploratory pair executes JSON followed by MessagePack; the repeat reverses the order.
The repeat uses local writer options and encoder bindings instead of repeated attribute lookup in the exploratory prototype.
This loop adjustment does not change the stored values.

| Run | Complete command | Simulation | Peak resident memory |
| --- | ---: | ---: | ---: |
| Exploratory JSON | 56.207 s | 55.766 s | 1,104,953,344 bytes |
| Exploratory MessagePack | 51.906 s | 51.468 s | 1,127,415,808 bytes |
| Repeat MessagePack | 51.434 s | 50.992 s | 1,113,243,648 bytes |
| Repeat JSON | 56.096 s | 55.659 s | 1,136,754,688 bytes |

The exploratory MessagePack run reduces total command time by 7.7%, corresponding to 1.083x speedup.
The reverse-order pair reduces command time by 8.3%, corresponding to 1.091x speedup.
The two JSON runs average 56.152 seconds, and the two MessagePack runs average 51.670 seconds.
The ratio of those means is 1.087x, corresponding to an 8.0% reduction in total command time.
Two observations per format provide a small sample without a confidence interval.
These short runs do not establish the speedup of the original 100-step command or the complete plotting experiment sweep.

| Trace representation | Bytes |
| --- | ---: |
| Original JSON | 2,588,215,556 |
| MessagePack | 2,110,891,135 |

The binary file is 18.4% smaller and preserves all 540,760 events.
The JSON conversion takes 15.655 seconds with a peak resident memory of 71,155,712 bytes in a separate fresh interpreter.
Adding that conversion to the exploratory binary command takes 67.561 seconds, exceeding the direct JSON command's 56.207 seconds.
This workflow saves execution time when binary traces remain the stored artifact and JSON conversion is requested for selected traces later.
It does not save total command-plus-conversion time when every trace immediately needs JSON.
These measurements use ordinary buffered writes and atomic rename, without a durable filesystem synchronization requirement.

## Equivalence and consumers

The converted JSON is byte-identical to the original JSON.
Both have SHA-256 `540c00510aca1d7127275a2443048fa1bdd5f117a8e43265f7df1f2523da7046`.
All iteration rows, provider request and cache counters, and template counters match.
The four configuration JSON files have matching hashes.
The summary JSON matches after normalizing the trace filename extension.
The two binary outputs have matching hashes, and the repeated JSON output matches the original and converted JSON hashes.
The user's `plot_tps_user_vs_tps_mw.py` reads the small `results.json` metrics list and does not consume trace events.
Changing the stored trace representation therefore requires no modification to that plotting script.

MessagePack output requires a compatible reader or conversion before use in a Chrome JSON timeline viewer.
Perfetto's native protobuf format is another possible viewer-compatible output, but this probe does not implement or measure it.
Mapping arbitrary nested detail fields and preserving existing event semantics would require separate validation.
See [Perfetto trace generation](https://perfetto.dev/docs/getting-started/converting) and [trace conversion](https://perfetto.dev/docs/quickstart/traceconv).

The preceding runtime profile attributes approximately 20.4% of sampled complete time to JSON conversion, encoding and prefix output.
Removing that measured work entirely would have a 1.256x isolated speedup bound under that profile's partition.
The binary probe still performs event mapping, encoding, buffering and publication, so it cannot realize that elimination bound.
Trace creation, validation and SQLite archival remain outside this format replacement.
A separate design could persist a single binary history instead of both the private archive and a public export, but this probe does not establish that design's performance or API compatibility.

## Validation

Four new unit tests cover complete and selected event round trips, empty selection, nested dataclasses, Unicode, integer mapping keys, malformed and truncated streams, failed publication cleanup, installation cleanup and the prototype's integer range limit.
Black formatting, the 500-line source limit, all 639 unit tests and whitespace checks pass.

## Reproduction

Install the optional native encoder in the diagnostic environment.

```bash
python -m pip install msgpack==1.1.1
python scripts/profile_af_probe.py \
  --steps 8 \
  --keep-trace \
  --output outputs/trace-format-20261017/json
python scripts/profile_af_probe.py \
  --kernels msgpack \
  --production-path \
  --steps 8 \
  --keep-trace \
  --output outputs/trace-format-20261017/msgpack
python -m scripts.af_probe_msgpack \
  outputs/trace-format-20261017/msgpack/decode-layers61-b32.trace.msgpack \
  outputs/trace-format-20261017/msgpack/converted.trace.json
```

Repeat the binary command and then the JSON command with fresh output directories to reproduce the reverse-order pair.
Keep `--keep-trace` for the binary probe because the generic harness's cleanup assumes a JSON trace filename.
Large trace files are diagnostic artifacts outside version control.
Only the initial MessagePack trace is retained after verification; the other large traces and diagnostic checkpoint directories are removed.
The checked-in `results.json` records timings, sizes, counters and complete output hashes.
