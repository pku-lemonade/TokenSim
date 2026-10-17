# Configurable AF traces

The DeepSeek decode runner and the shared `simulate_decode` API support independent event collection, invariant validation and export choices.
The default remains a complete validated trace with the existing output schema.
The DeepSeek command prepares finalized JSON during simulation and publishes the complete file after execution by default.
This default preserves every trace event and field and needs no additional flag.
Partial or repeated exports, changed export settings and edits to archived events use the complete archive as a fallback.
See [the integrated performance measurements](af-integrated-defaults.md) for the implementation scope and validation.
These options do not change modeled execution, iteration timing rows or latency-provider statistics.
The general serving CLI and `AFService` retain their existing trace interfaces.

## Collection modes

| CLI option | Construct events | Validate events | Completed event storage | Chrome trace file |
| --- | --- | --- | --- | --- |
| `--trace full` | Yes | Yes by default | Temporary disk archive with bounded read caching | Yes |
| `--trace validate` | Yes | Yes | Released after each iteration | No |
| `--trace off` | No | No | None | No |

Use `off` for fixed-batch throughput experiments that need only `summary.json` and provider statistics.
Use `validate` when an experiment must check event invariants without retaining a complete event history or writing Chrome JSON.
Validation mode retains all active iterations, so its peak event memory depends on simultaneous work.
Use `full` for detailed timeline, communication, dependency or resource analysis.
The `--no-trace-validation` flag disables automatic event invariant checks in full mode while preserving complete event recording and export.
Configuration validation, execution errors and capacity admission checks remain active in every mode.
The combination `--trace validate --no-trace-validation` is rejected.

Append the chosen options to the existing command, for example:

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
  --trace off \
  --output outputs/v3-throughput
```

## Export selection

All export selectors require full collection mode.
Different selectors intersect, and values within one selector form a union.
Omitted selectors include every event, and an explicit empty list in Python selects no events.
Selections without matching events produce a valid empty Chrome trace.

| Option | Meaning |
| --- | --- |
| `--trace-event-types compute transfer collective` | Export the selected event types |
| `--trace-iterations 0 1 10` | Export selected zero-based iteration identifiers |
| `--trace-layers -1 0 1` | Export model-boundary events and selected zero-based layers |
| `--trace-detail full` | Export every existing event field |
| `--trace-detail timing` | Export event identifiers, resource identifiers, operation names, counts, dependency identifiers and timestamps |
| `--trace-fields node_id predecessor_ids payload_bytes` | Export exactly these event argument fields while retaining the Chrome timeline envelope |

Event types are `compute`, `transfer`, `collective`, `control_transfer`, `control_device`, `operation_service`, `host_submit`, `host_prepare` and `synchronize`.
The parser derives this list from the event record's declared type.
Boundary layer `-1` includes model-level operations such as embedding and output processing.
Iteration identifiers restart for each batch-size run in the fixed-batch runner.

An experiment exporting communication timing for a few iterations can use:

```bash
python scripts/run_deepseek_mla.py \
  --batches 2 \
  --context 128 \
  --steps 4 \
  --fallback analytical_only \
  --trace full \
  --trace-event-types transfer collective \
  --trace-iterations 0 3 \
  --trace-layers 0 \
  --trace-detail timing \
  --output outputs/communication-timing
```

Filters apply during export, after complete dependency patching and validation.
They reduce JSON conversion work and output size while full event construction and temporary archival still occur.
They do not provide the event-construction savings of `--trace off`.
Filtered events preserve their original order and predecessor identifiers, including references to omitted events.
A filtered trace cannot support complete dependency-chain reconstruction or complete resource-utilization analysis.
Timing detail omits provenance, arbitrary details, routing edges, tensor layouts, tensor slices and other payload descriptions.
Timing detail preserves the Chrome timeline envelope and the exact fields listed in `TIMING_FIELDS` in `TokenSim/af/trace/config.py`.
Explicit fields select names declared by `AFTraceEvent` and are an alternative to timing detail.
The Chrome event name, category, resource lane, timestamp and duration remain present even when their corresponding argument fields are omitted.
An empty Python or JSON `fields` list exports an empty argument object for every selected event.
Omitting `predecessor_ids` prevents dependency reconstruction and is recorded in trace configuration metadata.
Resource release details can differ from event completion time, so timing-only exports are insufficient for reconstructing exact resource occupancy.

## Saved representation

`--trace-format json` saves the existing Chrome JSON and is the default.
`--trace-format msgpack` saves a versioned MessagePack stream with the same selected events, fields and metadata.
`--trace-format archive` saves an indexed SQLite snapshot of the retained internal events with the export configuration and metadata.
The archive retains all internal event fields and applies event and field selections when generating JSON.
JSON and MessagePack apply those selections while writing their saved event stream.
All formats preserve the modeled results and support the existing collection and validation choices.
Validate and off modes produce no saved trace in any representation.

The [experiment runner](af-experiment-runner.md#saved-trace-formats-and-json-conversion) accepts `trace_format` in its constructor and individual `run` calls.
Its `export_trace_json(source, output)` method converts a completed point or saved file to Chrome JSON without rerunning simulation.
MessagePack requires the `trace` installation extra.
Archive uses compatible TokenSim record definitions during conversion.
The saved suffixes are `.trace.json`, `.trace.msgpack` and `.trace.sqlite3`.

## Configuration files and experiment reuse

`--trace-config path.json` loads an object containing `mode`, `validate`, `detail`, `event_types`, `iterations`, `layers` and `fields` as needed.
Explicit trace flags override matching file entries, and unknown keys are rejected.
The [reusable experiment runner](af-experiment-runner.md) accepts the same settings as a Python configuration or mapping and checkpoints each complete point independently.
The DeepSeek command uses that runner automatically and supports `--no-resume`, `--verify-checkpoints sha256` and `--seed`.
`trace_format` is a separate runner option selected by `--trace-format` and is not accepted inside the trace selection file.

## Output contracts

Default runs preserve the previous summary and trace contents.
Nondefault runs record configured and effective trace choices in `summary.json` under `trace_config`.
Nondefault saved formats also record `trace_format` in the command's `summary.json`.
Full traces with nondefault settings record the same information in `metadata.trace_config`.
This metadata key is reserved for the exporter configuration.
Validation and off modes set each summary row's `profile` to JSON `null` and create no trace file.
Use separate output directories for experiments; disabled modes leave any files from earlier runs untouched.
Only full mode supports historical event readback after an iteration completes.
Close the returned recorder after reading or exporting its history.

Disabling detailed events preserves the fixed-batch timing and throughput statistics already produced by this runner.
It does not add request-arrival, active-user, resource-utilization or time-between-token statistics to an unrelated serving producer.
An experiment that needs those metrics must retain its required events or compute the metrics independently through its serving metric sink.

## Python interface

The collection choices belong to the trace layer and do not modify model definitions or execution plans.
The shared helper supports every model accepted by `simulate_decode`.
Existing `trace=True` and `trace=False` calls retain their original behavior when no `trace_config` is provided.
A legacy `trace=False` combined with an enabled explicit configuration is rejected.

```python
from TokenSim.af.factory import simulate_decode
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.export import write_chrome_trace

options = AFTraceConfig(
    event_types=("transfer", "collective"),
    iterations=(0, 3),
    detail="timing",
)
executor, rows, compute, transfer = simulate_decode(
    runtime,
    2,
    128,
    steps=4,
    trace_config=options,
)
try:
    write_chrome_trace(
        "outputs/communication.trace.json",
        executor.trace.events,
        trace_config=options,
    )
finally:
    executor.trace.close()
```

`scripts.deepseek_v3_runtime.write_profile` automatically uses the explicit configuration attached by `simulate_decode`.
Direct `write_chrome_trace` calls accept export choices explicitly and reject configurations whose mode has no retained trace.
The export selections can also be changed after a full recording without rerunning the simulation.
