# Reusable AF experiments

`AFExperimentRunner` executes fixed-batch decode experiments for any `AFModelRuntime` accepted by `simulate_decode`.
It uses the shared factory's planning caches, host collection policy, bounded trace storage and finalized-prefix export.
Models, latency providers and scheduling extensions keep their existing interfaces.
The DeepSeek command uses this runner and preserves its default six public JSON files.

## Python interface

```python
from TokenSim.af.experiments import AFExperimentRunner
from TokenSim.af.trace.config import AFTraceConfig
from scripts.deepseek_v3_runtime import configuration

runtime = configuration(
    full_model=True,
    a_tp=4,
    f_tp=32,
    af_batch_size=32,
)
runner = AFExperimentRunner(
    runtime,
    "outputs/decode-experiments",
    fallback="analytical_only",
    trace_config=AFTraceConfig(mode="off"),
    resume=True,
)
completed = runner.run(
    batch_size=32,
    context_len=8192,
    steps=100,
)
print(completed.result["summary"])
print(completed.directory, completed.reused)
```

`result["iterations"]` contains each decode iteration's exact timing row.
`result["trace_config"]` describes the effective trace behavior.
The returned values use JSON-compatible types in both fresh and resumed runs.
Each result directory contains `result.json`, a completion manifest and the selected trace file when enabled.
`result["summary"]["profile"]` names that file relative to `completed.directory`.

Pass a configuration to one call to override the runner's trace default:

```python
diagnostic = runner.run(
    batch_size=32,
    context_len=8192,
    steps=100,
    trace_config=AFTraceConfig(
        event_types=("transfer", "collective"),
        iterations=(0, 1),
        fields=("node_id", "predecessor_ids", "payload_bytes", "resource_ids"),
    ),
)
```

`trace_config` accepts an `AFTraceConfig` or a mapping of its constructor fields.
`seed`, `fallback` and `resume` can also be overridden per call.
`run_many` accepts a sequence of keyword mappings for `run` and returns results in the supplied order.
Every point commits independently, including points completed before a later point fails.

```python
points = runner.run_many([
    {"batch_size": 16, "context_len": 8192, "steps": 100},
    {"batch_size": 32, "context_len": 8192, "steps": 100},
])
```

## Command-line use

Existing invocations use completed-point reuse by default.
Append `--no-resume` to force execution or `--verify-checkpoints sha256` to verify every output byte on reuse.
`--seed` defaults to the previous value of 42.
The command's point directories live under `<output>/.experiments`.
Its public trace files are atomic hard-link publications of the corresponding cached files, so publication does not copy large traces.
Deleting the cache does not remove an already published trace.
Editing a published trace also changes its linked cached artifact, which causes the normal modification-time check to reject reuse.

The [communication trace configuration](../examples/af/trace-communication.json) selects event kinds, iterations and argument fields:

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
  --trace-config examples/af/trace-communication.json \
  --output outputs/decode-communication
```

Explicit trace flags override the corresponding JSON fields.
Omitted settings retain their defaults.
Unknown keys and incompatible choices fail before output creation.
See [trace configuration](af-trace-configuration.md) for collection, validation and field selection semantics.

## Checkpoint behavior

A checkpoint represents one completed independent experiment point.
A process interruption restarts the unfinished point from its seed and reuses preceding completed points.
The runner does not restore a partially executed SimPy environment or skip individual iterations within a point.
Changing the requested step count creates a different point.

The identity includes the model, actual cluster and worker settings, compiled plan, placement, device and link specifications, trace options, seed, fallback and output metadata.
Measured operator-package identities include actual loaded table values and source metadata.
The runner also fingerprints relevant operator-data files, TokenSim Python source, the callback's source file, Python and dependency versions.
Uncommitted source edits invalidate affected checkpoints.
New dataclass configuration fields enter the runtime identity automatically.
Additional external inputs can be declared with `inputs=[path, ...]`.
Use `code_paths=[...]` to provide a complete set of code dependencies when an external callback or exporter imports additional modules.
The default includes the TokenSim package and the callback's source file; the DeepSeek command also includes its command and runtime helper modules.
Construct a new runtime to load edited on-disk tables into a new hardware snapshot.

Completion requires a valid manifest, a matching result identity and all recorded artifact checks.
The default `verify="stat"` checks artifact size and modification time and always verifies the small `result.json` hash.
It avoids reading a large trace again and does not detect changes that deliberately preserve both file size and modification time.
`verify="sha256"` hashes every artifact at commit and reuse, adding a complete read of each trace.
Upgrading a stat-only checkpoint to SHA-256 verification reruns the point to establish hashes.
Missing, changed, incomplete or malformed outputs are recomputed.

Each point uses an operating-system file lock on Linux and macOS.
Two processes requesting the same point share its committed result after one executes it.
The operating system releases locks when a process exits.
Files are staged privately, and the completion manifest is prepared only after the callback and artifact writes succeed.
Handled failures clean their staging directory; staging left by a killed process is removed when the same point is retried.
This is process-interruption recovery and does not provide a filesystem power-loss durability guarantee.

## Other experiment producers

`TokenSim.experiments.ExperimentRunner` provides the same checkpoint mechanism for other producers without depending on AF models.
Supply a callback that writes its artifacts into the provided directory and returns a finite JSON dictionary.
`result.json` and `manifest.json` are reserved filenames.
Artifacts must be regular files and must not contain symbolic links.
Put all per-point settings in `parameters` and declare external input files and additional code dependencies.

```python
from TokenSim.experiments import ExperimentRunner

experiments = ExperimentRunner(
    "outputs/figure-points",
    inputs=["data/my-workload.json"],
)

def evaluate(output):
    # Call the existing producer with the settings recorded below.
    # Write any large artifacts beneath output and return the plot metrics.
    return {"tokens_per_s": 100.0, "active_users": 32}

point = experiments.run(
    {"batch_size": 32, "context_len": 8192, "seed": 42},
    evaluate,
)
```

Callbacks with the same identity must use only the declared parameters and inputs.
Cached results preserve JSON values; live executors and arbitrary Python objects are not checkpointed.
The existing LPX runner's process scheduling remains available independently.

## Validation

The 2026-10-17 comparison used Python 3.12 on the local Apple M5 with 10 logical CPUs.
It retained the original full-model geometry: A tensor parallelism 4, F tensor parallelism 32, batch 32, AF batch 32, context 8192, A100 SXM 80 GB, NVLink 3 and analytical latency.
The step count was reduced to 4 and complete validated traces remained enabled.
The reference was integrated revision `8cdbfb36ac6488a6fdc2a66452a6fb0a4fa896d4`.
Each version ran three times sequentially in fresh interpreters with identical dependencies.
All six public output hashes and every iteration row matched across all six executions, including each 270,380-event trace.

| Measurement | Reference | Runner |
| --- | ---: | ---: |
| Median fresh execution | 29.099 s | 29.518 s |
| Median peak resident memory | 889,094,144 bytes | 910,655,488 bytes |
| Median time outside simulation and trace export | 0.299 s | 0.396 s |

The observed fresh-execution difference was 1.44%, with individual executions ranging from 28.686 to 29.875 seconds.
A separate completed-point reuse took 0.303 seconds, including runtime construction and checkpoint checks.
This comparison does not establish a fresh-execution speedup or remeasure the 100-step workload.
Local measurement records are under `outputs/runner-validation-20261017`.

All 627 unit tests passed, including exact trace fields, per-point trace overrides, generic model use, table-content invalidation, interrupted sweeps, forced process termination, stale staging cleanup and competing processes requesting the same point.
