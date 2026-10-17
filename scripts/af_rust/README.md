# Rust AF kernel experiment

This optional experiment implements four runtime kernels in Rust through PyO3.
The production runtime does not import this directory or require Rust.
Model definitions, provider queries, configuration and simulation command-line arguments retain their existing interfaces.
The experiment measures whether moving existing operations across the language boundary improves complete runs.
It does not implement a complete native simulation engine.

## Implemented kernels

`src/fingerprints.rs` implements structural-key traversal, scalar type tags, bounded identity memoization and batched query-field normalization.
Mutable descendants prevent identity memoization.
The default identity capacity is 4,096 entries, and record layouts are bounded at 128 types.

`src/templates.rs` compiles validated Python template recipes into a sequence of Rust instructions and binds execution identities in one native call per layer.
Immutable geometry remains shared, and dynamic records retain the existing Python dataclass types.
The compiled representation has the same lifetime as its owning Python template.
The reference planner continues to validate new shapes and control template admission.

`src/events.rs` implements the SimPy event-dispatch loop in Rust while preserving the Python heap, callbacks and event objects.
It preserves time, priority and insertion order, callback-list growth, stop/resume behavior and failed-event propagation.
Temporary Python references are released after every dispatched event.
Custom environment subclasses and customized `step` methods retain their original execution path.
This kernel requires the tested SimPy 4.1.2 behavior.

`src/domains.rs` owns communication resource queues, stable priorities, resource ownership, cancelled-entry cleanup and atomic FIFO arbitration in native containers.
The Python adapter retains `DomainRequest`, SimPy events, priority-2 arbitration turns and the existing service generator.
Arbitration operates on native state without Python callbacks inside its queue scan.
Endpoint resource identifiers can be added as the existing Python endpoint registry discovers them.
The diagnostic boundary supports string resource identifiers, signed 64-bit integer or Unicode string identity components and finite floating-point timestamps.
Integer timestamps must be within the inclusive range from `-2**53` to `2**53`.
Unsupported identity values and integer timestamps raise an explicit error without changing queue state.
The `owners` property returns a snapshot, and custom scheduler subclasses that mutate private Python queue fields are outside this prototype's scope.
The reference scheduler remains available for those implementations.

The default Rust selection enables `keys` and `templates`.
The `events` kernel is selectable for measurement because its initial complete-run result was slower.
The `domains` kernel is separately selectable and does not change production defaults.
Python callbacks and object construction continue to hold the global interpreter lock.
These kernels do not provide parallel execution of one simulation.

## Build and run

The tested environment uses Python 3.12.12, Rust 1.94.0 and PyO3 0.20.3 on Apple silicon.
The lockfile pins the Rust dependencies.
Build artifacts are stored under `.codex/af-rust`, partitioned by Python version and Rust source digest.
The build command does not launch a simulation.
Add `--offline` when the pinned crates are already cached.

```bash
python scripts/profile_af_rust.py \
  --build
```

Run a bounded comparison using the same arguments and different backend selections and output directories.
The wrapper records host phase timings separately from result JSON and trace contents.

```bash
python scripts/profile_af_rust.py \
  --backend python \
  --report outputs/rust-probe-python.json \
  -- \
  --full-model \
  --a-tp 4 \
  --f-tp 32 \
  --batches 32 \
  --af-batch-size 32 \
  --context 8192 \
  --steps 1 \
  --device a100_sxm_80g \
  --network nvlink3 \
  --fallback analytical_only \
  --output outputs/rust-probe-python

python scripts/profile_af_rust.py \
  --backend rust \
  --report outputs/rust-probe-native.json \
  -- \
  --full-model \
  --a-tp 4 \
  --f-tp 32 \
  --batches 32 \
  --af-batch-size 32 \
  --context 8192 \
  --steps 1 \
  --device a100_sxm_80g \
  --network nvlink3 \
  --fallback analytical_only \
  --output outputs/rust-probe-native
```

Select individual kernels with `--kernels keys`, `--kernels templates`, `--kernels domains` or `--kernels events` before the argument separator.
Select the combined dispatch experiment with `--kernels keys templates domains events`.
The wrapper restores its substitutions when execution finishes or raises an exception.
Build the extension before running the optional native contract tests.

```bash
python -m unittest \
  tests.af.test_rust_probe \
  tests.af.test_rust_domains
```

The [native scheduling follow-up](../../experiments/performance/rust-scheduling-20261017/README.md) compares this broader queue boundary with the current integrated runtime.
It records complete-run measurements separately from an isolated scheduling workload and instrumented profiles.

## Adoption gate and remaining work

The requested adoption threshold is at least 2x throughput for representative complete runs with equal result JSON and full trace values.
The measured small-kernel prototype has not met that threshold.
The measurements and validation results are recorded in [AF runtime performance](../../docs/af-runtime-performance.md).

A complete native engine would require a larger boundary around planning, dependency tracking, communication phases and resource scheduling.
The internal adapter could convert runtime geometry and routing snapshots into typed tables and obtain provider estimates through the existing Python interfaces.
Native arrays would then hold dependency counters, resource queues and the event heap without per-event Python callbacks.
Trace data would need bounded chunk conversion that retains every existing field and permits the current dependency patches.
Provider cache misses and dynamic routing would cross the boundary at explicit synchronization points.
Custom providers, cancellation and unsupported execution modes would need the reference fallback.
That design requires a separate exactness and complete-run performance demonstration before replacing production execution.
