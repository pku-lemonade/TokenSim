# Follow-up profiling of the integrated runtime

## Scope and decision

This exploration compares new prototypes with the integrated runtime at `e10f8a7` on branch `codex/perf-integrated-defaults`.
Production model, planner, provider, execution and trace behavior remain unchanged.
The prototypes remain opt-in diagnostic modules under `scripts/af_probe_*.py`.
The measured gains do not justify adding generated constructors, a second dependency implementation or a runtime upgrade to the default execution path.

The workload uses the original 61-layer DeepSeek model, batch 32, attention tensor parallelism 4, feed-forward tensor parallelism 32, context 8,192, A100 SXM 80 GB, NVLink 3, seed 42 and analytical-only estimates.
Every timing retains the full trace, invariant validation, archival and all six public JSON files.
Each measurement uses a fresh process and disables experiment checkpoint reuse.
The timed interval includes configuration, simulation, prepared trace output, final publication and archive cleanup.
Imports, prototype installation and diagnostic hashing are outside that interval.
Processes run sequentially on an Apple M5 host with 10 logical CPUs and 32 GiB of memory.

[results.json](results.json) records the individual measurements, output hashes, iteration observations, provider counters, runtime versions and comparison decisions.
The normal runtime is Python 3.12.12.
The Python 3.14.8 experiment uses a separate environment with the same package versions, including transitive dependencies.
The just-in-time compilation experiment changes `PYTHON_JIT` only in its child process environment.
The existing project environment and Python requirement remain unchanged.

## Explored implementations

| Prototype | Implementation | Adoption decision |
| --- | --- | --- |
| `bulkinit` | Generate a field-complete initializer for the frozen trace record and populate its dictionary in one update. | Extra constructor machinery provides a small total-runtime gain. |
| `barrier` | Replace the nested dependency conditions with two ordinary events while preserving both scheduling turns. | A second dependency implementation provides a small total-runtime gain. |
| `codegen` | Compile each existing template recipe graph into straight-line Python binding code. | Compilation and additional stored code make the measured command slower. |
| `bindcopy` | Copy a template record's static dictionary and attach the completed dictionary once. | Alternating comparisons do not establish a useful gain. |
| `batchjson` | Encode at most 64 complete Chrome trace records in each JSON encoder call. | The bounded batching experiment does not establish a substantial gain. |
| `coldbind` | Execute already validated source nodes immediately after admitting their template. | The prototype changes the admission result type and loses static sharing across bindings. |
| Python 3.14 | Run the same command and dependency versions in a separate interpreter. | Short comparisons do not justify changing the supported runtime. |
| Python 3.14 with JIT | Enable the available JIT in the isolated interpreter. | Short comparisons do not establish a useful gain. |

The `coldbind` timings measure a contract-changing approximation of skipping admission binding and are excluded from the compatible-candidate recommendations.
A variant wrapping its source nodes in `BoundDAG` restores the result type but still fails the existing cross-binding identity test for shared physical workers.
That variant is rejected before timing.
The retained `coldbind` module documents these limitations and is never imported by production code.

## Measured results

| Candidate | Steps | Current runtime | Candidate | Speedup | Sampling |
| --- | ---: | ---: | ---: | ---: | --- |
| Bulk trace initialization | 4 | 29.694 s | 28.721 s | 1.034x | One candidate between two reference runs |
| Dependency barrier | 4 | 29.694 s | 29.002 s | 1.024x | One candidate between two reference runs |
| Generated template binder | 4 | 29.694 s | 39.789 s | 0.746x | One candidate between two reference runs |
| Template dictionary copying | 4 | 29.395 s | 29.327 s | 1.002x | Two runs per version with alternating order |
| Bounded JSON batches | 4 | 29.290 s | 29.438 s | 0.995x | Two runs per version with alternating order |
| Python 3.14, JIT disabled | 4 | 29.352 s | 28.853 s | 1.017x | Two fresh processes per interpreter |
| Python 3.14, JIT enabled | 4 | 29.352 s | 29.335 s | 1.001x | Two fresh processes per interpreter |
| Bulk initialization and barrier | 16 | 110.475 s | 106.573 s | 1.037x | One candidate between two reference runs |

Times with multiple measurements use the median for each version.
The two 16-step reference runs take 110.500 and 110.449 seconds.
The compatible combination uses 3.53% less time in its single longer comparison.
Its peak resident memory is 1,526,251,520 bytes, compared with 1,535,918,080 and 1,536,557,056 bytes for the two references.
These measurements show a modest improvement and do not establish a large speedup or a statistically characterized distribution.
The evidence file contains 29 normal command measurements and separates the initial JSON batching variant from its revised filtering implementation.

## Validation and limits

Every normally timed run compares all six public JSON hashes, every iteration row, event counts and service-query request and hit counts against the corresponding current-runtime reference.
The comparisons include complete trace bytes and preserve array and object-key order.
Equal exported values alone do not qualify the source-node reuse prototype because the existing internal sharing contract also matters.
The full AF test suite is additionally exercised with `bulkinit`, `barrier`, `codegen` and `bindcopy` installed separately.
The existing prepared-export tests exercise `batchjson`, including selections, Unicode, snapshots, late edits, partial export and failure cleanup.
All 629 default-runtime tests pass after formatting.
The 466 AF tests also pass with `bulkinit`, `barrier`, `bindcopy` and `batchjson` installed together.
Black 24.2.0, the 500-line source limit, documentation links, example shell syntax and whitespace checks pass.

The instrumented profile captures a complete two-step run and the last environment-run interval of an eight-step run.
Profiling shows remaining work across structural fingerprints, template compilation and binding, communication execution, JSON encoding, invariant validation and scheduling.
Cumulative profile entries overlap and must not be summed as independent costs.
The profiler can combine generated dataclass initializer entries with matching generated filenames and line numbers.
Instrumented timings are stored separately from normal command timings.

The longer screening cases use 16 steps and 1,081,520 events.
This round does not repeat the complete 100-step command or a concurrent cache-saturation run.
Its memory observations do not replace the existing concurrent per-process memory validation.
The original plot scan launcher and scan data are unavailable, so these measurements do not establish a complete plot-scan speedup.
The existing Rust experiments already cover smaller native kernels; this round does not establish a new benefit for a broader native engine.

## Reproduction

The diagnostic harness now accepts `--production-path` to keep the current prepared-prefix export enabled while installing a prototype.
Without that flag, nonempty kernel selections retain the older archive-and-export comparison path used by earlier experiments.
Use the flag for every new prototype in this report.

```bash
python scripts/profile_af_probe.py \
  --steps 4 \
  --production-path \
  --output outputs/follow-up-reference

python scripts/profile_af_probe.py \
  --steps 4 \
  --production-path \
  --kernels bindcopy \
  --output outputs/follow-up-bindcopy

python scripts/profile_af_probe.py \
  --steps 16 \
  --production-path \
  --kernels bulkinit,barrier \
  --output outputs/follow-up-combined
```

The harness hashes each complete output before deleting the large trace and private checkpoint directory.
Use `--keep-trace` when the full output is needed for independent inspection.
The companion JSON files and `measurement.json` remain available in each output directory.

An optional Python 3.14.8 comparison can use an isolated environment and the recorded package versions.
The pinned file describes the experiment environment and adds no production dependency.

```bash
python3.14 -m venv outputs/follow-up-python314
outputs/follow-up-python314/bin/python -m pip install \
  -r experiments/performance/follow-up-20261017/requirements-runtime.txt

PYTHON_JIT=0 outputs/follow-up-python314/bin/python scripts/profile_af_probe.py \
  --steps 4 \
  --production-path \
  --output outputs/follow-up-python314-nojit

PYTHON_JIT=1 outputs/follow-up-python314/bin/python scripts/profile_af_probe.py \
  --steps 4 \
  --production-path \
  --output outputs/follow-up-python314-jit
```

The JIT comparison requires an interpreter build that provides the experimental JIT.
