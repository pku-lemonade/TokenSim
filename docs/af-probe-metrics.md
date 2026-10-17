# Fixed-batch timing-only execution

This prototype is isolated on branch `codex/probe-metrics-only`.
Production imports and default execution paths remain unchanged.
The diagnostic context temporarily substitutes private functions within one fresh benchmark process.

The existing executor disables trace construction, trace validation, archival and complete event export.
Iteration timing rows and provider observations remain available.
The Chrome trace has an empty event array.
This experiment does not provide resource-utilization aggregates that require trace events.

Run a complete original-geometry short benchmark with:

```bash
python scripts/profile_af_probe.py \
  --kernels metrics \
  --steps 4 \
  --output outputs/probe-metrics
```

The available kernel selection is `metrics`.
The benchmark uses the 61-layer DeepSeek model, attention tensor parallelism 4, feed-forward tensor parallelism 32, batch 32, context 8192 and the analytical provider.
The timed interval includes configuration, simulation, validation, archival, all six JSON outputs and cleanup.
Imports and output hashing run outside that interval.
Trace files are removed after hashing unless `--keep-trace` is passed.
Use `--semantic-digest` to hash parsed complete trace values with object-key order ignored and array order preserved.
The runner records every iteration timing and provider/cache counters.
These fixed-batch measurements do not reproduce the unavailable request-rate scan that produced the supplied plotting data.

## Four-step screen

The same-process reference command takes 30.303 seconds, with 23.311 seconds in simulation and 6.702 seconds in final export.
Each row is one fresh-process measurement on the local Apple M5 with Python 3.12.12.

| Kernel | Simulation | Final export | Complete command | Speedup | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `metrics` | 18.219 s | 0.000 s | 18.510 s | 1.637x | 0.554 GB |

Differences near one percent do not establish a repeatable gain from this single screen.
Timing-only execution changes trace availability and preserves the complete summary document in this benchmark.
