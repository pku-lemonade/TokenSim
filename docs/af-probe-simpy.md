# SimPy dispatch and condition callbacks

This prototype is isolated on branch `codex/probe-simpy-loop`.
Production imports and default execution paths remain unchanged.
The diagnostic context temporarily substitutes private functions within one fresh benchmark process.

The loop substitutes unbounded stock environments and falls back for custom environments, custom stepping and bounded runs.
The condition variant retains every event and caches bound callbacks within construction and cleanup.
Both variants require the tested SimPy 4.1.2 semantics.

Run a complete original-geometry short benchmark with:

```bash
python scripts/profile_af_probe.py \
  --kernels loop \
  --steps 4 \
  --output outputs/probe-simpy
```

The available kernel selection is `loop or conditions`.
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
| `loop` | 23.410 s | 6.839 s | 30.537 s | 0.992x | 0.871 GB |
| `conditions` | 23.146 s | 6.734 s | 30.167 s | 1.005x | 0.911 GB |

Differences near one percent do not establish a repeatable gain from this single screen.
All six public JSON files are byte-identical to the reference in this screen.
