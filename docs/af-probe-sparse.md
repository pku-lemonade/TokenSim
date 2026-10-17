# Sparse query normalization

This prototype is isolated on branch `codex/probe-sparse-query`.
Production imports and default execution paths remain unchanged.
The diagnostic context temporarily substitutes private functions within one fresh benchmark process.

The key plan precomputes immutable default-field fingerprints and normalizes supplied fields on each query.
Mutable defaults are revisited.
The existing query validator, namespace, service cache, LRU eviction, counters and fallback behavior remain in place.

Run a complete original-geometry short benchmark with:

```bash
python scripts/profile_af_probe.py \
  --kernels sparse \
  --steps 4 \
  --output outputs/probe-sparse
```

The available kernel selection is `sparse`.
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
| `sparse` | 23.061 s | 6.711 s | 30.057 s | 1.008x | 0.881 GB |

Differences near one percent do not establish a repeatable gain from this single screen.
All six public JSON files are byte-identical to the reference in this screen.
