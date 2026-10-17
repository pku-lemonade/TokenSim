# Compatible orjson export

This prototype is isolated on branch `codex/probe-json-encoder`.
Production imports and default execution paths remain unchanged.
The diagnostic context temporarily substitutes private functions within one fresh benchmark process.

The encoder checks mutable descendants on every use and memoizes eligibility only for deeply immutable values.
Non-finite floats, non-string dictionary keys and unsupported values use the reference encoder.

Install the optional diagnostic dependency with `python -m pip install "orjson>=3.10"`.

Run a complete original-geometry short benchmark with:

```bash
python scripts/profile_af_probe.py \
  --kernels json \
  --steps 4 \
  --output outputs/probe-json
```

The available kernel selection is `json`.
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
| `json` | 23.248 s | 10.661 s | 34.198 s | 0.886x | 0.879 GB |

Differences near one percent do not establish a repeatable gain from this single screen.
The summary is byte-identical, and complete parsed trace comparison is recorded separately by the aggregate exploration.
