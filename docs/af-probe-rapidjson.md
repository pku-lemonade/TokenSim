# Compatible RapidJSON export

This prototype is isolated on branch `codex/probe-rapidjson`.
Production imports and default execution paths remain unchanged.
The diagnostic context temporarily substitutes private functions within one fresh benchmark process.

The encoder supports standard non-finite numbers and arbitrary integers.
Restricted mapping and iterable modes route unsupported values through the reference encoder.
Tuple conversion and dataclass attribute reads preserve the existing value contract.

Install the optional diagnostic dependency with `python -m pip install "python-rapidjson==1.25"`.

Run a complete original-geometry short benchmark with:

```bash
python scripts/profile_af_probe.py \
  --kernels rapidjson \
  --steps 4 \
  --output outputs/probe-rapidjson
```

The available kernel selection is `rapidjson`.
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
| `rapidjson` | 23.478 s | 6.579 s | 30.350 s | 0.998x | 0.881 GB |

Differences near one percent do not establish a repeatable gain from this single screen.
All six public JSON files are byte-identical to the reference in this screen.
