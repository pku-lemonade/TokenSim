# Finalized trace prefix export

This prototype is isolated on branch `codex/probe-trace-pipeline`.
Production imports and default execution paths remain unchanged.
The diagnostic context temporarily substitutes private functions within one fresh benchmark process.

The recorder writes validated finalized prefixes in global append order and retains the existing SQLite archive for complete readback.
Late replacements invalidate the prefix and use the reference final export.
The thread variant queues immutable UTF-8 byte blocks with a four-block bound.
Publication uses a temporary file and atomic destination replacement.

Run a complete original-geometry short benchmark with:

```bash
python scripts/profile_af_probe.py \
  --kernels pipeline \
  --steps 4 \
  --output outputs/probe-pipeline
```

The available kernel selection is `pipeline or thread`.
The benchmark uses the 61-layer DeepSeek model, attention tensor parallelism 4, feed-forward tensor parallelism 32, batch 32, context 8192 and the analytical provider.
The timed interval includes configuration, simulation, validation, archival, all six JSON outputs and cleanup.
Imports and output hashing run outside that interval.
Trace files are removed after hashing unless `--keep-trace` is passed.
Use `--semantic-digest` to hash parsed complete trace values with object-key order ignored and array order preserved.
The runner records every iteration timing and provider/cache counters.
These fixed-batch measurements do not reproduce the unavailable request-rate scan that produced the supplied plotting data.
