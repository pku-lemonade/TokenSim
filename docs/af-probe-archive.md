# Larger trace archive blocks

This prototype is isolated on branch `codex/probe-archive-chunks`.
Production imports and default execution paths remain unchanged.
The diagnostic changes the default archive block size from 256 events to 1,024 or 4,096 events.
Explicit caller block sizes and cache capacities retain their original meaning.

Larger blocks can share more immutable object references within each pickle and reduce database statements.
Readback and late replacements still use the existing archive implementation.
Each cached block contains more events, so resident memory must be measured alongside elapsed time.

```bash
python scripts/profile_af_probe.py \
  --kernels archive \
  --steps 4 \
  --output outputs/probe-archive-1024
python scripts/profile_af_probe.py \
  --kernels archive4k \
  --steps 4 \
  --output outputs/probe-archive-4096
```

The complete fixed-batch benchmark uses the original 61-layer, batch-32, context-8192 geometry.
Imports and output hashing run outside the timed interval.
Trace files are removed after hashing unless `--keep-trace` is passed.

The four-step reference takes 30.150 seconds with 0.891 GB peak resident memory.
The 1,024-event variant takes 30.194 seconds with 0.891 GB peak resident memory, giving a 0.999x complete-command speedup.
The 4,096-event variant takes 30.371 seconds with 0.927 GB peak resident memory, giving a 0.993x complete-command speedup.
Neither single measurement establishes a gain.
Both variants produce identical public JSON files, iteration rows and provider counters.
These are fresh-process measurements on the local Apple M5 with Python 3.12.12.
