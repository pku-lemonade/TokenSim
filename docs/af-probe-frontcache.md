# Query identity front cache

This prototype is isolated on branch `codex/probe-query-front-cache`.
Production imports and default execution paths remain unchanged.
The existing validator and structural service cache remain authoritative.

A 1,024-entry front cache reuses structural keys when every non-scalar field has already been proven deeply immutable by the existing fingerprint memo.
Entries retain the referenced objects while their identities serve as keys.
Structural keys retain their complete equality comparison and cache their hash for repeated front-cache hits.
Mutable or unproven inputs use full structural normalization.
Every hit checks and updates the existing service least-recently-used cache, preserving eviction and observation counters.

```bash
python scripts/profile_af_probe.py \
  --kernels frontcache \
  --steps 4 \
  --output outputs/probe-frontcache
```

The complete fixed-batch benchmark uses the original 61-layer, batch-32, context-8192 geometry.
Imports and output hashing run outside the timed interval.
Trace files are removed after hashing unless `--keep-trace` is passed.

The four-step reference takes 30.150 seconds, including 23.176 seconds of simulation, with 0.891 GB peak resident memory.
The front-cache prototype takes 32.712 seconds, including 25.833 seconds of simulation, with 0.900 GB peak resident memory.
The measured complete-command speedup is 0.922x, so this prototype is slower.
All six public JSON files, iteration rows and provider counters are identical to the reference.
These are single fresh-process measurements on the local Apple M5 with Python 3.12.12.
