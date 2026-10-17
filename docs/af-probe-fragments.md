# Immutable JSON-fragment reuse

This prototype is isolated on branch `codex/probe-json-fragments`.
Production imports and default execution paths remain unchanged.
The diagnostic encoder uses the reference trace writer and its atomic replacement behavior.

Deeply immutable dataclasses can reuse a complete encoded JSON fragment through a bounded 4,096-entry identity cache.
Mutable descendants are revisited through the reference encoder.
A single regular-expression substitution pass inserts the cached fragments.
Literal marker collisions select the reference encoder, and inserted fragments are never scanned again.
The cache retains objects while their identities serve as keys.
The encoder requires no native dependency.

```bash
python scripts/profile_af_probe.py \
  --kernels fragments \
  --steps 4 \
  --output outputs/probe-fragments
```

On the aggregate exploration branch, `--kernels pipeline,fragments` also uses the fragment encoder during finalized-prefix export.
The complete fixed-batch benchmark uses the original 61-layer, batch-32, context-8192 geometry.
Imports and output hashing run outside the timed interval.
Trace files are removed after hashing unless `--keep-trace` is passed.

The four-step reference takes 30.150 seconds, including 6.684 seconds of export, with 0.891 GB peak resident memory.
The fragment prototype takes 31.966 seconds, including 8.571 seconds of export, with 0.899 GB peak resident memory.
The measured complete-command speedup is 0.943x, so this prototype is slower.
All six public JSON files and iteration rows are identical to the reference.
These are single fresh-process measurements on the local Apple M5 with Python 3.12.12.
