# Trace persistence without repeated serialization

This diagnostic extends the MessagePack probe at `a897c98` on `codex/perf-integrated-defaults`.
The workload retains the original 61-layer DeepSeek model, batch 32, attention tensor parallelism 4, feed-forward tensor parallelism 32, context 8,192, A100 SXM 80 GB, NVLink 3, analytical-only estimates and seed 42.
The step count is reduced from 100 to 8.
All 540,760 trace events, fields, invariant validation, event order and simulation results remain enabled.
The host is an Apple M5 running Python 3.12.12.
No production defaults, model definitions, provider interfaces or scheduler APIs change.

## Implemented probes

`scripts/af_probe_archiveonly.py` completes iterations with the existing indexed binary archive and publishes a SQLite backup of that archive.
This removes the prepared-prefix event traversal and the additional JSON or MessagePack event encoder.
The published snapshot contains the existing pickle chunks, position and iteration indexes, export metadata, trace selection and event count.
Snapshot backup and publication remain inside the complete command timer.
The original live archive remains separate, so later readback or replacement cannot mutate an already published file.
The converter preserves global position order across interleaved iterations and accepts a fixed set of TokenSim record constructors.
This format depends on compatible project record definitions; adding another record class requires extending the diagnostic reader's constructor list.

`scripts/af_probe_sharedpack.py` stores repeated nested dataclass identities as MessagePack definitions and references.
Its record memo holds at most 8,192 records and resets between synchronous finalized-prefix writes.
The decoder expands references into ordinary maps before JSON conversion.
This probe assumes stable values during a synchronous prefix export and does not reuse references across simulation intervals that can mutate values.
It retains both the production archive and the additional public event stream.

`scripts/af_probe_datapickle.py` batches up to 1,024 Chrome event mappings and uses the native pickler's memo within each chunk.
Dataclasses and immutable provider dictionaries become plain dictionaries through the existing field extractor during serialization.
The output therefore stores field names and values without references to TokenSim class definitions or a per-model schema registry.
It supports large Python integers and new dataclass detail fields without a reader constructor update.
Its reader allows only the ordinary dictionary constructor and validates bounded frames and complete input consumption.
The format uses Python's pickle protocol and a versioned chunk envelope, with a 16-mebibyte frame limit.
It retains the existing private archive for full runtime readback, so event data is still encoded for both the archive and the public stream.
This probe requires no extra package dependency.

## Measurements

Each measurement starts in a fresh interpreter with checkpoint reuse disabled.
The complete command timer includes configuration, simulation, trace preparation, snapshot publication and checkpoint work after imports.
Diagnostic hashes and conversion to JSON run afterward.
Benchmarks and conversions run sequentially without another test suite or benchmark competing with them.
The plain MessagePack reference is the average of the preceding report's two runs on the same production runtime.
These eight-step observations do not establish the exact runtime of the original 100-step command or the complete plot sweep.

| Representation | Complete command | Binary bytes | Peak resident memory |
| --- | ---: | ---: | ---: |
| Fresh JSON baseline | 56.511 s | 2,588,215,556 | 1,128,792,064 bytes |
| Plain MessagePack, previous two-run mean | 51.670 s | 2,110,891,135 | 1,127,415,808 bytes maximum |
| MessagePack with record references | 50.531 s | 1,925,830,076 | 1,120,501,760 bytes |
| Data-only native pickle chunks | 48.349 s | 467,613,691 | 1,119,682,560 bytes |
| Existing indexed archive snapshot | 45.221 s | 460,959,744 | 1,103,052,800 bytes |
| Repeated indexed archive snapshot | 46.061 s | 460,959,744 | 1,115,324,416 bytes |

The archive probe spends 0.351 seconds publishing its snapshot after simulation; this time is included in the 45.221-second total.
The earlier complete JSON reference takes 56.152 seconds on average and writes 2,588,215,556 bytes.
Relative to that reference, the archive snapshot takes approximately 19.5% less total time, corresponding to 1.242x speedup.
The data-only stream takes approximately 13.9% less time, corresponding to 1.161x speedup.
The archive snapshot is 82.2% smaller than JSON, and the data-only stream is 81.9% smaller.
The fresh JSON measurement takes 56.511 seconds, and the repeated archive takes 46.061 seconds.
The two archive observations average 45.641 seconds, corresponding to 1.238x speedup and 19.2% less time relative to the fresh JSON baseline.
The repeated snapshot has the same complete binary hash as the snapshot whose JSON conversion is verified.
Relative to the fresh JSON baseline, the data-only stream reduces time by 14.4% and shared MessagePack reduces time by 10.6%.
Each of these latter probes has one complete timing observation, and the archive mean uses two observations against one fresh baseline.
The run order is archive, shared MessagePack, data-only chunks, fresh JSON and repeated archive.
These observations provide a small sample without a confidence interval.

| Conversion to byte-identical JSON | Conversion time | Separate-process peak resident memory |
| --- | ---: | ---: |
| Archive snapshot | 13.360 s | 129,138,688 bytes |
| Shared MessagePack | 15.066 s | 80,822,272 bytes |
| Data-only stream | 11.970 s | 97,026,048 bytes |

Adding immediate JSON conversion gives 58.581 seconds for the exploratory archive command and 60.319 seconds for the data-only command.
Those combined times exceed the preceding direct JSON reference.
These persistence strategies save complete experiment time when binary traces remain the stored artifact and only selected traces need JSON later.
The plotting script reads `results.json` metrics and does not require trace conversion.

## Equivalence and development scope

Every converted eight-step trace is byte-identical to the original JSON with SHA-256 `540c00510aca1d7127275a2443048fa1bdd5f117a8e43265f7df1f2523da7046`.
Iteration rows, cache counters, four configuration JSON hashes and summaries with normalized trace filename extensions match.
The archive snapshot tests cover interleaved order, filtering, empty histories, independent snapshots, late replacements, partial JSON fallback, atomic publication failure and rejected reader globals.
The shared-reference tests cover nested sharing, mutation between prefixes, bounded definitions and invalid references.
The data-only tests cover new nested record fields, large integers, selection, empty output, chunk boundaries, snapshot mutation, malformed and oversized frames, rejected reader globals and unsupported values.
The existing MessagePack tests continue to pass after extracting optional diagnostic reader hooks.
Black formatting, the 500-line source limit, all 648 unit tests and whitespace checks pass.

The archive snapshot has the lowest measured time and requires compatible project record classes when reopened.
The data-only stream provides a smaller dependence on future model code because its stored values are ordinary maps, lists, tuples and scalar fields.
A production format choice still requires consistent fallback output, checkpoint identity, command-line configuration and documented reader support.
All probes remain opt-in diagnostics and leave production behavior unchanged.

The preceding wall profile attributes approximately 20.4% of time to JSON conversion, encoding and prefix output.
The archive probe removes most of that measured work while retaining event construction, invariant validation, indexed archival and the simulation itself.
The result approaches that category's isolated 1.256x elimination bound and does not demonstrate a 2x trace-format speedup.
Larger gains would need changes to other measured categories or a separate redesign of the remaining archive work.

## Reproduction

Run each command in a fresh interpreter with a distinct output directory.
The ordinary JSON baseline uses the same command without `--kernels` and without `--production-path`.

```bash
python scripts/profile_af_probe.py \
  --kernels archiveonly \
  --production-path \
  --steps 8 \
  --keep-trace \
  --output outputs/trace-faster-20261017/archiveonly
python -m scripts.af_probe_archiveonly \
  outputs/trace-faster-20261017/archiveonly/decode-layers61-b32.trace.sqlite3 \
  outputs/trace-faster-20261017/archiveonly/converted.trace.json
```

Use `--kernels sharedpack` or `--kernels datapickle` with fresh output directories for the other implemented probes.
Plain and shared MessagePack require the optional `msgpack==1.1.1` package.
The data-only converter is available as `scripts.af_probe_datapickle.convert(source, destination)`, and the shared converter is available as `scripts.af_probe_sharedpack.convert(source, destination)`.
Keep `--keep-trace` because the generic harness's cleanup assumes a JSON trace filename.
Large diagnostic files remain outside version control, and redundant converted files and checkpoint directories are removed after verification.
The checked-in `results.json` records complete measurements, counters, file sizes and output hashes.
