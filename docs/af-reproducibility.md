# AF trace reproducibility

## Seeded simulation

The fixed-batch runner and native serving tests compare complete trace bytes across fresh Python processes with different `PYTHONHASHSEED` values.
The simulation seed remains fixed during those comparisons.
Python hash randomization changes string set iteration order independently of the simulation's local random streams.

Host submission resources follow physical worker order when host timing or CUDA graph submission is enabled.
The previous construction used a set of string worker identifiers, so serialized resource capacity metadata could change key order between processes.
The resource values, execution order and event records remain unchanged by this correction.

The regression covers concurrent eager and CUDA graph serving, serialized native control, random expert selection, dependency records and exported resource metadata.
It also repeats execution after changing the global Python random state and verifies that changing the explicit simulation seed changes trace contents.
Run it with:

```bash
python -m unittest tests.af.trace.test_determinism
```

## Measured checks

The [recorded observations](../experiments/reproducibility/seed-20261017/results.json) use Python 3.12 on the same Apple M5 host.
Every fixed-batch execution disables checkpoint reuse with `--no-resume`.
The source revision before the resource ordering correction is `ba14123`.

| Workload | Repetitions | Observation |
| --- | --- | --- |
| DeepSeek one layer, batches 1/8/32, context 8,192, 3 steps | Two executions per process across five hash settings | All public JSON bytes and complete trace values match |
| Three concurrent batch-32 streams, one layer, 3 steps per stream | Two executions per process across five hash settings | Complete trace bytes and iteration timings match |
| Original DeepSeek 61-layer geometry, batch 32, context 8,192, 2 steps | Three fresh processes | All six public JSON files match byte for byte |
| Native serving with eager/CUDA graph execution, control on/off, cyclic/random/hot routing | Twelve cases across five hash settings | Event trace bytes match; host resource mapping order differed before the correction |

Repeating the twelve native cases after the correction preserves event bytes and also makes resource mapping order identical across all five hash settings.
A further original-geometry two-step execution after the correction matches every public JSON hash from the three preceding executions.
Native control cases use one stream because the runtime rejects multiple in-flight iterations with native control enabled.
The other native serving cases use two concurrent streams.

These checks do not execute the complete 100-step command or the eight-point QPS=1 serving scan.
The supplied plotting script consumes saved metrics; its archived `launch_scan.py` and original `results.json` are absent from this checkout.
The observations establish repeatability for the tested inputs and runtime versions.

## Artifact metadata

Runner manifests report host elapsed time and output modification times.
Those fields vary between forced executions independently of the seed.
Compare simulation results and traces separately from those manifest fields.

The supplied plot is maintained in [scripts/plot_tps_user_vs_tps_mw.py](../scripts/plot_tps_user_vs_tps_mw.py).
It reads saved metrics and writes three PNG/SVG pairs without running the simulation.
SVG export uses a fixed salt for each figure's element identifiers and omits Matplotlib's export date.
The salt setting is scoped to the export call and restores the caller's Matplotlib configuration afterward.
PNG export and plot coordinates retain the supplied script's behavior.
Repeated exports with identical input match byte for byte in the same Matplotlib and font environment.
An eight-point synthetic fixture produces six identical output files across three fresh processes with different Python hash settings.
PNG bytes match the original script, and normalized SVG content matches after excluding the export date and element identifiers.

```bash
python scripts/plot_tps_user_vs_tps_mw.py \
  --input path/to/results.json \
  --output outputs/qps1-figures
```
