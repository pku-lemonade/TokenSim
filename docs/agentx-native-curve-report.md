# DEP8 native curve checkpoint

The first executable curve slice is complete with the same native DSV4 stack
as the main checkpoints: MegaMoE, CSA/HCA attention, MTP q=4 and native mHC,
SimpleCPU KV, all 393 traces, seed 0, warmup and a 3,600-second profile.
Every point passed the curve validator's request, cancellation, final-page,
DP-time and component-conservation checks.

| GPU | concurrency | output token/s | p90 TTFT (s) | p90 ITL (s) | GPU hit | CPU hit |
|---|---:|---:|---:|---:|---:|---:|
| B200 | 32 | 292.934 | 24.625 | 0.108 | 95.642% | 0.000% |
| B200 | 64 | 410.413 | 33.694 | 0.155 | 95.708% | 0.210% |
| B200 | 96 | 546.894 | 48.626 | 0.190 | 95.241% | 0.390% |
| B200 | 128 | 568.873 | 75.153 | 0.337 | 74.245% | 19.704% |
| B200 | 160 | 565.853 | 138.349 | 0.382 | 51.467% | 42.650% |
| B300 | 128 | 901.075 | 47.037 | 0.195 | 94.352% | 0.000% |
| B300 | 256 | 925.983 | 75.010 | 0.330 | 95.455% | 0.000% |
| B300 | 512 | 958.838 | 134.580 | 0.623 | 91.904% | 3.306% |
| B300 | 576 | 791.616 | 159.871 | 0.843 | 82.532% | 12.552% |

The values are native simulator results for the DEP8 slice and are not mixed
with reference values where the exact curve artifact/configuration has not yet
been independently imported. The machine-readable summary and plot are:

- `results/agentx-dsv4-full/native-curve/20261002-dep8-seed0/summary.csv`
- `results/agentx-dsv4-full/native-curve/20261002-dep8-seed0/summary.png`
- `results/agentx-dsv4-full/native-curve/20261002-dep8-seed0/validation.json`

The executable B300 TP4 slice is also complete and passed the same lifecycle
validator:

| point | cache | output token/s | p90 TTFT (s) | p90 ITL (s) |
|---|---|---:|---:|---:|
| B300 c8 TP4 | HBM | 189.946 | 2.683 | 0.045 |
| B300 c16 TP4 | SimpleCPU | 217.335 | 4.172 | 0.085 |

The B300 TP4+EP4 attention-DP slice is also complete:

| point | output token/s | p90 TTFT (s) | p90 ITL (s) |
|---|---:|---:|---:|
| B300 c48 TP4/EP4 | 397.753 | 27.225 | 0.119 |
| B300 c64 TP4/EP4 | 394.361 | 37.242 | 0.167 |

The remaining TP8/HBM points are recorded with exact reference topology
metadata but have not been counted as completed full windows; the native
HBM-only path is substantially slower during first long-prefill admission.
