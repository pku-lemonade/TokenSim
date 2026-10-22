# Current Gemma experiment with FP8 weights

The active experiment uses eight-bit floating-point (FP8) weights following the user's selection on October 20, 2026.
Activations and the key/value (KV) cache use bfloat16 (BF16).
The earlier BF16 studies retain their original settings and results.
The measured NVIDIA endpoint's weight format remains undisclosed.
FP8 is an experiment assumption, and equivalent model quality has not been verified.
The analytical matrix-multiplication path uses FP8 throughput and assumes fused format conversion.

## Request and timing configuration

| Setting | Value |
| --- | --- |
| Model | Gemma 4 31B text model |
| Weight precision | FP8 |
| Activation precision | BF16 |
| KV cache precision | BF16 |
| Input and output lengths | 1,000 / 1,000; 10,000 / 1,500; 100,000 / 2,000 |
| Client concurrency | 1 |
| Request ordering | Each request starts after the previous request completes |
| Warmup requests | 5 per workload and selected layout |
| Measured requests | 50 per workload and selected layout |
| Reported metric | Median generation tokens per second (TPS), excluding time to first token |
| Native tokens per Artificial Analysis token | 1 for inputs and outputs |
| Logical-operator fixed overhead | 0.2 microseconds |
| Memory reserve | Maximum of catalog reserve and 5% of capacity |
| Hardware and link parameters | Existing catalog values |
| Search limits | 1,024 LP30 chips and 144 Rubin graphics processing units (GPUs) |

The requests repeat the fixed lengths without generating text.
The five warmup requests are excluded from the median.
The two collective profiles are evaluated separately.
The native-ring profile uses the existing ring communication formula.
The scheduled-direct profile uses an uncalibrated direct-collective timing assumption and does not generate a compiler communication schedule.

## Results for one inventory across all three lengths

Attention/feed-forward network (AF) disaggregation places attention on Rubin and the feed-forward network on LPX.
Global-only AF places global attention on Rubin and retains local attention and the feed-forward network on LPX.
Each row selects the inventory with the highest minimum generation speed across the three workloads within that mode and collective profile.

| Collective profile | Layout | Rubin GPUs | LP30 chips | 1K TPS | 10K TPS | 100K TPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Native ring | Pure LPX | 0 | 160 | 3,232.3 | 3,211.0 | 3,017.0 |
| Native ring | Conventional AF | 4 | 48 | 1,054.4 | 1,041.6 | 931.7 |
| Native ring | Global-only AF | 4 | 80 | 2,320.0 | 2,259.2 | 1,798.9 |
| Scheduled direct | Pure LPX | 0 | 176 | 4,498.1 | 4,469.9 | 4,212.1 |
| Scheduled direct | Conventional AF | 4 | 56 | 1,236.1 | 1,218.6 | 1,070.8 |
| Scheduled direct | Global-only AF | 8 | 64 | 2,829.0 | 2,739.1 | 2,090.5 |

The smallest searched pure-LPX inventory meeting 3,400 TPS at all three lengths uses 160 LP30 chips under the scheduled-direct assumption.
Its generation speeds are 4,212.7, 4,176.6, and 3,854.2 TPS.
Neither AF variant reaches that target at all three lengths with the unchanged timing parameters.
These values are analytical projections within the searched layouts and assumptions.

## Reproduction

Run the FP8 experiment explicitly to preserve the selected precision independently of command-line defaults.

```bash
.venv/bin/python scripts/explore_lpx_aa.py \
  --weight-dtypes fp8 \
  --kv-dtype bf16 \
  --collectives native-ring scheduled-direct \
  --kernel-us 0.2 \
  --rho-in 1 \
  --rho-out 1 \
  --timing-window full \
  --requests 50 \
  --warmup 5 \
  --output outputs/lpx-fp8-20261020
```

The fresh validation run additionally used `--no-resume`.
The [run manifest](../outputs/lpx-fp8-20261020/summary.json) records the workload, settings, and selected layouts.
The [ring result](../outputs/lpx-fp8-20261020/fp8-native-ring.json) and [scheduled-direct result](../outputs/lpx-fp8-20261020/fp8-scheduled-direct.json) contain all candidate points and selected-layout replay timestamps.
The [hardware evidence audit](lpx-parameter-evidence-audit.md) records confirmed values and unverified assumptions.
