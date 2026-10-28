# Cross-tray latency and the TP32 AA match

Record date: 2026-10-26.
The [article-condition sweep](lpx-article-sweep.md) found that TP32/PP8 with decode context parallelism (DCP) of 8 on 256 LP30 chips places both Artificial Analysis (AA) speeds within 5% if each step costs 14 to 25 µs more than modeled.
A TP32 group spans two 16-chip trays, so its collectives use the middle fabric tier.
That tier's 0.75 µs latency comes from the Hot Chips reachable-capacity chart, which states the latency for reaching 64 GB of SRAM and does not describe a collective.
The [attention-group record](lpx-attention-group.md) attributed the 87 µs longer TP32 base step to this tier.
This record tests how the match depends on that latency.

## Method

[`scripts/run_lpx_tray_latency.py`](../scripts/run_lpx_tray_latency.py) repeats the batch-one TP32/PP8, DCP 8 point of the article-condition sweep at attention matrix efficiencies 0.30 and 0.50.
It sets the middle-tier latency to 0.50, 0.65, 0.75, 0.85, 1.00 and 1.15 µs and keeps the 0.35 µs tray tier, the 1.15 µs rack tier and every other setting.
For each point it reports the AA errors and the range of constant extra time per step that places both AA speeds within 5% at one native token per AA token.
The [token-ratio record](lpx-token-ratio.md) gives the corresponding values for other ratios.

## Expectations recorded before the run

1. The step grows by about 25 µs for each 0.1 µs of middle-tier latency at both contexts, because the TP-group collectives took 188.5 µs at 0.75 µs in the attention-group breakdown.
2. The 10K-to-100K increment stays at 14.2 µs at every latency.
3. At one native token per AA token, a middle-tier latency of about 0.81 to 0.85 µs matches both AA values without extra time.
4. At 1.00 and 1.15 µs, the 100K speed falls below the band, so no extra time can produce a match.
5. At 0.50 and 0.65 µs, the 10K speed is more than 15% too fast.

## Results

The run took 1 minute 31 seconds with a peak resident memory of 124 MB.
[`experiments/lpx/tray-latency-results.json`](../experiments/lpx/tray-latency-results.json) holds every step.

```bash
.venv/bin/python scripts/run_lpx_tray_latency.py \
  --summary experiments/lpx/tray-latency-results.json
```

| Middle-tier latency | 10K step | 100K step | Increment | AA errors without extra time | Extra time per step for a match |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.50 µs | 205.0 to 205.1 µs | 219.2 to 219.3 µs | 14.2 µs | +44.2%, +33.0% | 76.5 to 87.6 µs |
| 0.65 µs | 242.4 to 242.5 µs | 256.6 to 256.7 µs | 14.2 µs | +22.0%, +13.6% | 39.1 to 50.2 µs |
| 0.75 µs | 267.3 to 267.4 µs | 281.5 to 281.6 µs | 14.2 µs | +10.6%, +3.5% | 14.2 to 25.3 µs |
| 0.85 µs | 292.2 to 292.3 µs | 306.4 to 306.5 µs | 14.2 µs | +1.2%, −4.9% | 0 to 0.4 µs |
| 1.00 µs | 329.5 to 329.6 µs | 343.7 to 343.8 µs | 14.2 µs | −10.3%, −15.2% | none |
| 1.15 µs | 366.9 to 367.0 µs | 381.1 to 381.2 µs | 14.2 µs | −19.4%, −23.5% | none |

Values cover attention matrix efficiencies 0.30 and 0.50, which differ by at most 0.1 µs.
The errors assume one native token per AA token.

### Comparison with the expectations

All five expectations held.

1. The step grew by 24.9 µs for each 0.1 µs of middle-tier latency, from 205.1 µs at 0.50 µs to 367.0 µs at 1.15 µs.
2. The increment stayed at 14.2 µs at every latency.
3. At 0.85 µs both AA values matched without extra time, at +1.2% and −4.9%.
   Interpolating the linear step, the matching latencies are 0.81 to 0.85 µs.
4. At 1.00 and 1.15 µs both AA speeds fell below the band.
5. At 0.50 and 0.65 µs the 10K speed was 44.2% and 22.0% too fast.

## Interpretation

The TP32 step is almost a linear function of the middle-tier latency and grows by about 249 µs per microsecond of it.
The 14 to 25 µs of extra time that the article-condition sweep needs is therefore equivalent to a middle-tier latency of 0.81 to 0.85 µs instead of 0.75 µs, an 8% to 13% change in a value that comes from a reachable-capacity chart.
With the 1.031 to 1.048 Gemma tokens per `o200k_base` token measured on Gemma 4 31B's own writing outputs in the [token-ratio record](lpx-token-ratio.md), the matching range moves to about 0.76 to 0.81 µs, and at the 1.073 measured on another model's reasoning traces it moves to about 0.73 to 0.77 µs, which includes the catalog value.

The match therefore rests on one unmeasured number.
The TP32/PP8, DCP 8 layout places both AA values within 5% for any middle-tier latency between about 0.73 and 0.85 µs, depending on the token ratio, and for about 0.76 to 0.81 µs at the ratios of Gemma's own writing outputs, and no extra time per step is needed inside that range.
Below that range the layout needs extra time per step, and above it the 100K speed falls below the band.
A measured collective latency between two LP30 trays would decide whether this layout is consistent with the hardware.
