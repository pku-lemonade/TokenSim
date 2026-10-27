# Cross-tray latency and the TP32 AA match

Record date: 2026-10-26.
The [article-condition sweep](lpx-article-sweep.md) found that TP32/PP8 with decode context parallelism (DCP) of 8 on 256 LP30 chips reproduces both Artificial Analysis (AA) speeds if each step costs 14 to 25 µs more than modeled.
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

Pending.
