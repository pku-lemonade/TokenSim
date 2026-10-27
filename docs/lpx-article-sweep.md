# Batch-one AA sweep under the published LPX test conditions

Record date: 2026-10-26.
The [test-configuration record](lpx-aa-test-configuration.md) separates what public sources establish about the Artificial Analysis (AA) Gemma 4 31B measurement from what they leave open.
This sweep keeps only setups consistent with those sources and asks which of them reproduce both AA speeds.

## Conditions

| Condition | Value | Basis |
| --- | --- | --- |
| Requests per forward | One AA request; no background requests | AA footnotes: 50 sequential requests at client concurrency 1 to a private endpoint |
| Speculative decoding | None | No source mentions it for the AA run; a later arm can add it |
| Configuration | One layout for both workloads | Both footnotes name the same deployment |
| Hardware | LP30 chips only | The Register reports a model "running entirely on LPUs" |
| Precision | FP8 weights and FP8 KV cache, converted to BF16 for attention on the vector unit | The Register reports FP8; the KV conversion is the existing model assumption |
| Chip count | 256, one LPX rack | Upper bound from the Hot Chips "single rack" statement |
| Tensor parallelism (TP) | 32, the largest head-parallel TP for 32 query heads, and 16 for comparison | NVIDIA attributes the flat speed to "high tensor parallelism" |
| Decode context parallelism (DCP) | 1, 2, 4 and 8 on TP32; 1, 2 and 4 on TP16 | NVIDIA describes "parallelism for both model weights and long-context KV cache"; the code calls DCP `KVP` |
| Chip-to-chip (C2C) bandwidth | 96 links of 112 Gb/s, or 14 GB/s each, with a per-chip injection limit of 96 × 14 GB/s = 1.344 TB/s per direction | NVIDIA's long-context and architecture articles |
| C2C latency | 0.35 µs inside a 16-chip tray, 0.75 µs within 128 chips and 1.15 µs across the rack | Hot Chips 2026 interconnect slide, as in earlier records |
| Attention matrix efficiency | 0.10, 0.30 and 0.50 | Unmeasured; same values as earlier records |

The previous fabric used 1.25 TB/s per direction, half of the published 2.5 TB/s bidirectional figure.
Each link still carries 14 GB/s, and messages between two chips still use 6 lanes.

[`scripts/run_lpx_article_sweep.py`](../scripts/run_lpx_article_sweep.py) captures one native decode step at the mean decode context of each AA workload and of the 16K coding prompt.
A point matches when both AA speeds lie within 5% of the published values.
The sweep also reports the range of constant extra time per step that would make both speeds match.
Constant per-step time stands in for costs outside the model, such as token streaming, sampling or step synchronization.

## Expectations recorded before the run

1. The injection change from 1.25 to 1.344 TB/s changes no step by more than 0.5 µs.
   A batch-one activation message of about 10.75 KB is limited by the 6-lane path between two chips, which the change does not touch.
2. TP32/PP8 with DCP 8 at efficiency 0.30 and 0.50 repeats the [attention-group](lpx-attention-group.md) steps of about 267.4 µs at 10K and 281.6 µs at 100K.
   Its 10K speed is about 10.6% too fast, so it does not match without extra time.
   Its constant-time window is about 14.2 to 25.2 µs.
3. TP32/PP8 with DCP 4 has an increment of about 25.7 µs, above the 25.2 µs bound, so it has no window.
   DCP 1 and DCP 2 have larger increments and no window.
4. No point at efficiency 0.10 has a window, because the global attention increment is about 30 µs.
5. TP16/PP16 has no window at any DCP, because its smallest increment is about 28.4 µs.
6. No point matches without extra time.

## Results

Pending.
