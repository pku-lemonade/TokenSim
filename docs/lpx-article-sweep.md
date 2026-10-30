# Batch-one AA sweep under the published LPX test conditions

Record date: 2026-10-26.
The [test-configuration record](lpx-aa-test-configuration.md) separates what public sources establish about the Artificial Analysis (AA) Gemma 4 31B measurement from what they leave open.
This sweep keeps only setups consistent with those sources and asks which of them place both AA speeds within 5%.

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

The sweep ran on 2026-10-26 in 3 minutes 13 seconds with a peak resident memory of 262 MB.
[`experiments/lpx/article-sweep-results.json`](../experiments/lpx/article-sweep-results.json) holds every step time and the fabric settings.

```bash
.venv/bin/python scripts/run_lpx_article_sweep.py \
  --summary experiments/lpx/article-sweep-results.json
```

| Layout on 256 chips | Efficiency | 10K step | 100K step | Increment | 10K TPS | 100K TPS | Extra time per step for a match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TP32/PP8, DCP 1 | 0.10 to 0.50 | 271.0 to 271.3 µs | 366.1 to 366.3 µs | 95.0 µs | 3,686 to 3,689 (+9.1%) | 2,730 to 2,732 (−20.4%) | none |
| TP32/PP8, DCP 2 | 0.10 to 0.50 | 270.0 to 270.3 µs | 318.9 to 319.2 µs | 48.8 µs | 3,699 to 3,703 (+9.5%) | 3,133 to 3,136 (−8.6%) | none |
| TP32/PP8, DCP 4 | 0.10 | 268.6 µs | 298.7 µs | 30.1 µs | 3,723 (+10.1%) | 3,348 (−2.4%) | none |
| TP32/PP8, DCP 4 | 0.30 and 0.50 | 267.7 to 267.8 µs | 293.5 µs | 25.7 µs | 3,734 to 3,735 (+10.4%) | 3,407 to 3,408 (−0.7%) | none |
| TP32/PP8, DCP 8 | 0.10 | 269.5 µs | 299.6 µs | 30.1 µs | 3,711 (+9.7%) | 3,338 (−2.7%) | none |
| TP32/PP8, DCP 8 | 0.30 | 267.4 µs | 281.6 µs | 14.2 µs | 3,740 (+10.6%) | 3,551 (+3.5%) | 14.2 to 25.2 µs |
| TP32/PP8, DCP 8 | 0.50 | 267.3 µs | 281.5 µs | 14.2 µs | 3,742 (+10.6%) | 3,553 (+3.6%) | 14.3 to 25.3 µs |
| TP16/PP16, DCP 1 | 0.10 to 0.50 | 186.2 to 186.7 µs | 283.9 to 284.3 µs | 97.7 µs | 5,357 to 5,370 (+58.6%) | 3,517 to 3,523 (+2.6%) | none |
| TP16/PP16, DCP 2 | 0.10 | 187.3 µs | 247.4 µs | 60.1 µs | 5,340 (+57.9%) | 4,043 (+17.8%) | none |
| TP16/PP16, DCP 2 | 0.30 and 0.50 | 185.7 to 185.8 µs | 237.1 to 237.2 µs | 51.5 µs | 5,383 to 5,386 (+59.2%) | 4,216 to 4,217 (+22.9%) | none |
| TP16/PP16, DCP 4 | 0.10 | 188.1 µs | 248.2 µs | 60.1 µs | 5,316 (+57.2%) | 4,028 (+17.4%) | none |
| TP16/PP16, DCP 4 | 0.30 and 0.50 | 183.8 to 184.0 µs | 212.2 to 212.3 µs | 28.4 µs | 5,436 to 5,440 (+60.7%) | 4,710 to 4,713 (+37.3%) | none |

TPS values assume one committed token per step and no extra time.
The extra-time column gives the range of constant time per step that places both AA speeds within 5%.

### Comparison with the expectations

All six expectations held.

1. The injection change shortened the TP32/PP8 DCP 8 steps by 0.09 to 0.11 µs against the [attention-group results](lpx-attention-group.md).
2. TP32/PP8 with DCP 8 at efficiency 0.30 and 0.50 took 267.3 to 267.4 µs at 10K and 281.5 to 281.6 µs at 100K, with windows of 14.2 to 25.3 µs.
3. TP32/PP8 with DCP 4 had an increment of 25.7 µs and no window; DCP 1 and DCP 2 had increments of 95.0 and 48.8 µs.
4. Every point at efficiency 0.10 had an increment of at least 30.1 µs and no window.
5. TP16/PP16 had a smallest increment of 28.4 µs and no window.
   Its 10K step of about 184 µs would need at least 97.6 µs of extra time, while its 100K step allows at most 94.5 µs.
6. No point matched without extra time.

## Interpretation

Under the published conditions, one setup places both AA speeds within 5%: TP32/PP8 on 256 chips with DCP 8, attention matrix efficiency of at least 0.30, and 14 to 25 µs of constant time per step that the model does not include.
The extra time is 5% to 9% of the step.
Without it, the 100K speed lies within 5% of the published value and the 10K speed is 10.6% too fast.
The 25.2 µs bound on the increment from 10K to 100K excludes every other layout and every point at efficiency 0.10.

The constant time has no identified source.
Candidates are token streaming to the host, sampling, step synchronization and collectives slower than the assumed 0.75 µs between trays.
In earlier records, background requests supplied this time, but the AA footnotes make other requests on the endpoint unlikely.

TP16 gives a shorter step than TP32, because its tensor-parallel collectives stay inside a 16-chip tray at 0.35 µs.
NVIDIA's description of "high tensor parallelism" and the 25.2 µs increment bound both point to TP32, which pays the 0.75 µs tier on each collective.

The coding values do not fit this setup without speculative decoding.
At the matching TP32/PP8 DCP 8 points, the 16K coding step is 268.3 µs plus the same 14.2 to 25.2 µs.
The 4,767 TPS SPEED-Bench median would need 1.35 to 1.40 committed tokens per step, and the 10,996 TPS example would need 3.11 to 3.23.
If the AA endpoint drafted tokens as well, the AA prompts would need nearly zero acceptance for this setup to match, so the coding values either come from a build with speculative decoding or imply a longer AA step than this sweep models.
This sweep does not test speculative decoding.
