# LPX attention-group control for the AA Gemma pair

Record date: 2026-10-25.
This control follows the [batched-serving control](lpx-batched-serving.md) and the [speculative-decoding control](lpx-speculative-decoding.md).
Both found that the Artificial Analysis (AA) request's own 10K-to-100K attention increment prevents a match when the request shares its forward with background traffic.

## Terms

The attention group is the set of `N` chips that compute one global attention layer for one request.
In these layouts it is the tensor-parallel (TP) group of a pipeline stage, so `N` equals the TP size.
Decode context parallelism (DCP) factors that group as `N = TPA * DCP`.
TPA ranks split the four global key/value (KV) heads, and DCP ranks split the context by token position.
The scripts and earlier records call the DCP size `KVP`, and the layout field is `a_cp`.
PP denotes pipeline parallelism.

## Question

Does a larger attention group bring the simulated AA request to the published 3,382 tokens per second (TPS) at 10K input and 3,431 TPS at 100K input?
A point matches when both simulated speeds lie within 5% of the published values, which means 3,213 to 3,551 TPS at 10K and 3,259 to 3,603 TPS at 100K.

## Why the attention group matters

A breakdown of the TP16/PP8 `KVP = 4` serial forward at attention matrix efficiency 0.50 places the whole 28.35-microsecond increment in the attention core of the 10 global layers.
Each global layer's core takes 0.40 microseconds at 10K and 3.24 microseconds at 100K.
At 100K the core's unit service times are 3.17 microseconds on the vector unit, 1.35 microseconds on the matrix unit and approximately 0.2 microseconds each on the shuffle unit and static random-access memory (SRAM).
FP8-to-BF16 conversion of KV-cache elements accounts for most of the vector-unit time.
Raising the matrix efficiency therefore leaves the increment unchanged above 0.30.

Once no KV head is replicated, each chip converts and multiplies `1 / N` of the layer's KV work.
DCP inside a fixed TP16 group removes replication and keeps `N = 16`, which explains why `KVP = 2`, `4` and `8` gave nearly identical speeds in the [context-parallel results](lpx-context-parallel.md#results).
TP32 with `KVP = 8` keeps TPA at 4 and doubles `N`.

Without speculative decoding, both AA values fit their bands only if the increment is at most 25.2 microseconds.
That bound is the difference between the slowest admissible 100K step, 306.8 microseconds, and the fastest admissible 10K step, 281.6 microseconds.

## Method

[`scripts/run_lpx_attention_group.py`](../scripts/run_lpx_attention_group.py) uses the settings and step capture of the [speculative-decoding driver](../scripts/run_lpx_speculative.py).
Every point captures one native forward with the LPX streaming and fabric refinement.
The AA request sits at its mean generation context, 10,750 or 101,000 tokens, with a resident KV reserve of 11,500 or 102,000 tokens.
Background requests decode at a fixed context and reserve exactly that context.
Pipeline layer counts follow the capacity partition of each stage count.

| Dimension | Values |
| --- | --- |
| Layout | TP32/PP4 on 128 LP30 chips with `KVP = 1`, `2`, `4` and `8`; TP32/PP8 with `KVP = 8` on a 256-chip rack |
| Background requests `n` | 0, 1, 2, 3, 7 |
| Background context | 1,000, 4,000, 16,000 tokens |
| Attention matrix efficiency `compute.attention_mfu` | 0.10 (study value), 0.30, 0.50 |

A second part repeats the speculative-decoding analysis on TP32/PP4 with `KVP = 8`.
It verifies `q = K + 1` query tokens per request with `q` of 2, 4 and 8, background counts of 0, 1, 3 and 7, and efficiencies of 0.10 and 0.50.
It keeps the limit of 16 query rows per forward and the acceptance arithmetic of the speculative-decoding control.
Points without background traffic also price the 17,000-token coding context.

```bash
systemd-run --user --scope -q -p MemoryMax=3500M -p MemorySwapMax=0 \
  .venv/bin/python scripts/run_lpx_attention_group.py \
  --summary experiments/lpx/attention-group-results.json
```

## Expectations recorded before the sweep

Scratch runs on TP32/PP4 with `KVP = 8` gave these values before the driver existed.

| Case | 10K step | 100K step | 10K TPS | 100K TPS |
| --- | ---: | ---: | ---: | ---: |
| Efficiency 0.50, `n = 0` | 263.44 µs | 277.64 µs | 3,796 | 3,602 |
| Efficiency 0.10, `n = 0` | 265.67 µs | 295.77 µs | 3,764 | 3,381 |
| Efficiency 0.50, two 1,000-token requests | 289.67 µs | 303.99 µs | 3,452 | 3,290 |
| Efficiency 0.50, two 4,000-token requests | 290.81 µs | 305.09 µs | 3,439 | 3,278 |
| Efficiency 0.50, two 16,000-token requests | 294.65 µs | 308.83 µs | 3,394 | 3,238 |

TP32/PP4 with `KVP = 1` was rejected because resident KV needs 215,711,744 bytes on a rank with 194,830,336 available.

The global attention core gives these expected increments for 10 global layers:

| Layout | Efficiency 0.10 | Efficiency 0.30 | Efficiency 0.50 |
| --- | ---: | ---: | ---: |
| TP16 `KVP = 4`, reference | 60.1 µs | 28.4 µs | 28.3 µs |
| TP32 `KVP = 2` | 48.8 µs | 48.8 µs | 48.8 µs |
| TP32 `KVP = 4` | 30.1 µs | 25.7 µs | 25.7 µs |
| TP32 `KVP = 8` | 30.1 µs | 14.2 µs | 14.2 µs |

Background requests add less than 1 microsecond to these increments.
The expected outcomes are:

- TP32/PP4 with `KVP = 1` is infeasible at every point.
- TP32/PP4 with `KVP = 8` matches with two 1,000-token or 4,000-token background requests at efficiencies 0.30 and 0.50 and matches nowhere at efficiency 0.10.
- TP32/PP4 with `KVP = 2` or `KVP = 4` matches nowhere, because each increment exceeds 25.2 microseconds.
- TP32/PP8 with `KVP = 8` has the same increment as PP4 and matches with one to three background requests at efficiencies 0.30 and 0.50.
- Speculative decoding on TP32/PP4 with `KVP = 8` adds matches, including efficiency 0.10 points whose batched step exceeds 336 microseconds, because the step-time ratio limit of 1.0896 then admits a 30.1-microsecond increment.
- The 10,996 TPS coding example needs approximately 2.9 or more committed tokens per step at zero draft cost, because the serial TP32 step is slower than the TP16 step.
  Three draft tokens then need an acceptance of approximately 0.8, and two draft tokens need an acceptance near 1.

## Results

The sweep completed all 237 points on 2026-10-25 with source fingerprint `8365461d4e3e825c21a6e24d6356fa485f0a5c4a6139bca805bcbed15df3a57e`, the fingerprint of the speculative-decoding results.
[`experiments/lpx/attention-group-results.json`](../experiments/lpx/attention-group-results.json) holds every step time, the one-token analysis, the serial speculative analysis and the batched speculative analysis.
The scratch values in the expectations table reproduce to 0.01 microseconds.

### Admission

Resident KV admission rejected 74 points.
TP32/PP4 with `KVP = 1` was rejected at all 39 points, as expected.
`KVP = 2` was rejected at 27 points: every point with three or seven background requests, two requests of 4,000 or 16,000 tokens, and one 16,000-token request.
`KVP = 4` and `KVP = 8` on PP4 rejected seven 16,000-token background requests.
The speculative part on PP4 with `KVP = 8` also rejected seven 16,000-token requests at `K = 1`.
TP32/PP8 with `KVP = 8` admitted every point.

### One token per forward

| Layout | Efficiency | 10K step without background | 100K step without background | Increment |
| --- | ---: | ---: | ---: | ---: |
| TP32/PP4, `KVP = 2` | 0.10 to 0.50 | 266.2 to 266.5 µs | 315.0 to 315.3 µs | 48.8 µs |
| TP32/PP4, `KVP = 4` | 0.10 | 264.81 µs | 294.88 µs | 30.1 µs |
| TP32/PP4, `KVP = 4` | 0.30 and 0.50 | 263.9 to 264.0 µs | 289.6 to 289.7 µs | 25.7 µs |
| TP32/PP4, `KVP = 8` | 0.10 | 265.67 µs | 295.77 µs | 30.1 µs |
| TP32/PP4, `KVP = 8` | 0.30 and 0.50 | 263.4 to 263.6 µs | 277.6 to 277.8 µs | 14.2 µs |
| TP32/PP8, `KVP = 8` | 0.10 | 269.58 µs | 299.68 µs | 30.1 µs |
| TP32/PP8, `KVP = 8` | 0.30 and 0.50 | 267.4 to 267.5 µs | 281.6 to 281.7 µs | 14.2 µs |

Every increment matches its expectation to 0.1 microseconds, and background requests add at most 1.1 microseconds to it.
The TP32 step without background is approximately 87 microseconds longer than the TP16/PP8 step with `KVP = 4`.

### Base step

A breakdown on 2026-10-26 removed one category of work at a time from the modeled 10K step at efficiency 0.50 and recorded how much the step shrank.

| Removed work | TP16/PP8, `KVP = 4` | TP32/PP4, `KVP = 8` | Difference |
| --- | ---: | ---: | ---: |
| TP-group collectives: all-reduce, all-to-all and all-gather | 89.5 µs | 188.5 µs | +99.0 µs |
| Attention compute | 36.6 µs | 34.0 µs | −2.6 µs |
| Feed-forward compute | 18.8 µs | 14.3 µs | −4.5 µs |
| Normalization, residual and output-head compute | 20.3 µs | 20.0 µs | −0.3 µs |
| Pipeline-stage transfers | 6.2 µs | 2.6 µs | −3.5 µs |
| Whole step | 176.4 µs | 263.4 µs | +87.0 µs |

The removals sum to the whole step within 5 microseconds, most of which is the 4-microsecond graph submission.
The TP-group collectives account for the whole difference, while compute shrinks by about 7 microseconds because batch-one operators are dominated by fixed per-operator costs.
The collective cost ratio of 2.11 is close to the ratio of 2.14 between the 0.75-microsecond middle fabric tier and the 0.35-microsecond tier inside a 16-chip tray.
A TP32 group spans two trays, so its collectives probably use the middle tier, but this record has not repriced the collectives tier by tier.
The TP32 matches therefore also depend on the assumed middle-tier latency.

Six one-token points match both AA values within 5%:

| Layout | Efficiency | Background | 10K TPS | 100K TPS |
| --- | ---: | --- | ---: | ---: |
| TP32/PP4, `KVP = 8`, 128 chips | 0.30 | 2 × 1,000 | 3,448.9 (+2.0%) | 3,285.8 (−4.2%) |
| TP32/PP4, `KVP = 8`, 128 chips | 0.30 | 2 × 4,000 | 3,435.2 (+1.6%) | 3,273.9 (−4.6%) |
| TP32/PP4, `KVP = 8`, 128 chips | 0.50 | 2 × 1,000 | 3,452.2 (+2.1%) | 3,289.6 (−4.1%) |
| TP32/PP4, `KVP = 8`, 128 chips | 0.50 | 2 × 4,000 | 3,438.7 (+1.7%) | 3,277.7 (−4.5%) |
| TP32/PP8, `KVP = 8`, 256 chips | 0.30 | 1 × 16,000 | 3,540.8 (+4.7%) | 3,371.2 (−1.7%) |
| TP32/PP8, `KVP = 8`, 256 chips | 0.50 | 1 × 16,000 | 3,543.6 (+4.8%) | 3,373.9 (−1.7%) |

TP32/PP4 with `KVP = 8` matched at the four expected points.
TP32/PP8 matched only with one 16,000-token request, because its longer base step leaves a narrower background window than expected.
No point at efficiency 0.10 matched, and no `KVP = 2` or `KVP = 4` point matched, as expected.
The nearest `KVP = 4` point, one 16,000-token request at efficiency 0.30 or 0.50, misses the 10K band by 0.6%.

### Speculative decoding on TP32/PP4 with `KVP = 8`

At efficiency 0.50, 14 of 17 admitted batched points match the AA pair with one shared acceptance:

| `K` | Background | Step-time ratio | Shared acceptance |
| ---: | --- | ---: | --- |
| 1 | 1 × 1,000 to 16,000 | 1.055 to 1.056 | 0.039 to 0.083 |
| 1 | 3 × 1,000 to 16,000 | 1.046 to 1.048 | 0.210 to 0.290 |
| 1 | 7 × 1,000 or 4,000 | 1.036 to 1.037 | 0.552 to 0.651 |
| 3 | 1 × 1,000 to 16,000 | 1.068 to 1.069 | 0.180 to 0.206 |
| 3 | 3 × 1,000 to 16,000 | 1.052 to 1.054 | 0.363 to 0.408 |
| 7 | 1 × 1,000 to 16,000 | 1.106 to 1.108 | none |

At efficiency 0.10 no batched point matches, and the ratios range from 1.123 to 1.502.
The serial part matches at efficiency 0.50 for `K = 1` with a draft-cost ratio of 0.05 to 0.20 and for `K = 3` at every draft-cost ratio:

| `K` | Draft-cost ratio | Serial step-time ratio | Shared AA acceptance | 10,996 TPS example | 4,767 TPS median |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 0.05 | 1.061 | 0.000 to 0.016 | unreachable | 0.369 |
| 1 | 0.20 | 1.060 | 0.125 to 0.156 | unreachable | 0.559 |
| 3 | 0 | 1.081 | 0.045 to 0.052 | 0.870 | 0.304 |
| 3 | 0.05 | 1.078 | 0.155 to 0.164 | 0.952 | 0.395 |
| 3 | 0.10 | 1.075 | 0.244 to 0.254 | unreachable | 0.473 |
| 3 | 0.20 | 1.072 | 0.385 to 0.397 | unreachable | 0.601 |
| 7 | 0 | 1.139 | none | 0.774 | 0.401 |

No serial point matches at efficiency 0.10.
At efficiency 0.50 the coding example needs 0.870 at `K = 3` and 0.774 at `K = 7` without draft cost, and 0.910 and 0.810 at efficiency 0.10.

### Comparison with the expectations

The one-token expectations held except that TP32/PP8 matched with only one 16,000-token background request.
The speculative expectation for efficiency 0.10 was wrong.
Verification multiplies the matrix-unit attention work by the number of query rows, so at efficiency 0.10 the `K = 1` increment is approximately 60 microseconds, not the 30.1 microseconds of one-token decode.
The expected acceptance of approximately 0.8 for the coding example at `K = 3` was low, and the measured value is 0.870.

### Interpretation

A larger attention group reproduces the AA pair without speculative decoding, on the same 128 chips as TP16/PP8, if attention matrix work reaches an efficiency of 0.30 or more.
With `KVP = 8` each chip handles one KV head over one eighth of the context, so the AA request's increment falls to 14.2 microseconds, below the 25.2-microsecond bound.
`KVP = 2` and `KVP = 4` on TP32 replicate KV heads and keep the increment above the bound.

On TP32/PP4 with `KVP = 8` at efficiency 0.50, `K = 3` matches the AA pair with one or three background requests and also reaches the 10,996 TPS coding example at an acceptance of 0.870 without background and without draft cost.
Within the 16-row limit, one draft length therefore covers the AA match and the coding example on one layout, which the TP16 layouts did not allow.
The required AA acceptances of 0.05 to 0.41 are lower than the coding acceptances of 0.30 to 0.87, which is consistent with drafting that works far better on code than on AA's prompts.
These acceptances are fitted to the published speeds and are not measured.
Every result at efficiency 0.10, the study value, fails, so the TP32 explanations depend on an attention matrix efficiency of at least 0.30.
