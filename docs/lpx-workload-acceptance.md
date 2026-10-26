# LPX per-workload acceptance fits for the AA Gemma pair and the coding values

Record date: 2026-10-26.
This control extends the [speculative-decoding control](lpx-speculative-decoding.md) and the [attention-group control](lpx-attention-group.md).
Both fitted one acceptance rate shared by the Artificial Analysis (AA) 10K and 100K workloads and priced the 16K coding prompt only without background traffic.

## Questions

1. Which serving setups reproduce all four published speeds with one acceptance rate per workload: AA 3,382 tokens per second (TPS) at 10K input, AA 3,431 TPS at 100K input, the 4,767 TPS SPEED-Bench coding median and the 10,996 TPS Hot Chips coding example?
2. Do the fitted acceptance rates of those setups follow a plausible order?

## Why the shared acceptance is relaxed

With one acceptance shared by both AA workloads, each step commits the same number of tokens at 10K and 100K.
The AA speed is then proportional to the reciprocal of the step time, and the 100K step is never shorter than the 10K step.
A shared acceptance therefore cannot reproduce the published 1.45% rise from 10K to 100K.
Drafting acceptance depends on the text, and the earlier records already gave the coding values their own acceptance.
This control gives each of the four published speeds its own acceptance.

Each fit is exact: the acceptance `alpha` solves `tau(alpha, K) = TPS * step`, where `tau = (1 - alpha^(K+1)) / (1 - alpha)` is the expected number of committed tokens per verification step with `K` draft tokens.
A speed is reachable when `1 <= TPS * step <= K + 1`.
The AA 10K speed needs a 10K step of at least 295.7 microseconds, and the coding example needs a coding step of at most `(K + 1) / 10,996` seconds: 181.9 microseconds at `K = 1`, 363.8 at `K = 3` and 727.5 at `K = 7`.
Exact fits make every reachable setup match by construction, so the informative results are reachability and the fitted acceptance values.

## Method

[`scripts/run_lpx_workload_fit.py`](../scripts/run_lpx_workload_fit.py) captures native verification forwards with the settings and step capture of the [speculative-decoding driver](../scripts/run_lpx_speculative.py).
Each point prices the AA request at 10,750 and 101,000 tokens and the coding request at 17,000 tokens under the same background traffic.
Every request in a forward verifies `q = K + 1` query tokens, and no forward exceeds 16 query rows.
The fits carry no draft cost.

| Dimension | Values |
| --- | --- |
| Layout and attention matrix efficiency | TP16/PP8 with `KVP = 4` at 0.50 and 0.10; TP16/PP16 with `KVP = 4` at 0.50; TP16/PP8 at 0.50; TP32/PP4 with `KVP = 8` at 0.50 |
| Draft tokens `K` and background requests | `K = 1` with 0, 1, 3 or 7; `K = 3` with 0, 1 or 3; `K = 7` with 0 or 1 |
| Background context | 1,000, 4,000, 16,000 tokens |

TP denotes tensor parallelism, PP pipeline parallelism and `KVP` the decode-context-parallel size.
The sweep has 105 points.

```bash
systemd-run --user --scope -q -p MemoryMax=3500M -p MemorySwapMax=0 \
  .venv/bin/python scripts/run_lpx_workload_fit.py \
  --summary experiments/lpx/workload-acceptance-results.json
```

## Expectations recorded before the sweep

A scratch run of TP16/PP8 with `KVP = 4`, efficiency 0.50, `K = 7` and one 1,000-token background request gave steps of 417.9, 514.0 and 424.5 microseconds at AA 10K, AA 100K and coding.
Its fitted acceptances are 0.29 at AA 10K, 0.43 at AA 100K, 0.51 for the coding median and 0.84 for the coding example.
The expected outcomes follow from the step times of the earlier records and the reachability bounds above:

- `K = 1` reaches the coding example nowhere, because every coding step exceeds 181.9 microseconds.
- Without background, TP16 at efficiency 0.50 reaches the AA 10K speed nowhere, because its 10K steps are 204 to 231 microseconds.
  TP16/PP8 with `KVP = 4` at efficiency 0.10 reaches all four speeds at `K = 7` with an AA 10K acceptance near 0.14 and an AA 100K acceptance near 0.66.
- TP32/PP4 with `KVP = 8` reaches all four speeds without background at `K = 3` and `K = 7`, with AA acceptances below 0.15.
- `K = 3` with one background request reaches all four speeds on TP16/PP16 with `KVP = 4`, on TP32 and on TP16/PP8 with `KVP = 4` only with the 16,000-token background, and the coding example then needs an acceptance near 0.9 or more.
- `K = 3` with three background requests reaches the coding example nowhere, because its steps exceed 363.8 microseconds.
- `K = 7` with one background request reaches all four speeds on every layout, with coding-example acceptances of 0.83 to 0.86.
- Every fit gives the AA 100K workload a higher acceptance than the AA 10K workload, because every 100K-to-10K step-time ratio exceeds the published speed ratio of 1.0145.

## Results

The sweep completed all 105 points on 2026-10-26 with source fingerprint `8365461d4e3e825c21a6e24d6356fa485f0a5c4a6139bca805bcbed15df3a57e`.
[`experiments/lpx/workload-acceptance-results.json`](../experiments/lpx/workload-acceptance-results.json) holds every step time, the committed tokens each published speed needs and the fitted acceptances.
The AA steps reproduce the earlier batched and serial records, including 428.3 and 461.9 microseconds for `K = 1` with seven 1,000-token requests on TP16/PP8 with `KVP = 4`.
Resident KV admission rejected five points: four on TP16/PP8 without `KVP` and seven 16,000-token requests at `K = 1` on TP32.

### Setups that reach all four speeds

29 setups reach all four published speeds.
The table lists them with acceptances in the order AA 10K, AA 100K, coding median and coding example.

| Layout | Efficiency | `K` | Background | Acceptances |
| --- | ---: | ---: | --- | --- |
| TP16/PP8, `KVP = 4` | 0.50 | 3 | 1 × 16,000 | 0.003, 0.155, 0.307, 0.872 |
| TP16/PP8, `KVP = 4` | 0.50 | 7 | 1 × 1,000 to 16,000 | 0.292–0.319, 0.434–0.451, 0.508–0.527, 0.838–0.850 |
| TP16/PP8, `KVP = 4` | 0.10 | 3 | 1 × 1,000 or 4,000 | 0.078–0.101, 0.518–0.528, 0.393–0.410, 0.950–0.966 |
| TP16/PP8, `KVP = 4` | 0.10 | 7 | none | 0.138, 0.659, 0.443, 0.798 |
| TP16/PP8, `KVP = 4` | 0.10 | 7 | 1 × 1,000 to 16,000 | 0.392–0.480, 0.720–0.749, 0.603–0.664, 0.901–0.944 |
| TP16/PP16, `KVP = 4` | 0.50 | 3 | 1 × 1,000 to 16,000 | 0.024–0.049, 0.170–0.190, 0.322–0.342, 0.886–0.904 |
| TP16/PP16, `KVP = 4` | 0.50 | 7 | none | 0.022, 0.269, 0.321, 0.729 |
| TP16/PP16, `KVP = 4` | 0.50 | 7 | 1 × 1,000 to 16,000 | 0.329–0.353, 0.458–0.474, 0.534–0.551, 0.854–0.866 |
| TP16/PP8 | 0.50 | 3 | 1 × 16,000 | 0.013, 0.300, 0.325, 0.889 |
| TP16/PP8 | 0.50 | 7 | 1 × 1,000 to 16,000 | 0.251–0.291, 0.450–0.473, 0.482–0.511, 0.822–0.839 |
| TP32/PP4, `KVP = 8` | 0.50 | 3 | none | 0.005, 0.093, 0.304, 0.870 |
| TP32/PP4, `KVP = 8` | 0.50 | 3 | 1 × 1,000 to 16,000 | 0.155–0.166, 0.222–0.232, 0.422–0.431, 0.977–0.985 |
| TP32/PP4, `KVP = 8` | 0.50 | 7 | none | 0.147, 0.262, 0.401, 0.774 |
| TP32/PP4, `KVP = 8` | 0.50 | 7 | 1 × 1,000 to 16,000 | 0.338–0.350, 0.411–0.421, 0.537–0.546, 0.856–0.862 |

No `K = 1` setup reaches the coding example, and no setup with three or seven background requests reaches it.
Every reachable fit gives the AA 100K workload a higher acceptance than the AA 10K workload.

### Gap between the AA acceptances

The difference between the AA 100K and AA 10K acceptances measures how much the drafter must improve at long context.

| Setup at efficiency 0.50, `K = 7` | Background | AA 10K | AA 100K | Gap |
| --- | --- | ---: | ---: | ---: |
| TP16/PP8 | 1 × 1,000 | 0.251 | 0.450 | 0.199 |
| TP16/PP8, `KVP = 4` | 1 × 1,000 | 0.292 | 0.434 | 0.142 |
| TP16/PP16, `KVP = 4` | 1 × 1,000 | 0.329 | 0.458 | 0.129 |
| TP32/PP4, `KVP = 8` | 1 × 1,000 | 0.338 | 0.411 | 0.073 |
| TP32/PP4, `KVP = 8` | none | 0.147 | 0.262 | 0.115 |

The gap shrinks as the AA request's own 100K increment shrinks, from TP16 without `KVP` to TP32 with `KVP = 8`.
At efficiency 0.10 the gap on TP16/PP8 with `KVP = 4` grows to 0.27 to 0.52.

### Comparison with the expectations

- `K = 1` reached the coding example nowhere, as expected.
- The expectation that TP16 at efficiency 0.50 reaches the AA 10K speed nowhere without background was wrong for TP16/PP16 with `KVP = 4` at `K = 7`, whose 10K step of 302.3 microseconds reaches it with an acceptance of 0.022.
- TP16/PP8 with `KVP = 4` at efficiency 0.10 and `K = 7` reached all four speeds without background with AA acceptances of 0.138 and 0.659, as expected.
- TP32 reached all four speeds without background at `K = 3` and `K = 7`, as expected, but at `K = 7` the AA 100K acceptance is 0.262 and exceeds the expected bound of 0.15.
- `K = 3` with one background request reached all four speeds on TP16/PP16 with `KVP = 4`, on TP32 and on TP16/PP8 with `KVP = 4` only with the 16,000-token request, as expected, with coding-example acceptances of 0.872 to 0.985.
  TP16/PP8 without `KVP` also reached them with the 16,000-token request, and TP16/PP8 with `KVP = 4` reached them at efficiency 0.10 with 1,000-token and 4,000-token requests.
- `K = 3` with three background requests reached the coding example nowhere, as expected.
- `K = 7` with one background request reached all four speeds on every layout, as expected.
  The coding-example acceptances at efficiency 0.50 are 0.822 to 0.866, slightly wider than the expected 0.83 to 0.86, and 0.901 to 0.944 at efficiency 0.10.
- Every fit gave the AA 100K workload the higher acceptance, as expected.

### Interpretation

With one acceptance per published speed, many setups reproduce all four values, so a fit alone no longer discriminates between setups.
The fitted acceptances do.
`K = 7` with one background request gives the most even acceptances on every layout, with AA values of 0.25 to 0.35 at 10K and 0.41 to 0.47 at 100K at efficiency 0.50.
The batch-one fits, which match a private endpoint without other traffic, need AA 10K acceptances of 0.005 to 0.147 at efficiency 0.50.
TP32/PP4 with `KVP = 8` at `K = 7` is the batch-one setup with the most even acceptances: 0.147, 0.262, 0.401 and 0.774.

`KVP` is no longer required for a fit, because a larger AA acceptance gap can absorb the AA request's own 100K increment.
`KVP` and a larger attention group reduce that gap from 0.20 on TP16/PP8 without `KVP` to 0.07 on TP32/PP4 with `KVP = 8`.
Whether any of these gaps is plausible depends on the drafter.
Lookup drafting on long prompts whose answers reuse the prompt could raise acceptance at 100K, and model drafters usually lose acceptance at long context.

The fits carry no draft cost, assume every request verifies `K` draft tokens in every step, and use acceptances fitted to published speeds.
A lookup drafter that proposes no tokens when it finds no match would run cheaper steps than these fits assume.
