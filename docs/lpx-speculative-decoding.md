# LPX speculative-decoding control for the AA Gemma pair

Record date: 2026-10-23.
This control tests reason 3 of the [placement plausibility record](lpx-placement-plausibility.md), together with the batched-serving outcome of the [batched-serving control](lpx-batched-serving.md).
Reason 3 notes that the same Gemma4-31B-LPX deployment name is reported at 10,996 tokens per second (TPS) on a 16K-input coding example, which one committed token per forward pass cannot reach.

## Questions

1. Can batch-one speculative decoding, with one acceptance rate for both Artificial Analysis (AA) workloads, place the simulated AA request within 5% of the published 3,382 TPS at 10K input and 3,431 TPS at 100K input?
2. Which draft lengths and acceptance rates reach the 10,996 TPS coding example and the 4,767 TPS SPEED-Bench coding median on the same deployment?
3. Can speculative verification shared with background requests match the AA pair with one acceptance rate?

The 5% bands are 3,213 to 3,551 TPS at 10K and 3,259 to 3,603 TPS at 100K, as in the batched-serving control.

## Method

[`scripts/run_lpx_speculative.py`](../scripts/run_lpx_speculative.py) captures native verification forwards with the settings of the batched-serving control: the bundled [fidelity configuration](../experiments/lpx/fidelity-study.json), compiled execution, a 0.35-microsecond chip-to-chip hop, zero collective launch cost and the LPX streaming and fabric refinement.
A verification forward checks `K` draft tokens plus one target token, so each request in it carries `q = K + 1` query tokens.
The AA request sits at its mean generation context, 10,750 or 101,000 tokens, and the coding request at 17,000 tokens.
The coding context assumes the published 16K input plus a mean generation position of approximately 600 tokens, because SPEED-Bench output lengths are not published with the slide.

[`scripts/lpx/speculative_fit.py`](../scripts/lpx/speculative_fit.py) converts a step time into speeds.
Chain drafting with an independent per-position acceptance probability `alpha` commits `tau = (1 - alpha^(K+1)) / (1 - alpha)` tokens per step on average, between 1 and `K + 1`.
The request's speed is `tau` divided by the step time.
One shared acceptance matches the AA pair when a single `tau` places both workloads inside their bands.
Because `tau` divides both step times, a shared match requires the 100K-to-10K step-time ratio to be at most 3,551 / 3,259, which is 1.0896.

The serial part adds a draft cost of `K * r * T1(c)` to each verification step, where `T1(c)` is the same deployment's one-token step at the same context.
The ratio `r` takes the values 0, 0.05, 0.10 and 0.20.
A value of 0 represents host-side n-gram or prompt-lookup drafting, and larger values represent draft heads or small draft models on the LPX chips.
The batched part carries no draft cost, and every background request verifies the same `q` tokens.
Rejected positions are assumed to need no extra KV rollback time.

| Part | Dimension | Values |
| --- | --- | --- |
| Serial | Layout | TP16/PP8, TP16/PP8 with `KVP = 4`, TP16/PP16, TP16/PP16 with `KVP = 4` |
| Serial | Attention matrix efficiency | 0.10, 0.50, 1.00 |
| Serial | Query tokens `q` | 1, 2, 3, 4, 5, 6, 8 |
| Batched | Layout | TP16/PP8, TP16/PP8 with `KVP = 4`, TP16/PP16 with `KVP = 4` |
| Batched | Attention matrix efficiency | 0.50 |
| Batched | Query tokens and background requests | `q = 2` with 1, 3 or 7; `q = 4` with 1 or 3; `q = 8` with 1 |
| Batched | Background context | 1,000, 4,000, 16,000 tokens |

TP denotes tensor parallelism, PP pipeline parallelism and KVP the Helix-style KV-parallel size.
The serial part has 84 points and the batched part 54.

### Verification-width support and limits

The LPX streaming refinement previously rejected decode forwards with more than one query token.
A verification forward produces one output row per query token, which is the same shape as a batch of single-token requests.
The refinement now accepts any decode forward whose final matrix multiplication produces the complete all-reduce payload, and single-token results are unchanged.
The native decode workspace holds `q` tokens per request because the runtime reserves `(1 + n) * q` tokens for `n` background requests.

The batched part keeps every forward within 16 query rows.
The fidelity configuration reserves live scratch for 16 rows, so wider forwards are rejected by the configured memory policy.
Each additional row also enlarges the host-side streaming graph, and a 32-row forward exceeded the 4.5 GB memory available to the sweep process on the study host.

## Expectations recorded before the sweep

A smoke run on TP16/PP8 gives the following verification step times in microseconds at the 10,750, 17,000 and 101,000-token contexts.

| Layout and efficiency | `q = 1` | `q = 2` | `q = 4` | `q = 8` |
| --- | --- | --- | --- | --- |
| `KVP = 4`, efficiency 0.50 | 176.4, 178.4, 204.8 | 188.1, 190.5, 221.7 | 220.5, 223.9, 268.6 | 287.7, 294.4, 383.9 |
| `KVP = 4`, efficiency 1.00 | 176.3, 178.3, 204.7 | not run | 219.6, 222.7, 263.7 | 283.5, 288.0, 348.5 |
| `KVP = 1`, efficiency 0.50 | 178.8, 185.6, 276.4 | not run | 218.1, 226.0, 331.5 | 276.2, 285.5, 410.5 |

The AA request's 10K-to-100K increment grows with `q` under `KVP = 4`, from 28.4 microseconds at `q = 1` to 96.2 microseconds at `q = 8`, because global attention work grows with the number of query tokens.
Under `KVP = 1` the increment grows less, from 97.6 to 134.3 microseconds, because vector-unit KV conversion does not scale with query tokens.

1. Serial AA: no point matches with one shared acceptance.
   Every smoke-tested step-time ratio lies between 1.16 and 1.55, above the 1.0896 limit, and a draft cost proportional to `T1(c)` keeps the ratio within the same range.
   The published 10K time of 295.7 microseconds per token also exceeds every `q ≤ 8` verification step without draft cost, so the 10K level is reachable only with draft cost or wasted drafting.
2. Serial coding: the 10,996 TPS example needs `K ≥ 3` without draft cost.
   It needs an acceptance of approximately 0.68 at `K = 3` and 0.70 to 0.71 at `K = 7` in each smoke-tested TP16/PP8 setting.
   `K = 1` cannot reach it, because at most two tokens per approximately 190-microsecond step give at most approximately 10,500 TPS.
   The 4,767 TPS median needs an acceptance below approximately 0.3 on the same deployment, approximately 0.06 at `K = 3` and 0.27 to 0.29 at `K = 7`.
   One batch-one deployment therefore cannot give both coding values with similar acceptance, which again suggests different load.
3. Batched AA: background requests add the same time at both contexts and leave the AA increment unchanged.
   A shared match requires a 10K step of at least the increment divided by 0.0896: approximately 375 microseconds at `q = 2`, 537 at `q = 4` and 1,074 at `q = 8` under `KVP = 4` at efficiency 0.50.
   The smoke point at `q = 2` with seven 1,000-token background requests on TP16/PP8 with `KVP = 4` gives 428.3 and 461.9 microseconds, a ratio of 1.078, and matches with an acceptance of approximately 0.51.
   Matches are therefore expected at `q = 2` with seven background requests under `KVP = 4`.
   They are not expected at `q = 4` or `q = 8`, where 16 rows hold too little background to reach the required step, or under `KVP = 1`, which needs a step above 1 millisecond.

## Results

The sweep completed all 138 points on 2026-10-23 with source fingerprint `8365461d4e3e825c21a6e24d6356fa485f0a5c4a6139bca805bcbed15df3a57e`.
[`experiments/lpx/speculative-results.json`](../experiments/lpx/speculative-results.json) holds every step time, the serial analysis for each draft-cost ratio and the batched analysis.
Resident KV admission rejected four batched points on TP16/PP8: `K = 1` with three 16,000-token, seven 4,000-token or seven 16,000-token background requests, and `K = 3` with three 16,000-token requests.
The serial `q = 1` point on TP16/PP8 at efficiency 0.10 reproduces the earlier serial step times, 179.3 and 276.9 microseconds.

### Serial AA pair

No serial point matches the AA pair with one shared acceptance, as expected.
The smallest serial 100K-to-10K step-time ratio is 1.154, on TP16/PP16 with `KVP = 4`, against the 1.0896 limit.
The ratio ranges from 1.486 to 2.470 on TP16/PP8, 1.161 to 2.402 on TP16/PP8 with `KVP = 4`, 1.462 to 2.408 on TP16/PP16 and 1.154 to 2.344 on TP16/PP16 with `KVP = 4`.
The upper end exceeds the smoke range of 1.55 because low attention efficiency with many query tokens makes 100K verification slow.
With separate acceptances, the 100K workload always needs a higher acceptance than the 10K workload.

### Serial coding values

The table gives the acceptance needed for the 10,996 TPS example and the 4,767 TPS median, separated by a slash, without draft cost.
A dash marks a value that no acceptance between 0 and 1 reaches.

| Layout | Efficiency | `K = 1` | `K = 2` | `K = 3` | `K = 4` | `K = 5` | `K = 7` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP16/PP8 | 0.10 | — / — | 0.82 / 0.08 | 0.76 / 0.17 | 0.76 / 0.25 | 0.76 / 0.32 | 0.78 / 0.42 |
| TP16/PP8 | 0.50 | — / — | 0.75 / 0.01 | 0.69 / 0.07 | 0.68 / 0.13 | 0.68 / 0.18 | 0.70 / 0.27 |
| TP16/PP8 | 1.00 | — / — | 0.75 / 0.01 | 0.69 / 0.07 | 0.67 / 0.13 | 0.68 / 0.18 | 0.70 / 0.26 |
| TP16/PP8, `KVP = 4` | 0.10 | — / — | 0.86 / 0.11 | 0.79 / 0.21 | 0.78 / 0.28 | 0.78 / 0.35 | 0.80 / 0.44 |
| TP16/PP8, `KVP = 4` | 0.50 | — / — | 0.73 / — | 0.68 / 0.06 | 0.68 / 0.13 | 0.69 / 0.19 | 0.71 / 0.29 |
| TP16/PP8, `KVP = 4` | 1.00 | — / — | 0.73 / — | 0.68 / 0.06 | 0.67 / 0.12 | 0.68 / 0.18 | 0.70 / 0.27 |
| TP16/PP16 | 0.10 | — / — | 0.86 / 0.12 | 0.79 / 0.21 | 0.78 / 0.28 | 0.78 / 0.34 | 0.80 / 0.44 |
| TP16/PP16 | 0.50 | — / — | 0.79 / 0.05 | 0.72 / 0.11 | 0.70 / 0.17 | 0.70 / 0.22 | 0.72 / 0.30 |
| TP16/PP16 | 1.00 | — / — | 0.79 / 0.05 | 0.72 / 0.11 | 0.70 / 0.17 | 0.70 / 0.22 | 0.72 / 0.30 |
| TP16/PP16, `KVP = 4` | 0.10 | — / 0.03 | 0.90 / 0.15 | 0.81 / 0.24 | 0.80 / 0.31 | 0.80 / 0.37 | 0.81 / 0.46 |
| TP16/PP16, `KVP = 4` | 0.50 | — / — | 0.78 / 0.03 | 0.71 / 0.11 | 0.70 / 0.17 | 0.71 / 0.23 | 0.73 / 0.32 |
| TP16/PP16, `KVP = 4` | 1.00 | — / — | 0.77 / 0.03 | 0.71 / 0.10 | 0.70 / 0.16 | 0.70 / 0.22 | 0.72 / 0.31 |

`K = 1` never reaches the 10,996 TPS example, as expected.
The expectation that the example needs `K ≥ 3` was wrong: `K = 2` reaches it with an acceptance of 0.73 to 0.90.
At efficiency 0.50 or above, the example needs 0.67 to 0.73 at `K = 3` to `K = 7`, close to the expected 0.68 and 0.70 to 0.71.
The smallest acceptance for the example is 0.67 without draft cost, 0.74 with a draft-cost ratio of 0.05, 0.81 with 0.10 and 0.91 with 0.20.
The median needs at most 0.32 at efficiency 0.50 or above and at most 0.46 at efficiency 0.10, in line with the expected bound of approximately 0.3 at the higher efficiency.

### Batched AA pair

Steps are in microseconds and the background column gives the number of requests times their context.

| Layout | `K` | Background | 10K step | 100K step | Ratio | Shared acceptance |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| TP16/PP8 | 1 | 1 × 1,000 | 223.6 | 326.5 | 1.460 | none |
| TP16/PP8 | 1 | 1 × 4,000 | 227.1 | 330.0 | 1.453 | none |
| TP16/PP8 | 1 | 1 × 16,000 | 240.8 | 343.7 | 1.427 | none |
| TP16/PP8 | 1 | 3 × 1,000 | 286.7 | 389.6 | 1.359 | none |
| TP16/PP8 | 1 | 3 × 4,000 | 297.2 | 400.1 | 1.346 | none |
| TP16/PP8 | 1 | 7 × 1,000 | 413.2 | 516.1 | 1.249 | none |
| TP16/PP8 | 3 | 1 × 1,000 | 280.6 | 393.9 | 1.404 | none |
| TP16/PP8 | 3 | 1 × 4,000 | 284.4 | 397.8 | 1.399 | none |
| TP16/PP8 | 3 | 1 × 16,000 | 299.5 | 412.9 | 1.379 | none |
| TP16/PP8 | 3 | 3 × 1,000 | 399.8 | 513.1 | 1.284 | none |
| TP16/PP8 | 3 | 3 × 4,000 | 411.4 | 524.8 | 1.276 | none |
| TP16/PP8 | 7 | 1 × 1,000 | 394.8 | 529.1 | 1.340 | none |
| TP16/PP8 | 7 | 1 × 4,000 | 399.4 | 533.7 | 1.336 | none |
| TP16/PP8 | 7 | 1 × 16,000 | 417.3 | 551.6 | 1.322 | none |
| TP16/PP8, `KVP = 4` | 1 | 1 × 1,000 | 224.8 | 258.4 | 1.149 | none |
| TP16/PP8, `KVP = 4` | 1 | 1 × 4,000 | 226.0 | 259.6 | 1.149 | none |
| TP16/PP8, `KVP = 4` | 1 | 1 × 16,000 | 230.5 | 264.1 | 1.146 | none |
| TP16/PP8, `KVP = 4` | 1 | 3 × 1,000 | 292.5 | 326.1 | 1.115 | none |
| TP16/PP8, `KVP = 4` | 1 | 3 × 4,000 | 296.2 | 329.8 | 1.113 | none |
| TP16/PP8, `KVP = 4` | 1 | 3 × 16,000 | 309.5 | 343.1 | 1.109 | none |
| TP16/PP8, `KVP = 4` | 1 | 7 × 1,000 | 428.3 | 461.9 | 1.078 | 0.505 to 0.521 |
| TP16/PP8, `KVP = 4` | 1 | 7 × 4,000 | 436.9 | 470.5 | 1.077 | 0.533 to 0.551 |
| TP16/PP8, `KVP = 4` | 1 | 7 × 16,000 | 468.0 | 501.6 | 1.072 | 0.635 to 0.662 |
| TP16/PP8, `KVP = 4` | 3 | 1 × 1,000 | 288.3 | 336.4 | 1.167 | none |
| TP16/PP8, `KVP = 4` | 3 | 1 × 4,000 | 290.1 | 338.2 | 1.166 | none |
| TP16/PP8, `KVP = 4` | 3 | 1 × 16,000 | 296.5 | 344.6 | 1.162 | none |
| TP16/PP8, `KVP = 4` | 3 | 3 × 1,000 | 418.4 | 466.4 | 1.115 | none |
| TP16/PP8, `KVP = 4` | 3 | 3 × 4,000 | 423.7 | 471.8 | 1.114 | none |
| TP16/PP8, `KVP = 4` | 3 | 3 × 16,000 | 442.8 | 490.9 | 1.109 | none |
| TP16/PP8, `KVP = 4` | 7 | 1 × 1,000 | 417.9 | 514.0 | 1.230 | none |
| TP16/PP8, `KVP = 4` | 7 | 1 × 4,000 | 421.4 | 517.5 | 1.228 | none |
| TP16/PP8, `KVP = 4` | 7 | 1 × 16,000 | 434.1 | 530.3 | 1.222 | none |
| TP16/PP16, `KVP = 4` | 1 | 1 × 1,000 | 235.3 | 268.9 | 1.143 | none |
| TP16/PP16, `KVP = 4` | 1 | 1 × 4,000 | 236.5 | 270.1 | 1.142 | none |
| TP16/PP16, `KVP = 4` | 1 | 1 × 16,000 | 241.0 | 274.6 | 1.139 | none |
| TP16/PP16, `KVP = 4` | 1 | 3 × 1,000 | 307.1 | 340.7 | 1.109 | none |
| TP16/PP16, `KVP = 4` | 1 | 3 × 4,000 | 310.8 | 344.4 | 1.108 | none |
| TP16/PP16, `KVP = 4` | 1 | 3 × 16,000 | 324.1 | 357.7 | 1.104 | none |
| TP16/PP16, `KVP = 4` | 1 | 7 × 1,000 | 451.1 | 484.7 | 1.074 | 0.580 to 0.602 |
| TP16/PP16, `KVP = 4` | 1 | 7 × 4,000 | 459.6 | 493.2 | 1.073 | 0.608 to 0.632 |
| TP16/PP16, `KVP = 4` | 1 | 7 × 16,000 | 490.8 | 524.4 | 1.068 | 0.709 to 0.743 |
| TP16/PP16, `KVP = 4` | 3 | 1 × 1,000 | 302.9 | 351.0 | 1.159 | none |
| TP16/PP16, `KVP = 4` | 3 | 1 × 4,000 | 304.7 | 352.8 | 1.158 | none |
| TP16/PP16, `KVP = 4` | 3 | 1 × 16,000 | 311.1 | 359.2 | 1.155 | none |
| TP16/PP16, `KVP = 4` | 3 | 3 × 1,000 | 441.1 | 489.2 | 1.109 | none |
| TP16/PP16, `KVP = 4` | 3 | 3 × 4,000 | 446.4 | 494.5 | 1.108 | none |
| TP16/PP16, `KVP = 4` | 3 | 3 × 16,000 | 465.6 | 513.7 | 1.103 | none |
| TP16/PP16, `KVP = 4` | 7 | 1 × 1,000 | 440.6 | 536.8 | 1.218 | none |
| TP16/PP16, `KVP = 4` | 7 | 1 × 4,000 | 444.1 | 540.3 | 1.217 | none |
| TP16/PP16, `KVP = 4` | 7 | 1 × 16,000 | 456.9 | 553.1 | 1.210 | none |

Six points match the AA pair with one shared acceptance, all at `K = 1` with seven background requests under `KVP = 4`, as expected.
On TP16/PP8 with `KVP = 4` the acceptance range is 0.505 to 0.521 with 1,000-token background requests, 0.533 to 0.551 with 4,000 and 0.635 to 0.662 with 16,000.
On the full rack with `KVP = 4` the ranges are 0.580 to 0.602, 0.608 to 0.632 and 0.709 to 0.743.
The AA request's 10K-to-100K increment stays at 33.6 microseconds at `K = 1`, approximately 48.1 at `K = 3` and 96.1 to 96.2 at `K = 7` under `KVP = 4`, independent of the background.
`K = 3` with three background requests comes closest among the non-matching points, with ratios of 1.103 to 1.115.
Without KV parallelism the smallest batched ratio is 1.249.

### Interpretation

Speculative decoding at batch size one does not explain the AA pair.
Shared acceptance cancels from the step-time ratio, and the AA request's own verification attention keeps that ratio above 1.15.

Speculative verification shared with background traffic does reproduce the AA pair.
Committing 1.50 to 1.75 tokens per step lets a 428 to 491-microsecond batched step give the published per-token speed, and the AA request's 33.6-microsecond increment is then small enough relative to the step.
It needs KV parallelism, seven background requests and an acceptance of 0.51 to 0.74.

Within the 16-row limit, the matching settings do not also reproduce the coding example.
The AA matches need `K = 1`, and `K = 1` cannot reach 10,996 TPS at any acceptance.
The 16-row limit restricts `K = 3` to three background requests, whose ratios of 1.103 to 1.115 lie just above the match limit.
Larger verification batches could therefore allow one draft length for both, so this incompatibility is a consequence of the row limit and not evidence against a shared draft length.

The acceptances and background loads remain assumptions fitted to published speeds.
The match shows that the cost model can produce the published pair under a plausible serving regime, and it does not identify that regime.
