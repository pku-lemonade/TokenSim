# Is AA's higher 100K speed within measurement noise?

Record date: 2026-10-28.
Artificial Analysis (AA) reports 3,431 output tokens per second (TPS) at 100K input tokens and 3,382 TPS at 10K for the Gemma4-31B-LPX deployment, each the median of 50 sequential requests.
No setup tested in this study makes the 100K speed exceed the 10K speed, because every modeled cost either grows with context or stays constant.
The [step-cost audit](lpx-step-cost-audit.md) and the [fixed-cost evidence record](lpx-fixed-cost-evidence.md) add no cost that changes this.
This record checks whether a 1.45% difference between two 50-request medians lies within ordinary measurement variation.

## Sampling variation of a 50-request median

For normally distributed per-request speeds with coefficient of variation `c`, the standard error of a median of `n` requests is about `sqrt(pi / 2) * c / sqrt(n)`, which is `0.177 * c` for `n = 50`.
The difference between two independent medians has a standard error of `0.251 * c`.

| Per-request coefficient of variation | Standard error of one median | Standard error of the difference | 1.45% in standard errors |
| ---: | ---: | ---: | ---: |
| 1% | 0.18% | 0.25% | 5.8 |
| 2% | 0.35% | 0.50% | 2.9 |
| 3% | 0.53% | 0.75% | 1.9 |
| 5% | 0.89% | 1.25% | 1.2 |

A per-request variation of about 2.9% or more makes the 1.45% difference smaller than two standard errors, and 5.8% or more makes it smaller than one.
At batch size 1 without speculative decoding the simulated per-request spread is about 0.1%, as the [coding-spread record](lpx-coding-spread.md) shows, so a larger spread would have to come from the measurement path, such as network delays between streamed chunks, from content-dependent decoding, or from changes of the endpoint between the two runs.
AA does not publish the spread of the 50 requests.

## Method

The check reads AA's public pages for Gemma 4 31B and for its providers with WebFetch, looking for:

- Named provider speeds at 10K and 100K input tokens, to see whether any endpoint measures faster at 100K.
- Speeds over time for one endpoint, to estimate how much a median moves between measurements.
- AA's methodology page, for how output speed is timed and how requests vary.

The public providers are measured differently from the LPX endpoint: 72-hour and 14-day medians from shared production endpoints instead of 50 sequential requests to a private deployment.
Their variation is therefore a rough reference.

## Expectations recorded before the check

1. AA's Gemma 4 31B pages can be read and name the providers with their 10K speeds.
2. 100K speeds are available for at most half of the named providers.
3. Every named provider with both values is more than 5% slower at 100K than at 10K.
4. The speed-over-time charts cannot be read as numbers through WebFetch, so the run-to-run variation of a median cannot be measured from AA's pages.
5. The check ends without a measured spread for the LPX endpoint and leaves the per-request variation of about 2.9% above as the condition for the difference to be ordinary noise.

## Findings

The check ran on 2026-10-28.
WebFetch read AA's [Gemma 4 31B provider page](https://artificialanalysis.ai/models/gemma-4-31b/providers) and its [methodology page](https://artificialanalysis.ai/methodology/performance-benchmarking).
The provider page embeds each endpoint's median, quartiles and 5th and 95th percentiles of output speed for the selected workload, together with the medians of the 1K, 10K and 100K workloads.
[`scripts/extract_aa_provider_speeds.py`](../scripts/extract_aa_provider_speeds.py) extracts them from the saved pages of the 10K and 100K workloads into [`experiments/lpx/aa-provider-speeds-20261028.json`](../experiments/lpx/aa-provider-speeds-20261028.json).
The saved pages have SHA-256 `ac7f331ba0e20d509efd51c7725ee43bccf887d28db7bf5a403727af2c3885dc` (10K) and `eb7e9d3ea20a5e2572c7f5d31b2f7e27791400acf5e50607af8d42aa5760c38b` (100K); AA updates the page continuously, so a later fetch gives other values.

```bash
python -I scripts/extract_aa_provider_speeds.py \
  --page-10k p10k.html \
  --page-100k p100k.html \
  --source "artificialanalysis.ai Gemma 4 31B provider pages, fetched 2026-10-28" \
  --summary experiments/lpx/aa-provider-speeds-20261028.json
```

### Methodology

The methodology page states that output speed counts tokens per second after the first token, and that for reasoning models that do not expose all reasoning tokens it uses the last 80% of answer chunks.
It tests the 1K and 10K workloads about every 3 hours and the 100K workload once per week, and it reports the median over 72 hours and 14 days.
Every test run uses a unique prompt generated at the time of the test.
The page notes that speculative decoding makes output speed vary with the type of output.
It does not state how many requests one test sends or discuss variance.

### Public endpoints

All values are output tokens per second on shared public endpoints on 2026-10-28.
The spread columns give the interquartile range and the 5th-to-95th-percentile range of the 10K distribution, each divided by the median.

| Provider | 1K | 10K | 100K | 100K / 10K | 10K interquartile range | 10K 5th to 95th percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SambaNova | 203.8 | 542.7 | 287.9 | 0.531 | 20.9% | 38.3% |
| Modular (NVFP4) | 289.3 | 283.4 | 239.6 | 0.846 | 21.7% | 40.7% |
| FriendliAI | 120.3 | 112.6 | 100.1 | 0.889 | 36.4% | 144.1% |
| Parasail | 91.1 | 95.7 | 68.8 | 0.718 | 24.4% | 83.5% |
| CoreWeave | 51.0 | 66.0 | 35.7 | 0.541 | 56.6% | 91.1% |
| Crusoe | 54.6 | 62.0 | 18.7 | 0.301 | 44.1% | 117.6% |
| DigitalOcean | 48.1 | 43.6 | 30.8 | 0.708 | 14.1% | 30.9% |
| Google (AI Studio) | 35.2 | 35.5 | — | — | 5.2% | 17.4% |
| Novita | 25.4 | 29.7 | 10.7 | 0.361 | 53.9% | 356.9% |
| DeepInfra | 27.4 | 24.6 | 10.9 | 0.442 | 31.1% | 88.9% |
| SiliconFlow (FP8) | 25.4 | 9.1 | — | — | 17.1% | 95.0% |

Every endpoint with a 100K value is 11% to 70% slower at 100K than at 10K.
For normally distributed speeds the interquartile range is 1.35 standard deviations, so the 10K spreads correspond to coefficients of variation of about 4% to 42%.
Six of the 11 endpoints are faster at 10K than at 1K over the same 72 hours, by 0.9% for Google and by 5% to 166% for the other five, although a longer prompt cannot shorten a decode step.

### Expectations

1. Held: the provider page names 11 providers with their 10K speeds.
2. Missed: 9 of the 11 providers have 100K speeds.
3. Held: each of the 9 is more than 5% slower at 100K.
4. Missed in part: the speed-over-time charts require a paid tier, but the page embeds the quartiles and 5th and 95th percentiles of each endpoint's speed, which describe the spread of its measurements.
5. Held: no spread is available for the LPX endpoint, and the 2.9% condition remains the test.

## Interpretation

On AA's public endpoints the spread of measured speeds, about 4% to 42% as a coefficient of variation, is far above the 2.9% at which a 1.45% difference between two 50-request medians falls within two standard errors.
AA's per-workload medians also order against decode physics on 6 of 11 endpoints, so AA's workloads differ in more than context length, for example in load at the time of each test, in the prompts and outputs that each test generates, and in content-dependent speculative decoding, which AA's methodology names.
These endpoints are shared and loaded, while the LPX endpoint was private and served one request at a time, so its spread is likely smaller and the public spreads do not measure it.
The LPX measurement keeps the parts of that variation that do not depend on load: a unique prompt and output for every request, streaming through Google Cloud Private Service Connect, the reasoning-chunk rule and any content-dependent decoding.

The data therefore do not show that the 1.45% is noise, and they do not support reading it as a context effect.
A per-request spread of about 3% or more on the LPX endpoint would make it ordinary, and only AA's per-request LPX data can settle that.
Until then, the higher 100K speed does not require a mechanism that makes long contexts faster.
