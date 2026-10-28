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

Pending.
