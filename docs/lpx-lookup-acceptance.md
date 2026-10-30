# Prompt-lookup acceptance on Gemma 4 31B outputs

Record date: 2026-10-27.
The [article-condition speculation record](lpx-article-speculative.md) found that lookup drafting on TP32/PP8 with decode context parallelism (DCP) of 8 places both Artificial Analysis (AA) values within 5%, including the higher speed at 100K, if acceptance is higher at 100K than at 10K.
At 1.036 Gemma tokens per `o200k_base` token, seven draft tokens need a per-position acceptance of about 0.19 at 10K and 0.30 at 100K, and five draft tokens need 0.13 and 0.23.
This record measures lookup acceptance on Gemma 4 31B's own outputs.

## Method

A prompt-lookup drafter proposes the tokens that followed the most recent earlier occurrence of the current suffix in the prompt and the text generated so far.
The outputs in the WritingBench and AIME datasets of the [token-ratio record](lpx-token-ratio.md) were generated at temperature 0, so the target model's verification accepts a draft token exactly when it equals the token Gemma produced.
Replaying a lookup drafter over the recorded token sequence therefore measures its acceptance without running the model.

[`scripts/measure_lpx_lookup_acceptance.py`](../scripts/measure_lpx_lookup_acceptance.py) tokenizes each prompt and raw output with Gemma 4's tokenizer and replays the drafter step by step:

- The drafter matches the longest suffix of 3 down to `n_min` tokens, with `n_min` of 1 or 2, and proposes up to `K` following tokens that are already known.
- A step accepts the longest prefix of the proposal that equals the recorded continuation and commits that many tokens plus one target token.
- A step without a match proposes nothing and commits one token.
- Chat-template tokens are omitted.

It reports the mean tokens committed per step `tau`, the share of steps with a proposal, and the per-position acceptance `alpha` that gives the same `tau` with `K` proposals at every step, which is the quantity the speculative records fit.
It repeats the replay with the full prompt, with the first half of the prompt and without the prompt, to show how acceptance depends on the text available for copying.
After the first run, the script was extended to record the share of steps with each proposal length; a rerun reproduced every earlier value.

| Data | Prompts | Outputs |
| --- | --- | --- |
| WritingBench, 100 rows | Supplied document and writing request, median 798 tokens, at most 2,605 | Thinking and final answer, median 2,476 tokens |
| AIME, 100 rows | Short mathematics problem, about 100 tokens | Thinking and final answer |

## Expectations recorded before the measurement

1. On WritingBench with the full prompt, `n_min = 1` and `K = 7`, the equivalent per-position acceptance lies between 0.15 and 0.35.
2. Removing the prompt lowers the WritingBench acceptance at `K = 7` by at least 0.03.
3. AIME gives a lower acceptance than WritingBench at the same settings.
4. With `n_min = 1`, at least 70% of steps make a proposal.

## Results

The replay took 15 seconds.
[`experiments/lpx/lookup-acceptance-results.json`](../experiments/lpx/lookup-acceptance-results.json) holds every setting, the SHA-256 of both row files and the proposal-length shares.

```bash
python -I scripts/measure_lpx_lookup_acceptance.py \
  --gemma-tokenizer tokenizer.json \
  --data writingbench=g4-2.json \
  --data aime=g4-3.json \
  --summary experiments/lpx/lookup-acceptance-results.json
```

The WritingBench rows hold 84,531 prompt tokens and 279,607 output tokens, and the AIME rows 10,133 and 546,975.

| Data | Prompt | `n_min` | `K` = 1 | `K` = 3 | `K` = 5 | `K` = 7 | Steps with a proposal at `K` = 7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WritingBench | Full | 1 | 0.179 | 0.209 | 0.222 | 0.225 | 68% |
| WritingBench | Half | 1 | 0.171 | 0.200 | 0.213 | 0.217 | 66% |
| WritingBench | None | 1 | 0.158 | 0.187 | 0.199 | 0.203 | 62% |
| WritingBench | Full | 2 | 0.121 | 0.153 | 0.164 | 0.167 | 25% |
| AIME | Full | 1 | 0.513 | 0.541 | 0.557 | 0.571 | 90% |
| AIME | None | 1 | 0.505 | 0.534 | 0.551 | 0.565 | 89% |
| AIME | Full | 2 | 0.456 | 0.495 | 0.516 | 0.532 | 68% |

Values are the equivalent per-position acceptance.
At `K = 7` with the full prompt and `n_min = 1`, WritingBench commits 1.291 tokens per step and AIME 2.305.
On WritingBench, 31.8% of steps propose nothing and 65.0% propose seven tokens.

### Comparison with the expectations

1. Held: WritingBench at `K = 7` gave 0.225.
2. Missed: removing the prompt lowered the acceptance by 0.022, less than the expected 0.03, although in the expected direction.
3. Missed: AIME gave 0.571, more than twice the WritingBench value.
   Gemma's mathematics reasoning repeats its own expressions, so the drafter copies from the output, and the prompt contributes only 0.006.
4. Missed on WritingBench, where 62% to 71% of steps propose with `n_min = 1`, and held on AIME at 89% to 94%.

## Effect on the AA match

Each lookup step was priced on TP32/PP8 with DCP 8 at attention matrix efficiency 0.50 with the verification forward of its actual proposal length, taken from the [article-condition speculation record](lpx-article-speculative.md); seven-row verification is interpolated between six and eight rows.
At 1.036 Gemma tokens per `o200k_base` token, the WritingBench acceptance applied to both AA workloads gives:

| `n_min` | `K` | Tokens per step | 10K step | 100K step | AA 10K TPS | AA 100K TPS |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1 | 1.179 | 274 µs | 290 µs | 4,148 (+22.7%) | 3,919 (+14.2%) |
| 1 | 3 | 1.261 | 291 µs | 312 µs | 4,180 (+23.6%) | 3,899 (+13.6%) |
| 1 | 5 | 1.285 | 308 µs | 337 µs | 4,023 (+19.0%) | 3,677 (+7.2%) |
| 1 | 7 | 1.291 | 325 µs | 362 µs | 3,836 (+13.4%) | 3,445 (+0.4%) |
| 2 | 7 | 1.201 | 289 µs | 311 µs | 4,018 (+18.8%) | 3,727 (+8.6%) |

With the same acceptance at both contexts, lookup drafting makes the endpoint faster than AA measured, and more so at 10K.
It also widens the gap between the contexts, because each extra verified row costs more attention at 100K.
No constant extra time per step fixes the `K = 7` row: the 10K value needs at least 25.9 µs, and the 100K value allows at most 20.3 µs.
The other rows match only with 34 to 61 µs of extra time per step, two to four times the 14 to 25 µs that plain decoding needs.

The derived values in this record are reproduced from the committed results by [`scripts/analyze_lpx_published_conditions.py`](../scripts/analyze_lpx_published_conditions.py), which writes [`experiments/lpx/published-conditions-analysis.json`](../experiments/lpx/published-conditions-analysis.json).

## Interpretation

Lookup drafting on Gemma's writing outputs accepts about 0.2 per position, inside the published range, and acceptance rises with the text available for copying: 0.203 without a prompt, 0.217 with half and 0.225 with the full prompt of about 800 tokens.
A lookup drafter can reproduce AA's higher 100K speed only if its acceptance is about 0.1 higher at 100K than at 10K, for example 0.19 and 0.30 at `K = 7`.
The measured gain of about 0.008 per doubling of the prompt is too small to supply that difference if it continued to 100K tokens, and a 10K acceptance of 0.19 would be below the 0.225 measured on prompts of only 800 tokens.
AA's prompts and answers may copy differently from WritingBench, so the measurement weakens the lookup explanation and does not exclude it.
Gemma's mathematics output gives 0.57, which shows that the acceptance depends strongly on the content of the answer.
The [lookup-context record](lpx-lookup-context.md) tests the context effect on Gemma 4 31B agent turns with up to 66K context tokens and finds less than 0.005 gain per doubling of the lookback beyond 16K tokens.
