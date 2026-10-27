# Prompt-lookup acceptance on Gemma 4 31B outputs

Record date: 2026-10-27.
The [article-condition speculation record](lpx-article-speculative.md) found that lookup drafting on TP32/PP8 with decode context parallelism (DCP) of 8 reproduces both Artificial Analysis (AA) values, including the higher speed at 100K, if acceptance is higher at 100K than at 10K.
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

Pending.
