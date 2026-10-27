# Coding speed spread under plain decoding and lookup drafting

Record date: 2026-10-27.
NVIDIA reports three statistics of one SPEED-Bench coding run of Gemma 4 31B on LPX: a median of 4,767 output tokens per second (TPS), an 80th percentile (P80) of 5,520 TPS and a maximum of 6,981 TPS, as the [evidence audit](lpx-parameter-evidence-audit.md) lists.
The P80 is 1.158 times the median and the maximum 1.464 times.
Without speculative decoding, the decode step depends only on the context length, so requests of similar length run at nearly the same speed.
With a drafter, each request's speed depends on how many draft tokens its output accepts, so speeds spread across requests.
This record tests which of the two explanations of the [lookup-coding record](lpx-lookup-coding.md), plain decoding or lookup drafting, produces NVIDIA's spread.

## Method

[`scripts/measure_lpx_lookup_acceptance.py`](../scripts/measure_lpx_lookup_acceptance.py) with `--per-row` records each request's output tokens and the number of steps with each proposal length, for the full prompt.
The replay uses the 500 CodeX rows of the lookup-coding record, with the `combined` part, which includes the thinking, and the `answer-alone` part, which approximates generation without thinking.

[`scripts/analyze_lpx_coding_spread.py`](../scripts/analyze_lpx_coding_spread.py) prices each request on TP32/PP8 with decode context parallelism (DCP) of 8 at attention matrix efficiency 0.50.
A step that verifies `n` rows costs the `n`-row forward of the [article-condition speculation record](lpx-article-speculative.md).
The step is linear in context between the AA 10K and 100K captures, and each request is priced at an input of 16,000 tokens plus half its output.
A constant extra time of 0, 20 or 50 µs per step covers the unmodeled time of the plain-decoding and lookup explanations.
A request's speed is its output tokens divided by its decode time.

NVIDIA does not state its number of coding requests.
SPEED-Bench's qualitative split holds 880 prompts across 11 categories, about 80 per category if the categories are balanced.
The script draws 2,000 random subsets of 40, 80 and 160 of the 500 requests with seed 20261027 and reports the 10th, 50th and 90th percentiles of each subset's P80/median and maximum/median ratios.

The ratios exclude measurement noise, differences in prompt length and the time to the first token, which widen any measured spread.
Plain decoding therefore gives a lower bound on the spread it can produce.
The CodeX problems approximate and do not reproduce SPEED-Bench's coding prompts.

## Expectations recorded before the measurement

1. Plain decoding gives a P80/median ratio of at most 1.02 and a median maximum/median ratio of at most 1.06 for subsets of 80.
2. Lookup drafting on the `combined` part with `n_min = 1` and `K` of 3 to 7 gives a P80/median ratio between 1.08 and 1.25.
3. The same settings give a median maximum/median ratio between 1.3 and 1.8 for subsets of 80.
4. At least one lookup setting gives a P80/median ratio within 0.05 of 1.158 and a median maximum/median ratio for subsets of 80 within 0.15 of 1.464.
