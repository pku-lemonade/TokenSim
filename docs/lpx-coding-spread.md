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
According to one of the authors of [SPEED-Bench](https://proceedings.mlr.press/v306/abramovich26a.html), its qualitative split holds 880 prompts across 11 categories, about 80 per category if the categories are balanced.
The script draws 2,000 random subsets of 40, 80 and 160 of the 500 requests with seed 20261027 and reports the 10th, 50th and 90th percentiles of each subset's P80/median and maximum/median ratios.

The ratios exclude measurement noise, differences in prompt length and the time to the first token, which widen any measured spread.
Plain decoding therefore gives a lower bound on the spread it can produce.
The CodeX problems approximate and do not reproduce SPEED-Bench's coding prompts.

## Expectations recorded before the measurement

1. Plain decoding gives a P80/median ratio of at most 1.02 and a median maximum/median ratio of at most 1.06 for subsets of 80.
2. Lookup drafting on the `combined` part with `n_min = 1` and `K` of 3 to 7 gives a P80/median ratio between 1.08 and 1.25.
3. The same settings give a median maximum/median ratio between 1.3 and 1.8 for subsets of 80.
4. At least one lookup setting gives a P80/median ratio within 0.05 of 1.158 and a median maximum/median ratio for subsets of 80 within 0.15 of 1.464.

## Results

The per-request replay took about two minutes, and the analysis about one.
[`experiments/lpx/lookup-coding-rows.json`](../experiments/lpx/lookup-coding-rows.json) holds each request's output tokens and proposal-length counts, and its aggregate values equal those of the lookup-coding record.
[`experiments/lpx/coding-spread-analysis.json`](../experiments/lpx/coding-spread-analysis.json) holds every setting, extra time and subset size.

```bash
python -I scripts/measure_lpx_lookup_acceptance.py \
  --gemma-tokenizer tokenizer.json \
  --data codex-combined=codex-500.json#combined \
  --data codex-answer-alone=codex-500.json#answer-alone \
  --per-row \
  --summary experiments/lpx/lookup-coding-rows.json
python scripts/analyze_lpx_coding_spread.py \
  --summary experiments/lpx/coding-spread-analysis.json
```

Ratios to the median without extra time, with `n_min = 1`; the subset columns give the 10th, 50th and 90th percentiles over subsets of 80 requests:

| Setting | Median TPS | P80/median, all 500 | P80/median, subsets of 80 | Maximum/median, subsets of 80 |
| --- | ---: | ---: | --- | --- |
| NVIDIA SPEED-Bench coding | 4,767 | — | 1.158 | 1.464 |
| Plain decoding | 3,727 | 1.000 | 1.000 to 1.001 | 1.001 |
| Lookup, `combined`, `K` = 1 | 4,991 | 1.036 | 1.028 to 1.044 | 1.080 to 1.131 |
| Lookup, `combined`, `K` = 3 | 5,601 | 1.066 | 1.053 to 1.084 | 1.159 to 1.270 |
| Lookup, `combined`, `K` = 7 | 5,310 | 1.088 | 1.067 to 1.105 | 1.218 to 1.376 |
| Lookup, `answer-alone`, `K` = 3 | 4,831 | 1.066 | 1.052 to 1.079 | 1.249 to 1.492 |
| Lookup, `answer-alone`, `K` = 5 | 4,668 | 1.076 | 1.059 to 1.093 | 1.305 to 1.561 |
| Lookup, `answer-alone`, `K` = 7 | 4,465 | 1.084 | 1.064 to 1.101 | 1.324 to 1.671 |

The median of the maximum/median ratio over subsets of 80 is 1.262 for `combined` at `K = 7` and 1.492 and 1.525 for `answer-alone` at `K` of 5 and 7.
Extra time of 20 or 50 µs per step lowers every median speed and changes the ratios by at most 0.025.
With `n_min = 2` the ratios change by at most 0.011 in P80/median and 0.039 in maximum/median.
Subsets of 40 requests lower the median maximum/median ratio by up to 0.14, subsets of 160 raise it by up to 0.07, and both leave the P80/median ratio within 0.003.

At `K = 7`, the per-request committed tokens per step range from 1.55 at the 10th percentile to 2.53 at the maximum for `combined` and from 1.32 to 2.49 for `answer-alone`.
The thinking makes `combined` requests more alike, because most of their tokens come from long thinking sections that accept similarly.

### Comparison with the expectations

1. Held: plain decoding gives a P80/median ratio of 1.000 and a maximum/median ratio of 1.001.
2. Partly held: `combined` with `n_min = 1` gives 1.082 and 1.088 at `K` of 5 and 7, and 1.066 at `K = 3`, below the expected range.
3. Missed: the same settings give median maximum/median ratios of 1.197 to 1.262, below the expected 1.3 to 1.8.
4. Missed: `answer-alone` at `K` of 5 and 7 reproduces the maximum/median ratio, but no setting gives a P80/median ratio above 1.105 in 90% of the subsets, against NVIDIA's 1.158.

## Interpretation

Plain decoding cannot produce NVIDIA's coding spread.
Its step depends on the context alone, so its speeds differ by 0.1% across requests, while NVIDIA's P80 is 15.8% and its maximum 46.4% above the median.
Measurement noise or prompt-length differences of that size are unlikely on a dedicated system, because 100K tokens of context add only about 5% to the step.
The SPEED-Bench coding run therefore most likely used speculative decoding or another mechanism whose speed depends on the content.

Lookup drafting on the CodeX answers without thinking reproduces the maximum/median ratio, and with thinking it reaches about 1.26.
Its P80/median ratio of 1.04 to 1.09 is about half of NVIDIA's 1.158, so NVIDIA's requests differ more from each other in the middle of the distribution.
SPEED-Bench selects its prompts for semantic diversity, while the CodeX prompts are all competitive-programming tasks, and more varied prompts would make acceptance vary more across requests.
A drafter whose acceptance varies more between requests, such as a trained draft model, could also widen it.
The data does not separate these causes.

The result supports drafting in the build that produced the coding values.
It strengthens the lookup explanation of the [lookup-coding record](lpx-lookup-coding.md) when the AA deployment used the same build, which the AA footnotes' description of an "updated" deployment leaves open.
