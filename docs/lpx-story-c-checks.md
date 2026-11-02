# Two checks of one lookup drafter on every run

Record date: 2026-11-01.
Status: both checks ran on 2026-11-01; results below.
The expectations were committed as `d13f810` before either check ran.

## Question

Three explanations of NVIDIA's Gemma4-31B-LPX speeds remain on the closest simulated setup, TP32/PP8 with decode context parallelism (DCP) of 8 on 256 LP30 chips at attention efficiency 0.5.
The third, story C, uses one lookup drafter with a draft length `K` of 7 on every run and at most 6 to 7 µs of unmodeled time per step.
The [302-prompt code-completion record](lpx-cc16k-peak.md) found that it gives the 4,767 TPS SPEED-Bench coding median and reaches the 10,996 TPS Hot Chips example on one of 302 prompts.
Two parts of story C remain weak, and this record checks both with data already in the repository.

1. **The Artificial Analysis (AA) end.**
   At `K = 7`, no extra time places both AA values within 5% on the AA-like book answers of the [API-outputs record](lpx-api-outputs.md).
   Story C would need AA prompts that give the drafter less to copy than these answers on average.
   The check asks whether the book answers that copy least fit both AA values at the extra time story C allows.
2. **The coding maximum.**
   On all 89 H20 coding turns without thinking, story C gives a maximum of 1.885 to 1.95 times the median, against NVIDIA's 1.464, and a P80/median of 1.154 to 1.195, against 1.158.
   These ratios were computed after the 302-prompt results.
   A maximum depends strongly on sample size, and NVIDIA's request count is unknown, so the check asks how often a set of 80 turns gives NVIDIA's ratios.

## Method

Both checks reuse the step model of the earlier records: verification steps of 1 to 8 rows on the closest setup, from the published-conditions sweeps, with each lookup step priced at its own number of verified rows plus a constant extra time.

### Check 1: book answers that copy least

- **Data.** The lookup replays of Gemma 4 31B's answers to 16 books at 10K and 100K input tokens, with a summary question and a quotation question each, from `api-lookup-results.json` (thinking and answer) and `api-lookup-answer-alone-results.json` (answer alone, the stand-in for thinking off).
  Each replay row holds one answer's committed tokens and its count of steps at each verified-row count.
- **Settings.** Minimum n-gram 1 and 2 at `K = 7`, the two settings that reach the example.
- **Extra time.** From 0 µs to the upper end of the same setting's coding window in `gpu-cc302-peak.json`: 7 µs at minimum n-gram 1 and 6 µs at minimum n-gram 2, on a 0.5 µs grid.
- **Pricing.** Each answer is priced at the fixed AA contexts of the earlier AA analysis, and its AA speed counts Gemma tokens at the measured token ratio of its data set: 1.0055 for thinking and answer, 0.9950 for the answer alone.
- **Copy share.** Each book's copy share at a setting is its mean equivalent acceptance over its 10K and 100K answers.
- **Per-book fit.** A book fits when its 10K answer and its 100K answer both lie within 5% of the AA values at one extra time in the allowed range.
- **Low-copy subsets.** The books of each style are sorted by copy share, and for the 4, 8 and 16 books that copy least, the median 10K speed and the median 100K speed are tested for a shared extra time in the allowed range.

### Check 2: the coding spread in sets of 80

- **Data.** The lookup replay of the 89 H20 coding turns without thinking, `speed-coding-plain` in `gpu-cc302-lookup-results.json`.
- **Settings.** Story C: minimum n-gram 1 at `K = 7` with 7 µs and minimum n-gram 2 at `K = 7` with 6 µs.
  For comparison, story A's lookup setting: minimum n-gram 1 at `K = 3` with 13 and 21 µs.
- **Sampling.** 2,000 sets of 80 turns, drawn with replacement with seed 20261101.
  For each set, the P80/median and the maximum/median of the priced speeds.
- **Consistency.** A NVIDIA ratio is consistent with a setting when it lies between the 5th and 95th percentiles of the 2,000 sets.

## Expectations recorded before the checks

1. At minimum n-gram 1 and `K = 7`, no book fits in either data set, because almost every step then verifies 8 rows and the 100K step grows more than the 10K step.
2. At minimum n-gram 2 and `K = 7`, at most 4 of the 32 book-and-style pairs fit in each data set, and no quotation answer fits.
3. Every fitting book, if any, commits at most 1.085 tokens per step on its 10K answer.
4. The 8 summary answers that copy least have no shared extra time at minimum n-gram 2 in either data set.
5. At both story C settings, NVIDIA's maximum/median of 1.464 lies below the 5th percentile of the sets of 80.
6. At both story C settings, NVIDIA's P80/median of 1.158 lies between the 5th and 95th percentiles.
7. At story A's setting, minimum n-gram 1 and `K = 3` at 13 and 21 µs, NVIDIA's maximum/median lies between the 5th and 95th percentiles.

## How the results bear on story C

| Outcome | Meaning |
| --- | --- |
| Some low-copy books or subsets fit both AA values | Story C's AA end needs prompts like the least-copying book answers, which measured answers do produce |
| No book or subset fits | Story C needs AA prompts that copy less than any answer measured here, which only new generations or AA's prompts can test |
| Expectation 5 holds | Story C's coding maximum is too wide for NVIDIA's run at sets of 80 turns |
| Expectation 5 misses | NVIDIA's maximum is an ordinary result of sampling 80 turns under story C |

## Differences from the published runs

| Item | Published runs | These checks |
| --- | --- | --- |
| AA prompts | Not published | 16 well-known books with two questions written for this work; Gemma has likely seen the books in training |
| Model build | Private pre-release build | Public checkpoint on Parasail's FP8 deployment for the books, and in BF16 on one H20 for the coding turns |
| Thinking and sampling | Not stated | Thinking on for the books, with the answer alone as the stand-in for thinking off; thinking off for the coding turns; temperature 0 |
| Drafter and `K` | Not stated | Lookup replay at `K = 7`, and `K = 3` for comparison |
| Coding request count | Not stated | Sets of 80 turns drawn from 89 |
| Step times | Measured on LP30 | Simulated by TokenSim, with a constant extra time |

## Results

[`scripts/analyze_lpx_story_c.py`](../scripts/analyze_lpx_story_c.py) wrote [`story-c-checks.json`](../experiments/lpx/story-c-checks.json), which holds every book, subset and coding draw.

### Check 1: the book answers

Books whose 10K and 100K answers both fit at one extra time up to the coding window's upper end, out of 16:

| Data set | Minimum n-gram | Upper extra time | Summary books fitting | Quotation books fitting | Low-copy subsets with a window |
| --- | ---: | ---: | ---: | ---: | --- |
| Thinking and answer | 1 | 7 µs | 1 | 0 | none of 4, 8 or 16 |
| Thinking and answer | 2 | 6 µs | 0 | 0 | none |
| Answer alone | 1 | 7 µs | 0 | 0 | none |
| Answer alone | 2 | 6 µs | 0 | 0 | none |

AA errors with no extra time for the 4 summary answers that copy least, and the share of steps on which the drafter proposes tokens:

| Data set | Minimum n-gram | Tokens per step, 10K and 100K | AA 10K error | AA 100K error | Steps with a draft over all 16 books, 10K and 100K |
| --- | ---: | --- | --- | --- | --- |
| Thinking and answer | 1 | 1.23–1.26 | +8.6% to +11.5% | −6.1% to −7.5% | 78.9% and 87.7% |
| Thinking and answer | 2 | 1.15–1.17 | +14.9% to +16.0% | +0.6% to +1.9% | 32.7% and 43.7% |
| Answer alone | 1 | 1.14–1.17 | +2.5% to +4.6% | −12.0% to −13.4% | 77.2% and 86.9% |
| Answer alone | 2 | 1.08–1.10 | +9.8% to +11.6% | −2.5% to −4.5% | 30.8% and 42.4% |

The extra time slows both contexts, so it cannot repair an error with opposite signs at the two contexts.
At minimum n-gram 1, the drafter proposes on most steps, and an 8-row step costs 353.5 µs at 10K and 401.6 µs at 100K, so the 100K answers are too slow while the 10K answers are close or too fast.
At minimum n-gram 2, the drafter proposes less often, the steps stay closer to the one-row step, and the 10K answers are 10% to 16% too fast even at the lowest copy shares.
The drafter also proposes more often at 100K than at 10K, because a longer context holds more matches, while the tokens committed per step stay nearly the same.

The one book that fits, book 5 at minimum n-gram 1 with thinking and answer, ranks 12th of 16 by copy share.
It fits only at 7.0 µs, the edge of the allowed range, because its 100K answer commits 1.34 tokens per step against 1.22 at 10K.
Its fit comes from more copying at 100K than at 10K, the same per-workload acceptance gap that the earlier fits to all four published numbers required.

### Check 2: the coding spread in sets of 80

| Setting | Extra time | Median TPS, 89 turns | P80/median, 89 turns | P80/median, 5th–95th percentile | Maximum/median, 89 turns | Maximum/median, 5th–95th percentile | Sets at or below NVIDIA's maximum/median |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| Story C, minimum n-gram 1, `K = 7` | 7 µs | 4,531 | 1.195 | 1.111–1.264 | 1.950 | 1.425–1.999 | 15.6% |
| Story C, minimum n-gram 2, `K = 7` | 6 µs | 4,533 | 1.154 | 1.110–1.240 | 1.885 | 1.431–1.964 | 14.8% |
| Story A, minimum n-gram 1, `K = 3` | 13 µs | 4,686 | 1.155 | 1.103–1.194 | 1.601 | 1.294–1.627 | 41.0% |
| Story A, minimum n-gram 1, `K = 3` | 21 µs | 4,567 | 1.155 | 1.102–1.197 | 1.602 | 1.292–1.627 | 41.3% |

NVIDIA's P80/median of 1.158 and maximum/median of 1.464 lie between the 5th and 95th percentiles at all four settings.
Under story C, about one set of 80 turns in seven gives a maximum/median at or below NVIDIA's.
The 89-turn maximum is one request that copies heavily, and a set of 80 drawn with replacement leaves it out about 40% of the time.

### Comparison with the expectations

| Expectation | Outcome | Evidence |
| --- | --- | --- |
| 1: no book fits at minimum n-gram 1 | Missed | One summary book fits with thinking and answer, at 7.0 µs only |
| 2: at most 4 pairs fit at minimum n-gram 2, and no quotation answer | Held | No pair fits in either data set |
| 3: every fitting book commits at most 1.085 tokens per step at 10K | Missed | The fitting book commits 1.22 at 10K and 1.34 at 100K |
| 4: the 8 lowest-copy summary answers have no window at minimum n-gram 2 | Held | No window in either data set; their 10K medians are 10.4% and 15.4% too fast |
| 5: NVIDIA's maximum/median lies below the 5th percentile under story C | Missed | It lies at the 15th percentile at both story C settings |
| 6: NVIDIA's P80/median lies between the 5th and 95th percentiles under story C | Held | 1.158 lies inside 1.111–1.264 and 1.110–1.240 |
| 7: NVIDIA's maximum/median lies between the 5th and 95th percentiles under story A | Held | 1.464 lies inside 1.294–1.627 at both extra times |

## Interpretation

The coding spread does not count against story C.
At sets of 80 turns, both NVIDIA ratios are ordinary outcomes of the H20 coding answers under story C, as they are under story A.
The wide 89-turn maximum computed after the 302-prompt results came from one heavily copying turn and does not distinguish the stories.

The AA end of story C does not come from answers that copy little.
Among the measured book answers, those that copy least fit worse than average at minimum n-gram 2 and give opposite errors at the two contexts at minimum n-gram 1.
At minimum n-gram 2, answers that copy even less would draft less and approach plain decoding, which needs at least 15.8 µs on the answers alone, above the 6 µs story C allows.
At minimum n-gram 1, they would still draft on most steps, because single-token matches are nearly always available, and would commit fewer tokens, which slows both contexts while 100K is already too slow.
The exploratory bound of the comprehensive summary, which allowed a fit with 11% to 39% of steps drafting and at most about 1.085 tokens per step, assumed drafts of 7 tokens or none; the measured answers draft on 31% to 88% of steps with mixed lengths and do not reach that region.

At `K = 7` with at most 7 µs, the AA values fit only when the 100K answers commit more tokens per step than the 10K answers, as for book 5.
On average the book answers show no such gap: the paired acceptance difference between 100K and 10K is about zero, and Gemma's own MTP drafter shows the same.
Story C therefore needs AA's 100K prompts to lead to answers that copy more from their context than its 10K prompts do, by about 0.1 tokens per step.
This is the same condition that every earlier fit to the faster 100K speed needed, and the measured data do not show it on average.

Story C remains the only story that reaches the 10,996 TPS example with the coding median and an ordinary coding spread, while its AA end rests on a content difference between AA's two prompt sets that no measurement here supports.
Story A fits AA without that condition and keeps an ordinary coding spread, but misses the example.
A measured LP30 step time or host loop would still separate the stories, since they need at most 7 µs, 13 to 21 µs and 387 to 576 µs of extra time per step.

## Validation

The unit tests in [`tests/test_lpx_story_c.py`](../tests/test_lpx_story_c.py) cover the per-step pricing, the extra-time window, the ordering by copy share and the sampled percentiles.
