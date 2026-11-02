# How much do the three explanations depend on the assumed LP30 attention costs?

Record date: 2026-11-02.
Status: simulated and analyzed on 2026-11-02; results below.
The expectations were committed as `a278b67` before the simulation ran.

## Question

The [story C checks](lpx-story-c-checks.md) found that one lookup drafter with a draft length `K` of 7 on every run does not fit both Artificial Analysis (AA) values on the AA-like book answers, even for the answers that copy least.
The failure comes from the cost of verifying several rows at long context: on the closest setup, TP32/PP8 with decode context parallelism (DCP) of 8 on 256 LP30 chips at attention matrix efficiency 0.5, the 100K step is 14.2 µs longer than the 10K step with one row and 48.1 µs longer with 8 rows.
Drafting at `K = 7` therefore makes 100K slower relative to 10K, while AA measured 100K slightly faster.

None of the LP30 attention costs behind that growth is measured.
The cost formula of the LP30 device model attributes it as follows, per global attention layer and chip, where each chip holds 8 query heads and one KV head over one eighth of the context:

| Rows | Context | Matrix unit | Vector unit | Shuffle unit | Operator time |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 10,750 | 0.07 µs | 0.17 µs | 0.01 µs | 0.18 µs |
| 1 | 101,000 | 0.67 µs | 1.59 µs | 0.10 µs | 1.60 µs |
| 8 | 10,750 | 0.57 µs | 0.39 µs | 0.09 µs | 0.65 µs |
| 8 | 101,000 | 5.38 µs | 3.64 µs | 0.81 µs | 5.47 µs |

With one row, the vector unit's FP8-to-BF16 conversion of the cached keys and values sets the time, and its cost does not depend on the number of rows.
With 8 rows, the matrix unit at efficiency 0.5 sets the time, and its work grows with rows times context, as do the exponentials, the softmax arithmetic and the score movement.
Over the 10 global layers these give increments of 14.2 µs and 48.2 µs, matching the simulated 14.2 µs and 48.1 µs.

This record asks whether story C's failure at AA, and the windows of the other explanations, survive plausible alternative LP30 attention costs, or whether they depend on one assumed rate.

### What AA publishes about its prompts

AA's [performance methodology](https://artificialanalysis.ai/methodology/performance-benchmarking) describes the public benchmark:

- "Every test run uses a unique prompt that we generate at the time of the test and run on all endpoints we cover."
- "Prompts combine a range of long-form input content (e.g., articles)", with tasks including "explanation/summarization, question and answer generation, comparative analysis", translation and visual artifact generation.
- The 10K workload has "at least 1,500 answer tokens" and the 100K workload "at least 2,000 answer tokens".
- The 1K and 10K workloads run 8 times per day and the 100K workload once per week, with medians over 72 hours and 14 days.
- "Techniques like speculative decoding mean that output speeds vary with the type of output."

The page does not name the article sources, the generation method or whether the 100K prompts use the same task mix as the 10K prompts.
The LPX values are medians of 50 sequential requests to a private deployment, so their schedule differs from the public one, and the page does not say whether the two sets of 50 shared a task mix.
Story C needs AA's 100K answers to copy about 0.1 tokens per step more than its 10K answers; the published methodology neither supports nor excludes such a difference.

## Method

[`scripts/run_lpx_attention_sensitivity.py`](../scripts/run_lpx_attention_sensitivity.py) captures native verification steps of 1 to 8 rows on TP32/PP8 with DCP 8, with the conditions of the [published-conditions speculative sweep](lpx-article-speculative.md): one request, FP8 weights and KV cache, the article's link bandwidth and no background requests.
Each step is captured at the AA 10K context of 10,750 tokens, the AA 100K context of 101,000 tokens and the 16K coding context of 17,000 tokens.
The 7-row step is captured directly; the earlier sweep interpolated it from the 6-row and 8-row steps.

| Variant | Attention matrix efficiency | Exponential rate | Vector efficiency | Shuffle efficiency | Kind |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.5 | 1 × 10¹² per s | 0.5 | 0.8 | Simulated; the study values |
| Matrix | 1.0 | 1 × 10¹² per s | 0.5 | 0.8 | Simulated |
| Exponential | 0.5 | 4 × 10¹² per s | 0.5 | 0.8 | Simulated |
| Fast units | 1.0 | 4 × 10¹² per s | 1.0 | 1.0 | Simulated |
| Row-independent | — | — | — | — | Analytical bound: baseline steps with every row count's 100K and coding increments set to the one-row increments |

The vector efficiency, exponential rate and shuffle efficiency apply to every operator on those units, so the fast-units variant also shortens normalizations, elementwise work and data movement outside attention.
The row-independent bound has no hardware mechanism behind it; it is the most favorable case for drafting at long context.

[`scripts/analyze_lpx_attention_sensitivity.py`](../scripts/analyze_lpx_attention_sensitivity.py) recomputes the explanations' tests under each variant with the methods of the earlier records:

- **Plain decoding.** The extra time per step that places both AA values within 5% at the measured token ratios of the book answers, 1.0055 for thinking and answer and 0.9950 for the answer alone.
- **Lookup settings.** For minimum n-gram 1 and 2 and `K` of 1, 3, 5 and 7, the coding window from the H20 coding answers without thinking, the fastest of the 302 code-completion requests at the window's upper end, and the number of book-and-style pairs that fit both AA values at an extra time inside the coding window, in both book data sets.
- **Low-copy subsets.** The medians of the 4, 8 and 16 summary and quotation answers that copy least, tested for a shared extra time inside the coding window.
- **Joint condition.** A lookup setting satisfies story C under a variant when one extra time lies in its coding window, gives a code-completion request at or above 10,996 TPS, and places the 16-book median of one question style within 5% of both AA values in one data set.

## Expectations recorded before the simulation

1. The baseline variant reproduces the committed verification steps of 1 to 6 and 8 rows within 0.1 µs at all three contexts, and the captured 7-row step lies within 1 µs of the earlier interpolation.
2. The 100K-minus-10K increment of the 8-row step is 32.6 ± 2 µs under the matrix variant, within 1 µs of the baseline under the exponential variant, and 24 ± 3 µs under the fast-units variant; the one-row increment stays within 0.5 µs of 14.2 µs under the matrix and exponential variants and falls to 6.4 ± 1 µs under the fast-units variant.
3. The fast-units variant shortens the one-row 10K step by 5 to 20 µs, which raises the lower end of the plain-decoding AA window by the same amount.
4. Under no simulated variant does any lookup setting satisfy the joint condition in either data set.
5. Under every simulated variant, at most 2 of the 32 book-and-style pairs fit at each `K = 7` setting in each data set.
6. Under the row-independent bound, at minimum n-gram 1 and `K = 7` with the answer alone, at least one of the 4-book or 8-book low-copy summary subsets fits both AA values inside the coding window.
7. Under the row-independent bound, no setting at minimum n-gram 2 has any book-and-style pair that fits.

## How the results bear on the explanations

| Outcome | Meaning |
| --- | --- |
| Expectation 4 holds and expectation 6 holds | Story C's failure at AA holds under every simulated LP30 attention cost, and only attention whose cost does not grow with rows at long context would let low-copy answers fit |
| Expectation 4 holds and expectation 6 misses | Story C fails at AA even in the most favorable bound; its AA end needs AA's 100K answers to copy more than its 10K answers |
| Expectation 4 misses | Story C fits AA under a plausible LP30 attention cost, so its failure depends on the assumed rates, and a measured multi-row attention cost on LP30 becomes the deciding number |

## Differences from the published runs

| Item | Published runs | This record |
| --- | --- | --- |
| LP30 attention costs | Measured on LP30, not published | Four assumed sets of unit rates and one analytical bound |
| AA prompts | Generated per test from long-form content, not published | 16 books with two questions, answered by a public Gemma 4 31B checkpoint on a third-party FP8 deployment |
| Coding answers | NVIDIA's build, not published | Public checkpoint in BF16 on one H20, thinking off, temperature 0 |
| Drafter and `K` | Not stated | Lookup replay at minimum n-gram 1 and 2 and `K` of 1, 3, 5 and 7 |
| Step times | Measured on LP30 | Simulated by TokenSim, with a constant extra time per step |

## Results

[`run_lpx_attention_sensitivity.py`](../scripts/run_lpx_attention_sensitivity.py) wrote the captured steps to [`attention-sensitivity-steps.json`](../experiments/lpx/attention-sensitivity-steps.json), and [`analyze_lpx_attention_sensitivity.py`](../scripts/analyze_lpx_attention_sensitivity.py) wrote every window, book and subset to [`attention-sensitivity.json`](../experiments/lpx/attention-sensitivity.json).
The 32 captures took about 3.5 minutes.

### Verification steps

| Variant | 1 row, 10K | 1 row, 100K | 8 rows, 10K | 8 rows, 100K | 1-row increment | 8-row increment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 267.3 µs | 281.5 µs | 353.5 µs | 401.6 µs | 14.2 µs | 48.2 µs |
| Matrix | 267.2 µs | 281.4 µs | 351.3 µs | 383.8 µs | 14.2 µs | 32.6 µs |
| Exponential | 266.9 µs | 279.8 µs | 352.0 µs | 400.2 µs | 12.8 µs | 48.1 µs |
| Fast units | 264.3 µs | 270.8 µs | 344.5 µs | 368.6 µs | 6.4 µs | 24.1 µs |
| Row-independent bound | 267.3 µs | 281.5 µs | 353.5 µs | 367.7 µs | 14.2 µs | 14.2 µs |

The increments come from the unrounded steps; the earlier records quoted the baseline 8-row increment as 48.1 µs from the rounded steps.
The baseline captures equal the committed steps of 1 to 6 and 8 rows to the printed precision, and the captured 7-row step lies within 0.002 µs of the interpolation.

The faster exponential rate shortens the one-row increment by 1.4 µs, because the vector unit runs the exponentials as well as the conversions in the LP30 model, and the one-row operator is vector-unit bound.
At 8 rows it changes almost nothing, because the matrix unit sets the time.
The fast-units variant shortens the one-row 10K step by only 3.0 µs, so the operators outside attention that it also speeds up take little of the step.

### Plain decoding and story A

Extra time per step that places both AA values within 5% with plain decoding, and its overlap with the coding window of minimum n-gram 1 at `K = 3`, story A's setting:

| Variant | Plain AA, thinking and answer | Plain AA, answer alone | Story A, thinking and answer | Story A, answer alone |
| --- | --- | --- | --- | --- |
| Baseline | 12.8–23.7 µs | 15.8–26.9 µs | 12.8–23.5 µs | 15.8–23.5 µs |
| Matrix | 12.9–23.8 µs | 15.8–27.0 µs | 12.9–23.8 µs | 15.8–24.0 µs |
| Exponential | 13.1–25.3 µs | 16.1–28.6 µs | 13.1–24.0 µs | 16.1–24.0 µs |
| Fast units | 15.7–34.4 µs | 18.7–37.6 µs | 15.7–28.5 µs | 18.7–28.5 µs |
| Row-independent bound | 12.8–23.7 µs | 15.8–26.9 µs | 12.8–23.7 µs | 15.8–24.0 µs |

Story A keeps a window of at least 7.7 µs under every variant.
The fast-units variant moves the lower end of the plain window up by 2.9 µs, the amount it shortens the one-row 10K step.

### Lookup settings at `K = 7`

The coding window comes from the H20 coding answers without thinking, and the fastest request is the fastest of the 302 code-completion requests at the window's upper end.
Pairs count the book-and-style pairs, out of 32, that fit both AA values at one extra time inside the coding window.

| Variant | Minimum n-gram | Coding window | Fastest request | Pairs, thinking and answer | Pairs, answer alone | Joint condition |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Baseline | 1 | 0–7.0 µs | 11,606 TPS | 1 | 0 | none |
| Baseline | 2 | 0–6.0 µs | 11,735 TPS | 0 | 0 | none |
| Matrix | 1 | 0–9.0 µs | 11,633 TPS | 1 | 0 | none |
| Matrix | 2 | 0–7.5 µs | 11,762 TPS | 0 | 0 | none |
| Exponential | 1 | 0–8.0 µs | 11,615 TPS | 1 | 0 | none |
| Exponential | 2 | 0–7.0 µs | 11,739 TPS | 0 | 0 | none |
| Fast units | 1 | 0–14.5 µs | 11,677 TPS | 1 | 6 | answer alone, 7.0–10.0 µs |
| Fast units | 2 | 0–12.0 µs | 11,817 TPS | 0 | 0 | none |
| Row-independent bound | 1 | 0–8.0 µs | 11,639 TPS | 0 | 9 | answer alone, 0–8.0 µs |
| Row-independent bound | 2 | 0–7.0 µs | 11,755 TPS | 0 | 0 | none |

Every fitting pair in the answer-alone data set is a summary answer; no quotation answer fits under any variant.
At `K` of 1, 3 and 5, no setting meets the joint condition under any variant, because none reaches the 10,996 TPS example; the fastest request at `K = 5` is 9,983 to 10,022 TPS.
At minimum n-gram 1 and `K = 5`, 2 to 8 answer-alone summary pairs fit inside the coding window, depending on the variant.

Under the fast-units variant, the joint condition holds through the 16-book median of the answer-alone summary answers:

| Extra time | Median 10K | Error at 10K | Median 100K | Error at 100K | Fastest of 302 | Requests at or above 10,996 TPS | Coding median |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7.0 µs | 3,549 TPS | +4.9% | 3,287 TPS | −4.2% | 11,933 TPS | 2 | 4,959 TPS |
| 10.0 µs | 3,518 TPS | +4.0% | 3,260 TPS | −5.0% | 11,829 TPS | 1 | 4,916 TPS |

The fit lies near both edges of the 5% band in opposite directions.
The simulated 100K speed is 7% below the 10K speed, while AA measured 100K 1.4% faster than 10K.
The 8-book low-copy summary subset fits at 3.0–4.5 µs, and the 4-book subset does not fit.

Under the row-independent bound, the 8-book low-copy summary subset fits at 0–3.5 µs and the 16-book median at 0–8.0 µs; the 4-book subset does not fit.

### Scan of the growth with rows, added after these results

To find where story C begins to fit, the analysis also keeps the baseline 10K steps and scales the growth from 10K to 100K beyond the one-row growth, from none, the row-independent bound, to the full baseline growth, in steps of 5%.
This scan was not part of the recorded plan.

| Share of the baseline growth beyond one row | 8-row increment | Coding window at minimum n-gram 1, `K = 7` | Answer-alone pairs | Joint condition, answer alone |
| ---: | ---: | --- | ---: | --- |
| 0% | 14.2 µs | 0–8.0 µs | 9 | 0–8.0 µs |
| 10% | 17.6 µs | 0–8.0 µs | 8 | 0–6.0 µs |
| 20% | 21.0 µs | 0–8.0 µs | 7 | 0–3.5 µs |
| 30% | 24.4 µs | 0–8.0 µs | 5 | 0–0.5 µs |
| 35% | 26.1 µs | 0–7.5 µs | 3 | none |
| 50% | 31.2 µs | 0–7.5 µs | 1 | none |
| 100% | 48.2 µs | 0–7.0 µs | 0 | none |

No other setting and neither other data set meets the joint condition at any share.
With the baseline 10K steps, story C fits AA only when the 8-row step grows by at most about 25 µs from 10K to 100K, about half the assumed 48 µs.
The fast-units variant reaches 24.1 µs, at that threshold.

### Comparison with the expectations

| Expectation | Outcome | Evidence |
| --- | --- | --- |
| 1: baseline reproduces the committed steps; 7 rows within 1 µs of the interpolation | Held | Differences of 0.0 µs and at most 0.002 µs |
| 2: 8-row increments of 32.6 ± 2, baseline ± 1 and 24 ± 3 µs; one-row increments of 14.2 ± 0.5, 14.2 ± 0.5 and 6.4 ± 1 µs | Missed in one of six parts | 32.6, 48.1 and 24.1 µs at 8 rows; 14.2 and 6.4 µs at one row, but 12.8 µs under the exponential variant |
| 3: fast units shorten the one-row 10K step by 5 to 20 µs and raise the plain lower end by the same amount | Missed | 3.0 µs shorter; the plain lower end rises by 2.9 µs |
| 4: no simulated variant meets the joint condition | Missed | Fast units meet it at minimum n-gram 1, `K = 7`, answer alone, 7.0–10.0 µs |
| 5: at most 2 of 32 pairs fit at each `K = 7` setting under every simulated variant | Missed | 6 answer-alone pairs fit at minimum n-gram 1 under fast units |
| 6: under the row-independent bound, the 4-book or 8-book low-copy summary subset fits at minimum n-gram 1, `K = 7`, answer alone | Held | The 8-book subset fits at 0–3.5 µs; the 4-book subset does not |
| 7: under the row-independent bound, no pair fits at minimum n-gram 2 | Held | No pair fits at any `K` in either data set |

## Interpretation

Expectation 4 missed, so by the table recorded with the plan, story C's failure at AA depends on the assumed LP30 attention costs, and a measured multi-row attention cost on LP30 becomes the deciding number.

Three qualifications limit how far that goes.

- **Only the most favorable simulated rates fit.**
  Raising the attention matrix efficiency from 0.5 to 1.0 alone leaves an 8-row increment of 32.6 µs and no fit.
  The fit needs the matrix unit at its peak together with a fourfold exponential rate and the vector and shuffle units at their peak rates, or another combination that brings the 8-row increment to about 25 µs.
  None of these rates is measured, so the result says how fast LP30's multi-row attention at 100K must be for story C, not that it is that fast.
- **The fit is marginal.**
  It sits within 1% of the band's edges at both contexts in opposite directions, and the simulated 100K speed stays 7% below the 10K speed, while AA measured the reverse.
  Story A's plain decoding also relies on the band for this difference, over a window of at least 7.7 µs; story C's window under fast units spans 3 µs.
- **Only one kind of answer fits.**
  The fit comes from the answer-alone summary answers, the stand-in for thinking off.
  The quotation answers and the answers with thinking do not fit under any variant.

Combined with the [story C checks](lpx-story-c-checks.md), story C at AA needs one of two conditions:

1. AA's 100K answers copy about 0.1 tokens per step more than its 10K answers, at the assumed attention costs; or
2. LP30's 8-row verification step grows by at most about 25 µs from 10K to 100K, half the assumed growth, and AA's answers resemble summary answers without thinking.

Neither condition can be checked without NVIDIA's measurements or AA's prompts.
Story A keeps a window under every variant, so these attention costs do not separate stories A and C.
They still differ in the extra time per step they need: where story C meets the joint condition, at 0–8.0 µs under the bound and 7.0–10.0 µs under fast units, story A needs at least 12.8 µs and 15.7 µs, so a measured LP30 host loop would still separate them.

## Validation

The unit tests in [`tests/test_lpx_attention_sensitivity.py`](../tests/test_lpx_attention_sensitivity.py) cover the scaled growth, the grouping of the captures, the comparison with the committed steps and the window overlap.
The window and book functions shared with the [story C checks](lpx-story-c-checks.md) gained a lower extra time and a draft length; the story C checks rerun with them reproduce `story-c-checks.json` unchanged.
