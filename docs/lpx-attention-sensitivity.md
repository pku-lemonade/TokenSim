# How much do the three explanations depend on the assumed LP30 attention costs?

Record date: 2026-11-02.
Status: expectations recorded; no simulation has run.

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
