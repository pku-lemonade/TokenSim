# Two checks of one lookup drafter on every run

Record date: 2026-11-01.
Status: expectations recorded; no check has run.

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
