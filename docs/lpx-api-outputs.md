# Gemma 4 31B outputs on SPEED-Bench and AA-like prompts

Record date: 2026-10-27.
The earlier records measured lookup acceptance and token ratios on public Gemma 4 31B outputs whose prompts differ from the published workloads.
This record generates Gemma 4 31B outputs for three prompt sets that match those workloads more closely and repeats the measurements on them:

| Set | Prompts | Published value it addresses |
| --- | --- | --- |
| `speed-coding` | The 80 coding prompts of SPEED-Bench's qualitative split, with thinking on and off | NVIDIA's SPEED-Bench coding median of 4,767 TPS, 80th percentile (P80) of 5,520 TPS and maximum of 6,981 TPS |
| `speed-cc16k` | 40 of the 302 code-completion prompts of SPEED-Bench's 16K throughput split, with thinking off | The Hot Chips coding example of 10,996 TPS at a 16K input |
| `aa-docs` | 16 Project Gutenberg books cut to 10K and 100K Gemma tokens, each with a summary question and a quotation question, with thinking on | The Artificial Analysis (AA) 10K and 100K speeds |

## Prompts

[`scripts/build_lpx_api_prompts.py`](../scripts/build_lpx_api_prompts.py) reads SPEED-Bench from the `nvidia/SPEED-Bench` parquet files and writes 264 requests.

The qualitative coding prompts come from HumanEvalPack (61), Spec-Bench (9), CodeContests (5) and LiveCodeBench (5); 9 of them have two turns.
They hold 30 to 445 Gemma tokens.

The code-completion prompts come from RepoBench and the JetBrains project-level code-completion benchmark.
Each asks for exactly one line of code for every `[COMPLETE]` marker in a source file, given the repository context.
The 40 prompts are drawn with seed 20261027 and hold 18,253 to 21,286 Gemma tokens; SPEED-Bench sizes them with a different tokenizer.

AA does not publish its prompts.
The document set uses eight novels and eight nonfiction books: *Moby Dick*, *Pride and Prejudice*, *Great Expectations*, *A Tale of Two Cities*, *Middlemarch*, *Jane Eyre*, *Dracula*, *Anna Karenina*, *On the Origin of Species*, *The Wealth of Nations*, *The Republic*, *Walden*, *Leviathan*, *The Federalist Papers*, *The Critique of Pure Reason* and *The Voyage of the Beagle*.
Each book is cut after the Project Gutenberg header to its first tokens, so the prompt holds 10,000 or 100,000 Gemma tokens including the question, and the 10K prompt is the start of the 100K prompt.
The summary question asks for a report of at least 2,000 words that summarizes the text in order and explains its themes or arguments.
The quotation question asks for the five most important moments or arguments, with direct quotations and explanations, in at least 2,000 words.
AA's 100K workload requires at least 2,000 answer tokens, so both questions ask for long answers.
Gemma has likely seen these books in training, so its answers may copy less from the prompt than answers about unseen documents.

## Generation

[`scripts/generate_lpx_gemma_outputs.py`](../scripts/generate_lpx_gemma_outputs.py) sends each request to `google/gemma-4-31b-it` through OpenRouter at temperature 0 with seed 1.
It pins the Parasail FP8 endpoint, which serves the full 262,144-token context, and disables fallback to other providers, so every output comes from one FP8 deployment.
Multi-turn prompts send each user turn after the previous answer, as a chat client does.
The generation limit is 16,384 tokens, and 4,096 for code completion.
A provider-side speculative decoder would not change the generated text.

The script exports one rows file per set, thinking mode and, for the document set, context and question style.
Each row holds the user text, including earlier turns, and the thinking and answer of one turn.

## Measurements

- **Token ratio:** [`scripts/measure_lpx_token_ratio.py`](../scripts/measure_lpx_token_ratio.py) measures Gemma tokens per `o200k_base` token on the thinking and the answers of the document set.
- **Lookup acceptance:** [`scripts/measure_lpx_lookup_acceptance.py`](../scripts/measure_lpx_lookup_acceptance.py) replays the lookup drafter with `--per-row`. It uses the `combined` part for outputs with thinking and the `answer-alone` part for outputs without thinking.
- **AA pricing:** each AA workload is priced on TP32/PP8 with decode context parallelism (DCP) of 8 at attention matrix efficiency 0.50, with the proposal lengths measured at its own context, as in the [lookup-acceptance record](lpx-lookup-acceptance.md). The extra time per step that places both AA values within 5% uses the measured token ratio.
- **Coding spread:** [`scripts/analyze_lpx_coding_spread.py`](../scripts/analyze_lpx_coding_spread.py) prices each coding request as in the [coding-spread record](lpx-coding-spread.md). For SPEED-Bench's own 80 prompts it reports the ratios over all turns rather than over random subsets.

## Expectations recorded before the generation

1. On the document set, Gemma's answers give 1.00 to 1.06 Gemma tokens per `o200k_base` token and its thinking 1.02 to 1.08.
2. With `n_min = 1`, `K = 7` and the `combined` part, the summary answers at 100K accept less than 0.05 more than those at 10K.
3. The quotation answers accept at least 0.05 more than the summary answers at both contexts.
4. At the measured acceptances and token ratio, no summary-question lookup setting places both AA values within 5% without extra time per step.
5. On SPEED-Bench's coding prompts, at least one thinking mode and `K` gives a P80/median ratio within 0.05 of 1.158 and a maximum/median ratio within 0.15 of 1.464.
6. On the same prompts, at least one lookup setting places the median within 5% of 4,767 TPS with 0 to 60 µs of extra time per step.
7. On the code-completion prompts, lookup at `K = 7` commits at least 2.5 tokens per step on average, and at least one request reaches 10,996 TPS without extra time.

## Generation results

All 264 requests finished on Parasail within about one hour on 2026-10-27 and cost $0.87 in total.
[`experiments/lpx/api-gemma-outputs.jsonl`](../experiments/lpx/api-gemma-outputs.jsonl) holds every turn's thinking, answer, finish reason, provider, usage and latency, without the prompts.
The request file has SHA-256 `7abb2f93587b01649c27e262e041afe96dd68cc055908d80f35610b5de9506ae`, and the SPEED-Bench parquet files `4f76bc45bdffb38712a5c26d7a7c9a9a791484b392a7b64a6ffb9abe48e02e26` (qualitative) and `7e72b4b352c9ebb65f7847c62c446d93a8d2ff049c74d9a158a3c7e3d6ceb8b1` (16K throughput).
One run stopped on a truncated HTTP response after 259 requests; the script now retries that error, and the resumed run finished the remaining 5.

```bash
python -I scripts/build_lpx_api_prompts.py \
  --gemma-tokenizer tokenizer.json \
  --speed-qualitative speed-qualitative.parquet \
  --speed-16k speed-16k.parquet \
  --gutenberg gutenberg \
  --output requests.jsonl
OPENROUTER_API_KEY=... python scripts/generate_lpx_gemma_outputs.py \
  --requests requests.jsonl \
  --outputs outputs.jsonl \
  --concurrency 8
python scripts/generate_lpx_gemma_outputs.py \
  --requests requests.jsonl \
  --outputs outputs.jsonl \
  --export rows
```

| Rows file | Turns | Median thinking tokens | Median answer tokens | Finish |
| --- | ---: | ---: | ---: | --- |
| `aa-docs-think-10k-summary` | 16 | 1,157 | 2,497 | all stopped |
| `aa-docs-think-10k-quotes` | 16 | 1,370 | 2,522 | all stopped |
| `aa-docs-think-100k-summary` | 16 | 1,173 | 2,458 | all stopped |
| `aa-docs-think-100k-quotes` | 16 | 1,333 | 2,484 | all stopped |
| `speed-coding-think` | 89 | 875 | 405 | 4 reached the 16,384-token limit while thinking |
| `speed-coding-plain` | 89 | 0 | 359 | all stopped |
| `speed-cc16k-plain` | 40 | 0 | 524 | all stopped |

The 80 coding prompts give 89 turns because 9 have two turns.
The four truncated turns stay in the measurements as generated, with thinking and no answer.

## Measurement results

```bash
python -I scripts/measure_lpx_token_ratio.py \
  --gemma-tokenizer tokenizer.json \
  --text aa-thinking=rows/aa-docs-think-10k-summary.json#thinking \
  --text aa-answer=rows/aa-docs-think-10k-summary.json#answer \
  --text aa-output=rows/aa-docs-think-10k-summary.json#thinking \
  --text aa-output=rows/aa-docs-think-10k-summary.json#answer \
  --summary experiments/lpx/api-token-ratio-results.json
python -I scripts/measure_lpx_lookup_acceptance.py \
  --gemma-tokenizer tokenizer.json \
  --data aa-docs-think-10k-summary=rows/aa-docs-think-10k-summary.json#combined \
  --data speed-coding-think=rows/speed-coding-think.json#combined \
  --data speed-coding-plain=rows/speed-coding-plain.json#answer-alone \
  --data speed-cc16k-plain=rows/speed-cc16k-plain.json#answer-alone \
  --per-row \
  --summary experiments/lpx/api-lookup-results.json
python scripts/analyze_lpx_api_outputs.py \
  --lookup experiments/lpx/api-lookup-results.json \
  --ratio experiments/lpx/api-token-ratio-results.json \
  --summary experiments/lpx/api-aa-analysis.json
python scripts/analyze_lpx_coding_spread.py \
  --rows experiments/lpx/api-lookup-results.json \
  --summary experiments/lpx/api-coding-spread-analysis.json
```

The token-ratio and lookup commands list the arguments of all four document files in the same way, and the token-ratio command also adds an `aa-10k-answer` and an `aa-100k-answer` category.

### Token ratio

| Text | Gemma tokens | `o200k_base` tokens | Gemma tokens per `o200k_base` token |
| --- | ---: | ---: | ---: |
| Thinking | 83,052 | 80,928 | 1.026 |
| Answers | 159,040 | 159,839 | 0.995 |
| Thinking and answers | 242,092 | 240,767 | 1.006 |

The answers give 0.995 at both 10K and 100K.
[`experiments/lpx/api-token-ratio-results.json`](../experiments/lpx/api-token-ratio-results.json) holds every file.

### Lookup acceptance on the document answers

Equivalent per-position acceptance with the full prompt, `n_min = 1` and the `combined` part:

| Question | Context | `K` = 1 | `K` = 3 | `K` = 7 | Steps with a proposal at `K` = 7 |
| --- | --- | ---: | ---: | ---: | ---: |
| Summary | 10K | 0.174 | 0.197 | 0.209 | 79% |
| Summary | 100K | 0.186 | 0.206 | 0.218 | 88% |
| Quotation | 10K | 0.253 | 0.287 | 0.315 | 81% |
| Quotation | 100K | 0.249 | 0.279 | 0.305 | 89% |

At `K = 7`, the 100K answer accepts more than the 10K answer of the same book in 10 of 16 books for the summary question and in 7 of 16 for the quotation question.
At 100K the drafter finds a match more often and the matches are accepted less often, so the acceptance changes by about 0.01.
[`experiments/lpx/api-lookup-results.json`](../experiments/lpx/api-lookup-results.json) holds every setting with per-request counts.

### AA pricing

[`experiments/lpx/api-aa-analysis.json`](../experiments/lpx/api-aa-analysis.json) prices both AA workloads on TP32/PP8 with DCP 8 at efficiency 0.50 and 1.006 Gemma tokens per `o200k_base` token.
Plain decoding needs 12.8 to 23.7 µs of extra time per step.

| Question | `n_min` | `K` | Acceptance 10K / 100K | Error without extra time, 10K / 100K | Extra time that fits both |
| --- | ---: | ---: | --- | --- | --- |
| Summary | 1 | 1 | 0.174 / 0.186 | +25.5% / +17.6% | 53.7 to 69.4 µs |
| Summary | 1 | 3 | 0.197 / 0.206 | +24.1% / +13.6% | 53.5 to 63.0 µs |
| Summary | 1 | 7 | 0.209 / 0.218 | +11.2% / −4.0% | none |
| Summary | 2 | 5 | 0.143 / 0.158 | +19.5% / +8.4% | 39.7 to 44.7 µs |
| Quotation | 1 | 3 | 0.287 / 0.279 | +38.4% / +24.4% | 94.2 to 99.4 µs |
| Quotation | 2 | 1 | 0.188 / 0.187 | +28.6% / +19.4% | 61.0 to 74.1 µs |

Of the 16 settings, 5 summary and 5 quotation settings have an extra-time window, all between 39.6 and 99.4 µs.

### SPEED-Bench coding spread

[`experiments/lpx/api-coding-spread-analysis.json`](../experiments/lpx/api-coding-spread-analysis.json) prices each of the 89 coding turns at a 16K input without extra time, with `n_min = 1`:

| Outputs | `K` | Median TPS | P80 TPS | Maximum TPS | P80/median | Maximum/median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA SPEED-Bench coding | — | 4,767 | 5,520 | 6,981 | 1.158 | 1.464 |
| Without thinking | 1 | 4,609 | 5,027 | 5,777 | 1.091 | 1.254 |
| Without thinking | 3 | 4,863 | 5,633 | 7,830 | 1.158 | 1.610 |
| Without thinking | 7 | 4,651 | 5,542 | 9,017 | 1.192 | 1.939 |
| With thinking | 3 | 5,515 | 5,934 | 9,691 | 1.076 | 1.757 |
| With thinking | 7 | 5,282 | 5,894 | 12,211 | 1.116 | 2.312 |

Without thinking at `K = 3`, the median is 2.0% above NVIDIA's, the P80 2.0% above and the maximum 12.2% above.
The extra time that places the median within 5% of 4,767 TPS is 0 to 21 µs at `K = 3` without thinking, and 30 to 64 µs at `K = 3` with thinking.
With thinking, `K = 3` and `n_min = 1`, the window shares 53.5 to 63.0 µs with the AA summary settings; with `n_min = 2` and `K = 5` it shares 39.7 to 44.7 µs.
No setting without thinking shares an extra time with any AA setting.

### Code completion at 16K

Lookup on the 40 code-completion answers commits 1.42 tokens per step at `K = 1` and 1.96 at `K = 7`, with per-request values from 1.24 to 3.39 at `K = 7`.
Priced as above, the fastest request reaches 9,785 TPS at `K = 7`, 11% below 10,996 TPS.
With thinking, the fastest qualitative coding request reaches 11,716 TPS at `K = 5` and 12,211 TPS at `K = 7`.

### Answers without thinking

After the results above, the document answers were also replayed with the `answer-alone` part, which approximates generation without thinking; no expectation was recorded for this check.
[`experiments/lpx/api-lookup-answer-alone-results.json`](../experiments/lpx/api-lookup-answer-alone-results.json) and [`experiments/lpx/api-aa-answer-alone-analysis.json`](../experiments/lpx/api-aa-answer-alone-analysis.json) hold the results, priced at the answers' ratio of 0.995.
The summary answers accept 0.148 at 10K and 0.151 at 100K at `K = 3`, and the quotation answers 0.241 and 0.230.
Plain decoding then needs 15.8 to 26.9 µs, and the lookup settings with a window need 26.3 to 77.5 µs.
The SPEED-Bench median without thinking allows at most 21 µs, so no lookup setting without thinking fits the coding median and both AA values with one extra time.

### Comparison with the expectations

1. Partly held: the thinking gives 1.026, and the answers give 0.995, just below the expected 1.00 to 1.06.
2. Held: the summary answers accept 0.009 more at 100K than at 10K at `K = 7`.
3. Held: the quotation answers accept 0.106 more than the summary answers at 10K and 0.087 more at 100K.
4. Held: every summary setting makes the 10K value at least 11.2% too fast without extra time.
5. Held: without thinking at `K = 3`, the P80/median ratio is 1.158 and the maximum/median ratio 1.610, within 0.15 of 1.464.
6. Held: without thinking at `K = 3`, the median is 4,863 TPS without extra time.
7. Missed: code completion commits 1.96 tokens per step at `K = 7`, and its fastest request reaches 9,785 TPS.

## Interpretation

Lookup drafting without thinking on TP32/PP8 with DCP 8 reproduces all three of NVIDIA's SPEED-Bench coding statistics on SPEED-Bench's own prompts: the median and P80 within 2% and the maximum within 13%, with 0 to 21 µs of extra time per step.
That fits the [coding-spread record](lpx-coding-spread.md), whose competitive-programming stand-ins gave about half of NVIDIA's P80 spread.

On AA-like document questions, lookup acceptance is the same at 10K and 100K within about 0.01, so lookup drafting does not explain AA's higher speed at 100K.
The token ratio of about 1.0 leaves the plain-decoding gap at 13 to 24 µs per step.
With lookup at the measured acceptance, the AA values need 40 to 99 µs of extra time with thinking and 26 to 78 µs without, while the coding values without thinking allow at most 21 µs.

The published values are therefore consistent with one TP32/PP8 DCP 8 build and one extra time of about 13 to 21 µs per step only if the coding run drafted and the AA run did not, or drafted with almost no gain.
A lookup drafter accepts about 0.15 to 0.3 per position on AA-like answers, so an AA run with the same drafter would be faster than measured unless AA's questions give it much less to copy than these books.
The coding values with thinking fit the AA values with lookup at 40 to 63 µs, but their P80 spread is 1.08 to 1.12, below NVIDIA's 1.158.
Gemma has likely seen these books in training, AA's prompts remain unpublished, and the replay assumes a lookup drafter, so these results narrow the explanations without identifying the deployment.
