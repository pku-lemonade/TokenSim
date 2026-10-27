# Gemma 4 tokens per AA `o200k_base` token

Record date: 2026-10-26.
Artificial Analysis (AA) counts output tokens with the `o200k_base` tokenizer for every model.
TokenSim counts Gemma 4's native tokens, and every LPX record so far assumes one native token per AA token.
This record measures that ratio on public text.

## Why the ratio matters

If Gemma 4 produces `r` native tokens for every `o200k_base` token of the same text, the endpoint generated `r` times as many native tokens per second as AA reports.
The simulated native step must then be `r` times shorter than the AA time per token.
A ratio of 1.05 shifts the AA targets by 5%, the width of the match band.
The [fidelity configuration](../experiments/lpx/fidelity-study.json) already holds the ratio as `token_units.native_per_aa_output`, set to an unmeasured 1.0.

## Method

[`scripts/measure_lpx_token_ratio.py`](../scripts/measure_lpx_token_ratio.py) tokenizes the same text with Gemma 4's `tokenizer.json` from the `google/gemma-4-31B-it` repository and with `tiktoken`'s `o200k_base` encoding, without special tokens.
It runs in a separate environment with `tokenizers` 0.21.1 and `tiktoken` 0.9.0, which the project does not depend on.

| Category | Text | Source |
| --- | --- | --- |
| Prose | *Pride and Prejudice*, between the Project Gutenberg markers | [Project Gutenberg eBook 1342](https://www.gutenberg.org/cache/epub/1342/pg1342.txt) |
| Encyclopedic prose | Plain-text extracts of seven English Wikipedia articles: Black hole, Economy of Japan, Photosynthesis, Python (programming language), Quantum mechanics, Roman Empire and World War I | Wikipedia API |
| Reasoning output | Assistant turns of OpenThoughts-114k, which contain long reasoning traces followed by solutions, split into the thinking and solution parts | [Hugging Face dataset server](https://datasets-server.huggingface.co/rows?dataset=open-thoughts%2FOpenThoughts-114k&config=default&split=train&offset=0&length=100) |
| Code | Five Python standard-library modules: `argparse.py`, `dataclasses.py`, `typing.py`, `json/decoder.py` and `asyncio/base_events.py` from Python 3.12 | Local installation |

AA's own prompts and outputs are not public, so these categories bracket the likely text and do not reproduce it.
AA's Gemma 4 31B (Reasoning) workloads ask questions about long documents, so the output resembles the reasoning and prose categories.

## Expectations recorded before the measurement

1. Gemma 4 produces 0.98 to 1.08 native tokens per `o200k_base` token on both prose categories.
2. The reasoning category lies between 0.98 and 1.10.
3. Code lies between 1.00 and 1.20, above prose, because whitespace and indentation tokenize differently.

## Results

[`experiments/lpx/token-ratio-results.json`](../experiments/lpx/token-ratio-results.json) holds the token counts and the SHA-256 of every input file and of the Gemma tokenizer.

The first 100 OpenThoughts rows turned out to be Python programming problems only.
After the first measurement, the corpus was extended with 40 rows at each of the offsets 20,000, 45,000, 70,000, 95,000 and 113,000, which hold mathematics, science and puzzle problems, and each answer was split into its thinking and solution parts.
The programming rows are reported separately.

| Category | `o200k_base` tokens | Gemma tokens per `o200k_base` token | Range across files |
| --- | ---: | ---: | --- |
| Prose, *Pride and Prejudice* | 170,260 | 1.006 | — |
| Encyclopedic prose, seven Wikipedia articles | 97,063 | 1.035 | 0.973 to 1.131 |
| Reasoning thinking, mathematics, science and puzzles | 903,145 | 1.073 | 1.053 to 1.106 |
| Reasoning solutions, mathematics, science and puzzles | 89,163 | 1.027 | 1.005 to 1.051 |
| Reasoning thinking, programming problems | 695,308 | 1.100 | — |
| Reasoning solutions, programming problems | 67,452 | 1.088 | — |
| Python standard-library code | 80,697 | 1.170 | 1.123 to 1.208 |

All three expectations held.
The prose categories gave 1.006 and 1.035, the reasoning categories 1.027 to 1.100, and code 1.170.
The Wikipedia article on quantum mechanics gave 1.131, outside the expected prose range; it contains many formulas.

## Gemma 4 31B's own outputs

The corpus above contains no Gemma output.
Three public Hugging Face datasets hold generations of `google/gemma-4-31B-it` with the thinking and the final answer stored separately:

| Dataset | Content | Rows used |
| --- | --- | ---: |
| `ZachW/gemma-4-31b-it_writingbench-en100` | WritingBench writing tasks on supplied documents of about 2,000 tokens, temperature 0 | 100 |
| `ZachW/gemma-4-31b-it_aime-all` | AIME mathematics problems, temperature 0 | 100 |
| `trjxter/Gemma-4-31B-Reasoning-1000x` | Mixed reasoning prompts, including programming, with `<think>` sections | 100 |

The first 100 rows of each dataset were read through the Hugging Face dataset server.
WritingBench is the closest available proxy for AA's questions about long documents.
The WritingBench and AIME rows also record the serving engine's `completion_tokens`, which checks that the local Gemma tokenizer counts the raw output the same way.

Expectations recorded before the measurement:

4. Gemma's WritingBench thinking gives 1.03 to 1.09 Gemma tokens per `o200k_base` token, and its final answers 1.00 to 1.06.
5. Gemma's AIME thinking gives 1.05 to 1.15, above the writing tasks, because of formulas.
6. The local tokenizer's count of each raw output lies within 1% of the recorded `completion_tokens` on average.

[`experiments/lpx/token-ratio-gemma-outputs.json`](../experiments/lpx/token-ratio-gemma-outputs.json) holds the counts and the SHA-256 of each downloaded row file: `g4-1.json` is the mixed reasoning set, `g4-2.json` WritingBench and `g4-3.json` AIME.

| Gemma 4 31B output | `o200k_base` tokens | Gemma tokens per `o200k_base` token |
| --- | ---: | ---: |
| WritingBench thinking | 83,000 | 1.048 |
| WritingBench final answers | 186,470 | 1.031 |
| WritingBench, thinking and answers together | 269,470 | 1.036 |
| AIME thinking | 417,497 | 1.115 |
| AIME final answers | 75,178 | 1.080 |
| Mixed reasoning outputs with `<think>` sections | 50,820 | 1.069 |

All three expectations held.
The local tokenizer counted 279,607 tokens in the WritingBench raw outputs against 279,714 recorded, and 546,975 in the AIME raw outputs against 547,074, a difference of 0.04% and 0.02%.
Gemma's own writing outputs give lower ratios than the reasoning traces of the first table, and its mathematics gives higher ones.

## Effect on the AA match

The ratio scales both AA targets.
With `r` native tokens per AA token, the native step must lie between `1 / (r × 1.05 × TPS)` and `1 / (r × 0.95 × TPS)`.
Applied to the [article-condition sweep](lpx-article-sweep.md), which needed 14 to 25 µs of extra time per step at `r = 1`:

| Ratio `r` | Source of the value | TP32/PP8, DCP 8, efficiency 0.30: AA errors without extra time | Extra time per step for a match |
| ---: | --- | --- | --- |
| 1.000 | Previous assumption | +10.6% at 10K, +3.5% at 100K | 14.2 to 25.2 µs |
| 1.035 | Wikipedia prose | +6.9%, +0.0% | 4.7 to 14.9 µs |
| 1.053 | Lowest ratio that matches without extra time | +5.0%, −1.7% | 0.1 to 9.8 µs |
| 1.073 | Reasoning thinking | +3.1%, −3.5% | 0 to 4.4 µs |
| 1.090 | Highest ratio that matches without extra time | +1.5%, −5.0% | 0 µs only |
| 1.100 | Programming reasoning | — | no match |

The same arithmetic on Gemma's own outputs:

| Ratio `r` | Gemma 4 31B output | AA errors without extra time | Extra time per step for a match | Equivalent middle-tier latency |
| ---: | --- | --- | --- | --- |
| 1.031 | WritingBench final answers | +7.3%, +0.4% | 5.8 to 16.0 µs | 0.77 to 0.81 µs |
| 1.036 | WritingBench, all output | +6.7%, −0.1% | 4.4 to 14.5 µs | 0.77 to 0.81 µs |
| 1.048 | WritingBench thinking | +5.5%, −1.2% | 1.3 to 11.2 µs | 0.76 to 0.80 µs |
| 1.069 | Mixed reasoning | +3.5%, −3.2% | 0 to 5.4 µs | up to 0.77 µs |
| 1.115 | AIME thinking | −0.8%, −7.2% | no match | — |

The latency column uses the 24.9 µs per 0.1 µs slope of the [tray-latency record](lpx-tray-latency.md).

The ratio range that matches without extra time is 1.053 to 1.090, and the reasoning categories lie at or inside it.
No other layout of the sweep matches at any ratio, because a larger ratio also tightens the bound on the 10K-to-100K increment to `25.2 / r` µs.

## Interpretation

The one-to-one assumption biases the targets.
On Gemma 4 31B's own writing outputs, the closest available proxy for AA's long-document questions, Gemma produces 1.031 to 1.048 tokens per `o200k_base` token, so AA's 3,382 and 3,431 TPS correspond to about 3,500 to 3,600 native tokens per second.
At those ratios TP32/PP8 with DCP 8 still needs about 1 to 16 µs of extra time per step, down from 14 to 25 µs at a ratio of 1.0, or equivalently a middle-tier latency of about 0.76 to 0.81 µs.
Reasoning-heavy outputs with ratios of 1.053 to 1.090 would need no extra time, and mathematics-heavy outputs above 1.09 would make the layout too slow.

AA's prompts and Gemma's outputs on them are not public, and whether AA's streamed output includes the thinking is not stated.
The measured ratios therefore narrow the missing time and do not remove it.
Tokenizing Gemma 4 31B's own answers to AA-like long-document questions would settle the value.
