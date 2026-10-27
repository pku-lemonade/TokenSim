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

The ratio range that matches without extra time is 1.053 to 1.090, and the reasoning categories lie at or inside it.
No other layout of the sweep matches at any ratio, because a larger ratio also tightens the bound on the 10K-to-100K increment to `25.2 / r` µs.

## Interpretation

The one-to-one assumption biases the targets: on reasoning text Gemma 4 produces about 7% more tokens than `o200k_base`, so AA's 3,382 and 3,431 TPS correspond to about 3,630 and 3,680 native tokens per second.
If AA's Gemma 4 31B (Reasoning) outputs tokenize like the reasoning traces measured here, TP32/PP8 with DCP 8 on 256 chips reproduces both AA values at batch one, without speculative decoding, background requests or unmodeled time per step.

The corpus does not contain AA's prompts or Gemma 4's own outputs.
The traces come from another model, Gemma's thinking style may differ, and AA's long-document questions may produce more prose-like answers, whose ratio is 1.00 to 1.04.
Whether AA's streamed output includes Gemma's thinking is also not stated.
The ratio therefore narrows the missing time from 14 to 25 µs to somewhere between 0 and 25 µs, and a measurement on Gemma 4's own outputs for AA-like prompts would settle it.
