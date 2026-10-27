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
| Reasoning output | Assistant turns of the first 100 rows of OpenThoughts-114k, which contain long reasoning traces followed by solutions | [Hugging Face dataset server](https://datasets-server.huggingface.co/rows?dataset=open-thoughts%2FOpenThoughts-114k&config=default&split=train&offset=0&length=100) |
| Code | Five Python standard-library modules: `argparse.py`, `dataclasses.py`, `typing.py`, `json/decoder.py` and `asyncio/base_events.py` from Python 3.12 | Local installation |

AA's own prompts and outputs are not public, so these categories bracket the likely text rather than reproduce it.
AA's Gemma 4 31B (Reasoning) workloads ask questions about long documents, so the output resembles the reasoning and prose categories.

## Expectations recorded before the measurement

1. Gemma 4 produces 0.98 to 1.08 native tokens per `o200k_base` token on both prose categories.
2. The reasoning category lies between 0.98 and 1.10.
3. Code lies between 1.00 and 1.20, above prose, because whitespace and indentation tokenize differently.

## Results

Pending.
