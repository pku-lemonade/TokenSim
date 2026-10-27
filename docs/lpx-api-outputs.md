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
