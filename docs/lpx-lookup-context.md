# Prompt-lookup acceptance and context length

Record date: 2026-10-27.
The [lookup-acceptance record](lpx-lookup-acceptance.md) found that a prompt-lookup drafter reproduces Artificial Analysis (AA)'s higher speed at 100K than at 10K only if its per-position acceptance is about 0.1 higher at 100K.
Its WritingBench prompts held about 800 tokens, so it could not test whether acceptance rises with context length.
This record measures how lookup acceptance depends on the amount of earlier text on longer contexts of Gemma 4 31B.

## Data

A search of the Hugging Face Hub and the web on 2026-10-27 found no published Gemma 4 31B generations on 10K to 100K-token prompts.
The longest available contexts are in `open-athena/gemma4-31b-tpu-traces-consolidated`, which holds 16,672 terminal-agent trajectories.
Each trajectory alternates user turns, which carry the task and terminal output, with assistant turns, which hold a JSON object with an analysis, a plan and shell commands.
Every assistant turn was generated with all earlier turns as its context.

The dataset card does not describe the model or the sampling settings.
The `model` field names 12 `hosted_vllm` endpoints with numeric identifiers, so the turns may come from several checkpoints of the same Gemma 4 31B training run.
The assistant turns contain no thinking section.

Twenty pages of 100 rows were read through the Hugging Face dataset server at offsets 0, 830, 1,660 and so on up to 15,770.
They contain 19,792 assistant turns.
Measured with Gemma 4's tokenizer and without chat-template tokens, the context before an assistant turn has a median of 6,721 tokens, a 90th percentile of 21,472 and a maximum of 66,098 tokens, and an assistant turn has a median of 243 tokens.

| Context before the turn | Assistant turns |
| --- | ---: |
| Under 2,048 tokens | 4,468 |
| 2,048 to 4,095 | 2,727 |
| 4,096 to 8,191 | 3,872 |
| 8,192 to 16,383 | 4,789 |
| 16,384 tokens or more | 3,936 |

The content differs from AA's long-document questions.
Agent turns repeat the JSON structure, commands and paths from earlier turns, so their acceptance level does not transfer to AA's prompts.
The measurement addresses how acceptance changes with the amount of earlier text, which is the property the AA explanation needs.

## Method

[`scripts/measure_lpx_lookup_context.py`](../scripts/measure_lpx_lookup_context.py) tokenizes each message with Gemma 4's tokenizer and concatenates them in order.
It replays the drafter of the [lookup-acceptance record](lpx-lookup-acceptance.md) over each assistant turn, matching the longest suffix of 3 down to 1 tokens, while user turns only extend the text available for copying.

The replay uses a lookback window `W`: the drafter may copy only from an occurrence that ends within the `W` tokens before the current position.
Unlimited `W` uses the whole context, and `W` of 1,024, 4,096 and 16,384 tokens restricts the same turns to recent text.
Comparing windows on the same turns separates the effect of context length from the position of a turn in the trajectory, which also grows with context.

The script reports the tokens committed per step `tau`, the share of steps with a proposal, the proposal-length shares and the equivalent per-position acceptance `alpha` for `K` of 3 and 7 draft tokens.
It reports them for all turns and for each context-length bin in the table above, with a bin for 32,768 tokens or more.

The sampling temperature is unknown.
For a drafter that proposes fixed tokens, speculative sampling accepts a draft token with the probability that the target model samples it, and the committed text has the target model's distribution.
The probability that the replay matches a draft prefix on a sampled output therefore equals the probability that verification accepts it, so the replay estimates the acceptance at the temperature that produced the data.

## Expectations recorded before the measurement

1. With unlimited lookback and `K = 7`, the equivalent acceptance over all turns lies between 0.30 and 0.60.
2. Restricting the lookback from unlimited to 4,096 tokens lowers the `K = 7` acceptance by less than 0.03 over all turns and by less than 0.05 on turns with 16,384 or more context tokens.
3. Restricting the lookback to 1,024 tokens lowers the `K = 7` acceptance by at least 0.05 over all turns.
4. With unlimited lookback, the `K = 7` acceptance of turns with 16,384 or more context tokens exceeds that of turns with 2,048 to 4,095 context tokens by less than 0.10.

Expectations 2 and 4 together predict that more context does not add the 0.1 of acceptance that the lookup explanation of the AA values needs.
