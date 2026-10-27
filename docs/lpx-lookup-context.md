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

## Results

The replay took 73 seconds and used at most 0.77 GB.
[`experiments/lpx/lookup-context-results.json`](../experiments/lpx/lookup-context-results.json) holds every window, draft length and context bin, with the SHA-256 of each row file.
Without a window, the replay reproduces [`replay`](../scripts/measure_lpx_lookup_acceptance.py) of the earlier record exactly on 600 random prompt and output pairs.

```bash
python -I scripts/measure_lpx_lookup_context.py \
  --gemma-tokenizer tokenizer.json \
  --data oa-s0.json \
  --data oa-s1.json \
  --summary experiments/lpx/lookup-context-results.json
```

The command lists one `--data` argument for each of the 20 row files `oa-s0.json` to `oa-s19.json`.
The 19,792 assistant turns hold 5,677,110 tokens.

Equivalent per-position acceptance at `K = 7`:

| Context before the turn | Turn tokens | Unlimited | `W` = 16,384 | `W` = 4,096 | `W` = 1,024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Under 2,048 | 1,500,563 | 0.368 | 0.368 | 0.368 | 0.348 |
| 2,048 to 4,095 | 928,842 | 0.423 | 0.423 | 0.422 | 0.351 |
| 4,096 to 8,191 | 1,227,069 | 0.448 | 0.448 | 0.423 | 0.351 |
| 8,192 to 16,383 | 1,197,442 | 0.536 | 0.536 | 0.495 | 0.422 |
| 16,384 to 32,767 | 790,558 | 0.658 | 0.653 | 0.615 | 0.523 |
| 32,768 or more | 32,636 | 0.775 | 0.773 | 0.758 | 0.710 |
| 16,384 or more | 823,194 | 0.663 | 0.658 | 0.621 | 0.530 |
| All turns | 5,677,110 | 0.471 | 0.470 | 0.451 | 0.391 |

At `K = 3` the values are 0.02 to 0.04 lower and change with the window and the bin in the same way.
Without a window, 81% of steps make a proposal at `K = 7` over all turns and 93% on turns with 16,384 or more context tokens.

### Comparison with the expectations

1. Held: the unlimited `K = 7` acceptance over all turns is 0.471.
2. Held: a 4,096-token window lowers the acceptance by 0.020 over all turns and by 0.042 on turns with 16,384 or more context tokens.
3. Held: a 1,024-token window lowers the acceptance by 0.080 over all turns.
4. Missed: turns with 16,384 or more context tokens accept 0.663 and turns with 2,048 to 4,095 accept 0.423, a difference of 0.240.

## Interpretation

Acceptance rises strongly with the context length of a turn, and most of that rise comes from the turns themselves.
With a 1,024-token window, which gives every turn the same amount of text to copy from, turns with 16,384 or more context tokens still accept 0.530 and turns with 2,048 to 4,095 accept 0.351.
That 0.179 difference is 75% of the 0.240 difference without a window.
Late agent turns repeat earlier commands and JSON fields, so they are easier to draft regardless of how far back the drafter can look.

The text available for copying adds less with each increase in the window.
On turns with 16,384 or more context tokens, widening the window from 1,024 to 4,096 tokens adds 0.091, from 4,096 to 16,384 tokens 0.037 and from 16,384 tokens to the whole context 0.005.
That is about 0.045, 0.018 and less than 0.005 per doubling of the window, and the [WritingBench record](lpx-lookup-acceptance.md) found about 0.008 per doubling of a shorter prompt.

A lookup drafter reproduces AA's higher 100K speed only if its acceptance at 100K exceeds its acceptance at 10K by about 0.1.
Going from 10K to 100K tokens of context is about 3.3 doublings, and on this data the doublings beyond 16K tokens add less than 0.005 each, so more text to copy from would supply well under 0.1.
The data shows a second route that is large: answers whose content repeats more text.
AA's 100K questions would have to produce answers that repeat their context more than the 10K answers do.
AA does not publish its prompts, so this record cannot test that route, and the agent-trace acceptance levels do not transfer to AA's long-document questions.
