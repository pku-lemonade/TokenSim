# Prompt-lookup acceptance on Gemma 4 31B coding outputs

Record date: 2026-10-27.
NVIDIA reports 4,767 output tokens per second (TPS) as the SPEED-Bench coding median and 10,996 TPS as a coding example for the deployment name that Artificial Analysis (AA) measured at about 3,400 TPS.
The [article-condition sweep](lpx-article-sweep.md) found that the closest AA setup, TP32/PP8 with decode context parallelism (DCP) of 8 on 256 chips, needs 1.35 to 1.40 committed tokens per step for the coding median and 3.11 to 3.23 for the example.
Prompt-lookup drafting accepts more on text that repeats itself, so one lookup drafter could explain both the AA speeds and the faster coding speeds.
This record measures lookup acceptance on Gemma 4 31B's coding outputs and prices it on the same setup.

## Data

`MasonMac/CodeX-Thinking-Gemma-4-31B-IT` holds 138,683 programming problems from `Modotte/CodeX-2M-Thinking`, each answered once by `nvidia/Gemma-4-31B-IT-NVFP4`.
That checkpoint mixes BF16 and FP8 weights and uses an FP8 KV cache.
No system prompt was used, and the dataset card does not state the sampling temperature.
Each assistant message stores the thinking in a `reasoning` field and the answer, a fenced code block with explanation, in `content`.

Ten pages of 50 rows were read through the Hugging Face dataset server at offsets 0, 13,800, 27,600 and so on up to 124,200, and merged into one file with SHA-256 `30e6748532a5f113482147287d60ce9d044e0dd64269bae7429b66da983981e2`.
With Gemma 4's tokenizer, the 500 prompts have a median of 177 tokens, the thinking a median of 2,092 tokens and the answers a median of 454 tokens.

SPEED-Bench's coding prompts come from its qualitative split, as the [acceptance-evidence record](lpx-acceptance-evidence.md) describes, and NVIDIA does not state whether thinking was enabled.
The CodeX problems are competitive-programming tasks, so their acceptance approximates and does not reproduce SPEED-Bench's.

## Method

[`scripts/measure_lpx_lookup_acceptance.py`](../scripts/measure_lpx_lookup_acceptance.py) replays the drafter of the [lookup-acceptance record](lpx-lookup-acceptance.md) and accepts `name=path#part` for rows with a user and an assistant message.
The part selects the prompt and output of the replay:

| Part | Prompt | Output |
| --- | --- | --- |
| `combined` | Problem | Thinking followed by the answer, as a thinking model generates them |
| `thinking` | Problem | Thinking |
| `answer` | Problem followed by the thinking | Answer, which can copy from the thinking |
| `answer-alone` | Problem | Answer |

The parts join texts with a blank line, and chat-template and channel tokens are omitted.
The `answer-alone` part approximates generation without thinking, although the answers were written after thinking.
For a drafter that proposes fixed tokens, the replay on sampled outputs estimates the acceptance at the sampling temperature, as the [lookup-context record](lpx-lookup-context.md) explains.

[`scripts/analyze_lpx_published_conditions.py`](../scripts/analyze_lpx_published_conditions.py) prices each setting with the full prompt on TP32/PP8 with DCP 8 at attention matrix efficiency 0.50.
Each step costs the verification forward of its actual proposal length at the 16K coding context, taken from the [article-condition speculation record](lpx-article-speculative.md).
The coding speed is counted in native tokens.
The script reports the extra time per step that places the coding median within 5%, and whether one extra time also places both AA values within 5% at the WritingBench acceptance of the same setting.

## Expectations recorded before the measurement

1. With `n_min = 1`, `K = 7` and the full prompt, the `combined` part gives an equivalent per-position acceptance between 0.25 and 0.45.
2. The `answer` part accepts at least 0.05 more than the `answer-alone` part at `K = 7`.
3. At least one setting of the `combined` part places the coding median within 5% of 4,767 TPS with 0 to 25 µs of extra time per step.
4. No setting reaches 10,996 TPS within 5% without extra time.
5. No setting places the coding median and both AA values within 5% with one shared extra time.
