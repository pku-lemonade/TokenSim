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

## Results

The replay took 96 seconds and used at most 0.44 GB.
[`experiments/lpx/lookup-coding-results.json`](../experiments/lpx/lookup-coding-results.json) holds every part, prompt mode, `n_min` and `K` with the proposal-length shares.
The extended script reproduces every value of the earlier WritingBench and AIME results.

```bash
python -I scripts/measure_lpx_lookup_acceptance.py \
  --gemma-tokenizer tokenizer.json \
  --data codex-combined=codex-500.json#combined \
  --data codex-thinking=codex-500.json#thinking \
  --data codex-answer=codex-500.json#answer \
  --data codex-answer-alone=codex-500.json#answer-alone \
  --summary experiments/lpx/lookup-coding-results.json
python scripts/analyze_lpx_published_conditions.py \
  --summary experiments/lpx/published-conditions-analysis.json
```

Committed tokens per step `tau` and equivalent per-position acceptance with the full prompt and `n_min = 1`:

| Part | `K` = 1 | `K` = 3 | `K` = 5 | `K` = 7 | Acceptance at `K` = 7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `combined` | 1.415 | 1.738 | 1.854 | 1.906 | 0.477 |
| `thinking` | 1.418 | 1.745 | 1.864 | 1.917 | 0.480 |
| `answer` | 1.398 | 1.690 | 1.789 | 1.836 | 0.456 |
| `answer-alone` | 1.286 | 1.451 | 1.493 | 1.510 | 0.338 |

With `n_min = 2`, `tau` is 1.347 to 1.731 for `combined` and 1.189 to 1.328 for `answer-alone`.
Removing the prompt changes the `combined` and `thinking` acceptance by at most 0.011, because the problems are short, and the `answer-alone` acceptance by at most 0.043.
Removing the prompt of the `answer` part, which contains the thinking, lowers its `K = 7` acceptance from 0.456 to 0.294.

Coding speed on TP32/PP8 with DCP 8 at efficiency 0.50 and the 16K coding context, without extra time:

| Part | `n_min` | `K` | Step | Coding TPS | Against 4,767 | Extra time for the median | Extra time shared with AA |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `combined` | 1 | 1 | 277.2 µs | 5,105 | +7.1% | 5.5 to 35.3 µs | none |
| `combined` | 1 | 3 | 298.8 µs | 5,815 | +22.0% | 48.3 to 84.9 µs | 51.6 to 61.3 µs |
| `combined` | 1 | 7 | 342.1 µs | 5,573 | +16.9% | 38.8 to 78.9 µs | none |
| `combined` | 2 | 1 | 274.6 µs | 4,905 | +2.9% | 0 to 22.8 µs | none |
| `combined` | 2 | 3 | 288.1 µs | 5,560 | +16.6% | 31.9 to 65.6 µs | 44.6 to 56.5 µs |
| `combined` | 2 | 5 | 301.3 µs | 5,608 | +17.7% | 36.3 to 71.8 µs | 42.8 to 52.2 µs |
| `combined` | 2 | 7 | 314.2 µs | 5,510 | +15.6% | 31.7 to 68.1 µs | 38.0 to 44.6 µs |
| `answer-alone` | 1 | 3 | 293.7 µs | 4,941 | +3.7% | 0 to 26.8 µs | none |
| `answer-alone` | 1 | 5 | 311.6 µs | 4,791 | +0.5% | 0 to 18.1 µs | none |

The last column is the constant extra time per step that places the coding median and both AA values, priced at the WritingBench acceptance of the same `n_min` and `K`, within 5%.
[`experiments/lpx/published-conditions-analysis.json`](../experiments/lpx/published-conditions-analysis.json) holds all 32 settings under `coding_pricing`.
The fastest setting gives 5,841 TPS, 46.9% below the 10,996 TPS example.
If every step proposed `K` tokens, the example would need 3.34 committed tokens per step at `K = 3` and 3.92 at `K = 7`, per-position acceptances of 0.88 and 0.78.

### Comparison with the expectations

1. Missed: the `combined` part gives 0.477 at `K = 7`, above the expected 0.25 to 0.45.
2. Held: the `answer` part accepts 0.456 and the `answer-alone` part 0.338 at `K = 7`, a difference of 0.118.
3. Held: `combined` with `K = 1` places the median within 5% with 5.5 to 35.3 µs of extra time at `n_min = 1` and 0 to 22.8 µs at `n_min = 2`.
4. Held: no setting comes within 46% of 10,996 TPS.
5. Missed: 13 of 32 settings, including 4 of the 8 `combined` settings, place the coding median and both AA values within 5% with one shared extra time of 38.0 to 61.3 µs per step.

## Interpretation

Lookup drafting on Gemma's coding outputs commits 1.19 to 1.92 tokens per step.
Without extra time, the coding median needs 4,767 TPS times the step time: 1.31 committed tokens at the 274.6 µs step and 1.63 at the 342.1 µs step.
The 4,767 TPS median is therefore reachable with lookup drafting on the closest AA layout, with or without thinking.
The 10,996 TPS example needs an acceptance of about 0.8 per position, which competitive-programming answers do not reach; it would need an output that mostly copies its prompt, such as an edit of supplied code.

The coding median and the AA values now admit two consistent explanations on TP32/PP8 with DCP 8:

| Explanation | AA 10K and 100K | Coding median | Unmodeled time per step |
| --- | --- | --- | --- |
| Plain decoding | Within 5% | 22% to 29% too slow, so it needs another build | 14 to 25 µs at one Gemma token per AA token |
| Lookup drafting at the measured writing and coding acceptances | Within 5% | Within 5% | 38 to 61 µs |

Lookup drafting lets one build explain three of the four published speeds, at the cost of more than twice the unmodeled time per step.
A drafter that runs outside the LP30 chips, such as n-gram matching on the host CPU, would add time to every step, but no source describes where the deployment drafts, so the cause of the 38 to 61 µs remains unidentified.
Neither explanation reproduces AA's higher speed at 100K or the 10,996 TPS example.
The [API-outputs record](lpx-api-outputs.md) repeats this measurement on Gemma's answers to SPEED-Bench's own coding prompts and to questions about 10K and 100K-token books; there the coding values fit lookup drafting without thinking at 0 to 21 µs and the AA values fit lookup at 26 to 99 µs, so the two share no extra time on that data.
