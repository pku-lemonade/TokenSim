# Gemma 4 31B on LPX: final summary of the AA reproduction

Record date: 2026-10-27.
This record states the outcome of the work to reproduce the Artificial Analysis (AA) speeds of Gemma 4 31B on NVIDIA Groq 3 LPX in TokenSim.
The [progress summary](lpx-aa-reproduction-progress.md) links every experiment record behind it.

## Question

AA measured a median of 3,382 output tokens per second (TPS) at 10K input tokens and 3,431 TPS at 100K input tokens for the Gemma4-31B-LPX deployment.
The work asked whether TokenSim reproduces both values and which deployment setup does so.

## Test conditions

AA's chart footnotes state that each value is the median of 50 sequential client requests at concurrency 1 to one private deployment, so each request ran alone.
Press reports state that the model ran in FP8 on LP30 chips only.
No source publishes the chip count or layout, and one 256-chip rack is the most likely upper bound.
The [test-configuration record](lpx-aa-test-configuration.md) lists the established and the open conditions.

## Closest setup

Under these conditions on 256 chips, one layout comes close.
It uses tensor parallelism (TP) of 32 with 8 pipeline-parallel (PP) stages, splits each request's KV cache across 8 chips with decode context parallelism (DCP), runs attention matrix work at 30% efficiency or more and decodes one token per step without speculative decoding.
The [article-condition sweep](lpx-article-sweep.md) gives its step times.

| Value | 10K | 100K |
| --- | ---: | ---: |
| AA measurement | 3,382 TPS | 3,431 TPS |
| Simulated, in Gemma tokens | 3,740 TPS (+10.6%) | 3,551 TPS (+3.5%) |
| Simulated, in AA tokens at 1.036 Gemma tokens per AA token | about 3,610 TPS (+6.7%) | about 3,428 TPS (−0.1%) |

AA counts tokens with the `o200k_base` tokenizer, and the [token-ratio record](lpx-token-ratio.md) measures 1.031 to 1.048 Gemma tokens per `o200k_base` token on Gemma 4 31B's own writing outputs.
At those ratios the 100K value matches and the 10K value is too fast unless each step takes about 1 to 16 µs longer than modeled.
At one Gemma token per AA token the missing time is 14 to 25 µs per step.
The [tray-latency record](lpx-tray-latency.md) shows that a cross-tray collective latency of about 0.76 to 0.81 µs instead of the assumed 0.75 µs supplies the same time.
A token ratio from 1.053 to 1.090 on Gemma's answers to AA's prompts would also remove it.

## Speculative decoding

The [article-condition speculation record](lpx-article-speculative.md) finds matches on the same layout only at per-position acceptances of 0.006 to 0.335.
Gemma's multi-token prediction (MTP) drafter class reaches 0.57 to 0.68 on prose.
The [lookup-acceptance record](lpx-lookup-acceptance.md) measures prompt-lookup drafting on Gemma's own outputs, and that acceptance makes the endpoint 13% to 24% too fast at 10K.
The [lookup-context record](lpx-lookup-context.md) finds that widening the lookback beyond 16K tokens adds less than 0.005 acceptance on Gemma agent turns with up to 66K context tokens.
Plain decoding fits the AA values better than any drafter at its measured or published acceptance.

## Speed at 100K

No setup consistent with the test conditions reproduces AA's 1.45% higher speed at 100K, and the closest setup gives about equal speeds.
The difference may lie within the spread of the 50 requests, which AA does not publish.
A drafter could produce it only if AA's 100K answers repeat their context more than its 10K answers, which AA's unpublished prompts prevent checking.

## Simulator validation

On first-generation Groq hardware, the [BERT check](groqchip-v1-bert-validation.md) predicts a measured BERT-base layer within 6.5% to 9.5%.
The [GroqRack check](groqrack-llama2-7b-validation.md) matches a measured 72-card Llama-2-7B decode once a fixed host loop of 2.7 to 2.8 ms per token is added.
The same addition gives 264 to 286 TPS for Llama-2 70B, inside GroqCloud's published 241 to 300 TPS.
The LP30 cross-tray latency, matrix-unit operand bandwidth and FP8 attention path remain assumptions.

## Open questions

NVIDIA's chip count and layout remain unknown.
The token ratio and drafting acceptance of Gemma's answers to AA-style long-document prompts remain unmeasured.
The match shows that the cost model is compatible with the published values under one plausible setup and does not identify NVIDIA's deployment.

These measurements would settle the remaining questions:

| Measurement | Question it settles |
| --- | --- |
| Gemma 4 31B answers to AA-like 10K and 100K document questions, generated on a GPU host | Token ratio and lookup acceptance on AA-like content, using [`scripts/measure_lpx_token_ratio.py`](../scripts/measure_lpx_token_ratio.py) and [`scripts/measure_lpx_lookup_acceptance.py`](../scripts/measure_lpx_lookup_acceptance.py) |
| One all-reduce across two LP30 trays | Whether the cross-tray latency supplies the missing 1 to 16 µs per step |
| AA's per-request speed spread | Whether the 1.45% difference between 100K and 10K lies within the spread of the 50 requests |

## Reproduction

The committed results are in [`experiments/lpx/`](../experiments/lpx/).
[`scripts/analyze_lpx_published_conditions.py`](../scripts/analyze_lpx_published_conditions.py) recomputes the derived tables of the published-condition records from them.
