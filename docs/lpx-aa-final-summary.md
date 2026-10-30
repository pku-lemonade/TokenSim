# Gemma 4 31B on LPX: final summary of the AA reproduction

Record date: 2026-10-30.
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
Prompt-lookup drafting on the same layout can produce the SPEED-Bench coding values, as described under speculative decoding.
The [article-condition sweep](lpx-article-sweep.md) gives its step times.

| Value | 10K | 100K |
| --- | ---: | ---: |
| AA measurement | 3,382 TPS | 3,431 TPS |
| Simulated, in Gemma tokens | 3,740 TPS (+10.6%) | 3,551 TPS (+3.5%) |
| Simulated, in AA tokens at 1.006 Gemma tokens per AA token | about 3,718 TPS (+9.9%) | about 3,530 TPS (+2.9%) |

AA counts tokens with the `o200k_base` tokenizer.
The [API-outputs record](lpx-api-outputs.md) measures 1.006 Gemma tokens per `o200k_base` token on Gemma 4 31B's thinking and answers to questions about 10K and 100K-token books, and the [token-ratio record](lpx-token-ratio.md) measures 1.031 to 1.048 on its WritingBench outputs; AA's own prompts are not published.
At the ratio of the book answers the 100K value matches and the 10K value is too fast unless each step takes about 13 to 24 µs longer than modeled.
The [tray-latency record](lpx-tray-latency.md) shows that a cross-tray collective latency of about 0.80 to 0.85 µs instead of the assumed 0.75 µs supplies the same time.
The [step-cost audit](lpx-step-cost-audit.md) finds that the closest setup runs with the collective launch, the per-operator instruction floor and the per-step invocation at zero.
A collective launch of about 0.10 to 0.18 µs or a floor of about 0.012 to 0.023 µs per kernel component also supplies that time, while the larger values assumed in other records make the setup too slow, so the match requires LP30's fixed costs to total at most about 24 µs per step.
The [fixed-cost evidence record](lpx-fixed-cost-evidence.md) finds that first-generation Groq data leave about 1 to 2 µs of that to collective launches, instruction dispatch and synchronization, and that Groq's production host collects each output and invokes the next step, a per-token cost without a published size.

## Speculative decoding

The [article-condition speculation record](lpx-article-speculative.md) finds matches on the same layout without unmodeled time only at per-position acceptances of 0.006 to 0.335.
Published multi-token prediction (MTP) drafters reach 0.57 to 0.68 on prose, measured on Qwen3-Next.
The subsequent [public-checkpoint GPU measurements](lpx-gpu-followup-results.md) give document equivalent acceptance of 0.720 to 0.814 across draft lengths 1, 3, 5 and 7 with the released Gemma 4 assistant.
Pricing those measured means on the same LP30 layout makes every 10K result at least 72.26% too fast, and no common extra-time interval contains 25 microseconds or less.
These measurements strengthen the acceptance-based mismatch for the specified public checkpoint and book prompts; they do not identify the published model build or drafter.

NVIDIA also reports a SPEED-Bench coding median of 4,767 TPS, an 80th percentile of 5,520 TPS and a maximum of 6,981 TPS, and a coding example of 10,996 TPS, under the same deployment name.
Plain decoding on the closest setup is 22% to 29% too slow for the coding median and runs every request at nearly the same speed, while NVIDIA's 80th percentile is 1.158 times the median, as the [coding-spread record](lpx-coding-spread.md) shows.
The [API-outputs record](lpx-api-outputs.md) generates answers of the public Gemma 4 31B to SPEED-Bench's 80 coding prompts and to AA-like questions about 10K and 100K-token books.
Those answers differ from the published runs: the model build, thinking mode, drafter, sampling and AA's prompts are not published, and the record lists each difference.

On the SPEED-Bench prompts, the best of 16 lookup settings, chosen after the results, gives the coding median and 80th percentile within 2% and the maximum within 13% with 0 to 21 µs of unmodeled time per step.
That shows lookup drafting can produce NVIDIA's coding statistics, and is weak evidence of which drafter, if any, NVIDIA used.
On the book questions, lookup drafting accepts 0.15 to 0.32 per position and makes the AA 10K value too fast unless each step takes 26 to 99 µs longer than modeled.

The published sources do not say whether either run drafted.
One possibility is one build with about 13 to 21 µs of unmodeled time per step, in which the coding run drafted and the AA run did not.
AA's prompts may give a drafter less to copy than the books used here, and the AA build may differ from the SPEED-Bench build, so the data does not decide between plain decoding and drafting for AA.
No measured acceptance reaches the 10,996 TPS example on 16K code-completion prompts, where the fastest request reaches 9,785 TPS.

## Speed at 100K

No setup consistent with the test conditions reproduces AA's 1.45% higher speed at 100K, and the closest setup gives about equal speeds.
On the AA-like book answers and on agent traces, lookup acceptance at long context is about the same as at short context, so more context is unlikely to explain the difference through drafting.
The difference may lie within the spread of the 50 requests, which AA does not publish.
The [noise check](lpx-aa-noise-check.md) shows that a per-request spread of about 3% would make it smaller than two standard errors of the difference between the two medians.
AA's public Gemma 4 31B endpoints vary by about 4% to 42% between measurements and are faster at 10K than at 1K on 6 of 11 endpoints, which shows that AA's workloads differ in more than context length; those endpoints are shared, so they do not measure the private LPX endpoint's spread.

## Simulator validation

On first-generation Groq hardware, the [BERT check](groqchip-v1-bert-validation.md) predicts a measured BERT-base layer within 6.5% to 9.5%.
The [GroqRack check](groqrack-llama2-7b-validation.md) matches a measured 72-card Llama-2-7B decode once a fixed host loop of 2.7 to 2.8 ms per token is added.
The same addition gives 264 to 286 TPS for Llama-2 70B, inside GroqCloud's published 241 to 300 TPS.
The LP30 cross-tray latency, collective launch and per-kernel costs, matrix-unit operand bandwidth and FP8 attention path remain assumptions.

## Open questions

NVIDIA's chip count and layout remain unknown, and so do the model build, thinking mode and drafter of both the AA and the SPEED-Bench runs.
AA's prompts remain unpublished, so the token ratio and acceptance on them remain unmeasured; the book answers give about one Gemma token per AA token.
Which of the unmeasured LP30 fixed costs supplies the 13 to 24 µs of unmodeled time per step remains unknown.
The match shows that the cost model is compatible with the published values under one plausible setup and does not identify NVIDIA's deployment.

These measurements would settle the remaining questions:

| Measurement | Question it settles |
| --- | --- |
| One all-reduce across two LP30 trays, LP30's collective launch and per-kernel costs, and the host's per-token output collection and invocation | Which fixed cost supplies the missing 13 to 24 µs per step, and whether together they stay within about 24 µs |
| AA's prompts or per-request speed spread | Whether the AA run drafted, since drafting spreads per-request speeds as it does on SPEED-Bench, and whether the 1.45% difference between 100K and 10K lies within the spread |
| A statement of the deployment's speculative-decoding settings | Whether the AA and SPEED-Bench runs used the same build |

## Reproduction

The committed results are in [`experiments/lpx/`](../experiments/lpx/).
[`scripts/analyze_lpx_published_conditions.py`](../scripts/analyze_lpx_published_conditions.py) recomputes the derived tables of the published-condition records from them.
