# Published drafting acceptance compared with the fitted LPX acceptances

Record date: 2026-10-26.
The [workload-acceptance control](lpx-workload-acceptance.md) found 29 serving setups whose fitted acceptances reproduce the Artificial Analysis (AA) Gemma 4 31B speeds at 10K and 100K input and the 4,767 and 10,996 tokens-per-second (TPS) coding values.
This record collects published acceptance measurements and compares them with those fits.

## Conversion

Papers report the acceptance length (AL), the mean number of tokens committed per verification step including the target model's own token.
For a chain of `K` draft tokens with independent per-position acceptance `alpha`, `AL = (1 - alpha^(K+1)) / (1 - alpha)`, which is the `tau` of the speculative-decoding records.
The per-position acceptances below invert that relation with the reported draft length.
Long-context results that do not state a fixed draft length are converted with `K = 7`, and the geometric limit `1 - 1 / AL` differs from those values by at most 0.02.
Different target models, drafters, prompts and draft structures make every value an order-of-magnitude prior, not a Gemma measurement.

## Gemma 4 drafter

The Gemma 4 technical report (arXiv 2607.02770) describes a multi-token prediction (MTP) drafter of 500M parameters for Gemma 4 31B.
The drafter is a 4-layer Transformer with model dimension 1,024, three local and one global attention layer, and it cross-attends to the key/value (KV) cache of the main model.
Google's announcement of the drafters reports speedups of up to 3x and gives no acceptance rate.
No published source gives a Gemma 4 31B acceptance rate by task or context length.
Because the drafter's global layer reads the main model's KV cache, its drafting cost grows with context, which the fits of this study do not include.

## Published acceptance by drafter type

| Drafter | Source | Workload and context | AL | Per-position acceptance |
| --- | --- | --- | --- | --- |
| n-gram, `K = 3` | SPEED-Bench, Llama 3.3 70B | Roleplay, writing, mean, math, coding; short prompts | 1.15, 1.33, 1.41, 1.43, 1.54 | 0.13, 0.25, 0.30, 0.31, 0.36 |
| EAGLE-3, `K = 3` | SPEED-Bench, GPT-OSS 120B | Roleplay, writing, mean, coding | 1.87, 1.98, 2.25, 2.46 | 0.50, 0.54, 0.62, 0.68 |
| EAGLE-3, `K = 3` | SPEED-Bench, Llama 3.3 70B | Multilingual, roleplay, QA, mean, summarization, writing, RAG, coding | 1.73 to 3.00 | 0.44, 0.56, 0.64, 0.68, 0.72, 0.73, 0.75, 0.81 |
| MTP, `K = 3` | SPEED-Bench, Qwen3-Next | Roleplay, writing, mean, coding | 2.09, 2.46, 2.81, 3.34 | 0.57, 0.68, 0.77, 0.88 |
| MTP, `K = 1` | DeepSeek-V3 technical report | Second-token prediction across topics | — | 0.85 to 0.90 |
| Prompt lookup (PLD) | OWL, Llama 3.1 8B and 3.3 70B | WildChat conversations, 4K to 64K input | 2.75, 2.24 | 0.65, 0.56 |
| Suffix decoding | OWL, same models | Same | 3.41, 2.61 | 0.73, 0.63 |
| EAGLE-3 trained on 2K windows | OWL, same models | Same | 1.28, 1.35 | 0.22, 0.26 |
| PLD | LongSpec, five 7B to 13B models | Long-document summarization and code completion | 1.18 to 2.71 | approximately 0.15 to 0.65 |
| Long-context drafter reading the target KV | LongSpec, LongChat 7B | GovReport by prefill length, 0 to 32K | 3.97 to 4.45 | stable across lengths |

The SPEED-Bench values come from NVIDIA's SPEED-Bench announcement at a draft length of 3, and the OWL and LongSpec values come from their main tables and appendices.
SPEED-Bench prompts in these tables come from its qualitative split, whose categories include coding.
Its throughput split uses input buckets of 1K to 32K tokens without a separate coding category.

## Priors

| Drafter type | Prose at 10K or more input | Code |
| --- | --- | --- |
| Lookup (n-gram, PLD, suffix) | 0.55 to 0.73 in multi-turn chat at 4K to 64K; 0.15 to 0.65 on long documents | 0.36 on short prompts; high only when the output copies the input |
| Model drafter trained on short windows (EAGLE-3) | 0.22 to 0.26 | Not measured at long context |
| Model drafter trained with the model or reading the target KV (MTP, LongSpec) | 0.57 to 0.68 on short prompts, stable with context in LongSpec | 0.85 to 0.90 |

## Comparison with the fits

The fits need AA acceptances of 0.003 to 0.48 at 10K and 0.09 to 0.75 at 100K, and 0.73 to 0.99 for the coding example.

- **Fits with an AA 10K acceptance below 0.10** fall below every published prior for prose, including short-prompt n-gram drafting.
  They include TP32/PP4 with `KVP = 8` at `K = 3` without background, TP16/PP16 with `KVP = 4` at `K = 7` without background and the `K = 3` fits with one background request on TP16 at efficiency 0.50.
- **A drafter trained with the model** gives prose acceptances of about 0.57 to 0.68 in the published MTP results for Qwen3-Next, and LongSpec reports such drafters stable with context; no acceptance is published for Gemma 4 31B's own MTP drafter.
  No fit reaches that range at 10K, and with a stable acceptance the AA 100K workload cannot be faster than the 10K workload.
  At 0.57 to 0.68, the AA 10K speed needs a step of 615 to 726 microseconds at `K = 3` and 680 to 882 microseconds at `K = 7`.
  The simulated 10K steps are 218 to 297 microseconds at `K = 3` and 276 to 347 microseconds at `K = 7` without background, and at most 457 microseconds with one background request at efficiency 0.50.
  The coding example at an acceptance of 0.85 to 0.88 needs a step of at most 441 to 486 microseconds at `K = 7`.
- **An EAGLE-3 drafter trained on short windows** gives about 0.22 to 0.26 at long context.
  That range overlaps the 10K acceptance of several fits, but such a drafter degrades with context and cannot supply the higher 100K acceptance that every fit needs.
- **A lookup drafter** matches the direction of every fit, because its acceptance can rise when long answers reuse the prompt.
  The fits at efficiency 0.10 on TP16/PP8 with `KVP = 4` and `K = 7` have the closest shape: 0.138, 0.659, 0.443 and 0.798 without background, and 0.39 to 0.48, 0.72 to 0.75, 0.60 to 0.66 and 0.90 to 0.94 with one background request.
  Their 100K acceptances lie in the long-context lookup range, and their 10K acceptances lie below it.
  A lookup drafter proposes no tokens when it finds no match, so its steps at low acceptance would be cheaper than the fixed `K`-token verification that the fits assume.

## Interpretation

No drafter type matches any fit for all four published speeds.
Capable drafters reach about 0.55 to 0.73 on prose at 10K or more input, and the fits need at most 0.48 at 10K.
At those acceptances the simulated setups commit more tokens per step than the published AA speed allows, so the AA steps would have to be about 1.3 to 3.2 times longer than the simulated steps with at most one background request.
The coding example needs steps no longer than about 490 microseconds, so a capable drafter would also make the AA steps roughly 1.3 to 2 times longer than the coding demonstration's steps.

Three explanations remain:

1. The AA endpoint ran substantially more work per step than the simulated setups, through other traffic, serving overhead or a different configuration from the coding demonstration.
2. The LP30 cost model underestimates the decode step time on this deployment.
3. The deployment used a drafter that performs poorly on AA's prompts, such as a lookup drafter with little to copy at 10K.

The private, single-concurrency AA measurement weakens explanation 1, and explanation 3 conflicts with the published Gemma MTP drafters.
Explanation 2 is testable only against measured hardware, which makes cost-model calibration the next requirement.

## Gaps

- No published acceptance exists for Gemma 4 31B with its MTP drafter, by task or by context length.
- No source measures any drafter at 10K against 100K input on AA-like prompts.
- AA does not publish its prompts in enough detail to judge how much its answers reuse the prompt.
- The SPEED-Bench coding median on the Hot Chips slide most likely comes from the short-prompt qualitative split, while this study prices it at 17,000 tokens of context, and the slide does not state the load.

## Sources

- [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770)
- [Accelerating Gemma 4: faster inference with multi-token prediction drafters](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
- [Introducing SPEED-Bench](https://huggingface.co/blog/nvidia/speed-bench) and the [SPEED-Bench dataset card](https://huggingface.co/datasets/nvidia/SPEED-Bench)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [OWL: Overcoming Window Length-Dependence in Speculative Decoding for Long-Context Inputs](https://arxiv.org/abs/2510.07535)
- [LongSpec: Long-Context Lossless Speculative Decoding](https://arxiv.org/abs/2502.17421)
