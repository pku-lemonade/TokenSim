# Progress on reproducing the AA Gemma 4 31B LPX results

Record date: 2026-10-26.
This page summarizes the work to reproduce the Artificial Analysis (AA) Gemma 4 31B results on NVIDIA Groq 3 LPX (LP30) hardware with TokenSim.
Each section links to the detailed record that holds the method, the expectations recorded before each run and the complete results.
The [final summary](lpx-aa-final-summary.md) states the outcome of the work in one page.

## Goal

The published AA endpoint reaches a median output speed of 3,382 tokens per second (TPS) at 10K input tokens and 3,431 TPS at 100K input tokens.
The goal is to reproduce both values, ideally with the smallest LP30 deployment.
The published 100K speed is 1.45% higher than the 10K speed, so the endpoint's speed does not fall with context length.
A simulated point matches when both speeds lie within 5% of the published values.

TP denotes tensor parallelism and PP pipeline parallelism.
Decode context parallelism (DCP) splits one request's key/value (KV) cache by token position across the chips of a TP group, following NVIDIA's Helix parallelism.
The scripts and earlier records call the DCP size `KVP`.
The attention group is the TP group of `N` chips that computes one global attention layer, factored as `N = TPA * KVP`, where TPA ranks split the four global KV heads.
All results use the bundled [fidelity configuration](../experiments/lpx/fidelity-study.json), whose LP30 unit rates and efficiencies are uncalibrated assumptions.

## Summary of outcomes

| Step | Record | Outcome |
| --- | --- | --- |
| DCP for decode and prefill | [LPX context parallelism](lpx-context-parallel.md) | 100K speed rises approximately 15%; minimum inventory falls from 128 to 96 chips |
| TP8 with DCP | [TP8 section](lpx-context-parallel.md#tp8-with-kv-parallelism) | No gain at TP8; TP16/PP6 on 96 chips remains the smallest layout that clears all targets |
| Placement plausibility | [Placement plausibility record](lpx-placement-plausibility.md) | Six reasons why a dedicated batch-one deployment is unlikely to be the published one |
| Batched serving | [Batched-serving control](lpx-batched-serving.md) | Background traffic explains the 10K level but not the flat context dependence; no match in 156 points |
| Speculative decoding | [Speculative-decoding control](lpx-speculative-decoding.md) | Speculation shared with background traffic matches both AA values within 5% on TP16 |
| Attention-group size | [Attention-group control](lpx-attention-group.md) | TP32 with `KVP = 8` matches without speculation, and with `K = 3` speculation it also reaches the coding example, at attention matrix efficiency 0.30 or more |
| Per-workload acceptance | [Workload-acceptance control](lpx-workload-acceptance.md) | 29 setups reach both AA values and both coding values with one acceptance per speed; every fit needs a higher acceptance at 100K than at 10K |
| Published acceptance | [Acceptance evidence](lpx-acceptance-evidence.md) | No drafter type matches any fit for all four speeds; capable drafters imply AA steps 1.3 to 3.2 times longer than simulated |
| GroqChip v1 BERT check | [BERT validation](groqchip-v1-bert-validation.md) | The LP30 method with matrix efficiency 0.3 predicts the measured BERT-base layer within 6.5% to 9.5%; decode weight streaming, communication and fixed costs remain untested |
| GroqChip v1 Llama-2 70B check | [Llama-2 validation](groqchip-v1-llama2-validation.md) | With the direct all-reduce that production uses, simulated decode is 3.5 to 5.2 times faster than the published 241 to 300 tokens per second |
| AA test configuration | [Test-configuration record](lpx-aa-test-configuration.md) | Sequential single-request decode on one named deployment is established; LP30-only execution, FP8 and batch size 1 are likely; chip count, layout and speculative decoding are unknown |
| Published-condition sweep | [Article-condition sweep](lpx-article-sweep.md) | At batch one without speculation on 256 chips, only TP32/PP8 with `KVP = 8` at efficiency 0.30 or more matches, and only with 14 to 25 µs of unmodeled time per step |
| Published-condition speculation | [Article-condition speculation](lpx-article-speculative.md) | 11 of 378 drafter combinations match on TP32/PP8 with `KVP = 8`, all at shared AA acceptances of 0.006 to 0.335, below published MTP (0.57 to 0.68) and mostly below lookup (0.13 to 0.73) acceptance |
| Lookup acceptance on Gemma outputs | [Lookup-acceptance record](lpx-lookup-acceptance.md) | Prompt lookup accepts about 0.2 per position on Gemma 4 31B writing outputs and 0.57 on its mathematics; at the writing acceptance the TP32/PP8 endpoint is 13% to 24% too fast at 10K, and AA's higher 100K speed would need about 0.1 more acceptance at 100K than the measured prompt effect suggests |
| Lookup acceptance and context length | [Lookup-context record](lpx-lookup-context.md) | On Gemma 4 31B agent turns with up to 66K context tokens, widening the lookback beyond 16K tokens adds less than 0.005 acceptance; acceptance rises 0.240 from 2K to 16K+ contexts, but 75% of that rise remains with a 1,024-token lookback, so it comes from more repetitive late turns |
| Lookup acceptance on coding outputs | [Lookup-coding record](lpx-lookup-coding.md) | Lookup drafting on Gemma 4 31B coding outputs commits 1.19 to 1.92 tokens per step; 13 of 32 lookup settings place both AA values and the 4,767 TPS coding median within 5% with one shared 38 to 61 µs of unmodeled time per step, while plain decoding needs 14 to 25 µs and leaves the coding median 22% to 29% too slow |
| Coding speed spread | [Coding-spread record](lpx-coding-spread.md) | NVIDIA's coding P80 and maximum are 1.158 and 1.464 times the median; plain decoding gives 1.000 and 1.001, and lookup drafting on Gemma coding outputs gives 1.04 to 1.09 and up to 1.53; NVIDIA does not state the load, and a coding run that served one request at a time most likely used drafting or another content-dependent mechanism |
| Gemma outputs on SPEED-Bench and AA-like prompts | [API-outputs record](lpx-api-outputs.md) | The best of 16 lookup settings, chosen after the results, gives NVIDIA's SPEED-Bench coding median and P80 within 2% and maximum within 13% with 0 to 21 µs of extra time; answers to questions about 10K and 100K-token books give 1.006 Gemma tokens per `o200k_base` token and the same lookup acceptance at both contexts within 0.01; model build, thinking, drafter and AA's prompts differ from the published runs |
| Token ratio | [Token-ratio record](lpx-token-ratio.md) | Gemma 4 31B's own writing outputs give 1.031 to 1.048 tokens per `o200k_base` token, which cuts the TP32/PP8 extra time from 14 to 25 µs to about 1 to 16 µs; ratios of 1.053 to 1.090, seen on reasoning-heavy text, remove it |
| Cross-tray latency | [Tray-latency record](lpx-tray-latency.md) | The TP32/PP8 step grows by about 25 µs per 0.1 µs of cross-tray latency; 0.81 to 0.85 µs instead of the assumed 0.75 µs matches both AA values without extra time at one token per AA token, about 0.80 to 0.85 µs at the 1.006 tokens per AA token of AA-like book answers, and about 0.73 to 0.77 µs at another model's reasoning ratio of 1.073 |
| GroqRack Llama-2-7B check | [GroqRack validation](groqrack-llama2-7b-validation.md) | A measured 2.98 to 3.05 ms per token on 72 known first-generation cards exceeds the simulated forward by 2.70 to 2.83 ms; adding that per-token host cost to the 70B check gives 264 to 286 tokens per second, inside GroqCloud's 241 to 300 |

## DCP

The simulator supports Helix-style DCP on LP30 decode and prefill.
On TP16/PP8 with 128 chips, `KVP = 2` changes the speeds at 1K, 10K and 100K from 5,908.78, 5,578.41 and 3,611.17 TPS to 5,759.05, 5,560.63 and 4,167.93 TPS.
The 100K-to-10K speed ratio rises from 0.647 to 0.750, while the published ratio is 1.015.
DCP helps only when duplicated global KV heads make vector-unit KV conversion the bound, which holds at TP16 and not at TP8.
TP16/PP6 on 96 chips with DCP is the smallest layout found that clears the 1K, 10K and 100K speed targets.

## Why the dedicated batch-one placement is unlikely

The simulated dedicated deployment runs one request per forward pass without speculative decoding.
It is faster than published at 10K and slower than published at 100K, and its speed always falls with context length.
The [placement plausibility record](lpx-placement-plausibility.md) gives six reasons why this is unlikely to be the published deployment.
One reason is that the same Gemma4-31B-LPX deployment name is reported at 10,996 TPS on a 16K-input coding example at Hot Chips 2026, which one committed token per forward pass cannot reach.
The shared name does not establish that the coding example and the AA runs used the same configuration, so this summary treats the coding example as a consistency check.

## Batched serving

The [batched-serving control](lpx-batched-serving.md) places the AA request in each decode forward together with up to 15 background requests.
About seven short background requests slow the 10K speed to the published value.
Background requests add the same time at 10K and 100K, so the AA request's own 100K attention increment of 28.6 to 97.7 microseconds remains.
When the 10K speed matches, the 100K speed stays 10% to 26% below the published value.
Background requests represent any context-independent time per step, so a match shows how much such time a deployment needs and does not show that other users shared the endpoint.

## Speculative decoding

The [speculative-decoding control](lpx-speculative-decoding.md) prices native verification forwards that check `K` draft tokens plus one target token.
One shared acceptance rate matches the AA pair only when the 100K-to-10K step-time ratio is at most 1.0896.

Speculation at batch size one on TP16 cannot match the AA pair, because the smallest step-time ratio is 1.154.
On TP16, speculation shared with background traffic does match.
One-token drafting with seven background requests under `KVP = 4` places both AA values within 5% at an acceptance of 0.51 to 0.74, on 128 or 256 chips.
Each step then commits 1.50 to 1.75 tokens, so a 428 to 491-microsecond batched step gives the published per-token speed.
The 10,996 TPS coding example needs at least two draft tokens on TP16, so no single TP16 draft length covered both within the tested limit of 16 query rows per forward.

## The remaining 100K cost

A critical-path breakdown of TP16/PP8 with `KVP = 4` at attention matrix efficiency 0.50 places the whole 28.35-microsecond 10K-to-100K increment in the attention core of the 10 global layers.
At 100K each global core spends 3.17 microseconds on the vector unit, mostly converting FP8 KV entries to BF16, and 1.35 microseconds on the matrix unit.
The vector unit therefore bounds the core above efficiency 0.30, which is why higher matrix efficiency did not reduce the increment.
Without duplicated KV heads, each chip's attention work is the layer's work divided by `N`, so DCP inside a fixed TP16 group cannot reduce it further.

## Attention-group size

The [attention-group control](lpx-attention-group.md) doubles `N` with TP32/PP4 on the same 128 chips.
With `KVP = 8` the increment falls to 14.2 microseconds at efficiency 0.30 and 0.50, below the 25.2-microsecond bound that one-token decode allows.
Two short background requests then place both AA values within 5% without speculation, at 3,435 to 3,452 TPS and 3,274 to 3,290 TPS.
TP32/PP8 on 256 chips matches with one 16,000-token background request.
`KVP = 2` and `KVP = 4` on TP32 keep replicated KV heads and do not match, and TP32 without DCP does not fit in SRAM.

On TP32/PP4 with `KVP = 8` at efficiency 0.50, speculation matches at 14 of 17 batched points and at 7 serial points.
`K = 3` matches the AA pair with one or three background requests at an acceptance of 0.18 to 0.41.
Without background and draft cost, the same layout reaches the 10,996 TPS coding example at `K = 3` with an acceptance of 0.870.
One draft length therefore covers both on one layout.
No TP32 point matches at efficiency 0.10, the study value.

## Per-workload acceptance

A shared acceptance for the AA 10K and 100K workloads makes the AA speed proportional to the reciprocal of the step time, so it cannot reproduce the published rise from 10K to 100K.
The [workload-acceptance control](lpx-workload-acceptance.md) fits one acceptance to each of the four published speeds and prices the coding prompt under the same background traffic as the AA prompts.
29 of 105 setups reach all four speeds, all with `K = 3` or `K = 7`.
`K = 7` with one background request reaches them on every layout at efficiency 0.50, with AA acceptances of 0.25 to 0.35 at 10K and 0.41 to 0.47 at 100K and coding-example acceptances of 0.82 to 0.87.
Without background, TP32/PP4 with `KVP = 8` at `K = 7` needs 0.147, 0.262, 0.401 and 0.774.
Every fit needs a higher acceptance at 100K than at 10K, and the gap shrinks from 0.20 on TP16/PP8 without `KVP` to 0.07 on TP32/PP4 with `KVP = 8`.
Public reports describe the AA measurement as 50 sequential requests at client concurrency one against a private, pre-release endpoint, so background requests are better read as unidentified fixed time per step than as other users.

## Published acceptance

The [acceptance evidence record](lpx-acceptance-evidence.md) compares the fits with published drafting acceptance.
Capable drafters reach about 0.55 to 0.73 per position on prose at 10K or more input, while the fits need at most 0.48 at AA 10K.
Fits with an AA 10K acceptance below 0.10 fall below every published prior.
A drafter trained with the model, such as Gemma's MTP drafter, keeps acceptance stable with context and cannot make AA 100K faster than AA 10K.
At its acceptance, the AA 10K speed needs steps of 615 to 882 microseconds, 1.3 to 3.2 times the simulated steps with at most one background request.
Lookup drafting matches the direction of every fit but reaches the long-context range only at 100K.
The remaining explanations are more work per AA step than simulated, an underestimated LP30 decode step or a drafter that performs poorly on AA's prompts.

## GroqChip v1 BERT check

The [BERT validation](groqchip-v1-bert-validation.md) prices one BERT-base encoder layer on the first-generation chip and compares it with the 8.9-microsecond layer measured in ASAP 2022.
The LP30 method with the study's matrix efficiency of 0.3 predicts 9.49 to 9.76 microseconds when non-matrix work is hidden, as the paper describes, while the generic Groq model with efficiency 0.8 predicts about half of the measurement.
The check supports the matrix efficiency for compute-bound work on one chip.
It does not test decode weight streaming, collectives, chip-to-chip hops or fixed per-operator costs, where a decode-step underestimate would have to sit.
The paper's 20 TB/s operand bandwidth is one quarter of the chip's quoted 80 TB/s memory bandwidth, which makes weight streaming a candidate.

## GroqChip v1 Llama-2 70B check

The [Llama-2 validation](groqchip-v1-llama2-validation.md) simulates batch-one Llama-2 70B decode on 328 to 656 first-generation chips.
With the direct all-reduce that the production system uses, the simulation gives 1,036 to 1,256 tokens per second, 3.5 to 5.2 times the published 241 to 300.
The default 14-round ring all-reduce lands at 313 to 705 tokens per second, but the production system does not use it.
The published speeds come from an unknown production configuration under unknown load, so the check cannot size the error.
It agrees in direction with the published acceptance: both suggest that the simulated decode step is shorter than real decode on Groq hardware.

## Published-condition sweep

The [article-condition sweep](lpx-article-sweep.md) keeps one request per forward, no speculation, FP8 weights and KV cache, 256 chips and the published C2C link bandwidth of 96 links at 112 Gb/s.
Among TP32/PP8 with `KVP = 1` to `8` and TP16/PP16 with `KVP = 1` to `4`, only TP32/PP8 with `KVP = 8` at attention efficiency 0.30 or 0.50 can match both AA values.
Its steps are 267.3 to 267.4 µs at 10K and 281.5 to 281.6 µs at 100K, so the 10K speed is 10.6% too fast and the 100K speed is 3.5% fast, inside the 5% band.
A constant 14.2 to 25.3 µs per step that the model does not include would place both values within 5%.
Every other layout exceeds the 25.2 µs bound on the 10K-to-100K increment.
The published link bandwidth changed the steps by about 0.1 µs.

## Ablation

The table lists every tested combination and whether it matches both AA values within 5%.
Rows marked "one acceptance per speed" fit all four published speeds exactly instead.

| DCP | Background | Speculation | Attention group | Result |
| --- | --- | --- | --- | --- |
| none | none | none | TP16 | No: 5,578 and 3,611 TPS |
| `KVP = 4` | none | none | TP16 | No: 5,533 and 4,152 TPS |
| none | 7 requests | none | TP16 | No: 100K 2,542 TPS when 10K matches |
| `KVP = 4` | up to 15 requests | none | TP16 | No: best 3,331 and 3,041 TPS |
| none or `KVP = 4` | none | `K` from 1 to 7 | TP16 | No: step-time ratio at least 1.154 |
| none | 1 to 7 requests | `K` of 1, 3 or 7 | TP16 | No: step-time ratio at least 1.249 |
| `KVP = 4` | 7 requests | `K = 1` | TP16 | Yes: 6 points, acceptance 0.51 to 0.74 |
| none | any | none | TP32 | Infeasible: resident KV exceeds SRAM |
| `KVP = 2` or `KVP = 4` | 0 to 7 requests | none | TP32 | No: increment 25.7 µs or more |
| `KVP = 8` | none | none | TP32 | No: 10K speed 12% too high |
| `KVP = 8` | 1 or 2 requests | none | TP32 | Yes: 6 points at efficiency 0.30 or 0.50 |
| `KVP = 8` | none to 7 requests | `K = 1` or `K = 3` | TP32 | Yes: 21 points at efficiency 0.50 |
| `KVP = 8` | any | any | TP32, efficiency 0.10 | No |
| none to `KVP = 8` | none or 1 request | `K = 3` or `K = 7`, one acceptance per speed | TP16 or TP32 | Yes: 29 setups reach all four speeds, including the coding values |
| any | any | `K = 1`, one acceptance per speed | TP16 or TP32 | No: the coding example is unreachable |

With one acceptance shared by both AA workloads, every match uses DCP and some context-independent work per step, from background requests, verification of draft tokens or both.
With one acceptance per published speed, DCP is no longer required, and it reduces the acceptance gap between the AA workloads.
Speculation is required on TP16 and optional on TP32.
TP32 is required for a match without speculation and depends on an attention matrix efficiency of at least 0.30.

## Current position

The AA footnotes establish one request at a time on one named deployment, and press reports give FP8 weights on LP30 chips only.
Under those conditions on one 256-chip rack, only TP32/PP8 with `KVP = 8` at attention matrix efficiency 0.30 or more comes within 5% of both AA values, and only with unmodeled time per step.
At one native token per AA token it needs 14 to 25 µs of unmodeled time per step.
The [token-ratio record](lpx-token-ratio.md) measures 1.031 to 1.048 Gemma tokens per `o200k_base` token on Gemma 4 31B's own writing outputs, which cuts that need to about 1 to 16 µs, and any ratio from 1.053 to 1.090 removes it.
The [tray-latency record](lpx-tray-latency.md) shows the same effect from a cross-tray latency of 0.81 to 0.85 µs instead of 0.75 µs, or about 0.80 to 0.85 µs at the 1.006 tokens per AA token of AA-like book answers.
Speculative decoding matches on the same layout only at AA acceptances of 0.006 to 0.335, below the published acceptance of Gemma's drafter class, so the AA values fit best without effective drafting.
The [lookup-context record](lpx-lookup-context.md) finds that more text to copy from adds less than 0.005 acceptance per doubling beyond 16K tokens, so lookup drafting could produce AA's higher 100K speed only if AA's 100K answers repeat their context more than its 10K answers.
The [lookup-coding record](lpx-lookup-coding.md) shows that lookup drafting at the measured writing and coding acceptances, with 38 to 61 µs of unmodeled time per step, also fits the AA values and the coding median under one build in 13 of 32 settings; the published sources do not decide between that and plain decoding.
The [coding-spread record](lpx-coding-spread.md) finds that plain decoding at one request at a time cannot produce the spread of NVIDIA's coding speeds.
NVIDIA does not state the load of that run, so the spread points to drafting only if the run served one request at a time, and changing background traffic would also spread the speeds.
The [GroqRack Llama-2-7B check](groqrack-llama2-7b-validation.md) supports the multi-chip device model on first-generation hardware once a per-token host loop is added.
The [API-outputs record](lpx-api-outputs.md) generates Gemma 4 31B answers to SPEED-Bench's coding prompts and to AA-like questions about 10K and 100K-token books.
The best of 16 lookup settings gives NVIDIA's coding median, P80 and maximum with 0 to 21 µs of extra time, which shows that lookup drafting can produce them without showing that NVIDIA used it.
The book answers give 1.006 Gemma tokens per `o200k_base` token, which leaves the plain-decoding need at 13 to 24 µs, and the same lookup acceptance at 10K and 100K.
One possibility is one TP32/PP8 build with about 13 to 21 µs of unmodeled time per step in which the coding run drafted and the AA run did not; the model build, thinking mode, drafter and AA's prompts differ from the published runs, so the data does not establish it.
The remaining uncertainty sits in the source of that time, such as the LP30 cross-tray collective latency, and in whether the AA run drafted, and the match does not identify NVIDIA's deployment.

GPU measurement setup is recorded in the [GPU measurements record](lpx-gpu-measurements.md).
The H20 environment, exact input hashes, sequential streaming client and LP30 pricing analysis are available; inference measurements are in progress and do not yet change the conclusions in this page.

## Open work

| Item | Purpose |
| --- | --- |
| LP30 cross-tray collective latency | Measure an all-reduce across two LP30 trays; the TP32 match holds for about 0.80 to 0.85 µs at the 1.006 Gemma tokens per `o200k_base` token of AA-like outputs |
| LP30 operand bandwidth | Establish whether LP30's matrix-unit operand bandwidth is below its 150 TB/s SRAM bandwidth, as GroqChip v1's 20 TB/s is below its 80 TB/s |
| LP30 FP8 attention path | Establish whether KV conversion runs on the vector unit, which decides whether a larger attention group helps |
| Gemma 4 31B measurements on GPUs | Measure the acceptance of Gemma's own MTP drafter at 10K and 100K, the per-request speed spread it causes and the fixed costs of a real decode step, following the [GPU TODO](lpx-gpu-gemma-todo.md) |
| AA drafting status | Establish whether the AA run drafted, from AA's prompts, its per-request speed spread or a statement of the deployment's settings; lookup acceptance measured on AA-like book answers would make the AA values too fast without 26 to 99 µs of extra time |
| Lift the 16-row forward limit | Test longer drafts and heavier background traffic in one forward |

## Earlier records

These records come from the study phases of October 19 to 22, before the context-parallel work, and set up the model, evidence and baselines that the records above build on.

| Record | Content |
| --- | --- |
| [Study guide](lpx-study-guide.md) | Study plan, commands and metrics for the AA calibration and the Figure 11 deployments |
| [Groq hardware evidence](groq-hardware-evidence.md) | Groq literature and LP30 evidence audit |
| [Parameter evidence audit](lpx-parameter-evidence-audit.md) | Recorded LP30 and LPX parameters with their sources and status |
| [Public-evidence and model-validity audit](lpx-public-model-validity-audit.md) | Modeling tasks that public LP30 evidence requires |
| [Fidelity model](lpx-fidelity-model.md) | LP30 execution, fabric, workspace and token-unit model implementing those tasks |
| [Timing recommendations](lpx-timing-recommendations.md) | Settings for the coarse operator model and their evidence limits |
| [Model corrections](lpx-model-corrections.md) | Corrections to AF collectives, GPU graph prefill and QPS propagation, with retests |
| [Coverage-gap analysis](lpx-coverage-gap-analysis.md) | What the Figure 11 experiment covers and where it diverges |
| [Architecture study](lpx-aa-architecture-study.md) | Gemma 4 architecture and inventory exploration |
| [FP8 study](lpx-fp8-study.md) | The Gemma experiment with FP8 weights |
| [Compute-efficiency sensitivity](lpx-compute-efficiency-sensitivity.md) | Decode study with matrix efficiency 0.3 and attention efficiency 0.1 |
| [Context sensitivity](lpx-context-sensitivity.md) | Numerical explanations tested against the 10K and 100K speeds |
| [Conventional AF](lpx-conventional-af.md) | All attention on Rubin and all FFNs on LPX |
| [Native Gemma AF search](lpx-native-gemma-af-search.md) | AF layout search with the native Gemma model |
| [Native global-attention AF](lpx-native-global-attention-af.md) | AF with only the global attention layers on Rubin |
| [Native minimum-chip search](lpx-native-minimum-search.md) | Smallest LP30-only deployment with balanced stages |
| [Native context ordering](lpx-native-context-ordering.md) | Parameters that would make the 100K speed exceed the 10K speed, as published; operator startup and a per-request pause can, without a confirmed physical cause |
| [Execution boundaries](lpx-execution-boundaries.md) | Execution boundaries and placement sensitivity of the minimum search |
| [Latency ledger](lpx-latency-ledger.md) | Reconciliation of isolated and full-model decode latency |
| [Compute and communication overlap](lpx-compute-communication-overlap.md) | Output-vector overlap at 320-byte granularity |
| [Colocated dense reuse](lpx-colocated-dense-reuse.md) | Local tensor reuse when A and F share chips |
| [Integration verification](lpx-integration-20261021.md) | Branch integration checks of October 21 |
| [Context-parallel plan](lpx-context-parallel-plan.md) | Implementation plan for the context parallelism that the records above use |

## Commits

The work is on branch `lpx-experiment`, pushed to `origin/lpx-context-parallel`.

| Commit | Content |
| --- | --- |
| `f39fce1` | Context-parallel plan and KV-parallel configuration |
| `befd8a0` | KV-parallel decode and prefill with results |
| `69e767e` | Placement plausibility record |
| `bfa944b` | TP8 KV-parallel confirmation |
| `15e86e9` | Batched-serving control |
| `0aec3d4` | Speculative-decoding control |
| `d90008a` | Progress summary |
| `fddccce` | Attention-group control |
| `ac2491e` | Per-workload acceptance fits |
| `787fd5d` | Published drafter acceptance |
| `1916fcd` | GroqChip v1 BERT validation |
| `72967b2` | GroqChip v1 Llama-2 70B check |
| `a64c090` | AA measurement windows and press report in the evidence audit |
| `2cc4e1c` | AA test-configuration record |
| `de93f75`, `4b7692b` | Article-condition sweep: expectations and results |
| `8d7bb73`, `09991d1` | Article-condition speculation: expectations and results |
| `4909029`, `a176a52`, `c397c75`, `971c161` | Token ratio on public text and on Gemma outputs |
| `c34e57c`, `804b236` | Cross-tray latency sensitivity |
| `e548087`, `628e72d` | GroqRack Llama-2-7B check |
| `0c2e690` | Current position after the published-condition checks |
| `5454a49`, `3e26048` | Prompt-lookup acceptance on Gemma outputs |
| `851dbaf` | Script reproducing the derived published-condition tables |
| `a187b44` | Documentation of unreferenced inputs, shared scripts and earlier records |
| `92bc22e`, `4499b4b` | Lookup acceptance against context length |
| `e35a1ac`, `1170ced` | Lookup acceptance on coding outputs |
| `a5f48a6`, `90cd654` | Coding speed spread |
| `9a45292`, `1486ad7` | Gemma outputs on SPEED-Bench and AA-like prompts |
| `528eaf4`, `9d4f257` | Qualified conclusions on the API outputs, coding spread and lookup coding |
