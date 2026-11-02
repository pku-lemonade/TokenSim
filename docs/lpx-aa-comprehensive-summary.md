# Reproducing the AA Gemma 4 31B LPX speeds: comprehensive summary

Record date: 2026-11-01, updated the same day with the story C checks and on 2026-11-02 with AA's published prompt methodology and the sensitivity to LP30 attention costs.

This record describes, in one self-contained document, the whole effort to reproduce the Artificial Analysis (AA) output speeds of Gemma 4 31B on NVIDIA Groq 3 LPX hardware with the TokenSim simulator.
It covers every experiment that was run, the result of each, the expectations that held or missed, the corrections made along the way, and our interpretation of where the evidence stands.
It does not depend on any other record: every number needed to follow the argument is stated here.

## Contents

- [1. The question and the published numbers](#1-the-question-and-the-published-numbers)
- [2. Model, hardware and simulator](#2-model-hardware-and-simulator)
- [3. Working rules](#3-working-rules)
- [4. The starting point](#4-the-starting-point)
- [5. Exploration without the published test conditions](#5-exploration-without-the-published-test-conditions)
- [6. What the published sources establish about the AA run](#6-what-the-published-sources-establish-about-the-aa-run)
- [7. One setup for all four published numbers](#7-one-setup-for-all-four-published-numbers)
- [8. Attention and feed-forward separation](#8-attention-and-feed-forward-separation)
- [9. Checks against measured Groq hardware](#9-checks-against-measured-groq-hardware)
- [10. Sweeps under the published test conditions](#10-sweeps-under-the-published-test-conditions)
- [11. Token ratio and tray latency](#11-token-ratio-and-tray-latency)
- [12. Lookup drafting measured on public Gemma outputs](#12-lookup-drafting-measured-on-public-gemma-outputs)
- [13. Gemma outputs generated for the published workloads](#13-gemma-outputs-generated-for-the-published-workloads)
- [14. Where the missing time per step could come from](#14-where-the-missing-time-per-step-could-come-from)
- [15. Is the faster 100K speed noise?](#15-is-the-faster-100k-speed-noise)
- [16. Gemma's own MTP drafter measured on a GPU](#16-gemmas-own-mtp-drafter-measured-on-a-gpu)
- [17. One extra time per step under MTP](#17-one-extra-time-per-step-under-mtp)
- [18. One drafter, three kinds of content](#18-one-drafter-three-kinds-of-content)
- [19. Lookup on all 302 code-completion prompts](#19-lookup-on-all-302-code-completion-prompts)
- [20. AA at the drafter setting that reaches the example](#20-aa-at-the-drafter-setting-that-reaches-the-example)
- [21. Where the evidence stands](#21-where-the-evidence-stands)
- [22. Our assessment](#22-our-assessment)
- [23. Corrections made along the way](#23-corrections-made-along-the-way)
- [24. Assumptions and limits](#24-assumptions-and-limits)
- [25. Open questions and the tests that would answer them](#25-open-questions-and-the-tests-that-would-answer-them)
- [26. Overview of every tested combination](#26-overview-of-every-tested-combination)
- [27. Commit history](#27-commit-history)
- [28. Glossary](#28-glossary)

## 1. The question and the published numbers

### What AA measured

Artificial Analysis benchmarks hosted model endpoints by sending requests from one client and timing the streamed response.
Its output speed counts output tokens per second (TPS) after the first token arrives, so it excludes the time to first token and the prefill of the prompt.
AA counts tokens with the `o200k_base` tokenizer for every model, not with the model's own tokenizer.
Each published value is the median of 50 requests.

| Workload | Input tokens | Output tokens | Median TPS | Time per AA token |
| --- | ---: | ---: | ---: | ---: |
| AA 10K | 10,000 | 1,500 | 3,382 | 295.7 µs |
| AA 100K | 100,000 | 2,000 | 3,431 | 291.5 µs |

Both AA chart footnotes read: "NVIDIA Groq 3 LPX: median of 50 sequential client requests (concurrency 1) to the updated private, pre-release Gemma4-31B-LPX deployment via Google Cloud PSC".
PSC is Google Cloud's Private Service Connect.
Both charts are labelled "As of Aug 21, 2026".

AA's performance methodology page describes the prompts of its public benchmark.
It states that "every test run uses a unique prompt that we generate at the time of the test", that prompts "combine a range of long-form input content (e.g., articles)", and that tasks include explanation and summarization, question and answer generation, comparative analysis, translation and visual artifact generation.
The 10K workload asks for at least 1,500 answer tokens and the 100K workload for at least 2,000.
The page notes that speculative decoding makes output speeds vary with the type of output.
It does not name the article sources or the generation method, and it does not say whether the 100K prompts share the task mix of the 10K prompts.

The 100K speed is 1.45% higher than the 10K speed.
Attention over a longer context cannot be cheaper in a fixed setup, so this rise is either measurement variation or a sign that the two workloads ran differently.
Every simple simulation produces a clearly falling curve, while the published curve is flat.

### Other numbers published under the same deployment name

An NVIDIA slide presented at Hot Chips 2026 shows a model selector reading "Gemma4-31B-LPX", a callout of "10,996 TPS / User", the note "Example Coding Gemma 4 31B – 16K ISL Out of Max 264K Context" (ISL is input sequence length) and the note "SPEED-bench coding running at 4,767 median output tokens / second".
NVIDIA separately reports an 80th percentile of 5,520 TPS and a maximum of 6,981 TPS for the same SPEED-Bench coding run.
SPEED-Bench is NVIDIA's speculative-decoding benchmark, with a short-prompt qualitative split and long-prompt throughput splits.

| Published speed under the Gemma4-31B-LPX name | TPS | Time per token |
| --- | ---: | ---: |
| Hot Chips coding example, 16K input | 10,996 | 90.9 µs |
| SPEED-Bench coding maximum | 6,981 | 143.2 µs |
| SPEED-Bench coding 80th percentile | 5,520 | 181.2 µs |
| SPEED-Bench coding median | 4,767 | 209.8 µs |
| AA, 10K input | 3,382 | 295.7 µs |
| AA, 100K input | 3,431 | 291.5 µs |

Speed under one name varies by a factor of three across workloads and by 46% within one benchmark.
Plain decoding speed depends on sequence lengths and batch size and not on what the text says, so this spread points to a mechanism that depends on content, most plausibly speculative decoding.
At 91 µs per token, the 10,996 TPS example is far beyond any simulated setup that commits one token per forward pass.

The link between AA and the slide is a shared deployment name only.
The slide states no chip count, precision, batch size, token counting, sampling settings or drafting status.
AA calls its deployment "updated" and "pre-release", so the slide and AA may describe different builds.
A conference slide is also likely to show a selected case.
We therefore treated the 10,996 TPS example as a consistency check and as evidence that the deployment drafts tokens, without making it a hard constraint on the AA setup.

Another slide in the same talk, "External-Drafter Speculative Decode", shows an LPX rack drafting for a GPU rack that verifies a large model.
It confirms that NVIDIA presents speculative decoding as an LPX serving pattern, though it says nothing about the Gemma endpoint.

### The match criterion

A simulated point matches when both AA speeds lie within 5% of the published values.

| Workload | Lowest TPS | Highest TPS | Longest step | Shortest step |
| --- | ---: | ---: | ---: | ---: |
| 10K | 3,213 | 3,551 | 311.2 µs | 281.6 µs |
| 100K | 3,259 | 3,603 | 306.8 µs | 277.5 µs |

Two bounds follow and recur throughout the work.

- **The 25.2 µs increment bound.**
  When each step commits one token, the 100K step can be at most 25.2 µs longer than the 10K step: 306.8 µs minus 281.6 µs.
- **The 1.0896 ratio bound.**
  When both workloads commit the same mean number of tokens per step, as with one shared acceptance rate, the token count cancels and the 100K step can be at most 3,551 / 3,259 = 1.0896 times the 10K step.

## 2. Model, hardware and simulator

### Gemma 4 31B as simulated

The simulated model is the text part of Gemma 4 31B with FP8 weights, an FP8 key-value (KV) cache and BF16 arithmetic inside attention.
It has 60 decoder layers with a hidden size of 5,376.
Fifty windowed layers attend only to the most recent 1,024 tokens, so their cost hardly depends on the input size.
Ten global layers attend to the whole context, each with 32 query heads, 4 KV heads and a head dimension of 512.
The global layers carry all of the cost that grows with context.
The small number of global KV heads is central: tensor parallelism splits attention by head, and four KV heads can be split at most four ways without copying.

### The LP30 chip and the LPX rack

LP30 is the compute chip of NVIDIA Groq 3 LPX, descended from Groq's language processing unit.
It keeps weights and the KV cache in on-chip static random-access memory (SRAM), about 500 MB per chip in the simulated budget.
A full LPX rack holds 256 chips in 16-chip trays.

| Unit | Role | Simulated rate |
| --- | --- | --- |
| Matrix unit (MXM) | Matrix multiplications, including the score and value products of attention | Peak times an efficiency; attention matrix efficiency defaults to 0.10 |
| Vector unit (VXM) | Elementwise work, softmax, normalization, data-type conversion, exponentials | 20 trillion operations per second, 1 trillion exponentials per second, 0.5 efficiency |
| Switch and shuffle unit (SXM) | Moving and reshaping data between units | 10 TB/s at 0.8 efficiency |
| SRAM | Reading weights and the KV cache | On-chip bandwidth at 0.9 efficiency |

An operator is a pipeline over tiles: its time is the busiest unit's service time, plus the other units' time divided by the number of tiles, plus a 0.05 µs fill-and-drain overhead.
The chip-to-chip fabric costs 0.35 µs per hop inside a tray and 0.75 µs across trays, and the published-conditions sweeps add 1.15 µs across the rack.
None of these rates has been calibrated against a real LP30, and every result is conditional on them.

### How a step is simulated

TokenSim's native simulator builds the dependency graph of one forward pass across every chip, with per-layer attention and feed-forward work, collectives such as all-reduces between tensor-parallel ranks, and transfers between pipeline stages.
All-reduces are refined into 320-byte vectors that stream as the final matrix multiplication produces them, which models how LPX overlaps communication with compute.
The forward is a compiled static schedule with no per-operator host launch cost.

The AA request is placed at its mean generation context: 10,000 plus half of 1,500 gives 10,750 tokens, and 100,000 plus half of 2,000 gives 101,000 tokens.
Its KV reservation covers the maximum context of 11,500 or 102,000 tokens, and admission rejects any setup whose weights, KV cache and scratch memory do not fit every chip's SRAM.
When each forward commits one token, the speed is the reciprocal of the step time, and this mean-context shortcut reproduces the full integration over the output to within 0.1 TPS.
Prefill is excluded from every speed comparison because AA excludes it.
A one-off prefill of a 100,000-token prompt in 16-token chunks confirmed that the setup fits and runs: 6.76 simulated seconds with context parallelism of 4 against 6.40 seconds without.

### Parallelism notation

- **TP**, tensor parallelism, is the number of chips that split each layer's matrices within one pipeline stage.
- **PP**, pipeline parallelism, is the number of stages the 60 layers are divided into, so chips = TP × PP.
- **DCP**, decode context parallelism, written `KVP` in the simulator, is how many chips inside a TP group split one global layer's KV cache by token position.
  It adds no chips; the TP group is factored as TP = TPA × DCP, where TPA chips split the KV heads.

## 3. Working rules

Every experiment after the first exploratory runs followed the same rules.

- Expectations were written and committed before each run, and each result section reports which held and which missed.
- Every setting that ran is reported, including failed and capped runs.
- A fit chosen after seeing the results is labelled as such and called weak evidence, never a reproduction.
- Each proxy experiment lists how it differs from the published runs: model build, prompts, thinking mode, drafter, sampling and hardware.
- Long sweeps ran one at a time under a 5 GB memory limit, because the study host has 7 GB of memory and no swap.
- GPU measurements ran on a separate machine with one NVIDIA H20, operated by another agent from a written task list, and were synchronized into the repository.

## 4. The starting point

An earlier minimum-chip search had selected TP16/PP8 on 128 LP30 chips, at batch one and without speculative decoding.

| Setup | 1K TPS | 10K TPS | 100K TPS | 100K / 10K |
| --- | ---: | ---: | ---: | ---: |
| TP16/PP8, 128 chips, batch one | 5,908.78 | 5,578.41 | 3,611.17 | 0.647 |
| Published | — | 3,382 | 3,431 | 1.0145 |

The simulation took 179.3 µs per token at 10K and 276.9 µs at 100K.
It was 65% too fast at 10K and 5% too fast at 100K, and its 100K step was 97.7 µs longer than its 10K step against a published difference of −4.2 µs.
No uniform correction to hardware costs can fix an error that differs this much between the two contexts: scaling every cost slows both and pushes 100K further below its target.
The error is mostly context-dependent and comes from the ten global layers, so the work asked which serving setup, together with the cost model, could produce a flat curve near 3,400 TPS.

Before testing mechanisms, we wrote down six reasons why the dedicated batch-one setup was unlikely, ordered by strength.
The simulated speed cannot rise with context; the errors do not fit a uniformly scaled cost model; the same name is reported far faster on coding; the endpoint might share forward passes with other traffic; the chip count follows from assumed parameters; and the token conversion was assumed to be one to one.
Later evidence removed the shared-traffic reason, measured the token ratio, and left the coding reason standing if the coding run served one request at a time.

## 5. Exploration without the published test conditions

### Decode context parallelism

With TP16, each global layer's 32 query heads are spread over 16 chips while the layer has 4 KV heads, so each KV head's cache is held and read on 4 chips.
At 100K this duplicated work, mainly converting FP8 cache entries to BF16 on the vector unit, dominated the step.
DCP splits the cache by token position instead: every chip in a DCP group attends over its own share of tokens, and the partial outputs are merged exactly with their log-sum-exp (LSE) values, the same rescaling FlashAttention uses.
NVIDIA researchers published the idea as Helix parallelism, and vLLM implements it as decode context parallelism.

In the simulator, only the global layers are split by default, tokens are assigned to ranks in round-robin blocks of 16 positions, each rank's attention is priced over its own keys, admission charges each rank its peak KV, query, key and value projection weights are replicated across a DCP group, and the merge is an all-to-all followed by the usual all-reduce.
With DCP of 1 every result reproduced the previous simulator field by field.
Measured values exceeded the recorded expectations by 0.18% to 0.42%, within the 3% tolerance.

| TP16/PP8, 128 chips | 1K TPS | 10K TPS | 100K TPS | 100K / 10K |
| --- | ---: | ---: | ---: | ---: |
| DCP 1 | 5,908.78 | 5,578.41 | 3,611.17 | 0.647 |
| DCP 2 | 5,759.05 | 5,560.63 | 4,167.93 | 0.750 |
| DCP 4 | 5,730.77 | 5,533.30 | 4,152.44 | 0.750 |
| DCP 4, contiguous ownership | 5,632.84 | 4,945.85 | 4,141.26 | 0.837 |
| DCP 4 on all 60 layers | 4,913.07 | 4,767.24 | 3,705.58 | 0.777 |

DCP raised the 100K speed by about 15% and cost up to 3% at short contexts.
DCP 4 was no better than DCP 2, because once duplication is gone the per-chip attention work depends only on the size of the whole TP group.
Contiguous ownership hurt at 10K, where one rank holds every key, and splitting the windowed layers cost 29 µs per token at every context.
Without duplication the busiest chip's KV fell from 211.5 MB to 30.8 MB, which let TP16/PP6 with DCP 8 on 96 chips reach 5,716.56, 5,518.62 and 4,143.38 TPS, the smallest layout found that clears the speed targets.
On TP8, DCP made every context 1.8% to 3.0% slower, because the matrix unit already sets the attention time there.

Our reading: DCP is a real improvement for long contexts and appears in every later match, but it raised the 100K/10K ratio only to 0.750 and left 10K about 64% too fast.

### Background requests

A background request stands for another user's request decoded in the same forward pass, at a fixed context of 1,000, 4,000 or 16,000 tokens.
The streaming all-reduce refinement was extended to multi-row outputs so that batched steps could be priced.
A sweep of 156 points over four layouts, 0 to 15 background requests, three background contexts and attention efficiencies of 0.10, 0.30 and 0.50 found no match.

About seven short background requests bring 10K to 3,382 TPS, but the AA request's own 10K-to-100K increment stays in every step.
With DCP 4 that increment is 60.1 µs at efficiency 0.10 and 28.6 µs at 0.50, so whenever 10K matches, 100K is 10% to 26% too slow.
The best point gave 3,331 and 3,041 TPS.
SRAM also limits the batch: without DCP, TP16/PP8 rejects larger backgrounds that DCP 4 admits.

Our reading: context-independent time per step is the only lever found that sets the 10K level, but it cannot remove the AA request's own long-context cost.
Background requests also stand in for any fixed time per step, such as host overhead or launch costs, so a match shows that time is needed without showing where it comes from.

### Speculative decoding

A drafter proposes K tokens, the large model verifies the current token plus the K guesses in one forward pass over K + 1 rows, and keeps the longest agreeing prefix plus its own next token.
With independent per-position acceptance α, one step commits τ = (1 − α^(K+1)) / (1 − α) tokens on average.
Verification is cheap at decode because the forward pass is dominated by reading weights and the KV cache, which all rows share.

At batch one on TP16, one shared acceptance cancels from the step-time ratio, and the smallest ratio found was 1.154, above the 1.0896 bound, so no TP16 point matched.
The coding example needs at least two draft tokens: two reach it at acceptance 0.73 to 0.90, and three to seven need 0.67 to 0.73 at efficiency 0.50 or more.
The coding median needs at most 0.32.

Batched with background requests, 6 of 54 TP16 points matched both AA values with one shared acceptance, all with one draft token, seven background requests and DCP 4, at acceptance 0.505 to 0.743.
Each step commits 1.5 to 1.75 tokens, so a 428 to 491 µs step still gives about 295 µs per token, while the AA request's own increment grows only to 33.6 µs, about 7% of the step.
The 16-row forward limit of the study host excluded the K = 3 points most likely to match.

Our reading: speculation dilutes the context-dependent cost with fixed work per committed token, and it is the only mechanism that could make 100K faster, through a higher acceptance on the long prompts.

### Anatomy of the 100K cost

A critical-path breakdown of TP16/PP8 with DCP 4 at efficiency 0.50 placed the whole 28.35 µs increment in the attention core of the global layers: 0.404 µs per global layer at 10K and 3.24 µs at 100K.
At 100K the vector unit takes 3.17 µs per global layer, 82% of it converting 25.9 million FP8 values to BF16, while the matrix unit takes 1.35 µs at efficiency 0.50 and 6.72 µs at 0.10.
From efficiency 0.30 upward the vector unit sets the time, which is why higher matrix efficiency stopped helping.
Inside a TP group of fixed size N, per-chip attention work is the total divided by N, so only a larger group can shrink it.
The cost formula predicted an increment of 14.2 µs for TP32 with DCP 8, the only layout clearly under the 25.2 µs bound.

### A larger attention group: TP32

All 237 points of a TP32 sweep ran, and admission rejected 74: TP32 without DCP does not fit, because every KV head would be copied 8 times.

| Layout | Efficiency | 10K step | 100K step | Increment |
| --- | ---: | ---: | ---: | ---: |
| TP16/PP8, DCP 4 | 0.50 | 176.4 µs | 204.8 µs | 28.4 µs |
| TP32/PP4, DCP 4 | 0.30–0.50 | 263.9 µs | 289.6 µs | 25.7 µs |
| TP32/PP4, DCP 8 | 0.10 | 265.7 µs | 295.8 µs | 30.1 µs |
| TP32/PP4, DCP 8 | 0.30–0.50 | 263.4 µs | 277.6 µs | 14.2 µs |
| TP32/PP8, DCP 8 | 0.30–0.50 | 267.4 µs | 281.6 µs | 14.2 µs |

Every increment matched the formula to 0.1 µs.
Six points matched without speculation: TP32/PP4 with DCP 8 and two short background requests, and TP32/PP8 with DCP 8 and one 16,000-token background request, all at efficiency 0.30 or 0.50.
With speculation at efficiency 0.50, 14 of 17 batched points matched, and at batch one K = 3 matched AA at an acceptance of about 0.05 while reaching the coding example at 0.870 on the same layout.
Nothing matched at efficiency 0.10, because verification multiplies matrix-unit work by the number of rows; the expectation that such points would match was wrong.

### Why TP32 has a longer base step

Removing one category of work at a time from the 10K step showed that TP32/PP4 with DCP 8 spends 99 µs more on collectives than TP16/PP8 with DCP 4, while compute shrinks by only about 7 µs.
The collective cost ratio, 188.5 / 89.5 = 2.11, closely matches the hop-latency ratio 0.75 / 0.35 = 2.14, which suggests that a TP32 group spanning two trays pays the slower cross-tray latency on about 130 small, latency-bound collectives per step.
TP32 is therefore only about 12% too fast at 10K, because its longer base step acts as built-in fixed work, and that base step rests on the unmeasured 0.75 µs cross-tray latency.

## 6. What the published sources establish about the AA run

- AA's default scenario sends one prompt at a time, and its methodology does not control how a provider batches requests on its servers.
- AA's rules for public listings forbid dedicated resources, but this measurement was a "private demonstration endpoint for NVIDIA Groq 3 LPX".
- Press reports describe "a private, pre-release Gemma 4 31B endpoint served through Google Cloud" and "the median of 50 sequential client requests at a concurrency of one", in contrast with shared serverless public endpoints.
- NVIDIA hosted the hardware in its own data centers, The Register reports the model ran "entirely on LPUs" and "at FP8", and NVIDIA mentions "high tensor parallelism" and parallelism for "both model weights and long-context KV cache".
- Both workloads name the same deployment and date, and no source gives the chip count, batch size, number of users or drafting status.

From this we concluded that one AA request was in flight at a time, that the private endpoint probably served nothing else, and that background requests are better read as unidentified fixed time per step than as other users.
Prefill hardware does not affect the numbers, because AA's speed starts after the first token.

## 7. One setup for all four published numbers

The earlier sweeps required one acceptance shared by AA 10K and AA 100K, which makes 100K faster than 10K impossible by construction, while the coding numbers already had their own acceptance.
We relaxed this and gave every published speed its own acceptance.
Every such fit then needs a higher acceptance at 100K than at 10K, because every simulated 100K-to-10K step ratio exceeds the published 1.0145.

The literature splits by drafter type.
Lookup drafters, which copy from the context, can gain from longer context, and one study found n-gram acceptance length growing with sequence length on reasoning work.
Model drafters such as EAGLE usually do worse at long context, with reported acceptance lengths of 1.28 on long inputs and falling acceptance with context.
No paper measures acceptance at 10K against 100K on prompts like AA's.

A 105-point sweep priced AA 10K, AA 100K and the 16K coding prompt under the same background traffic with one acceptance per published speed and no drafting cost; 29 setups reached all four numbers.
The most even fit was K = 7 with one short background request, needing acceptances of 0.25 to 0.35 at AA 10K, 0.41 to 0.47 at AA 100K, 0.48 to 0.55 for the coding median and 0.82 to 0.87 for the example on every layout.
At batch one, TP32/PP4 with DCP 8 at K = 7 needed 0.147, 0.262, 0.401 and 0.774.
No K = 1 setup reaches the example, and DCP is no longer required once acceptances can differ, though it shrinks the 10K-to-100K acceptance gap from 0.199 to 0.073.

We then compared the fitted acceptances with published drafter acceptance on other models.
SPEED-Bench reports n-gram acceptance of 0.13 to 0.36 per position on Llama 3.3 70B, EAGLE-3 reaches 0.44 to 0.81, multi-token prediction (MTP) on Qwen3-Next reaches 0.57 to 0.68 on prose and 0.88 on coding, and long-chat prompt lookup and suffix decoding reach 0.56 to 0.73.
Gemma 4 ships a 500-million-parameter MTP drafter that cross-attends to the main model's KV cache, with speedups of up to 3x and no published acceptance.
No drafter type matched any fit across all four numbers.
Fits needing an AA 10K acceptance below 0.10 fall below every prior, and at MTP-class prose acceptance the AA steps would have to be 1.3 to 3.2 times longer than simulated.
This left three explanations: the AA endpoint did more work per step than simulated, the cost model underestimates the decode step, or the drafter did poorly on AA's prompts.

## 8. Attention and feed-forward separation

Running attention on Rubin GPUs and the feed-forward layers on LPX, or the reverse, was tested before the main work under an older cost model, at batch one without speculation.
The best variant reached 2,463 TPS at 10K and 2,162 TPS at 100K, against 3,538 and 3,417 TPS for pure LPX under the same model.
Every token crosses the GPU-LPX link 20 times, about 42 µs, and GPU global attention grows by 57 µs from 10K to 100K because GPUs read the KV cache from high-bandwidth memory.
We judged a retest low priority: the Hot Chips slide shows only an LPX rack, and the press describes the model running entirely on LPUs.

## 9. Checks against measured Groq hardware

### GroqChip v1 BERT layer

A published measurement of quantized BERT-base on one first-generation Groq chip gives 8.9 µs per encoder layer, about 28% of the 750 TOPS INT8 peak.
The LP30 method with the study's matrix efficiency of 0.3 predicts 9.49 to 9.76 µs, within 6.5% to 9.5%, when non-matrix work is hidden as the paper describes.
TokenSim's generic Groq model, at 0.8 matrix efficiency, is about twice too fast.
The check supports the matrix efficiency for compute-bound work on one chip and tests none of decode weight streaming, collectives or fixed costs.
No expectation was recorded before this run, so it is exploratory.

### GroqCloud Llama-2 70B

GroqCloud's published Llama-2 70B speeds are 241 to over 300 TPS per user, with an unknown configuration.
With direct all-reduces, which Groq's production system uses, simulated batch-one decode on four plausible layouts gives 1,036 to 1,256 TPS, 3.5 to 5.2 times faster.
With TokenSim's default 14-round ring the simulation lands in the published range, but only through an algorithm production does not use.

### GroqRack Llama-2-7B

A 2026 paper measures batch-one Llama-2-7B decode on a known GroqRack configuration of 72 first-generation cards at 2.98 to 3.05 ms per token at every length from 100 to 1,600 tokens.
Each token is sampled on the host CPU and broadcast before the next invocation.
The simulated device forward takes 0.224 to 0.278 ms with direct all-reduces, leaving a constant 2.70 to 2.83 ms that the host loop explains.
Adding that loop to the simulated 70B steps gives 264 to 286 TPS, inside GroqCloud's published 241 to 300.
Four of five expectations held.

Our reading: the multi-chip device model is consistent with first-generation hardware once a host loop is added, which weakens the earlier suggestion that the device model underestimates decode.
At about 290 µs per token, the AA endpoint cannot have a millisecond-scale host round trip per token.

## 10. Sweeps under the published test conditions

### Batch one, no speculation

This sweep kept only setups consistent with the published conditions: one AA request per forward, no speculative decoding, one layout for both workloads, LP30 only, FP8 weights and KV cache, at most 256 chips, TP32 or TP16, DCP of 1 to 8, the article's link bandwidth of 96 links of 112 Gb/s per chip, and link latencies of 0.35, 0.75 and 1.15 µs for the tray, 128-chip and rack tiers.
For each point it reports the range of constant extra time per step that would place both AA speeds within 5%.
All six expectations held.

| Layout, 256 chips | Efficiency | 10K step | 100K step | Increment | Extra time for a match |
| --- | ---: | ---: | ---: | ---: | --- |
| TP32/PP8, DCP 1 | 0.10–0.50 | 271.0–271.3 µs | 366.1–366.3 µs | 95.0 µs | none |
| TP32/PP8, DCP 4 | 0.30–0.50 | 267.7–267.8 µs | 293.5 µs | 25.7 µs | none |
| TP32/PP8, DCP 8 | 0.10 | 269.5 µs | 299.6 µs | 30.1 µs | none |
| **TP32/PP8, DCP 8** | **0.30–0.50** | **267.3–267.4 µs** | **281.5–281.6 µs** | **14.2 µs** | **14.2–25.3 µs** |
| TP16/PP16, DCP 4 | 0.30–0.50 | 183.8–184.0 µs | 212.2–212.3 µs | 28.4 µs | none |

One setup survives: TP32/PP8 with DCP 8 on 256 chips at attention efficiency 0.30 or more, with 14 to 25 µs per step that the model does not include.
Without that time its 100K speed is 3,551 TPS, 3.5% fast and inside the band, while its 10K speed is 3,740 TPS, 10.6% too fast.
We call this the closest setup, and every later pricing uses it at efficiency 0.5.
The coding numbers do not fit it without drafting: the 16K coding step is 268.3 µs, so the median needs 1.35 to 1.40 committed tokens per step and the example 3.11 to 3.23.

### Speculative decoding under the same conditions

Three drafter models were priced on top of the verification step: lookup with no drafting cost, and Gemma's MTP drafter at 5% or 10% of a one-token step per draft token, with one global layer's attention per draft token.
Of 378 combinations, 11 matched both AA values with one shared acceptance, all on TP32/PP8 with DCP 8, at AA acceptances of 0.006 to 0.335.
MTP adds 13 to 27 µs per draft token, nearly equal at both contexts, which plays the role of the missing fixed time.
At MTP's published prose acceptance, the 10K value would need 111 to 526 µs more per step than the model gives.
Five of seven expectations held fully.
Under the published conditions, the AA values fit best with no drafting or with drafting that almost never succeeds on AA's prompts.

## 11. Token ratio and tray latency

### Gemma tokens per AA token

If Gemma produces r native tokens for each `o200k_base` token of the same text, the endpoint generates r times as many native tokens as AA counts, and the simulated step must be r times shorter.

| Text | Gemma tokens per `o200k_base` token |
| --- | ---: |
| Novel | 1.006 |
| Seven Wikipedia articles | 1.035 |
| Reasoning traces of another model, thinking part | 1.073 |
| Python source code | 1.170 |
| Gemma 4 31B WritingBench thinking | 1.048 |
| Gemma 4 31B WritingBench final answers | 1.031 |
| Gemma 4 31B AIME mathematics thinking | 1.115 |

At Gemma's writing ratios of 1.031 to 1.048, the closest setup needs about 1 to 16 µs per step instead of 14 to 25 µs, ratios of 1.053 to 1.090 need no extra time, and ratios above 1.09 make the setup too slow.
The later measurement on Gemma's own answers to long-book questions gave 1.006, which leaves 13 to 24 µs.

### Latency between trays

The step grows by about 25 µs per 0.1 µs of cross-tray latency, and the 14.2 µs increment does not change.

| Cross-tray latency | 10K step | AA errors at ratio 1.0 | Extra time for a match |
| ---: | ---: | --- | --- |
| 0.65 µs | 242.5 µs | +22.0%, +13.6% | 39.1–50.2 µs |
| 0.75 µs | 267.4 µs | +10.6%, +3.5% | 14.2–25.3 µs |
| 0.85 µs | 292.3 µs | +1.2%, −4.9% | 0–0.4 µs |
| 1.00 µs | 329.6 µs | −10.3%, −15.2% | none |

A cross-tray latency of about 0.80 to 0.85 µs instead of 0.75 µs supplies the missing time at the book-answer ratio.
All five expectations held.
The surviving match therefore rests on one unmeasured number within a window about 0.1 µs wide.

## 12. Lookup drafting measured on public Gemma outputs

At temperature 0 a lookup drafter's token is accepted exactly when it equals the token the model wrote, so replaying a lookup drafter over recorded outputs measures its acceptance without running the model.
The replay proposes the continuation of the most recent match of the longest n-gram, with a minimum n-gram length of 1 or 2 and a draft length K.

### Writing outputs

On Gemma 4 31B WritingBench outputs with the full prompt available, lookup at K = 7 accepts 0.225 per position, 1.291 tokens per step, and on AIME mathematics it accepts 0.571.
Acceptance rises with the text available for copying by about 0.008 per doubling.
Priced on the closest setup, the writing acceptance makes AA 13% to 24% too fast at 10K and widens the 10K-to-100K gap, because verifying extra rows costs more at 100K; at K = 7 no constant extra time fixes it, and shorter drafts need 34 to 61 µs.
AA's shape would need about 0.1 more acceptance at 100K than at 10K.

### Does more context raise acceptance?

No public data has Gemma 4 31B answering 10K to 100K prompts, so we used 19,792 turns of Gemma 4 31B terminal-agent traces with contexts up to 66K tokens.
For a lookup drafter the sampling temperature does not matter, because verification accepts a token with the probability that the model samples it.

| Context before the turn | Whole context | Last 16K | Last 4K | Last 1K |
| --- | ---: | ---: | ---: | ---: |
| 2K–4K | 0.423 | 0.423 | 0.422 | 0.351 |
| 8K–16K | 0.536 | 0.536 | 0.495 | 0.422 |
| 16K or more | 0.663 | 0.658 | 0.621 | 0.530 |

Long-context turns accept 0.240 more, but 75% of that rise remains with a 1K lookback, because late agent turns repeat their earlier commands.
Beyond 16K tokens, a wider lookback adds 0.005.
Three of four expectations held.
More text to copy from cannot supply the 0.1 that AA's shape needs; only answers that repeat their context more at 100K could.

### Coding outputs

On 500 coding problems answered by Gemma 4 31B in NVIDIA's NVFP4 build, lookup commits 1.415 to 1.906 tokens per step with thinking and 1.286 to 1.510 on the answer alone.
Plain decoding on the closest setup gives about 3,400 to 3,730 TPS on coding, 22% to 29% below the 4,767 median, so either the coding numbers came from another build or the deployment drafts.
Several lookup settings reach the median, none reaches the example (best 5,841 TPS), and 13 of 32 settings fit AA 10K, AA 100K and the coding median with one shared 38 to 61 µs per step.
Three of five expectations held.

### The coding speed spread

NVIDIA's coding statistics give an 80th percentile of 1.158 times the median and a maximum of 1.464 times.
With plain decoding at one request at a time, the step depends only on context length, so priced speeds differ by 0.1%: 1.000 and 1.001.
Lookup gives 1.08 to 1.09 and maxima of 1.26 to 1.53 in random sets of 80 requests.
If NVIDIA served one request at a time, the coding run most likely drafted, though the slide does not state the load and a changing number of other requests would also spread speeds.

## 13. Gemma outputs generated for the published workloads

We generated Gemma 4 31B outputs through OpenRouter on Parasail's FP8 deployment, at temperature 0, for 264 requests costing $0.87.

| Prompt set | Content |
| --- | --- |
| SPEED-Bench coding | The 80 coding prompts of the qualitative split, thinking on and off |
| SPEED-Bench 16K code completion | 40 repository code-completion prompts of the 16K throughput split |
| AA-like documents | 16 public-domain books cut to 10K and 100K tokens, each with a summary question and a quotation question, thinking on |

On SPEED-Bench coding, the best of 16 lookup settings, chosen after the results, is minimum n-gram 1 with K = 3 and thinking off.
Without extra time it gives 4,863, 5,633 and 7,830 TPS against NVIDIA's 4,767, 5,520 and 6,981, within 2.0%, 2.0% and 12.2%, and it fits the median with 0 to 21 µs of extra time.
With thinking on, the median needs 30 to 64 µs and the spread is too narrow.
With 16 settings and a free extra time, matching three statistics is weak evidence of NVIDIA's drafter.

On the book questions the token ratio is 1.006 overall, 1.026 for thinking and 0.995 for answers.
Lookup acceptance at K = 7 does not rise at 100K: summaries move from 0.209 to 0.218 and quotations from 0.315 to 0.305.
Lookup makes AA 10K 11% to 38% too fast without extra time, and fitting both AA values needs 40 to 99 µs per step, or 26 to 78 µs on answers alone as a stand-in for thinking off.

On 16K code completion, lookup at K = 7 commits 1.96 tokens per step and the fastest request reaches 9,785 TPS, 11% short of the example.
Five of seven expectations held, one partly held and the code-completion expectation missed.

These outputs differ from the published runs in the AA prompts (well-known books with our own questions), the model build (a public checkpoint on a third-party FP8 deployment), thinking mode, drafter, sampling, and SPEED-Bench's split and load.
The short coding prompts were priced at 16K context; at their real length the median rises 1.2%.

From these data, one possibility is one build with about 13 to 21 µs of unmodeled time per step in which the coding run drafted and the AA run did not.
We call this story A.

## 14. Where the missing time per step could come from

### Audit of one simulated step

One decode step of the closest setup at 10K has 8,515 events, 393 of them on the critical path: a 4 µs host submission, the embedding, 254 compute operators made of 1,036 priced kernel components, 132 collectives, 7 stage hand-offs, the vocabulary head and sampling.
Compute is about 70 µs of the 267 µs, and collectives take most of the rest.
Three fixed costs are zero in this setup: the collective launch, the per-kernel instruction floor and the per-step invocation charge.

| Change | Step growth | AA 10K | AA 100K |
| --- | ---: | ---: | ---: |
| None | — | +10.0% | +2.9% |
| Collective launch 0.1 µs | +13.2 µs | +4.8% | −1.7% |
| Collective launch 0.5 µs | +66.0 µs | −11.8% | −16.6% |
| Kernel floor 0.025 µs | +25.9 µs | +0.3% | −5.7% |
| Kernel floor 0.05 µs and invocation 1 µs | +52.8 µs | −8.2% | −13.3% |

A collective launch of about 0.10 to 0.18 µs or a kernel floor of about 0.012 to 0.023 µs would supply the 13 to 24 µs, while the values used elsewhere in the repository make the setup too slow.
Returning the token to the first pipeline stage would add at most about 1.2 µs.
Four of six expectations held; the floor costs four times more than expected, because it applies per kernel component.

### Published fixed costs

Groq's production-serving paper and its multi-chip paper describe first-generation hardware.
An all-reduce on 8 chips has a fixed time of about 0.15 to 0.55 µs, close to the hop latency, with no visible launch cost; chip-to-chip links take about 0.24 µs; compile-time scheduling leaves no kernel-launch overhead; and synchronizing chips at the start of each program run costs about 1 to 2 µs per step if every token starts a new run.
The production host handles the initial invocation and the final output collection, which implies a per-token host loop at batch one whose size nobody publishes.
Three expectations held, one partly held and one missed: we had expected the production loop to stay on the chips.

Our reading: if LP30 behaves like the first-generation chip, launches, dispatch and synchronization supply about 1 to 2 µs, and the remaining candidates are a host loop costing roughly 10 to 24 µs more than the modeled 4 µs submission, or a cross-tray latency of about 0.80 to 0.85 µs.

## 15. Is the faster 100K speed noise?

For a median of 50 requests, the standard error of the difference between two medians is about 0.25 times the per-request coefficient of variation.
A per-request spread of 3% makes the 1.45% difference 1.9 standard errors, which is ordinary; 1% makes it 5.8.
Plain decoding at batch one varies about 0.1% in the simulator, so a larger spread would come from the measurement path or from content-dependent decoding.

AA's public Gemma 4 31B endpoints, fetched on 28 October, are 11% to 70% slower at 100K than at 10K, have interquartile ranges of 5% to 57% of the median at 10K, and 6 of 11 are faster at 10K than at 1K, by up to 2.7 times.
A longer prompt cannot speed up decoding, so AA's workloads differ in more than length: load at test time, the generated prompt and output of each test, and content-dependent drafting.
Three expectations held, one missed and one missed in part.
The public endpoints are shared and loaded, so they overstate a private endpoint's spread, but they show that the 1.45% is weak evidence of a real context effect.

## 16. Gemma's own MTP drafter measured on a GPU

Every argument against drafting on the AA run had rested on other models' drafters and on lookup.
Gemma 4 31B ships `google/gemma-4-31B-it-assistant`, so the GPU agent measured it on one H20 with vLLM 0.32.0, against eight recorded expectations.
The runs used the public checkpoint in BF16, a comparison with FP8 weights, K of 1, 3, 5 and 7, the prompts of section 13, one request at a time and temperature 0: 976 MTP requests and 1,048 turns, plus a plain control and vLLM's n-gram drafter.

With vLLM's default attention path, the plain 10K book request looped at temperature 0, repeating "al" until the 16K-token output cap.
It also looped in eager mode, with mixed FlashAttention versions and with the older model runner, while native Transformers and vLLM with Triton attention finished normally.
Every reported measurement therefore uses Triton attention; that choice was made after seeing the failures, and the failed outputs stay in the record.
Twenty-three coding turns with thinking reached the output cap without an answer and remain in the numbers.

Equivalent acceptance is the per-position acceptance that gives the measured mean tokens per step.

| K | Book questions | Coding, thinking off | Coding, thinking on | 16K code completion |
| ---: | --- | ---: | ---: | ---: |
| 1 | 0.781–0.814 | 0.931 | 0.932 | — |
| 3 | 0.746–0.785 | 0.913 | 0.918 | 0.886 |
| 5 | 0.732–0.768 | 0.906 | 0.911 | — |
| 7 | 0.720–0.761 | 0.896 | 0.901 | 0.871 |

- Gemma's drafter accepts more on prose than Qwen3-Next's published 0.57 to 0.68.
- Paired over the 16 books, 100K minus 10K averages −0.016 to +0.002, so acceptance does not rise with context.
- FP8 weights lower book acceptance by 0.008.
- vLLM's n-gram drafter agrees with our lookup replay within 0.0003 on average, which supports the replay used in every lookup experiment.

Priced on the closest setup at 1.006 Gemma tokens per AA token, with a drafter cost of 5% or 10% of a one-token step per draft token, every setting makes AA 10K at least 72% too fast.
Both AA values fit only with 195 to 576 µs of unmodeled time per step, and none fits with 25 µs or less.
On coding, plain decoding gives 3,765 TPS, 21% below 4,767, and every MTP setting gives a median of about 6,400 to 13,200 TPS.
The coding spread stays below NVIDIA's: P80/median is 1.066 for MTP at K = 3 and 1.006 for plain decoding on the H20, and 1.007 to 1.141 priced on LP30.

A profile of the H20 decode step in CUDA-graph mode gives 24.7 ms at 10K and 33.6 ms at 100K, a fixed 761 µs vocabulary projection, 32 µs for logits and sampling, and about 1.1 ms with no GPU activity that the record does not attribute.
These GPU costs do not transfer to LP30.
Seven of eight expectations held; greedy plain and MTP outputs were identical on 5 of 10 coding requests against an expected 9.

## 17. One extra time per step under MTP

If both runs used Gemma's drafter, is there one extra time per step that fits both AA values and the coding median?
Each SPEED-Bench coding turn was priced at its own context, with the verification of K + 1 rows, the drafter, and an extra time swept from 0 to 1,000 µs in 0.5 µs steps.

| K | Coding median needs | AA needs | Shared |
| ---: | --- | --- | --- |
| 1 | 86–140 µs | 195–242 µs | none |
| 3 | 326–451 µs | 387–494 µs | 387–451 µs, summary questions |
| 5 | 505–694 µs | 431–576 µs | 505–576 µs, quotation questions and coding without thinking |
| 7 | 595–848 µs | 407–507 µs, summaries only | none |

At the shared extra time the coding median is 4,601 to 5,003 TPS, its P80/median is 1.03 to 1.09 against 1.158, its maximum/median is 1.08 to 1.17 against 1.464, and the fastest 16K code-completion turn reaches 4,893 TPS against the 10,996 example.
A large constant time per step evens out per-request speeds.
Four expectations held and one missed in part: with no extra time, K = 3 code completion peaked at 10,945 and 9,807 TPS, and K = 7 reached 12,647 to 15,255.

We call this story B: Gemma's MTP drafter on both runs with 387 to 576 µs of unmodeled time per step, 1.4 to 2.2 times the whole modeled step.
Published first-generation data leave only 1 to 2 µs to launches and synchronization, but the GroqRack host loop of 2.7 to 2.8 ms shows that per-token costs outside the forward can be large, so story B cannot be excluded.

## 18. One drafter, three kinds of content

The user proposed a third reading.
One deployment with one drafter whose acceptance depends on content would accept little on prose answers (AA), more when writing code (the 4,767 median), and a lot when completing code inside a repository (the 10,996 example).
The example is a separate 16K code-completion prompt, not the top of the SPEED-Bench run, whose fastest request was 6,981 TPS.

With 13 to 21 µs of extra time per step on the closest setup, each number needs about this many committed tokens per step:

| Published speed | Content | K = 3 | K = 7 |
| --- | --- | ---: | ---: |
| AA 10K and 100K | Prose answers | 1.07–1.2 | 1.25–1.46 |
| Coding median 4,767 | Writing code | 1.51–1.55 | 1.76–1.8 |
| Example 10,996 | Completing code | 3.48–3.57 | 4.06–4.15 |

Gemma's MTP drafter accepts too much on prose to fit this pattern.
A lookup drafter has the right shape, because it copies from the prompt, but on the data available at that point it missed both ends narrowly: the fastest of 40 code-completion requests reached 9,785 TPS, and lookup on the book answers needed at least 26 µs of extra time while the coding median allowed at most 21 µs.
NVIDIA's example is one selected prompt, and the fastest of 302 prompts can exceed the fastest of 40, so we planned a test on all 302.

## 19. Lookup on all 302 code-completion prompts

### Question and expectations

The test asked whether lookup reaches the 10,996 TPS example on some SPEED-Bench 16K code-completion prompt at the extra time per step that also gives the coding median, under one drafter setting.
Seven expectations were committed before any run:

1. The request file built with all 302 completions has SHA-256 `c4815c8625696784735a924c51a042f554c1f27e309539f87126c30c134b776a` and contains the 40 earlier prompts.
2. On the 40 earlier prompts, the H20 answers commit 1.7 to 2.2 tokens per step on average at minimum n-gram 1 and K = 7.
3. Over all 302 prompts, that mean lies within 0.2 of the 40-prompt mean.
4. With no extra time, the fastest of 302 is at least 5% faster than the fastest of 40 and reaches 9,785 to 11,500 TPS.
5. No setting with a nonempty coding window has a request at or above 10,996 TPS at the upper end of that window.
6. Every coding window ends at 21 µs or less.
7. vLLM's native n-gram drafter at K = 7 gives an equivalent acceptance within 0.03 of the replay.

Expectation 5 stated the peak part of the hypothesis as a prediction that it fails, because only one of the 40 earlier requests exceeded 2.8 tokens per step and the example needs about 4.

### Method

On one H20, the public checkpoint in BF16 answered all 302 prompts with plain decoding, thinking off, temperature 0 and a 4,096-token limit.
The run produced 204,651 tokens, 677.65 per request, at a median H20 decode speed of 38.72 TPS; 301 answers stopped normally and one reached the limit and remains in every calculation.
Lookup was replayed at minimum n-gram 1 and 2 and K of 1, 3, 5 and 7 on the 302 answers, on the 40 earlier prompts as a separate set, and on the H20 plain coding answers without thinking.
For each setting the coding window is the range of extra time, on a 0.5 µs grid from 0 to 100 µs, that puts the median of the 80-request, 89-turn coding set within 5% of 4,767 TPS.
Every code-completion request was priced on the closest setup at 0, 13, 21 and 26 µs and at the window's upper end, with each step costed by its actual number of verified rows.
A separate run with vLLM's n-gram drafter at K = 7 on the same 302 prompts checked the replay.

### Results

Fastest code-completion request at the upper end of each setting's coding window:

| Minimum n-gram | K | Coding window | Fastest of 302 | Fastest of the earlier 40 | Requests ≥ 10,996 |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 1 | 0–7 µs | 6,208 | 6,208 | 0 |
| 1 | 3 | 0–23.5 µs | 8,739 | 8,739 | 0 |
| 1 | 5 | 0–17 µs | 9,983 | 9,838 | 0 |
| 1 | 7 | 0–7 µs | **11,606** | 10,751 | 1 |
| 2 | 1 | none | — | — | — |
| 2 | 3 | 0–6 µs | 8,868 | 8,868 | 0 |
| 2 | 5 | 0–6 µs | 10,290 | 9,821 | 0 |
| 2 | 7 | 0–6 µs | **11,735** | 10,508 | 1 |

At minimum n-gram 1 and K = 7 with no extra time, the fastest request reaches 11,841 TPS, and it is still 11,012 TPS at 26 µs, though the coding median no longer fits there.
Over all 302 prompts the median is 4,828 TPS at the 7 µs window end.
The fastest request at K = 7 is the same in both settings: a Python repository task whose answer reproduces a long table that maps message IDs to class names from the repository.
It commits 1,501 tokens in 367 steps, 4.09 tokens per step, and finished normally.
No other request reaches 10,996 TPS.
Computed for this record after the results, the 89 H20 coding turns at the K = 7 window ends give a median of 4,531 to 4,533 TPS, a P80/median of 1.195 at minimum n-gram 1 and 1.154 at minimum n-gram 2 against NVIDIA's 1.158, and a maximum/median of 1.95 and 1.885 against 1.464.
The mean is 1.90 tokens per step over all 302 prompts and 2.02 over the earlier 40, with arithmetic per-request means of 1.80 and 1.93.

vLLM's n-gram drafter at K = 7 gives an equivalent acceptance of 0.443768 against 0.442675 for the replay of its own 302 answers, a difference of +0.001094, and no request differs by 0.03 or more.
The native drafter fills missing draft slots differently and the replay tokenizes exported text without chat-template tokens, so the check supports the replay's acceptance for this configuration without establishing identical decoding paths.

| Expectation | Outcome |
| --- | --- |
| 1: input hash and inclusion | Held |
| 2: earlier-40 mean of 1.7 to 2.2 | Held at 2.02 |
| 3: 302-prompt mean within 0.2 | Held, a difference of 0.12 |
| 4: fastest of 302 at least 5% faster, 9,785 to 11,500 TPS | Missed: 7.97% faster at 11,841 TPS, above the upper bound |
| 5: no request reaches 10,996 at a window's upper end | Missed: one request at each K = 7 setting |
| 6: every coding window ends at 21 µs or less | Missed: minimum n-gram 1, K = 3 ends at 23.5 µs |
| 7: native acceptance within 0.03 of replay | Held, +0.0011 |

The pricing was reproduced independently from the committed replay data with identical values, and the unit tests, formatting and whitespace checks passed.

### What it means

The peak part of the third reading holds on the modeled layout.
A lookup drafter at K = 7 gives both the coding median and the 10,996 TPS example with at most 6 to 7 µs of extra time per step, through one request in 302.
A single hand-picked example on a slide is consistent with that, and the request that reaches it copies heavily from its repository, as the reading predicts.
The result also moves the extra time down: the K = 7 windows end at 6 to 7 µs, below the 13 to 21 µs of story A.

This test used the public checkpoint in BF16 on an H20, SPEED-Bench's prompts of 18,071 to 21,421 tokens priced as 16K inputs, thinking off and temperature 0, and simulated LP30 step times plus a constant extra time.
NVIDIA's build, example prompt, drafter, K and sampling are not published.

## 20. AA at the drafter setting that reaches the example

One deployment uses one drafter setting, so AA has to fit at K = 7 with at most 6 to 7 µs of extra time.

On the book answers, no extra time places both AA values within 5% at either K = 7 setting, and plain decoding on the answers needs at least 15.8 µs.
The reason is the cost of verification at long context: an 8-row step takes 353.5 µs at 10K and 401.6 µs at 100K, against 267.3 µs and 281.5 µs for one row.
Drafting at K = 7 therefore makes 100K slower relative to 10K, while AA measured 100K slightly faster.
At minimum n-gram 1 and K = 7, the summary answers give 282 µs per committed token at 10K, close to AA, and 325.5 µs at 100K, about 10% too slow.
At minimum n-gram 2 and K = 7, the 10K speed is about 12% too fast.

An exploratory bound, computed after the results and not recorded as a separate experiment, simplifies drafting to a full 7-token draft or none on each step.
With at most 7 µs of extra time, both AA values fit only if 11% to 39% of steps carry a draft and the steps commit at most about 1.085 tokens on average.
At a similar equivalent draft rate of about 31%, the book summaries at minimum n-gram 2 commit 1.104 tokens per step, while the bound allows about 1.03 to 1.055.
This suggested that AA's prompts would need roughly half as many accepted tokens per step as the book summaries, while still triggering drafts that are mostly rejected.
The checks below tested that suggestion on the measured answers and did not support it.

### Checks of the AA end and the coding spread

Two checks with seven recorded expectations then tested story C's weak points using data already collected.

The book check priced every book answer at K = 7, at both minimum n-gram settings, in both data sets (thinking and answer at a token ratio of 1.0055, and the answer alone at 0.9950).
A book fits when its 10K and 100K answers both lie within 5% of the AA values at one extra time up to 7 µs at minimum n-gram 1 or 6 µs at minimum n-gram 2.
Books were also sorted by copy share, the mean equivalent acceptance of their two answers, and the medians of the 4, 8 and 16 books that copy least were tested the same way.

| Data set | Minimum n-gram | Summary books fitting | Quotation books fitting | Low-copy subsets with a window |
| --- | ---: | ---: | ---: | --- |
| Thinking and answer | 1 | 1 of 16 | 0 of 16 | none |
| Thinking and answer | 2 | 0 | 0 | none |
| Answer alone | 1 | 0 | 0 | none |
| Answer alone | 2 | 0 | 0 | none |

For the 4 summary answers that copy least, with no extra time:

| Data set | Minimum n-gram | Tokens per step | AA 10K error | AA 100K error | Steps with a draft over all books, 10K and 100K |
| --- | ---: | --- | --- | --- | --- |
| Thinking and answer | 1 | 1.23–1.26 | +8.6% to +11.5% | −6.1% to −7.5% | 78.9% and 87.7% |
| Thinking and answer | 2 | 1.15–1.17 | +14.9% to +16.0% | +0.6% to +1.9% | 32.7% and 43.7% |
| Answer alone | 1 | 1.14–1.17 | +2.5% to +4.6% | −12.0% to −13.4% | 77.2% and 86.9% |
| Answer alone | 2 | 1.08–1.10 | +9.8% to +11.6% | −2.5% to −4.5% | 30.8% and 42.4% |

An extra time slows both contexts, so it cannot repair errors of opposite sign.
At minimum n-gram 1 the drafter proposes on most steps, and the costly 8-row steps make 100K too slow.
At minimum n-gram 2 it proposes less often and the 10K answers are 10% to 16% too fast even at the lowest copy shares; answers copying still less would approach plain decoding, which needs at least 15.8 µs.
The drafter also proposes more often at 100K than at 10K, because a longer context holds more matches, while the committed tokens per step barely change.
The one fitting book ranks 12th of 16 by copy share and fits only at 7.0 µs, because its 100K answer commits 1.34 tokens per step against 1.22 at 10K.

The coding-spread check drew 2,000 sets of 80 of the 89 H20 coding turns, with replacement and a fixed seed, and placed NVIDIA's ratios among them.

| Setting | Extra time | P80/median, 5th–95th percentile | Maximum/median, 5th–95th percentile | Sets at or below NVIDIA's maximum/median |
| --- | ---: | --- | --- | ---: |
| Story C, minimum n-gram 1, K = 7 | 7 µs | 1.111–1.264 | 1.425–1.999 | 15.6% |
| Story C, minimum n-gram 2, K = 7 | 6 µs | 1.110–1.240 | 1.431–1.964 | 14.8% |
| Story A, minimum n-gram 1, K = 3 | 13 µs | 1.103–1.194 | 1.294–1.627 | 41.0% |
| Story A, minimum n-gram 1, K = 3 | 21 µs | 1.102–1.197 | 1.292–1.627 | 41.3% |

NVIDIA's 1.158 and 1.464 lie inside every range, so the wide 89-turn maximum computed earlier came from one heavily copying turn, which a set of 80 omits about 40% of the time.

Four of seven expectations held.
The misses were one fitting book at minimum n-gram 1, where none was expected; that book committing 1.22 tokens per step at 10K, above the expected 1.085; and NVIDIA's maximum/median lying at the 15th percentile under story C instead of below the 5th.

These checks change the AA condition of story C.
Answers that copy little do not fit; at K = 7 with at most 7 µs the AA values fit only when the 100K answers commit about 0.1 more tokens per step than the 10K answers, as for the one fitting book.
On average the measured answers show no such gap, and neither does Gemma's MTP drafter.
Story C therefore needs AA's 100K prompts to lead to answers that copy more from their context than its 10K prompts do, the same condition every earlier fit to the faster 100K speed needed.
The coding spread, by contrast, does not count against story C.

### Sensitivity to the assumed LP30 attention costs

Story C's AA failure comes from how much an 8-row verification step grows from 10K to 100K, and none of the LP30 attention rates behind that growth is measured.
In the LP30 cost model, one row of attention is bound by the vector unit, which converts the FP8 cache to BF16 and computes the exponentials, while 8 rows are bound by the matrix unit at efficiency 0.5.

A further experiment, with seven expectations recorded before it ran, captured verification steps of 1 to 8 rows on the closest setup under four sets of LP30 rates, and added an analytical bound in which every row count grows with context like one row.
It recomputed the plain AA window, story A's window, the coding window of every lookup setting, the fastest of the 302 code-completion requests at each window's upper end, the book pairs that fit both AA values inside the coding window, and a joint condition: one extra time inside the coding window that reaches 10,996 TPS on a code-completion request and places the median of all 16 books of one question style within 5% of both AA values.

| Variant | Rates | 1-row increment | 8-row increment | Story A window, answer alone | Story C at AA |
| --- | --- | ---: | ---: | --- | --- |
| Baseline | Matrix efficiency 0.5, exponentials 10¹² per s, vector 0.5, shuffle 0.8 | 14.2 µs | 48.2 µs | 15.8–23.5 µs | no fit |
| Matrix | Matrix efficiency 1.0 | 14.2 µs | 32.6 µs | 15.8–24.0 µs | no fit |
| Exponential | Exponentials 4 × 10¹² per s | 12.8 µs | 48.1 µs | 16.1–24.0 µs | no fit |
| Fast units | All of the above at peak, vector and shuffle 1.0 | 6.4 µs | 24.1 µs | 18.7–28.5 µs | fits at 7.0–10.0 µs, minimum n-gram 1, summary answers alone |
| Row-independent bound | 8 rows grow like 1 row | 14.2 µs | 14.2 µs | 15.8–24.0 µs | fits at 0–8.0 µs, the same setting |

The baseline reproduced the committed steps exactly.
Three of seven expectations held.
The misses: the faster exponential rate shortened the one-row increment by 1.4 µs, because the vector unit also runs the exponentials; the fast-units variant shortened the one-row 10K step by 3.0 µs, below the expected 5 to 20 µs; and, most important, story C met the joint condition under the fast-units variant, where no simulated variant was expected to allow it, with 6 fitting pairs against the expected at most 2.

The fast-units fit is marginal.
At 7 to 10 µs the 16-book median of the summary answers without thinking lies at +4.0% to +4.9% at 10K and −4.2% to −5.0% at 100K, and the simulated 100K speed stays 7% below the 10K speed.
The quotation answers and the answers with thinking fit under no variant, and no lookup setting other than minimum n-gram 1 at K = 7 meets the joint condition, because the others do not reach the example.

A scan added after these results kept the baseline 10K steps and scaled the extra growth of multi-row steps from none to the full baseline.
Story C fits only when the 8-row increment is at most about 25 µs, half the modeled 48 µs; full matrix efficiency alone gives 32.6 µs.

Story C's AA end therefore needs one of two conditions: AA's 100K answers copying about 0.1 tokens per step more than its 10K answers at the modeled costs, or LP30 multi-row attention at 100K about twice as fast as modeled together with AA answers like summaries without thinking.
Story A keeps a window under every variant, and where story C fits, story A still needs more extra time, so a measured host loop would still separate them.

## 21. Where the evidence stands

Three stories now remain, all on TP32/PP8 with DCP 8 on 256 LP30 chips at attention efficiency 0.5.

| | A: plain decoding on AA, lookup on coding | B: Gemma's MTP on both runs | C: lookup at K = 7 on every run |
| --- | --- | --- | --- |
| Unmodeled time per step | 13–21 µs | 387–576 µs | at most 6–7 µs at the modeled attention costs |
| Both AA values | fit | fit | at the modeled attention costs, only if AA's 100K answers copy about 0.1 tokens per step more than its 10K answers; with 8-row growth from 10K to 100K of at most about 25 µs, marginally on summary answers without thinking |
| Coding median | fits, best of 16 settings chosen after the results | fits | fits |
| Coding spread | NVIDIA's P80 and maximum ratios inside the 5th–95th percentiles of sets of 80 | too narrow | NVIDIA's P80 and maximum ratios inside the 5th–95th percentiles of sets of 80 |
| 10,996 TPS example | not reached; best 8,806 of 302 at 21 µs | not reached; 4,893 | reached by one request in 302 |
| Drafter | weaker than Gemma's MTP, on coding only | Gemma's released MTP | lookup, on every run |
| Weakest point | two drafting behaviours in one deployment | an unmodeled cost larger than the modeled step | a copy gap between AA's two prompt sets that the data do not show, or multi-row attention at 100K twice as fast as modeled; a peak resting on one request |

Results we consider reasonably firm:

- AA measured one request at a time on one private deployment.
- Gemma produces about one native token per AA token on prose answers: 1.006 on book answers and 1.03 to 1.05 on writing outputs.
- Neither lookup nor MTP acceptance rises from 10K to 100K on the data measured.
- Plain decoding at batch one gives almost no per-request speed spread.
- Gemma's public checkpoint and released drafter accept 0.72 to 0.81 per position on book questions and about 0.9 on coding.
- Only TP32/PP8 with DCP 8 at efficiency of at least 0.30 comes close under the published conditions, and it needs a small constant time per step that the model does not include.

Results we consider weak:

- The SPEED-Bench coding fit, chosen as the best of 16 settings after the results.
- The picture that the coding run drafted and the AA run did not.
- Story C's example, which rests on one request in 302.
- Story C's AA end, which needs AA's 100K answers to copy more than its 10K answers, or LP30 multi-row attention at 100K about twice as fast as modeled, and then fits only near the band's edges.
- Every acceptance fitted in the early sweeps.

## 22. Our assessment

The simulator can produce the AA pair under the published test conditions, but only on one layout and only with a small amount of time per step that it does not model.
The closest setup is 10.6% too fast at 10K and inside the band at 100K, and the gap equals plausible small values of fixed costs that the setup sets to zero, such as a 0.1 µs collective launch, a slightly slower cross-tray link or a short per-token host loop.
We regard this as a meaningful agreement between the cost model and the published numbers, while recognizing that the agreement rests on unmeasured LP30 rates and that a match shows possibility, not identity.

The 1.45% higher speed at 100K is, in our judgement, most likely variation between two sets of 50 requests.
No mechanism in the simulator makes 100K faster, acceptance does not rise with context on any data we measured, and a per-request spread of about 3% would make the difference ordinary.

On drafting, the coding numbers most likely involve a content-dependent mechanism or a different build: plain decoding is 21% to 29% too slow for the coding median and gives no spread at batch one.
The AA numbers, by contrast, are hard to reconcile with an effective drafter: Gemma's own drafter would make AA at least 72% too fast unless each step carries hundreds of microseconds the model does not include.
This tension produced the three stories.

Story A needs the least unexplained time and fits AA directly, but it requires the AA run and the coding run to differ in drafting, which one named deployment makes awkward unless AA measured a different build or configuration.
Story B uses the drafter Google ships, which makes it natural for a production deployment, but it needs an unmodeled cost larger than the whole modeled step, it spreads coding speeds too little, and it misses the example; it survives only because the host loop on LPX has no published size.
Story C uses one mechanism for all three kinds of content, reaches the example and gives NVIDIA's coding spread as an ordinary outcome, and it needs the smallest fixed time of the three.
Its AA end does not come from prompts that copy little, as we had expected: at K = 7 the verification of 8 rows costs much more at 100K than at 10K, so the AA pair fits only if AA's 100K answers copy more than its 10K answers.
No measurement here shows that on average, so the AA end of story C has moved from untested to unsupported by the data we have, though AA's own prompts could still show it.
The attention-cost sensitivity check adds another route: if LP30 verifies 8 rows at 100K with about half the modeled attention growth, the summary answers without thinking fit both AA values, though only near the band's edges.
We read this as making story C's AA end depend on one unmeasured hardware number, not as support for it, because the simulated rates that reach it put every attention unit at its peak.
We find story C the most economical explanation of the coding numbers, and story A the most economical explanation of the AA numbers, and the two cannot both describe one deployment with one drafter setting unless the AA run drafted differently.

We do not think the data identify NVIDIA's deployment.
The model build, thinking mode, drafter, sampling and AA's prompts are unknown or differ in every proxy experiment, and the LP30 rates are assumptions.
The three stories need at most 7 µs, 13 to 21 µs and 387 to 576 µs of extra time per step, so one measured LP30 decode step or host loop would separate them.
A measured 8-row verification step at 10K and 100K would also decide whether story C's AA end needs a copy gap.

## 23. Corrections made along the way

- **TP32 speculation at efficiency 0.10.**
  The expectation that long batched steps would match at efficiency 0.10 was wrong, because verification multiplies matrix work by the number of rows.
- **Coding example draft length.**
  We expected the example to need at least three draft tokens; two reach it at acceptance 0.73 to 0.90.
- **What bounds the remaining 100K cost.**
  An earlier record blamed matrix work for the remaining increment under DCP; that holds only at efficiency 0.10, and the vector unit's FP8 conversion sets it from 0.30 upward.
- **The coding conflict.**
  Treating the example as a hard constraint and the K = 1 against K ≥ 2 difference as a conflict assumed too much; on TP32 the conflict disappears.
- **The shared acceptance.**
  Requiring one acceptance for 10K and 100K excluded the published rise by construction.
- **Background requests as other users.**
  The private endpoint at concurrency one makes background requests a stand-in for fixed time per step.
- **The one-to-one token assumption.**
  Gemma's writing outputs give 1.03 to 1.05; an earlier draft quoted 1.07 from another model's reasoning traces.
- **AA measurement windows.**
  The 72-hour and 14-day windows in the footnotes apply to the public providers, not to the LPX values.
- **Overclaimed stand-in results.**
  The original write-up called the best-of-16 SPEED-Bench fit a match of NVIDIA's run and said the coding run most likely drafted; both are now stated as weak evidence and conditional on the unstated load.
- **AA window range under MTP.**
  An earlier summary gave 195 to 510 µs for the AA window under MTP; the correct range over 14 of 16 settings is 195 to 576 µs.
- **The AA condition of story C.**
  This record originally concluded that story C needs AA prompts that copy less than the book answers; the checks of section 20 found that less copying does not help and that story C needs more copying at 100K than at 10K.
- **Story C under other attention costs.**
  We expected no simulated LP30 attention cost to let story C fit AA; with every unit at its peak rate it fits marginally on summary answers without thinking.
  We also expected a faster exponential rate to leave the one-row increment unchanged, but the vector unit runs the exponentials, and we expected the fast-units variant to shorten the one-row step by 5 to 20 µs, but it shortened it by 3.0 µs.
- **The coding maximum under story C.**
  The 89-turn maximum/median of 1.885 to 1.95 initially looked too wide for NVIDIA's 1.464; in sets of 80 turns NVIDIA's value is an ordinary outcome.
- **The code-completion peak.**
  The earlier conclusion that lookup cannot reach the example, drawn from 40 prompts, was revised by the 302-prompt study.
- **Terminology.**
  The work originally used the Helix paper's "KV parallelism"; decode context parallelism is the more widely used name for the same technique.

## 24. Assumptions and limits

- **Fitted values.**
  Acceptances in the early sweeps were solved for, and background loads were chosen to match.
- **Uncalibrated LP30 rates.**
  Every unit rate and efficiency, the 0.05 µs fill-and-drain time and the fabric latencies are assumptions; no LP30 operator cost was compared with hardware.
  Story C's AA end depends on the multi-row attention rates, as section 20 shows.
- **FP8 KV conversion on the vector unit.**
  The larger-attention-group result depends on it; if LP30 feeds FP8 directly to its matrix unit, the cost structure changes.
- **The DCP partition.**
  NVIDIA states that LPX parallelizes the long-context KV cache but does not disclose how.
- **Drafting model.**
  The early sweeps assume independent per-position acceptance, equal draft lengths for background requests and free rollback of rejected positions; the measured-acceptance studies cost each step by its actual verified rows.
- **Forward width.**
  The study's memory policy reserves scratch for at most 16 query rows per forward, so drafts longer than 7 tokens at batch one could not be priced.
- **Not modelled.**
  Requests arriving and leaving, a varying batch size, mixed prefill chunks, and network, API and host overheads outside the forward pass.
- **Proxy outputs.**
  Every measured acceptance comes from the public checkpoint or third-party builds on study prompts, never from NVIDIA's build or AA's prompts.
- **Code-completion pricing.**
  The 302 prompts hold 18,071 to 21,421 tokens but were priced at the nominal 16K context, which slightly favours their speed.

## 25. Open questions and the tests that would answer them

| Question | Test |
| --- | --- |
| How much time does an LP30 step carry beyond the modeled forward? | Measure one LP30 decode step or the host's per-token loop; the three stories need at most 7 µs, 13 to 21 µs and 387 to 576 µs |
| Does story C hold at AA's end? | Obtain AA's 10K and 100K prompts and measure whether the 100K answers copy about 0.1 tokens per step more than the 10K answers; or measure an 8-row LP30 verification step at 10K and 100K, where growth of at most about 25 µs lets summary answers without thinking fit |
| Does the collective latency between trays match the assumption? | Measure one all-reduce across two LP30 trays; the match holds for about 0.73 to 0.85 µs depending on the token ratio |
| Is the faster 100K speed noise? | Obtain AA's per-request speeds for the LPX endpoint; a spread of about 3% or more makes it ordinary, and a wide spread also points to drafting |
| Which build, thinking mode and drafter did each run use? | A statement of the deployment's settings for the AA and SPEED-Bench runs |
| Is LP30's operand bandwidth below its SRAM bandwidth? | Measure decode weight streaming; GroqChip v1's operand bandwidth was a quarter of its memory bandwidth |
| Where does FP8 attention conversion run on LP30? | Establish whether KV conversion runs on the vector unit, which decides whether a larger attention group helps |
| Do drafts longer than 7 tokens change the picture? | Lift the 16-row forward limit and price verification steps of 9 to 16 rows |
| How does dynamic serving change the median? | Model arrivals, departures and mixed prefill chunks and compare the token-gap distribution with the steady-state step |

## 26. Overview of every tested combination

| DCP | Background | Speculation | Layout or data | Result |
| --- | --- | --- | --- | --- |
| none | none | none | TP16/PP8 | No match: 5,578 / 3,611 TPS |
| 4 | none | none | TP16/PP8 | No match: 5,533 / 4,152 TPS |
| 4 | up to 15 | none | TP16 | No match: best 3,331 / 3,041 TPS |
| none or 4 | none | K 1–7 | TP16 | No match: step ratio at least 1.154 |
| 4 | 7 | K = 1 | TP16 | 6 matches at acceptance 0.51–0.74 |
| none | any | none | TP32 | Infeasible: KV exceeds SRAM |
| 2 or 4 | 0–7 | none | TP32 | No match: increment at least 25.7 µs |
| 8 | 1–2 | none | TP32 | 6 matches at efficiency 0.30 or more |
| 8 | 0–7 | K = 1 or 3 | TP32 | 21 matches at efficiency 0.50 |
| any | any | any | TP32 at efficiency 0.10 | No match |
| none to 8 | none or 1 | K = 3 or 7, one acceptance per published speed | TP16 or TP32 | 29 setups fit all four numbers with fitted acceptances |
| 1–8 | none | none | 256 chips, published conditions | Only TP32/PP8 DCP 8 at efficiency 0.30 or more, with 14–25 µs extra |
| 8 | none | none | Token ratio 1.031–1.048 | Needs 1–16 µs extra |
| 8 | none | none | Cross-tray latency 0.80–0.85 µs | Matches at the book-answer ratio |
| 8 | none | none | Collective launch 0.1 µs | Matches |
| 1–8 | none | lookup or MTP, K 1–7 | 256 chips, published conditions | 11 of 378 match, at AA acceptance 0.006–0.335 |
| 8 | none | lookup on Gemma writing outputs | TP32/PP8 | 13–24% too fast at 10K |
| — | — | lookup on agent traces up to 66K | data only | Lookback beyond 16K adds under 0.005 |
| 8 | none | lookup on Gemma coding outputs | TP32/PP8 | 13 of 32 fit AA and the coding median with 38–61 µs |
| 8 | none | plain against lookup, per request | TP32/PP8 | Plain spread 1.000 / 1.001 against NVIDIA's 1.158 / 1.464 |
| 8 | none | lookup on SPEED-Bench coding outputs | TP32/PP8 | Best of 16 after the results: median and P80 within 2%, maximum within 12.2%, with 0–21 µs |
| 8 | none | lookup on book answers | TP32/PP8 | AA needs 26–99 µs; acceptance equal at 10K and 100K |
| 8 | none | Gemma MTP measured on an H20 | TP32/PP8 | AA at least 72% too fast; fits only with 195–576 µs |
| 8 | none | Gemma MTP on AA and coding together | TP32/PP8 | Shared 387–576 µs at K = 3 or 5; spread too narrow; example missed |
| 8 | none | lookup on all 302 code-completion answers | TP32/PP8 | K = 7 reaches 11,606–11,735 TPS at 6–7 µs with the coding median inside 5% |
| 8 | none | lookup at K = 7 on book answers | TP32/PP8 | No extra time fits both AA values |
| 8 | none | lookup at K = 7 on each book answer and on low-copy subsets | TP32/PP8 | 1 of 64 book pairs fits, through more copying at 100K; no low-copy subset fits |
| 8 | none | lookup at K = 3 and 7 on sets of 80 coding turns | TP32/PP8 | NVIDIA's P80 and maximum ratios inside the 5th–95th percentiles for stories A and C |
| 8 | none | lookup at K = 1–7 under four sets of LP30 attention rates and a row-independent bound | TP32/PP8 | Story C fits AA only with every unit at its peak rate or under the bound, near the band's edges; story A keeps a window under all |
| attention on Rubin GPUs | none | none | Rubin and LPX, older cost model | No match: best 2,463 / 2,162 TPS |

## 27. Commit history

| Commits | Content |
| --- | --- |
| `a64c090`, `2cc4e1c` | Published test conditions |
| `de93f75`, `4b7692b` | Batch-one sweep on 256 chips |
| `8d7bb73`, `09991d1` | Speculative decoding under the published conditions |
| `4909029`, `a176a52`, `c397c75`, `971c161` | Gemma tokens per AA token |
| `c34e57c`, `804b236` | Latency between trays |
| `e548087`, `628e72d` | GroqRack Llama-2-7B check |
| `5454a49`, `3e26048` | Lookup acceptance on Gemma outputs |
| `92bc22e`, `4499b4b` | Lookup acceptance against context length |
| `e35a1ac`, `1170ced` | Lookup drafting on coding outputs |
| `a5f48a6`, `90cd654` | Coding speed spread |
| `9a45292`, `1486ad7` | Gemma outputs on SPEED-Bench and AA-like prompts |
| `528eaf4`, `9d4f257`, `a768c38` | Qualified conclusions |
| `c3af9dc`, `d897025` | Per-step cost audit |
| `13bfa7b`, `e429d26` | Published fixed costs |
| `7accd04`, `e888cca` | AA noise check |
| `fcf13ac`, `edab20d`, `bca4d8a`, `08d2dec`, `76d09dc` | Gemma's MTP drafter measured on an H20 |
| `2147ee4`, `7901597` | One extra time per step under MTP |
| `78ea6f1`, `7388167` | Expectations and plan for the 302-prompt code-completion study |
| `7540b49`, `1439744`, `fa90435`, `0514637`, `d3e8cb3` | The 302-prompt study on the GPU machine |
| `f68bfec`, `d835c7c` | Independent check of the 302-prompt results and AA at the example's drafter setting |
| `a387f3a`, `1da3aee` | This comprehensive summary |
| `d13f810`, `6d1d581` | Expectations and results of the two story C checks |
| `a278b67`, `4f3136c` | AA's prompt methodology, and expectations and results of the sensitivity to LP30 attention costs |

## 28. Glossary

| Term | Meaning |
| --- | --- |
| AA | Artificial Analysis, the independent benchmark whose published speeds are the target |
| Acceptance (α) | The probability that a drafted token matches the large model's own choice; equivalent acceptance is the per-position value that gives the measured mean tokens per step |
| Attention group | The chips that together compute one attention layer for one request; here the TP group of a pipeline stage |
| Background request | A simulated request that shares every decode step with the AA request; it stands for any fixed time per step |
| Closest setup | TP32/PP8 with DCP 8 on 256 LP30 chips at attention efficiency 0.5, with a 267.3 µs step at 10K and 281.5 µs at 100K |
| Coding window | The range of constant extra time per step that places the priced coding median within 5% of 4,767 TPS |
| DCP | Decode context parallelism: splitting one request's KV cache across chips by token position inside the TP group |
| Draft length (K) | The number of tokens a drafter proposes per step; verification covers K + 1 rows |
| Extra time | A constant time added to every simulated step to stand for costs the model does not include |
| Global and windowed layers | Gemma 4's attention kinds: global layers read the whole context, windowed layers the last 1,024 tokens |
| Increment | How much longer the AA request's step is at 100K than at 10K |
| KV cache | The stored keys and values of earlier tokens, read by attention at every decode step |
| Lookup drafter | A drafter that proposes the tokens that followed the most recent match of the current n-gram in the prompt or output |
| LP30 and LPX | NVIDIA's Groq 3 language processing chip, and the 256-chip rack system built from it |
| MTP | Multi-token prediction: a small drafter trained with the model; Gemma 4 31B ships a 500-million-parameter MTP assistant |
| Replay | Measuring a lookup drafter's acceptance by running it over recorded temperature-0 outputs |
| SPEED-Bench | NVIDIA's speculative-decoding benchmark, with a qualitative split and throughput splits such as 16K code completion |
| TP, PP, TPA | Tensor parallelism, pipeline parallelism, and the part of TP that splits attention heads |
| TPS | Output tokens per second for one request |
| τ | Mean tokens committed per verification step |
