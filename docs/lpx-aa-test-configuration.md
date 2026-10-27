# What is known about the AA Gemma 4 31B LPX test configuration

Record date: 2026-10-26.
This record lists what public sources establish, what they suggest and what they leave unknown about the Artificial Analysis (AA) measurement of Gemma 4 31B on NVIDIA Groq 3 LPX.
The [reproduction progress summary](lpx-aa-reproduction-progress.md) uses these conditions as its target.
The AA chart footnotes were read from the chart images in NVIDIA's article, because the article text does not contain them.

## Established by a primary source

| Item | Value | Source |
| --- | --- | --- |
| Model | Gemma 4 31B (Reasoning) | AA chart titles |
| Input lengths | 10K and 100K input tokens, counted with the `o200k_base` tokenizer | AA chart subtitles; Gemma's own tokenizer may count a different number |
| Result | Median 3,382 output tokens per second at 10K and 3,431 at 100K | AA charts and NVIDIA article |
| Metric | Output tokens per second after the first token arrives, which excludes prefill | AA methodology |
| Sample size | 50 requests per workload, reported as their median | Both chart footnotes |
| Request pattern | Sequential requests at client concurrency 1, so each request starts after the previous one finishes | Both chart footnotes; the AA methodology states that "one prompt is sent to the model's API at a time" |
| Endpoint | "the updated private, pre-release Gemma4-31B-LPX deployment via Google Cloud PSC", identical in both footnotes | Both chart footnotes; PSC denotes Private Service Connect |
| Host | "An NVIDIA Groq 3 LPX system NVIDIA has stood up in its own data centers" | NVIDIA article |
| Hardware family | NVIDIA Groq 3 LPX | AA charts and NVIDIA article |
| Date | "As of Aug 21, 2026" on both charts | Chart labels; the article's image descriptions give August 20, 2026 |
| Measurement window | 50 requests per workload; the 72-hour and 14-day medians in the footnotes apply only to the public providers | Footnote wording |
| 100K output length | At least 2,000 answer tokens | AA methodology for the 100K workload |

## Likely, from a secondary source or an inference

| Item | Likely value | Reason it is not established |
| --- | --- | --- |
| GPU involvement | None; the model ran on LP30 chips only | The Register states that "the performance figures shared by Nvidia today are for a model running entirely on LPUs"; NVIDIA and AA do not confirm it |
| Weight precision | FP8 | Only The Register states it; NVIDIA and AA state only that there was no loss of precision or model quality |
| Same configuration for 10K and 100K | Yes | Both footnotes name the same deployment, date and method, and both workloads fit in the same allocation; no source states it directly |
| Server batch size | 1 | AA kept one request in flight on a private endpoint without public traffic; no source mentions or excludes NVIDIA's own traffic or added load |
| Upper bound on chip count | 256 chips, one rack | The Hot Chips coverage says "a single rack" decodes Gemma 4 at about 11,000 tokens per second under the same deployment name; the AA endpoint is described as "updated", so it may be a different build |
| Lower bound on chip count | About 67 chips with FP8 weights and KV cache | About 30.7 GB of weights and 2.5 GB of 100K KV cache need about 67 chips of 500 MB, before activations and buffers; The Register's "just under 64" counts weights only |
| Tensor parallelism (TP) | High, with an unknown degree | NVIDIA attributes the flat speed to "high tensor parallelism"; head-parallel attention over 32 query heads caps TP at 32 |
| KV cache split across chips | Yes, in a form similar to decode context parallelism (DCP) | NVIDIA states that LPX uses "parallelism for both model weights and long-context KV cache"; the scheme and degree are not stated |
| Chip activity | Most chips idle at any instant | At batch size 1 with pipeline stages, a token passes through the stages in turn, so only one stage computes at a time; this follows from the batch-size assumption |
| 1K workload | Possibly run | Only a search excerpt of AA's paywalled X post mentions 1K prompts, and no 1K LPX value was published |

The KV cache estimate uses the repository's [Gemma 4 31B model file](../data/models/gemma-4-31b-text.yaml).
The 10 global layers store one shared key/value tensor of 4 heads by 512 dimensions, which is 20,480 bytes per token in FP8 and 2.05 GB at 100K tokens.
The 50 windowed layers store keys and values of 16 heads by 256 dimensions over a 1,024-token window, which is 0.42 GB in FP8.

## Unknown

| Item | Effect on the reproduction | Available evidence |
| --- | --- | --- |
| Exact chip count | Sets per-chip work and collective sizes | Bounded at about 67 to 256 chips |
| TP, pipeline-parallel (PP), DCP and replica degrees | Set the step time and its growth with context | NVIDIA names "high tensor parallelism" and "layer-wise partitioning" without degrees |
| KV cache and activation precision | Changes the 100K attention cost, including FP8-to-BF16 KV conversion | None |
| Speculative decoding, drafter, draft length and acceptance | Decide whether 3,400 tokens per second implies a step of about 290 microseconds or a longer step that commits several tokens | No source mentions it for the AA run; one deployment name spans about 3,400 tokens per second on AA prompts and 4,767 to 10,996 on coding, which suggests content-dependent speed |
| Other traffic on the endpoint | Decides whether background requests in the simulations can represent other users | No source mentions any |
| Times of the two request sets and changes between them | Could explain the 1.45% higher speed at 100K | Not stated |
| Spread of the 50 samples | Needed to judge whether the 1.45% difference is noise | Only medians are published |
| Output lengths and reasoning handling | AA computes speed from the last 80% of answer chunks when a model does not expose all reasoning tokens | Not stated for this endpoint |
| Prompt content | Affects drafting acceptance if speculative decoding was used | AA does not publish its prompts in detail |
| Sampling settings | Affect drafting acceptance | Not stated |
| Network and API overhead between Google Cloud and NVIDIA's data center | Could delay streamed chunks | Not stated; output speed is measured between chunks, so a constant connection delay mostly falls into the time to first token |
| Build used for the Hot Chips and SPEED-Bench values | Decides whether the coding values constrain the AA setup | The names match, and the AA footnotes call their deployment "updated" |
| Prefill hardware | None, because output speed excludes the first token | Not stated |

## Consequences for the reproduction

The simulations can rely on sequential single-request decode, the two input lengths, the `o200k_base` token count and the decode-only metric.
They assume LP30-only execution, FP8 weights, server batch size 1 and one configuration for both workloads; each assumption is likely but rests on a press report or an inference.
The chip count, parallel layout, KV precision and speculative decoding remain free parameters, and the sweeps in the [progress summary](lpx-aa-reproduction-progress.md) fit them.
Speculative decoding is the largest unknown, because it decides how long a step the published speed implies.

## Sources

- NVIDIA, [How NVIDIA Groq 3 LPX Unlocks Ultrafast Interactivity at Long Context on NVIDIA Vera Rubin](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/), with its [10K chart](https://developer-blogs.nvidia.com/wp-content/uploads/2026/08/nvidia-groq-3-lpx-gemma-4-10k-artificial-analysis.webp) and [100K chart](https://developer-blogs.nvidia.com/wp-content/uploads/2026/08/nvidia-groq-3-lpx-gemma-4-100k-artificial-analysis.webp)
- NVIDIA, [Inside NVIDIA Groq 3 LPX](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/)
- Artificial Analysis, [performance benchmarking methodology](https://artificialanalysis.ai/methodology/performance-benchmarking)
- The Register, [What Nvidia's first Groq 3 LPU benchmarks do and don't tell us about its $20B gamble](https://www.theregister.com/systems/2026/08/24/what-nvidias-first-groq-3-lpu-benchmarks-do-and-dont-tell-us-about-its-20b-gamble/5291880)
- ServeTheHome, [NVIDIA's Groq 3 LPU Accelerators for Heterogeneous AI Compute at Hot Chips 2026](https://www.servethehome.com/nvidias-groq-3-lpu-accelerators-for-heterogeneous-ai-compute-at-hot-chips-2026/)
- Artificial Analysis, [post on X](https://x.com/ArtificialAnlys/status/2092070577934925992), seen only through search excerpts
- The [parameter evidence audit](lpx-parameter-evidence-audit.md) records the same footnotes and press statements as sources S5 and S8.
