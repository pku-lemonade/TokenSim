# Plausibility of the simulated LPX placement for the published Gemma results

Record date: 2026-10-23.
This record explains why the deployment that TokenSim currently selects for Gemma 4 31B is unlikely to be the configuration that produced the published Artificial Analysis (AA) results.
It also records the public sources for the 10,996 tokens-per-second (TPS) Hot Chips example, which several of the arguments use.

## Scope of the claim

The simulated deployment has four parts:

| Part | Current simulation |
| --- | --- |
| Hardware | Pure LPX: every layer runs on LP30 chips, with attention and feed-forward network (FFN) work on the same chips |
| Layout | Tensor parallelism (TP) 16 and pipeline parallelism (PP) 8 on 128 LP30 chips, or TP16/PP6 on 96 chips with Helix-style KV parallelism (KVP) from the [context-parallel record](lpx-context-parallel.md#results) |
| Selection rule | The smallest memory-feasible chip count whose simulated speed clears 3,400 TPS at 1K and the published 3,382 and 3,431 TPS at 10K and 100K |
| Serving regime | One dedicated request per forward pass, batch size 1, no speculative decoding, identity conversion between Gemma tokens and AA `o200k_base` tokens |

The arguments below concern the combination of these parts.
The chip layout alone may still resemble the real one.
The record does not show that any part is wrong in isolation.
It shows that the published numbers are unlikely to come from this layout operated in this serving regime with the current cost model.

## Published and simulated numbers

Time per output token is the reciprocal of output TPS.

| Source | 10K input | 100K input | Change from 10K to 100K |
| --- | ---: | ---: | ---: |
| Published AA median | 3,382 TPS, 295.7 µs | 3,431 TPS, 291.5 µs | −4.2 µs, speed ratio 1.0145 |
| Simulated TP16/PP8, `KVP = 1` | 5,578.41 TPS, 179.3 µs | 3,611.17 TPS, 276.9 µs | +97.7 µs, speed ratio 0.647 |
| Simulated TP16/PP8, `KVP = 4` | 5,533.30 TPS, 180.7 µs | 4,152.44 TPS, 240.8 µs | +60.1 µs, speed ratio 0.750 |
| Simulated TP16/PP6, `KVP = 8`, 96 chips | 5,518.62 TPS, 181.2 µs | 4,143.38 TPS, 241.4 µs | +60.2 µs, speed ratio 0.751 |

The simulated 10K speed exceeds the published one by about 65%.
At 100K the simulated time per token is 5% to 17% below the published time.

## Source of the 10,996 TPS example

NVIDIA presented the example at Hot Chips 2026.
ServeTheHome published the presentation slides in [NVIDIA's Groq 3 LPU Accelerators for Heterogeneous AI Compute at Hot Chips 2026](https://www.servethehome.com/nvidias-groq-3-lpu-accelerators-for-heterogeneous-ai-compute-at-hot-chips-2026/), by Ryan Smith, dated 2026-08-25.
The slide image is [NVIDIA-Groq-3-LPU-Hot-Chips-2026-LPX-Performance.png](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-LPX-Performance.png).
The copy reviewed on 2026-10-23 is 1,930 by 1,086 pixels with SHA-256 `cd3f54fdfe0e103f7bac6ddc004048acf795b1e40a97c04caf6b83f1d4124c9f`.
The [evidence audit](lpx-parameter-evidence-audit.md) lists the same image as source S7.

The slide contains the following text:

| Slide element | Text |
| --- | --- |
| Title | "11,000 Tokens Per Second Sets a New Bar for Inference AI" |
| Subtitle | "NVIDIA Groq 3 LPX rack" |
| Callout beside the rack photograph | "10,996 TPS / User" |
| Chat interface screenshot | Model selector "Gemma4-31B-LPX" |
| Workload note | "Example Coding Gemma 4 31B–16K ISL Out of Max 264K Context" |
| Benchmark note | "SPEED-bench coding running at 4,767 median output tokens / second" |

ISL denotes input sequence length.
The article text states that "a single rack can decode 11,000 tokens per second on a 31B model (Gemma 4)".
One LPX rack holds 256 LP30 chips according to the same article and the official specification.
The slide shows one rack photograph and does not state the allocated chip count, precision, batch size, token-counting convention, sampling settings or speculative-decoding status.
The value comes from a single vendor example and has no published variance.

The [NVIDIA long-context benchmark article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) footnotes its AA charts with "an updated private, pre-release Gemma4-31B-LPX deployment via Google Cloud PSC".
PSC denotes Private Service Connect.
The same deployment name therefore appears with approximately 3,400 TPS on AA prompts and with 4,767 to 10,996 TPS on coding workloads.
Both sources use the name, but neither states that the two measurements used an identical build, allocation or serving configuration.

Another slide in the same article, [External-Drafter Speculative Decode](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-Speculative-Decode.png), shows an LPX rack of 256 LP30 chips running a small draft model and a Vera Rubin NVL72 rack of 72 Rubin GPUs verifying a large target model.
Its footer reads "Each rack keeps its own mode's KV cache | Only draft tokens cross the link", where "mode's" is printed on the slide.
The reviewed copy has SHA-256 `5d189b0c2df3df3e4201bb4a9f55a3623e4e739a4e4d752c86a53cbfc4c346df`.
The slide shows that NVIDIA presents speculative decoding as an LPX serving pattern.
It describes LPX as the drafter for a GPU-hosted target model and does not describe the Gemma endpoint.

## Reasons

The reasons are ordered by the strength of their evidence.

### Reason 1: the simulated configuration cannot become faster at longer context

Strength: strong.

In the simulated configuration, each decode step costs a context-independent part plus attention work.
Attention work cannot decrease when the context grows, so the simulated speed at 100K can equal the 10K speed only when attention costs nothing and can never exceed it.
The published speed rises by 1.45% from 10K to 100K.
The [context-sensitivity scan](lpx-context-sensitivity.md) found zero cases in which a fixed setting produced a higher 100K speed, across every searched layout and parameter combination.

The published pair has no variance data, so the 1.45% rise may be measurement noise.
A flat curve still conflicts with the simulation.
The simulated speed falls by 35% without KV parallelism and by 25% with it.
The Helix-style KV parallelism, which addresses the published statement that LPX parallelizes long-context KV cache, removes only part of that decline.
The remaining decline comes from global-attention matrix work, which KV parallelism cannot reduce.

A configuration that matches the published pair therefore needs a mechanism outside the simulated one.
Candidates include other requests in the same forward pass, speculative decoding with content-dependent acceptance, a measurement-window effect or different deployments for the two context lengths.

### Reason 2: the errors do not fit a uniformly scaled cost model

Strength: strong.

The simulated time per token is 116 µs too short at 10K and only 15 µs to 51 µs too short at 100K, depending on the KV-parallel setting.
If the endpoint used the simulated layout and serving regime, and the cost model only misjudged LP30 speed by a constant factor, the relative error would be similar at both contexts.

Matching both published values with this layout and regime requires two independent errors in opposite directions.
The context-independent cost per token would need roughly 116 µs more time.
The context-dependent cost from 10K to 100K would need to shrink from the simulated 60 µs to 98 µs to about zero.
The two errors would also need to cancel so closely that the 100K speed slightly exceeds the 10K speed.
These coincident errors are possible but less likely than a difference in serving regime.

The earlier [public-model validity audit](lpx-public-model-validity-audit.md) reached the related conclusion that adding the same constant to every token cannot reproduce both published context points and their ordering.

### Reason 3: the same deployment name is reported far faster on coding

Strength: strong for the direction, limited by vendor-reported values without variance.

| Reported speed | TPS | Time per token |
| --- | ---: | ---: |
| Hot Chips coding example, 16K input | 10,996 | 90.9 µs |
| SPEED-Bench coding maximum | 6,981 | 143.2 µs |
| SPEED-Bench coding 80th percentile | 5,520 | 181.2 µs |
| SPEED-Bench coding median | 4,767 | 209.8 µs |
| AA, 10K input | 3,382 | 295.7 µs |

Linear interpolation of the simulated `KVP = 1` times between the 10K and 100K points gives approximately 185 µs per token at 16K.
This interpolation is an estimate, not a simulated point.
The 10,996 TPS example is about twice as fast as that estimate, while the AA 10K value is about 1.65 times slower than the simulated 10K value.

With one committed token per forward pass, the per-token time of a fixed deployment depends only on the sequence lengths and the batch.
The simulated TP16/PP8 configuration reaches at most 5,908.78 TPS at 1K input, its shortest workload.
It therefore cannot reproduce 6,981 or 10,996 TPS at coding lengths.
It also cannot produce a spread from 4,767 to 6,981 TPS within one benchmark unless the samples differ greatly in length.

Speculative decoding produces content-dependent speed directly.
A target model that verifies several drafted tokens per forward pass commits more tokens per pass on predictable text, such as code, than on less predictable text.
This explanation is an inference.
No reviewed source states that the Gemma4-31B-LPX endpoint uses speculative decoding.
The metrics also differ: AA excludes the initial token and counts `o200k_base` tokens, while the Hot Chips value is a "TPS / User" figure with an unstated counting convention.

The simulated deployment contains no draft model.
A draft model would require SRAM and compute, so it would also change the memory-feasible layout.

### Reason 4: the endpoint probably shares forward passes with other traffic

Strength: moderate; inferred from the benchmark method.

AA runs 50 sequential requests with client concurrency 1.
That setting controls only AA's own client.
It does not reveal how many requests the endpoint batches into each forward pass.
A private, pre-release deployment may or may not have served other traffic during the measurement.

Batching other requests into the same forward pass has two effects that match the published pair.
It lengthens each step, which lowers per-user speed toward the published 10K level.
It also makes step time depend on the aggregate work of the whole batch, so the AA request's own context length matters less, which flattens the curve.
One mechanism could therefore explain both the level and the shape.

The simulation assumes a dedicated endpoint at batch size 1.
The native runtime supports fixed batches, so this hypothesis can be tested directly.

### Reason 5: the selected chip count follows from assumed parameters

Strength: moderate.

TP16/PP8 is the smallest layout that clears the targets under illustrative LP30 parameters from the [fidelity study](lpx-fidelity-model.md).
Changing one uncalibrated parameter changes the result.
A per-operator fill/drain residual of 0.10 µs instead of 0.05 µs lowers the 100K speed at the same layout to 3,046.75 TPS, below the target.
The selected layout is therefore a consequence of the assumed parameters and is not derived from deployment evidence.

The selection rule also assumes that the operator minimizes chip count.
LPX racks hold 256 LP30 chips.
The Hot Chips slide subtitle reads "NVIDIA Groq 3 LPX rack", and the ServeTheHome article describes the Gemma example as running on "a single rack".
A deployment that occupies a whole rack, uses part of it for other purposes or hosts several model replicas would not follow a minimum-chip rule.

### Reason 6: the token conversion is assumed

Strength: weak to moderate; affects the level more than the shape.

AA counts tokens with the `o200k_base` tokenizer.
The simulation counts native Gemma tokens and assumes a ratio of 1.0 between the two.
The [fidelity study](lpx-fidelity-model.md) shows that ratios of 0.8 and 1.2 change the 10K simulated speed from 5,578.41 TPS to 7,064.71 and 4,589.11 TPS.
A ratio between the two tokenizers changes the absolute level and the native KV length.
It cannot by itself turn the falling simulated curve into a flat or rising one, because the same ratio applies to both context points.
Different ratios for the 10K and 100K prompt sets could change the shape, but no evidence about the prompt text is available.

## Supporting observations

NVIDIA's statement that LPX employs "parallelism for both model weights and long-context KV cache" agrees qualitatively with the simulated KV-parallel layout.
That agreement does not identify the partition scheme or show that the published endpoint uses it.

The [AA architecture study](lpx-aa-architecture-study.md) found that the simulated mixed Rubin and LP30 attention/FFN placements miss the published targets.
The [ServeTheHome article](https://www.servethehome.com/nvidias-groq-3-lpu-accelerators-for-heterogeneous-ai-compute-at-hot-chips-2026/) states that NVIDIA offloads the attention part of decode to GPUs in its disaggregated platform.
The Gemma slide shows only an LPX rack, so the reviewed sources do not establish whether the Gemma endpoint uses GPU attention.

## Tests that distinguish the explanations

| Test | Question | Outcome that supports the explanation |
| --- | --- | --- |
| Batched serving with background requests | Does a plausible batch size with mixed background contexts give about 3,400 TPS at both 10K and 100K? | A batch range that flattens the curve and lowers the level together |
| Speculative-decoding sweep | Does one set of plausible acceptance rates and draft lengths give about 3,400 TPS on AA and 4,767 to 10,996 TPS on coding? | A region of draft settings consistent with all published values |
| GroqChip v1 validation | Does the same cost-modeling method reproduce measured results on the previous Groq chip? | Agreement with the published BERT single-chip latency and the GroqCloud Llama-2 70B throughput listed in the [data-collection checklist](data-collection-checklist.md) |

The batched and speculative tests check consistency with the published values.
Only the GroqChip v1 test checks the operator cost model against measured hardware.
