# Gemma 4 architecture and inventory exploration

## Result status

The active follow-up uses FP8 weights as described in the [current FP8 experiment](lpx-fp8-study.md).
The comparisons below retain their original BF16 baseline and FP8 sensitivity settings.

This study compares attention/feed-forward network (AF) disaggregation, prefill/decode (PD) disaggregation, homogeneous Rubin, and homogeneous LPX for one serial client.
The results are uncalibrated analytical projections from TokenSim operators and communication models.
They do not establish the real benchmark's hardware inventory, equal model quality, or a global minimum across all possible implementations.
Rubin remains the user-selected companion hypothesis because the measured endpoint's GPU inventory and division of work are undisclosed.
The experiment adds an isolated Gemma adapter and search entry point without changing the native AF executor or hardware catalog.

## Workload and published target

The [NVIDIA benchmark article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) reports median Gemma 4 31B output speeds of 3,382 tokens/s at 10K input and 3,431 tokens/s at 100K input.
The article reports preserved precision and quality for the measured endpoint.
Its deployment configuration and raw request records are unavailable.
The user-requested acceptance threshold is 3,400 generation tokens/s for every context length.
Published 10K and 100K measurements remain reference values and do not replace this common threshold.
A corresponding published numeric 1K LPX result was not found.

| Workload | Input tokens | Answer tokens used | Target output tokens/s |
| --- | ---: | ---: | ---: |
| 1K | 1,000 | 1,000 | 3,400 |
| 10K | 10,000 | 1,500 | 3,400 |
| 100K | 100,000 | 2,000 | 3,400 |

Answer lengths use the minimum workloads in the [Artificial Analysis methodology](https://artificialanalysis.ai/methodology/performance-benchmarking), which counts performance tokens with `o200k_base` and excludes input processing from output speed.
The primary metric is `(output_tokens - 1) / (last_output_s - first_output_s)` in AA token units.
The initial wait for prefill, initial key/value (KV) cache handoff, and the initial output token is excluded from this interval.
There is no time-to-first-token (TTFT) requirement in layout acceptance or inventory minimization.
The last 80% of native output intervals is exported separately, and `--timing-window tail80` selects that approximation of AA's hidden-reasoning rule as the primary metric.
The simulation cannot reproduce the actual answer-chunk window or hidden reasoning without the original stream.
Each selected layout replays five warmups and 50 measured requests at client concurrency one.
These are deterministic repetitions of token lengths and timing formulas, with no generated text or independent quality samples.
Additional replicas cannot increase this single user's speed in the experiment.

The actual prompts, answer lengths, native token conversion ratios, hidden reasoning lengths, precision, speculative decoding, compiler schedule, and streaming timestamps remain unknown.
Strict reproduction of the public endpoint therefore remains incomplete.

## Follow-up scope and context-length trend

The follow-up comparison focuses on homogeneous LPX and AF disaggregation.
The earlier PD results remain inventory references for the generation-only objective.

The published 100K speed of 3,431 tokens/s exceeds the 10K speed of 3,382 tokens/s by 1.449%.
These speeds correspond to 291.460 and 295.683 microseconds per AA output token, a difference of 4.223 microseconds.
The [NVIDIA article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) describes minimal speed variation with context length and attributes that behavior to deterministic scheduling and high TP.
It does not provide raw per-request distributions, confidence intervals, or enough deployment detail to establish a repeatable causal speedup from longer context.
Both the small positive difference and the nearly constant speed need to be considered when comparing architecture hypotheses.

The following rows keep each layout fixed between the 10K and 100K workloads.
Weights and KV use BF16, token conversion ratios remain one, and collectives use the scheduled-direct hypothesis.

| Fixed layout | Rubin + LP30 | 10K tokens/s | 100K tokens/s | 100K / 10K |
| --- | --- | ---: | ---: | ---: |
| Published endpoint, inventory undisclosed | Unknown | 3,382 | 3,431 | 1.0145 |
| LPX, minimum passing all lengths | 0 + 248 | 3,723 | 3,464 | 0.9306 |
| LPX, fastest searched baseline | 0 + 352 | 4,149 | 3,926 | 0.9462 |
| Conventional AF, fastest searched baseline | 8 + 108 | 1,047 | 936 | 0.8939 |
| Global-only AF, fastest searched baseline | 8 + 160 | 2,476 | 1,934 | 0.7810 |
| Global-only AF, lower fixed-cost sensitivity | 8 + 160 | 4,417 | 2,944 | 0.6666 |

The last row combines a 0.05-microsecond operator fixed cost with a 0.5-microsecond cross-rack startup in a separate follow-up run.
These values are unmeasured assumptions and do not demonstrate an implemented optimization.
The run is saved separately in `outputs/lpx-aa-20261019/af-combined-bf16/` and is excluded from the original 12-run validation totals below.
Derived ratios and timing components are saved in `outputs/lpx-aa-20261019/context-trend.json`.

The current decode model sums context-independent costs and nondecreasing attention costs for a fixed layout and token conversion ratio.
It therefore cannot explain the positive published ratio under those fixed assumptions.
Reducing constant operator overhead improves absolute TPS while increasing the relative influence of the growing attention term.
The 248-chip baseline satisfies the requested speed threshold but does not reproduce the published context-length trend.
Its inventory remains conditional on the uncalibrated execution model.

Candidate mechanisms for investigation include context parallelism, which partitions cached sequence positions across chips, and overlap between attention, communication, and other work where dependencies permit it.
The NVIDIA article describes compiler scheduling at 320-byte vector granularity, which this serial cost composition does not implement.
A revised model must account for parallel attention reductions, link traffic, memory ownership, and the extra chips required by these mechanisms.
Measurements should hold the allocated hardware, precision, and scheduling policy constant across context lengths and separately report any adaptive scheduling changes.
Prompt-dependent native/AA token ratios, actual answer lengths, post-initial-token streaming behavior, speculative decoding if present, and differences between the sampled requests can also affect measured ratios.
The [AA methodology](https://artificialanalysis.ai/methodology/performance-benchmarking) uses distinct answer-length minima for the two workloads and notes that output content can affect speed through speculative decoding.
The original request records are required to distinguish these explanations.
Changing prefill duration alone does not change the generation-only metric.
Matching absolute TPS and the context-length trend are separate calibration checks.
The [context-sensitivity follow-up](lpx-context-sensitivity.md) evaluates 55 numerical profiles and constructs both execution-cost and output-window explanations of the small reversal.

## Model, placement, and hardware assumptions

The [configuration snapshot](../experiments/lpx/aa-gemma4.json) transcribes the public [Gemma checkpoint configuration](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) and follows the [reference text implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py).
It models 60 layers with five sliding-window layers followed by one full-attention layer, repeated ten times.
Local layers have 32 query heads, 16 key/value heads, head dimension 256, and a 1,024-token window.
Global layers have 32 query heads, four key/value heads, and head dimension 512.
Global K=V shares the projection weights while retaining two distinct cache tensors.
The gated feed-forward network uses three weight matrices, hidden width 5,376, and intermediate width 21,504.
The vocabulary has 262,144 entries and tied embeddings.
Text linear and embedding weights contain 30,696,013,824 elements before normalization parameters.
The vision encoder is excluded from this text-only workload.
The transcribed input has a SHA-256 digest in the results; an immutable upstream checkpoint revision was not obtained.

| Parameter | Assumption |
| --- | --- |
| Rubin memory | 288 GB HBM, 19.2 TB/s, catalog compute and efficiency values |
| LP30 memory | 500 MB SRAM, 150 TB/s, catalog compute and efficiency values |
| Weight precision | BF16 primary; FP8 sensitivity with fused conversion assumed |
| Activations and default KV | BF16 |
| Native tokens / AA tokens | `rho_in = rho_out = 1`, varied separately |
| Operator fixed overhead | 0.2 microseconds, including the GPU graph-kernel hypothesis |
| GPU graph submission | 4 microseconds per decode step or prefill chunk |
| Memory reserve | Maximum of catalog reserve and 5% of chip memory |
| Workspace | Explicit BF16 chunk workspace in addition to weights, KV, and reserve |
| Prefill chunk | 256 tokens, serial chunks without pipeline overlap |
| Cross-rack link | Catalog Ethernet 1,600 Gb/s, 200 GB/s per direction, 2 microsecond startup |
| Search limits | At most 1,024 LP30 chips and 144 Rubin GPUs per layout |

All memory and bandwidth figures use decimal units.
Tensor parallelism (TP) for attention is searched over 1, 2, 4, 8, 16, and 32.
AF feed-forward TP also explores every positive divisor of 21,504 up to 256.
Pipeline parallelism (PP) uses a capacity-minimal contiguous partition of complete layers at the peak input-plus-output context.
The tied head can reuse the embedding stage and receive the final hidden states over an activation return transfer.
That schedule is a prototype assumption whose compiler feasibility has not been demonstrated.
PP increases available storage and does not divide the serial client's total layer compute time.
Weights and KV must fit in SRAM; external DRAM streaming is excluded.

The `native-ring` profile uses the flat alpha-plus-bytes-over-bandwidth ring formula of the current LPX transfer path.
The `scheduled-direct` profile uses the existing hierarchical collective model with direct collectives, no message-size bandwidth ramp, and no additional collective efficiency discount.
Its topology and latency inputs remain uncalibrated.
This option does not implement the real LPX compiler or its fine-grained compute/communication overlap.

## Minimum inventory for each context

The following table uses the scheduled-direct hypothesis and optimizes hardware separately for each workload.
Counts are allocated LP30 chips, and a PD inventory adds the listed Rubin GPU to the LP30 decoder.
The LP30 count must not be interpreted as a count of LPX racks or PCIe cards.

| Precision | Input | Pure LPX minimum | LPX TP x PP | Predicted tokens/s | PD minimum Rubin + LP30 |
| --- | --- | ---: | --- | ---: | --- |
| BF16 | 1K | 168 | 8 x 21 | 3,760 | 1 + 168 |
| BF16 | 10K | 168 | 8 x 21 | 3,731 | 1 + 168 |
| BF16 | 100K | 248 | 8 x 31 | 3,464 | 1 + 248 |
| FP8 | 1K | 80 | 16 x 5 | 4,505 | 1 + 80 |
| FP8 | 10K | 84 | 4 x 21 | 3,694 | 1 + 84 |
| FP8 | 100K | 160 | 8 x 20 | 3,854 | 1 + 160 |

FP8 rows do not demonstrate the published equal-quality requirement.
The public text checkpoint uses BF16, and neither quantized output quality nor service precision was verified.
Pure Rubin, conventional AF, global-only AF, reversed AF, and reversed PD have no passing layout for these targets under the searched assumptions.
The search excludes context parallelism, arbitrary tensor decompositions, different fusion schedules, and speculative decoding, so their possible benefits remain open.

## One inventory serving all three contexts

The following table selects the fastest layout in each architecture by its worst ratio to the three workload targets.
It uses BF16 weights and scheduled-direct collectives.
The minimum passing layout is shown separately because the fastest LPX layout uses more chips.

| Architecture | Rubin + LP30 | 1K tokens/s | 10K tokens/s | 100K tokens/s | Pass all |
| --- | --- | ---: | ---: | ---: | --- |
| Pure Rubin | 32 + 0 | 2,203 | 2,148 | 1,728 | No |
| Pure LPX, fastest | 0 + 352 | 4,174 | 4,149 | 3,926 | Yes |
| Pure LPX, minimum passing | 0 + 248 | 3,751 | 3,723 | 3,464 | Yes |
| AF: Rubin attention, LPX FFN | 8 + 108 | 1,060 | 1,047 | 936 | No |
| AF: only global attention and head on Rubin | 8 + 160 | 2,549 | 2,476 | 1,934 | No |
| AF: LPX attention, Rubin FFN | 8 + 88 | 924 | 922 | 905 | No |
| PD: Rubin prefill, LPX decode, minimum passing | 1 + 248 | 3,751 | 3,723 | 3,464 | Yes |
| PD: LPX prefill, Rubin decode | 32 + 248 | 2,203 | 2,148 | 1,728 | No |

The FP8 minimum serving all three workloads is 160 LP30 chips, or one Rubin GPU plus 160 LP30 chips for PD.
Its predicted speeds are 4,213, 4,177, and 3,854 tokens/s.
Under native-ring communication, no BF16 or FP8 layout passes all three workloads.
The best native-ring LPX decoder reaches 2,772 tokens/s at 100K with BF16 and 3,017 tokens/s with FP8.
FP8 native-ring passes the separately provisioned 1K target with 84 chips, while its 10K maximum of 3,393 tokens/s falls below the common 3,400 tokens/s threshold.

The conventional AF layout exchanges activations for all 60 layers and pays substantial serial communication overhead.
The global-only variant places local attention and all FFNs on LPX and exchanges activations around ten global attention layers plus embedding/head endpoints.
Its speed improves, but Rubin attention and head costs and the remaining boundaries still prevent it from meeting the targets in this model.
PD leaves the complete decode path on LPX, so changing its prefill allocation changes TTFT and inventory while preserving generation speed.

## Rubin-to-LPX ratio under the generation-only objective

The measured generation path determines the minimum decoder inventory.
One Rubin prefill GPU minimizes the searched PD chip count because the output-speed metric excludes prefill.
The minimum PD ratios are 1:168 at 1K and 10K, and 1:248 at 100K with BF16.
Additional prefill GPUs do not improve generation speed or change acceptance under this objective.
Pure LPX uses one fewer chip than PD for the same modeled generation speed when TTFT is unconstrained.
TTFT is retained as auxiliary diagnostic data.
The following BF16 table keeps the minimum 248-chip LPX decoder and varies Rubin TP.

| Rubin GPUs | Rubin : LP30 chips | TTFT at 1K | TTFT at 10K | TTFT at 100K |
| ---: | --- | ---: | ---: | ---: |
| 1 | 1 : 248 | 0.029 s | 0.272 s | 4.289 s |
| 2 | 1 : 124 | 0.016 s | 0.157 s | 2.350 s |
| 4 | 1 : 62 | 0.010 s | 0.097 s | 1.357 s |
| 8 | 1 : 31 | 0.007 s | 0.067 s | 0.867 s |
| 16 | 2 : 31 | 0.005 s | 0.053 s | 0.623 s |
| 32 | 4 : 31 | 0.005 s | 0.046 s | 0.507 s |

Pure LPX at the same 248-chip layout has approximately 0.018 s, 0.196 s, and 2.955 s TTFT.
PD with one Rubin GPU can have slower prefill than this LPX allocation.
KV handoff includes source and destination NIC budgets, pipeline placement, and target cache replication.
Host packing, API networking, hidden reasoning, and actual streaming overhead remain unmodeled, so these TTFT estimates are conditional.

A physical LPX rack contains 256 LP30 chips, and the modeled Rubin NVL72 rack contains 72 GPUs.
The 248-chip decoder fits within one LPX rack and leaves eight chips unallocated.
A mixed deployment requiring whole racks would occupy at least one LPX rack and one Rubin rack even when it allocates only a small part of the Rubin rack.
Rack purchasing, prices, power optimization, and multi-user pool sizing are outside this single-client chip-count objective.

## Sensitivity of the minimum passing LPX decoder

The following table requires one layout to satisfy all three workloads.
Each row changes one setting from the scheduled-direct baseline while retaining all other inputs.

| Changed assumption | BF16 minimum LP30 | FP8 minimum LP30 |
| --- | ---: | ---: |
| Baseline | 248 | 160 |
| Output native/AA token ratio 0.8 | 248 | 124 |
| Output native/AA token ratio 1.2 | No passing layout | 176 |
| Input native/AA token ratio 1.2 | 248 | 160 |
| Operator overhead 0.05 microseconds | 248 | 124 |
| Operator overhead 0.4 microseconds | No passing layout | No passing layout |
| FP8 KV cache | 208 | 104 |
| Memory reserve 10% | 248 | 160 |
| Cross-rack startup 0.5 microseconds | 248 | 160 |

The BF16 baseline clears the 100K target by approximately 1.9%, which is smaller than the uncertainty in its unmeasured inputs.
The inventory numbers therefore support architecture exploration and calibration priorities, with no demonstrated real-hardware minimum.
The most consequential missing measurements are native token conversion, operator and collective latency under the actual compiler, and the serving precision/quality configuration.
Changing architectures alone cannot resolve those uncertainties.

## Reproduction and evidence

Run the complete inventory scan with the default four precision/communication profiles:

```bash
python scripts/explore_lpx_aa.py \
  --requests 50 \
  --warmup 5 \
  --output outputs/lpx-aa-20261019/main
```

Optimize a separate deployment for a single workload:

```bash
python scripts/explore_lpx_aa.py \
  --workloads 1k \
  --output outputs/lpx-aa-20261019/single-1k
```

Repeat with `10k` and `100k` to reproduce the other per-context minima.
Sensitivity runs use `--rho-out 0.8`, `--rho-out 1.2`, `--rho-in 1.2`, `--kernel-us 0.05`, `--kernel-us 0.4`, `--kv-dtype fp8`, `--reserve-fraction 0.1`, and `--cross-latency-us 0.5` in separate output directories.
The runner reuses completed results only when its parameter, code, and input fingerprints match.
Use `--no-resume` to force a new execution.
Optional controls include `--weight-dtypes`, `--collectives`, `--timing-window`, `--prefill-chunk`, `--max-lpu`, and `--max-gpu`.
Every profile exports feasible points, rejected placement queries, selected inventories, and serial replay timestamps.
The summary records assumptions and input digests.

The [saved study results](../experiments/lpx/aa-architecture-results.json) contain compact selections for all 12 runs, settings, hardware/source digests, and validation totals.
Local detailed results are in `outputs/lpx-aa-20261019/`, including `points.csv`, `architecture-comparison.png`, and `architecture-comparison.svg`.
The 48 profiles evaluate 12,120 capacity-valid configuration points across the main, separate-context, and sensitivity runs.
Selected layouts produce 47,300 measured deterministic request replays with maximum relative replay/analytical TPS discrepancy below 1.4e-12.
This agreement checks timing composition and serial replay implementation, and does not calibrate the hardware model.
All 505 repository unit tests pass, including eight new tests for Gemma shapes, PP capacity, vectorized attention, PD semantics, generation-only acceptance, NIC limits, and invalid inputs.
The repository Black format check passes.
