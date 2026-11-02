# Complete 100K Gemma serving replay

Study date: 2026-11-03.
This experiment includes a 100,000-token prompt and 2,000 generated tokens on one fixed LP30 allocation.
It extends the [maximum MFU search](lpx-maximum-mfu-search.md) with prompt computation, bounded pipeline buffers, token feedback and request completion.
The existing-format resident-prefix decode incumbent remains 3.03914% model FLOP utilization (MFU).
The experimental shared-cache decode incumbent remains 3.41960%, conditional on matching model quality.
The complete-request metric includes prompt work and has its own ranking.
Every result below uses the existing separate key/value format.

## Work and denominator

The target is the repository's Gemma 4 31B text model with eight-bit floating-point (FP8) weights and key-value (KV) cache and bfloat16 (BF16) activations and attention.
General matrix multiplication (GEMM) efficiency is 0.30, and attention efficiency is 0.50.
All allocated transformer and boundary chips contribute their FP8 peak to the denominator.
The experiment preserves the 5,000,000-byte safety reserve on each chip and reserves every resident request through context 102,000.
Input and output lengths use model tokens under the existing identity token-unit assumption.

Prompt processing computes transformer layers for every input token and the language-model (LM) head for the terminal prompt row.
That terminal row produces output token one.
The remaining 1,999 output tokens require 1,999 autoregressive forwards at contexts 100,001 through 101,999.
The useful-work ledger sums exact causal attention pairs, including the attention diagonal and the sliding window of each layer.
Chunk boundaries do not change useful-work credit.
Padding, replicated projections, scalar normalization, rejected speculation and communication receive no additional useful credit.

| Useful matrix work per completed request | FLOPs |
| --- | ---: |
| Prompt transformer linear work | 5,857,345,536,000,000 |
| Prompt terminal LM head | 2,818,572,288 |
| Prompt attention | 3,443,746,773,401,600 |
| Decode transformer linear work | 117,088,337,264,640 |
| Decode LM heads | 5,634,326,003,712 |
| Decode attention | 135,670,294,118,400 |
| Total | 9,559,488,085,360,640 |

```text
complete_request_MFU = completed_requests * useful_FLOPs_per_request
                       / (elapsed_seconds * allocated_chips * FP8_peak_per_chip)
FP8_peak_per_chip = 315e15 / 256
```

Prompt work supplies approximately 97.30% of the useful FLOPs of this fixed workload.
Whole-request output tokens per second (TPS) includes the time spent processing the prompt.
Time to first token (TTFT) includes queueing from the request's recorded arrival.
Time per output token (TPOT) measures gaps between consecutive generated tokens of the same request.
The cohort begins with empty KV and ends after every request completes.
With multiple cycles, a client submits its replacement request when its previous request completes.
There is no work already in flight at the measurement start or remaining after its end.

Under the supplied shape-independent GEMM and attention efficiencies, a relaxed matrix-only ceiling is approximately 27.9099% for this workload.
This ceiling divides total useful work by `linear_work / 0.30 + attention_work / 0.25` in FP8-peak units.
The 0.25 factor combines attention efficiency 0.50 with the catalog assumption that BF16 peak is half FP8 peak.
It removes replication, communication, scalar pipelines, capacity placement and dependencies.
It is a bound within the stated compute assumptions and does not establish a physical LP30 ceiling.

## Execution and memory model

The [driver](../scripts/search_lpx_serving_mfu.py) captures native multiquery transformer operator graphs as causal prompt chunks.
The production aggregate-prefill route is outside this replay.
Whole-buffer collectives are selected with overlap and early output disabled.
This avoids expanding every prompt chunk into individual 320-byte collective fragments.
LP30 fabric wire service, routing resources and propagation remain explicit.

Each pipeline-parallel (PP) stage has one active chunk and one reserved incoming chunk.
A stage reserves its successor's incoming slot before execution.
An explicit entry copy completes before the stage releases its own incoming slot.
The outgoing handoff retains its source activation until transfer completion.
The terminal stage retains its hidden state until the boundary pool receives the output row.
The controller checks causal KV order separately for every request at every stage.
Known prompt chunks from the same request can occupy different PP stages concurrently.
Generated tokens enter the pipeline after their predecessor's greedy selection completes.
One shared boundary operation is active at a time, with output operations prioritized over new prompt chunks.

The transformer communication floor reserves two BF16 hidden-state buffers for the maximum chunk width.
The boundary pool reserves embedding workspace for the complete input chunk and logits workspace for one terminal output row.
Charging prompt-sized logits for every input row would reserve memory for work that this workload does not perform.
The default reservation behavior of the existing decode tools remains unchanged.
The native admission ledger still includes FP8 scale storage, compiler KV watermark, scratch and auxiliary weights.

Prompt contexts are grouped into bands to limit the number of captured graphs.
Every actual context in a band is priced analytically, and each captured attention-node budget is extended when any covered context is slower.
This check retains nonmonotonic effects of context sharding and vector padding.
The native multiquery attention cost uses its existing causal approximation, while the useful-work ledger counts exact integer pairs.
Physical SRAM banks, compiler execution, unit rates and matched model quality remain unverified.
External request transport and tokenization are outside the timing model.

Chunked prefill and interactions with PP scheduling are supported as research directions by the [SARATHI paper](https://arxiv.org/abs/2308.16369) and [Sarathi-Serve paper](https://arxiv.org/abs/2403.02310).
Their GPU measurements do not calibrate these LP30 programs.

## Capacity coverage

The capacity screen evaluates 36 tensor/context/ownership families at chunk widths 128, 256 and 512.
Tensor parallelism (TP) ranges over 1, 2, 4, 8, 16 and 32.
Context parallelism divides TP, with full-attention-only and all-attention ownership evaluated separately where distinct.
PP ranges from one to 60, reaching 1,924 allocated chips including a four-chip boundary pool.
The screen retains every feasible contiguous-cut PP depth under the supplied native memory envelopes.
Its 108 frontiers contain 97 nonempty families and 4,604 feasible PP points.
Three families fail native setup, and eight additional frontiers contain no feasible PP point.
No nonempty frontier reaches the 256-request search limit.
Capacity coverage does not imply timing coverage of every point or every possible partition.

At TP16, full-attention context width four and PP10, chunk widths 128 and 256 admit at most nine resident requests.
Chunk width 512 admits at most eight requests.
The boundary pool requires at least three chips at widths 128 and 256 and four chips at width 512.
These limits apply to the complete-serving workspace policy.
The one-row decode policy has a different capacity frontier.

## Completed timing probes

`N` is the resident client count.
The table uses four prompt context bands and one complete request per client.
All chips, including the shared boundary pool, contribute to MFU.
Latency values are modeled and include the finite cohort's queueing behavior.

| Chips | TP / PP / context width | Chunk | N | Cycles | End-to-end MFU | Output TPS | Mean TTFT (ms) | Mean TPOT (us) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 163 | 16 / 10 / 4 | 128 | 9 | 1 | 15.08248% | 6,328.9 | 2,414.203 | 215.007 |
| 163 | 16 / 10 / 4 | 256 | 1 | 1 | 6.83281% | 2,867.2 | 268.095 | 214.836 |
| 163 | 16 / 10 / 4 | 256 | 8 | 1 | 15.10364% | 6,337.8 | 2,093.352 | 215.652 |
| 164 | 16 / 10 / 4 | 512 | 8 | 1 | 15.22529% | 6,428.0 | 2,059.808 | 214.703 |
| 84 | 8 / 10 / 2 | 512 | 1 | 1 | 9.02510% | 1,951.6 | 500.919 | 262.064 |
| 164 | 8 / 20 / 2 | 512 | 6 | 1 | 12.62279% | 5,329.2 | 1,710.317 | 270.748 |
| 164 | 32 / 5 / 8 | 512 | 6 | 1 | 10.51047% | 4,437.4 | 2,043.997 | 330.231 |

The wider chunk trades a larger boundary pool and fewer KV slots for fewer collective and launch operations per prompt.
The result depends on that tradeoff, the partition and the complete request mix.
Further search includes additional partitions, independent context-band and request-replacement controls, mixed prefill/decode batches, heterogeneous stages and physical validation.

A refinement uses 16 context bands and two consecutive requests per client on the 164-chip TP16/PP10/context-width-four allocation with chunk 512 and eight clients.
It completes 16 requests, 1,600,000 input tokens, 32,000 output tokens and 31,984 autoregressive decode forwards in 4.796426477 modeled seconds.
Its end-to-end MFU is 15.80238%, and whole-request output throughput is 6,671.6 TPS.
Mean TTFT is 1,967.402 ms, and mean TPOT is 215.478 us.
The maximum inter-token gap is 23.418 ms, while the 99th-percentile gap is 213.887 us.
The experiment imposes no inter-token latency service-level objective.
The refinement changes context granularity and request replacement together; their individual effects require separate controls.
It retains 10,440,372 bytes of minimum transformer SRAM headroom and 109,113,280 bytes of boundary-pool headroom.
This result is the highest observed complete-request MFU in the supplied probes.
The resident-prefix decode ranking remains unchanged.

An independent sum of occupied resource time over every profile invocation bounds the refined fixed workload at 21.08620% MFU.
The sum includes stage entry copies, boundary work and every prompt and decode invocation.
It agrees with all collected occupied-resource counters for the two instrumented frozen-profile controls within numerical tolerance.
The sum relaxes dependencies, propagation and scheduling constraints and does not establish that the upper bound is attainable.

## Context and request-scheduling controls

Frozen transformer graphs separate prompt context granularity from the number of consecutive requests per client.
The allocation, admission, chunk width, target precision and useful-work ledger remain fixed at TP16/PP10/context width four, 164 chips, eight clients and chunk 512.

| Context bands | Requests per client | End-to-end MFU | Whole-request output TPS | Mean TTFT (ms) | Maximum token gap (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 15.22529% | 6,428.0 | 2,059.808 | 3.510 |
| 4 | 2 | 15.19663% | 6,415.9 | 2,062.997 | 23.418 |
| 16 | 1 | 15.86469% | 6,697.9 | 1,959.489 | 3.510 |
| 16 | 2 | 15.80238% | 6,671.6 | 1,967.402 | 23.418 |

Finer context bands reduce conservative attention reservations in both finite horizons.
The improvement comes from tighter timing approximation under the same workload.
Request replacement reduces measured throughput slightly in both supplied controls.
The one-cohort value of 15.86469% is the highest supplied finite complete-request value and does not establish sustained-serving throughput.
More request cycles and explicit arrival processes remain open.

The input scheduler exposes a prompt rotation quantum measured in chunks.
Quantum one rotates the prompt input queue after every chunk.
Quantum 196 covers a complete 100K prompt at chunk width 512, including its partial tail.
Optional decode priority chooses the earliest waiting decode task before waiting prompt tasks.
Every stage remains nonpreemptive and retains the same active and incoming buffer limits.
These controls order complete single-request chunks.
Mixed-request matrix and attention fusion remains outside them.

| Prefill quantum | Decode priority | End-to-end MFU | Output TPS | Mean TTFT (ms) | Mean TPOT (us) | Maximum token gap (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Disabled | 15.86469% | 6,697.9 | 1,959.489 | 214.703 | 3.510 |
| 16 | Enabled | 15.35629% | 6,483.3 | 2,001.148 | 233.224 | 16.757 |
| 196 | Disabled | 15.58727% | 6,580.8 | 1,130.596 | 650.624 | 1,746.324 |
| 196 | Enabled | 9.59072% | 4,049.1 | 1,595.638 | 1,162.093 | 15.935 |

Every row uses 16 context bands and one request per client.
All four rows execute identical operator counts and occupied physical-resource work.
The maximum busy-time difference across corresponding counters is below 8e-12 seconds.
The default scheduler reproduces the previous frozen one-cohort control's completion time, request timings, output TPS and operation counts exactly.
The supplied input-order changes do not improve MFU.
The complete-prompt policy without decode priority reduces mean TTFT while introducing a long generation stall.
Uniform mixed batching remains a separate research direction supported by the [Sarathi-Serve study](https://arxiv.org/abs/2403.02310).
Its GPU performance results do not calibrate LP30 scheduling or establish compiler support for these controls.

## Additional resident-prefix decode probes

Two shared-cache probes extend the existing decode search under its original fixed-context metric and one-row scratch policy.
Both use TP16, full-attention context width four, the strided shared-KV kernel and three boundary chips.
PP11 admits 14 requests on 179 chips, with a fixed-DAG bound of 3.61429% and a best short phase screen of 3.11848%.
PP13 admits 17 requests on 211 chips, with a fixed-DAG bound of 3.69464% and a best short phase screen of 0.80940%.
Each point checks three issue intervals with two warm-up rounds and eight measured rounds.
Their poor schedules leave the broader phase and partition search open.
They do not improve the experimental decode incumbent of 3.41960% or certify the remaining relaxed envelope.

## Reproduction and validation

```bash
python scripts/search_lpx_serving_mfu.py \
  --config experiments/lpx/fidelity-study.json \
  --output outputs/lpx-complete-serving/q512-n8 \
  --tp 16 \
  --kvp 4 \
  --pp 10 \
  --pool-chips 4 \
  --inputs 100000 \
  --outputs 2000 \
  --chunk 512 \
  --clients 8 \
  --cycles 1 \
  --context-bands 4 \
  --global-kv-layout separate
```

`result.json` records configuration, admission, program identities, context reservations, request timings, resource statistics, useful work and complete-request MFU.
Resource-statistics collection is enabled from simulation time zero.
The initial timing captures omitted that statistics window.
Frozen-graph replays of the TP16/PP10 and TP32/PP5 chunk-512 configurations retain identical completion times, request timings, output TPS and operation counts while collecting corrected resource statistics.
Their necessary fixed-work resource-demand MFU bounds are 19.97179% and 16.50624%, respectively.
These bounds constrain the captured programs and do not establish attainable schedules or exclude different kernels and partitions.
The driver fixes attention efficiency to 0.50 consistently with the decode search.
`--context-bands` controls prompt reservation granularity.
`--cycles` controls the number of complete requests served sequentially by each client.
`--prefill-quantum` sets the positive prompt-input rotation quantum in chunks and defaults to one.
`--decode-priority` gives waiting generated-token forwards priority over waiting prompt chunks at the pipeline input.
Boundary output copies and greedy selection retain their existing priority.
Both options preserve the SRAM admission and stage-buffer policy.
`--global-kv-layout` also permits the experimental shared representation; the timing table uses `separate`.
The CLI rejects nonpositive workload dimensions, invalid parallel dimensions, insufficient SRAM and requests extending beyond context 102,000.

The new tests cover exact causal work, chunk-independent credit, terminal-only logits, bounded PP wavefront timing, request replacement, partial prompt tails, one-output requests, resource conservation, cross-stage lifetime rejection and a nonmonotonic context reservation.
The existing-format and experimental strided-cache decode programs each retain all 96,574 original node rows exactly after the boundary-helper refactor.
The input-policy full-suite run executes 846 tests with 844 passes and the same two macOS-specific baseline failures involving `/tmp` normalization and missing Linux/NVIDIA process facilities.
All 570 Python files pass the pinned Black formatter check.
