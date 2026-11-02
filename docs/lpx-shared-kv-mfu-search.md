# Experimental shared global KV storage for maximum MFU

This experiment tests a different key/value (KV) cache representation for Gemma 4 31B at 100K context.
It extends the [maximum model FLOPs utilization (MFU) search](lpx-maximum-mfu-search.md) with explicit storage, reconstruction costs and native static random-access memory (SRAM) admission.
Model quality, matching-cache acceptance and compiled LP30 execution remain unverified.
The format is an experimental candidate with a separate result category.

## Representation and arithmetic

The [Transformers v5.20.0 reference](https://github.com/huggingface/transformers/blob/v5.20.0/src/transformers/models/gemma4/modeling_gemma4.py#L1108-L1116) derives global K and V from the same projection, applies learned normalization weights to K, and applies rotary position embedding (RoPE) to K.
V normalization has no learned scale.
In real arithmetic, a nonrotary key coordinate therefore equals the corresponding normalized value coordinate multiplied by its learned key-normalization weight.
Multiplying those query coordinates by the same weight preserves their dot product.
These observations motivate the experimental representation.

The [Google implementation](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/math/_positional_embeddings.py#L43-L69), checked on November 3, 2026, splits the complete head into two halves and pads inactive rotation angles with zero.
For head width 512 and rotary proportion 0.25, active coordinates are `[0:64]` and `[256:320]`.
The preceding contiguous-128-coordinate algebra example remains a generic identity example; this implementation uses the full-head coordinate order.

Each global KV head and token stores:

| Item | Bytes |
| --- | ---: |
| Full V in eight-bit floating point (FP8), E4M3FN | 512 |
| Active post-RoPE K coordinates in FP8 | 128 |
| Independent single-precision (FP32) scales for V and active K | 8 |
| Total | 648 |

The legacy two-array FP8 representation charges 1,024 bytes per global KV head and token.
The new physical format saves 36.71875% of those global-cache bytes.
Local sliding-window layers retain their existing representation.
Each scale belongs to one token and one KV head.
The reference codec uses maximum-absolute-value scaling, finite E4M3FN saturation and nearest-even rounding.

FP8 quantization and multiplication by learned normalization weights do not commute.
BF16 normalization and query-folding rounding introduce additional differences.
The representation therefore requires model-quality validation against the intended serving precision and workload.
Existing GPU acceptance measurements use another cache policy and cannot establish acceptance for this format.

## Native memory admission

`ModelSpec.global_kv_layout` defaults to `separate` and accepts experimental `value_rope_fp8` for Gemma 4 with FP8 KV, bfloat16 (BF16) activations, Q/K normalization and tied global projections.
The selected layout is recorded on each native layer.
Native head partitions use the physical bytes per token and block, including both FP32 scales.
The production resident-KV allocator retains block rounding and the compiler watermark.
The legacy cache allocator rejects this experimental policy.
Measured operator tables are rejected because their shape keys do not distinguish this cache representation.

Reusable scratch retains the original allocation and adds assembled-key buffers, query-folding space, scale buffers and append-quantization space.
The default `conservative` workspace policy uses the larger assembled-key envelope for both original kernels.
Assistant remote-attention slots additionally reserve independent cache-owner scratch.
At tensor parallelism (TP) 16 and decode context parallelism (DCP) 4, each assistant remote slot reserves 2 MiB per transformer chip for this format, compared with 1 MiB for the existing format.

The single-token placement uses ten pipeline stages, 16 transformer chips per stage and three separate embedding/logits chips, for 163 allocated chips.
Each stage holds six consecutive layers.
With the conservative workspace policy, fourteen independent requests reserve 102,000 KV positions each and pass native admission.
The busiest transformer chip holds 201,630,464 bytes of weights, 8,511,436 bytes of buffers and 268,072,448 bytes of KV.
Its physical headroom is 21,785,652 bytes.
The scratch increase over the existing format is 539,136 bytes per chip.

Fifteen requests fail that policy's compiled KV budget: 287,220,480 required bytes exceed 286,964,128 available bytes on the limiting rank.
Raw SRAM headroom alone does not establish admission because the compiler watermark reserves additional capacity.
The earlier 5/8 storage-only screen omitted scale bytes and reconstruction scratch and predicted fifteen requests for this placement.
The conservative physical-format policy supports fourteen.

## Kernel-specific tile workspaces

`memory.shared_kv_workspace` accepts `conservative`, `assemble` and `split_dot_packed`.
Its default preserves the original reservation.
The two kernel-specific policies require a matching `operator.shared_kv_kernel` and the experimental cache format.
Native compilation rejects a mismatch.
Default configuration serialization and the existing-cache baseline remain unchanged.

The explicit policies retain the input Q/K/V, online-softmax state and context-parallel communication buffers.
These policies assume query folding overwrites the existing Q buffer after normalization.
The cache tiles contain the physical FP8 payload and scales plus BF16 decoded V and active K coordinates.
The assembly policy additionally reserves a complete virtual K tile.
The packed policy reserves contiguous nonrotary key columns and an additional FP32 score tile.
Both retain FP32 append-quantization scratch.

For one query row at TP16/DCP4, the global rank owns one KV head and eight query heads.
It uses two buffers of 256 key positions each.

| Workspace component | Assemble (bytes) | Packed split dot (bytes) |
| --- | ---: | ---: |
| Physical FP8 payload and FP32 scales | 331,776 | 331,776 |
| Decoded V and active K | 655,360 | 655,360 |
| Complete or nonrotary key tiles | 524,288 | 393,216 |
| Additional score tile | 0 | 8,192 |
| Append quantization | 2,568 | 2,568 |
| Q/K/V, base scores, online state and combine buffers | 67,712 | 67,712 |
| Total live attention scratch | 1,581,704 | 1,458,824 |

Both explicit policies pass native admission for fifteen requests and reject sixteen on the six-layers-per-stage placement.
The explicit packed kernel charges BF16 gather reads and packed writes for the 384 nonrotary coordinates.
That adds 1,536 shuffle bytes per read KV-head vector to the original strided split-dot hypothesis.
The [NVIDIA architecture description](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) describes programmable switch execution modules for permutations and transposition.
It does not establish the throughput of either reconstruction kernel.

The [CPU tiled-attention oracle](../scripts/lpx/kv_attention.py) checks assembly and packed dot products with online softmax against the same quantized full-context reference.
It decodes one tile at a time and supports in-place logical Q folding.
The oracle uses FP32 arithmetic and does not measure BF16 LP30 execution, hardware scratch peaks or SRAM bank placement.

| Explicit workspace/kernel | Resident requests | Isolated forward (us) | Aggregate tokens/s | MFU | Fixed-DAG MFU bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| Assemble | 15 | 288.618678 | 39,898.36 | 2.57137% | 3.34947% |
| Packed split dot | 15 | 274.968893 | 42,196.39 | 2.71947% | 3.51575% |

These runs use four warm-up rounds and sixteen measured rounds on 163 chips.
The extra admitted request does not improve assembly throughput at the supplied issue schedule.
Charging the packed data path also removes much of the original strided hypothesis's timing benefit.
Neither control exceeds the 3.41960% strided-cache candidate.

## Family capacity and timing coverage

A native-footprint screen evaluates 36 tensor/context-width and ownership combinations for each of three workspace/kernel policies, for 108 frontiers.
Tensor widths are 1, 2, 4, 8, 16 and 32, and context width divides tensor width.
Ownership modes are global-only and all attention layers, with equivalent context-width-one modes evaluated once.
Pipeline depths range from one through sixty, with three separate boundary chips added to each inventory.
Each policy has 35 nonempty families.
None reaches the 256-request search limit.

The timing comparison checks 233 family/owner cases at 101,000 positions, including repeated shape/context keys across ownership modes.
The original strided split-dot estimate is no slower than the assembly or packed estimate at every checked key.
This is a comparison of the supplied analytical kernels.
It does not establish hardware latency dominance or cover larger batches.

Thirty-five native transformer captures retain admitted reference partitions.
They project onto 1,664 supplied pipeline points.
Their necessary envelope uses the fastest priced global kernel together with the largest supplied native capacity frontier across the three policies.
The envelope deliberately allows incompatible timing and memory choices, removes direct pipeline handoffs and external boundary execution, and retains at least three boundary-weight chips.
Projection across pipeline cuts assumes unchanged per-layer kernels and intra-stage collective costs.
These relaxations bound all three supplied batch-one policies within that footprint model.

Thirty of the 35 families have envelopes below 3.41960% MFU.
The five remaining families contain 24 pipeline points above that threshold.
They are TP16/DCP4 global-only, TP16/DCP8 global-only, TP32/DCP8 global-only, TP32/DCP16 global-only and TP16/DCP4 with context partitioning on all attention layers.
Within an inventory of at most 256 chips, only the two TP16 global-only families exceed that threshold in the necessary envelope.
The largest relaxed bound is 4.59297%.
It does not establish an attainable memory/timing combination.

Five full boundary-plus-transformer programs check the selected native reference partitions.
With three boundary chips, TP16/DCP4 on all attention layers at PP20 has a 3.20734% bound, and TP32/DCP16 at PP10 has a 3.37986% bound.
Those two supplied programs are excluded without a concurrent replay.
TP16/DCP8/PP10 reaches 3.37489% MFU at 52,366.09 tokens/s on 163 chips.
TP32/DCP8/PP10 reaches 2.73193% at 83,999.00 tokens/s on 323 chips.
The TP16/DCP4/PP12 reference has a much lower replay rate than its 4.01317% necessary bound under aggressive pacing.
Additional issue-interval and active-request screens retain that scheduling sensitivity explicitly.
Eighteen short phase probes vary active request count and issue spacing for the two TP16 references.
The PP12 reference reaches at most 3.00730% in those supplied probes, with -0.6314% half-window change at that point.
The DCP8/PP10 reference retains its 3.37489% plateau across the faster tested issue intervals.
Replays below a bound do not exclude improved schedules, and the paid three-chip boundary bounds do not exclude other boundary-pool sizes.

The supplied cuts, kernels and one-row workspace policies define this finite coverage.
Other batches, partitions with different buffer envelopes, strided-kernel implementation, model quality and executable schedules remain open.

## Independent request batching

The I/O-pool driver accepts `--batch` for independent requests, while `queries` in the reusable profile helper counts verification positions within each request.
The total scratch-row count is their product.
Every batch row retains a separate native KV lease and a separate context.
Resident requests must form complete fixed-width cohorts.
Boundary computation and communication use the total row count, and useful-token credit counts one committed output per independent request in decode.

`--mode bound` admits and captures the full I/O-plus-transformer program without a concurrent replay.
Its output contains necessary bounds and no observed MFU.
For the existing separate-cache TP16/DCP4/PP10 placement, batch two admits eight resident requests and has a 1.98941% MFU bound.
This excludes that supplied batch-two candidate from improving the 3.03914% existing-format incumbent.

The extended screen checks every independent cohort width from two through nine for that placement.
Each bound includes the complete boundary program, native scratch, full communication reservation and all 163 allocated chips.

| Batch width | Resident requests | Necessary MFU bound |
| ---: | ---: | ---: |
| 2 | 8 | 1.98941% |
| 3 | 9 | 1.86333% |
| 4 | 8 | 1.41853% |
| 5 | 5 | 0.77514% |
| 6 | 6 | 0.82638% |
| 7 | 7 | 0.86727% |
| 8 | 8 | 0.90068% |
| 9 | 9 | 0.92853% |

Batch ten has no admitted complete cohort under this memory policy.
The native footprint frontier admits at most nine resident requests when ten query rows are reserved.
Larger query-row reservations do not increase this memory frontier, so they cannot form a complete wider cohort in this placement.
These results cover this pipeline placement and fixed-width cohorts.
Other pipeline depths and mixed-width schedules require separate checks.

## Reconstruction costs and decode replay

The attention estimate charges the physical cache reads and appends, FP8 conversions, FP32 scale movement, vector multiplication and append-quantization reductions.
Query multiplication by learned key-normalization weights is a dependent prefix before matrix work.
It includes vector service, shuffle traffic, SRAM traffic and pipeline fill.
The matrix-work numerator remains the target model's original useful work per committed token.
Reconstruction and speculation receive no additional useful-token credit.

The `assemble` kernel reconstructs a complete BF16 virtual key tile and charges its reads and writes.
The `split_dot` kernel computes rotary and nonrotary score contributions separately and charges their FP32 sum and an additional invocation fill.
The split kernel's matrix FLOPs equal the original full-width dot product.
Its streaming and execution-unit efficiencies remain analytical hypotheses.

All rows use 101,000 positions for decode timing, fourteen resident requests, one query row, four warm-up rounds and sixteen measured rounds.
The denominator includes the FP8 peak of all 163 allocated chips.

| Kernel | Isolated forward (us) | Aggregate tokens/s | Tokens/s per request | MFU | Fixed-DAG MFU bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| Assemble complete key | 288.618678 | 39,898.36 | 2,849.88 | 2.57137% | 3.12618% |
| Split rotary/nonrotary dot | 226.448347 | 53,059.81 | 3,789.99 | 3.41960% | 3.98445% |

Measurement-half differences are below 2e-12 relative for these runs.
A separate fifteen-point phase screen varies active request count from ten through fourteen and changes the issue interval for both kernels.
The split kernel reaches the same 53,059.81 tokens/s plateau at thirteen and fourteen active requests within the tested schedules.
The necessary fixed-DAG bound remains above the observed plateau and does not certify schedule optimality.

The split-kernel candidate exceeds the existing-format 3.03914% decode incumbent under its experimental cache assumptions.
The assembled-key candidate has lower MFU despite admitting more requests.
These comparisons demonstrate why storage-only scaling cannot predict the throughput benefit.
The existing-format incumbent remains recorded separately because the new cache policy lacks a matching model-quality measurement.

## Shared-cache assistant combination

The combination uses the official BF16 assistant on two additional chips, K=3 proposals, three independent remote-attention slots and four target verification rows.
All fourteen target requests pass native admission, with minimum transformer headroom 12,080,820 bytes.
The total inventory is 165 chips, and the remote-attention reservations contribute 6,291,456 bytes per transformer chip.

| Issue interval (us) | Warm-up / measured rounds | Verification rounds/s | Half-window change | MFU at assumed alpha=0.9 |
| ---: | --- | ---: | ---: | ---: |
| 27.535753 | 4 / 16 | 22,495.33 | -1.1940% | 4.92536% |
| 45.0 | 8 / 32 | 21,179.11 | -0.3005% | 4.63717% |

Here alpha is an assumed identical conditional acceptance probability, giving 3.439 committed tokens per round at K=3.
The two runs change both pacing and measurement length, so their difference cannot be attributed to measurement length alone.
The longer run still has nonzero drift and does not establish a maximum.

Transporting the measured FP8-target/BF16-assistant GPU accepted lengths gives the following conditional estimates.
Both rows retain mismatches for KV precision and global cache policy.

| Replay | Quotes MFU | Summary MFU |
| --- | ---: | ---: |
| Initial 4 / 16 | 3.96807% | 3.86517% |
| Paced 8 / 32 | 3.73590% | 3.63901% |

These estimates use pooled committed tokens per verification round from the existing sixteen-book 100K document groups.
The new cache representation has no matching acceptance or output-quality measurement.
The supplied combination therefore remains an experimental transport and acceptance scenario.

## Numerical reference and regression controls

The [CPU reference codec](../scripts/lpx/kv_codec.py) stores exactly 648 bytes per token and global KV head.
The synthetic check uses 256 key positions beginning at position 100,000, four KV heads, 32 query heads, head width 512 and local random seed 42.
The full-precision dot-product identity has maximum absolute error 2.8422e-14.
After FP8 encoding, maximum score error is 2.33546 and mean absolute score error is 0.485489 against the synthetic full-precision scores.
This check uses FP32 query folding and does not measure BF16 LP30 arithmetic or model outputs.
Tests additionally cover the coordinate order, physical byte count, quantization bounds, rounding ties, zero vectors, invalid payloads, incompatible configurations and admission rejection.

A baseline capture using `separate` reproduces all 96,574 saved DAG node rows exactly.
Its isolated duration remains 212.052722 us, and its native memory admission report is unchanged.
The baseline configuration and result serialization omit the new optional field when its default is used.

## Reproduction

```bash
python scripts/search_lpx_io_pool.py \
  --tp 16 \
  --kvp 4 \
  --pp 10 \
  --pool-chips 3 \
  --global-kv-layout value_rope_fp8 \
  --shared-kv-kernel split_dot \
  --warmup-rounds 4 \
  --measured-rounds 16 \
  --output outputs/lpx-shared-kv/split-dot
```

`--shared-kv-kernel assemble` selects the reconstruction control.
Both switches are also available on `scripts/search_lpx_assistant.py`.
The workspace switch is available on both drivers and must match the selected kernel when it requests an explicit tile layout.
The assistant's global attention uses the same representation at the target cache owners, and its local layers retain their original cache policy.
The empirical-acceptance projection tool adds `global_kv_layout` to its mismatch list whenever this format is selected.

The explicit packed control is reproduced with:

```bash
python scripts/search_lpx_io_pool.py \
  --global-kv-layout value_rope_fp8 \
  --shared-kv-kernel split_dot_packed \
  --shared-kv-workspace split_dot_packed \
  --warmup-rounds 4 \
  --measured-rounds 16 \
  --output outputs/lpx-shared-kv/compact-packed
```

The existing-cache independent-batch bound uses:

```bash
python scripts/search_lpx_io_pool.py \
  --batch 2 \
  --mode bound \
  --output outputs/lpx-shared-kv/separate-batch2
```

The [portable search record](../experiments/lpx/maximum-mfu-search-progress.json) retains admission controls, timing results, phase probes, source links and regression evidence.
Full profiles and reproduction scripts remain under `outputs/lpx-maximum-mfu-20261103`.
Open work includes matching-cache model quality, accepted-token measurements, further layouts and kernels, complete prefill/serving behavior, physical bank placement and LP30 compiler validation.
