# Explicit stage buffers and additional decode search

Study date: 2026-11-03.
This study extends the [100K Gemma MFU search](lpx-maximum-mfu-search.md) with explicit pipeline-buffer lifetimes and inventories beyond 256 LP30 chips.
Resident-prefix decode retains its representative context of 101,000 tokens and its 102,000-token SRAM reservation.
Every timed decode point below uses the experimental shared global-cache format and strided attention kernel.
Matching model quality and physical compiler execution remain unverified.
The existing-format decode incumbent remains 3.03914% model FLOP utilization (MFU), and the earlier experimental shared-cache incumbent remains 3.41960%.
Complete-request results use a [separate prompt-inclusive metric](lpx-complete-serving-mfu.md#context-and-request-scheduling-controls).

## Additional partitions and pacing

Tensor parallelism (TP) is 16, full-attention context width is four, and the boundary embedding and language-model head pool uses three chips.
Pipeline parallelism (PP) changes the transformer inventory and contiguous layer partition.
The memory policy retains one-query scratch and the 5 MB safety reserve.
Each partition passes native admission for its resident count.

| PP | Allocated chips | Resident requests | Fixed-DAG MFU bound | Best short pacing screen |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 243 | 19 | 3.55825% | 0.69226% |
| 20 | 323 | 26 | 3.58873% | 0.37431% |

Each screen checks three issue intervals with two warm-up rounds and eight measured rounds.
These poor observed schedules do not exclude improved pacing or different partitions.
The 323-chip route inherits the existing uncalibrated cross-rack fabric hypothesis.

## Whole-buffer stage controller

A fixed-prefix benchmark uses native whole-buffer stage parts, one active frame and one incoming frame per stage, explicit entry copies and generated-token feedback.
It repeats the fixed-context snapshot convention of the existing resident decode benchmark.
It drains 80 forwards per request and measures a window after eight warm-up rounds, retaining 32 measured rounds.
All requests release their KV leases, and every stage buffer is empty after the complete drain.
The benchmark does not include prompt processing or increasing-context decode.

| PP | Chips | Requests | Aggregate tokens/s | MFU |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 163 | 14 | 44,992.9 | 2.89971% |
| 12 | 195 | 17 | 50,977.4 | 2.74625% |
| 15 | 243 | 19 | 58,798.1 | 2.54187% |
| 20 | 323 | 26 | 71,279.5 | 2.31825% |

The measurement halves agree within floating-point tolerance for these supplied points.
Bounded stage scheduling substantially improves the poorly paced PP15 and PP20 screens while remaining below the experimental incumbent.
Whole-buffer barriers retain a conservative restriction on overlap within a forward.

## Fine native dependencies and buffer lifetimes

A graph prototype retains every original transformer dependency and adds stage-buffer arena acquisitions and releases.
An arena is an exclusive logical resource held until its explicit release node completes.
Native host-launch work is priced once at the boundary pool.
One active arena covers a stage's transformer operations and outgoing transfer.
An incoming arena is acquired before the producer stage enters and released after an explicit BF16 hidden-state copy into the consumer's active workspace.
The boundary arena serializes embedding and output-head work.
Stage release nodes descend from their acquisitions, and the native replay kernel validates those lifetimes and graph acyclicity.
These graph checks establish the declared logical constraints; physical addresses and SRAM banks require compiler validation.

| PP | Active slots per stage | Incoming slots per stage | MFU |
| ---: | ---: | ---: | ---: |
| 10 | 1 | 1 | 2.92531% |
| 20 | 1 | 1 | 2.34534% |
| 10 | 2 | 1 | 2.92531% |
| 10 | 3 | 1 | 2.92531% |
| 10 | 2 | 2 | 3.41007% |
| 10 | 3 | 3 | 3.41007% |

The PP10 rows retain 14 resident requests on 163 chips.
The PP20 row retains 26 requests on 323 chips.
Multiple program variants assign requests cyclically to slot-specific arena names while sharing the same physical compute, fabric and boundary resources.
The supplied zero-pacing and latency-derived pacing runs reach the same displayed values.
Increasing active slots while retaining a single incoming slot does not improve these supplied schedules.
Paired active and incoming slots improve the captured schedule.
The extra entry-copy cost remains included.

Each additional active slot reserves 2,179,712 bytes of conservative live scratch and 10,752 bytes of colocated dense residual on every transformer chip.
This additional reserve is applied before native compilation and resident-KV admission.
The selected layer cuts remain six layers per stage.
The original 5,000,000-byte safety reserve remains enabled.

| Active / incoming slots | Additional reserve per chip | Minimum physical SRAM headroom |
| --- | ---: | ---: |
| 2 / 2 | 2,190,464 bytes | 19,595,188 bytes |
| 3 / 3 | 4,380,928 bytes | 17,404,724 bytes |

The existing 1,048,576-byte communication floor also covers the declared hidden-state slots.
Two active and two incoming BF16 hidden states require 43,008 bytes per chip at hidden width 5,376.
The corresponding three-plus-three arrangement requires 64,512 bytes.
Attention reduction storage and other protocol buffers remain charged through the unchanged native memory envelope.
One conservative logical partition for two slots assigns 131,328 bytes to four FP32 context fragments per eight-head attention group, 43,008 bytes to hidden states and 21,504 bytes to additional transfer staging.
Its total is 195,840 bytes within the communication floor.
The corresponding three-slot partition totals 283,008 bytes.
Physical bank placement and possible aliasing between protocol allocations remain unverified.

The two-plus-two program receives an extended replay with eight warm-up rounds and 32 measured rounds.
It measures 448 committed target tokens at 52,911.881644 tokens/s and 3.410066841% MFU.
Every one of the 14 requests contributes exactly 32 measured completions.
The two measurement halves differ by approximately 1.1e-13 relative.
The fixed-program relaxed bound is 3.97520%, so the experiment does not establish schedule optimality.
The earlier experimental 3.41960% result remains higher, while this prototype supplies an explicit admitted scratch and buffer arrangement for a nearby point.
The prototype remains a research artifact and does not establish executable LP30 support or matching cache quality.

## Conditional GPU-attention bandwidth bound

The search also considers attention on graphics processing units (GPUs) and feed-forward networks (FFNs) on LP30.
The [NVIDIA architecture article](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) describes that role split.
A necessary bandwidth bound applies when each committed decode token fetches at least one complete separate-format FP8 KV payload from GPU high-bandwidth memory (HBM).
At context 101,000, the target payload is 4,556,390,400 bytes before scale metadata.
Target FFN matrix work is 41,617,981,440 FLOPs per token.
Total useful target work remains 129,261,109,248 FLOPs per token.

For throughput `T`, GPU count `G`, LP30 count `L`, GPU bandwidth `B`, GPU FP8 peak `P_g`, LP30 FP8 peak `P_l`, KV payload `K` and FFN work `F`:

```text
G >= T * K / B
L >= T * F / (0.30 * P_l)
G * P_g + L * P_l >= T * (K * P_g / B + F / 0.30)
MFU <= useful_target_FLOPs / (K * P_g / B + F / 0.30)
```

This relaxation removes scale bytes, weights, GPU arithmetic, LP30 weight capacity, communication, integer placement and dependency costs.
It retains the supplied LP30 FFN efficiency of 0.30 and each catalog's GPU peak and HBM bandwidth.
The [Rubin product specifications](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) were rechecked on 2026-11-03 and confirm dense FP8 peak 17.5 PFLOPS and HBM bandwidth 19.2 TB/s per GPU.
The other rows use the repository's pinned device catalogs.

| Attention GPU | Catalog FP8 peak | Catalog HBM bandwidth | Conditional mixed-deployment MFU bound |
| --- | ---: | ---: | ---: |
| Rubin | 17.5 PFLOPS | 19.2 TB/s | 3.01190% |
| H100 SXM | 1.9789 PFLOPS | 3.352 TB/s | 4.56970% |
| B200 SXM | 4.5 PFLOPS | 7.7 TB/s | 4.61391% |

The Rubin row lies below the existing-format LP30 decode incumbent under this full-HBM-fetch and fixed-efficiency hypothesis for every GPU/LP30 count ratio.
The H100 and B200 rows remain above the incumbent and need additional bounds or simulations.
The bound does not cover persistent on-chip KV, shared prefixes, compressed or reconstructed cache, GPU FFN execution, other hardware roles or different efficiency assumptions.
The table is a conditional model result and does not establish a general physical ceiling for heterogeneous serving.

## Remaining work

The remaining search includes finer activation-buffer lifetimes, other partitions and arena assignments, mixed prefill/decode matrix batches, heterogeneous widths, and additional GPU/LP30 role splits.
Longer serving horizons and arrival controls remain necessary for sustained complete-request throughput.
The broader shared-cache family envelope remains open.
Hardware unit rates, memory banks, compiler schedules and matched model quality remain outstanding requirements for physical optimality.
