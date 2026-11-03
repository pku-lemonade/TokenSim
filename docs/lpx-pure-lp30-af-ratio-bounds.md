# Pure-LP30 A/F ratios and separate head pools

The search evaluates attention/feed-forward (A/F) separation using LP30 chips for both roles.
The [physical-group correction](lpx-multi-group-fidelity.md) gives the groups distinct vector resources and routes A/F traffic through the same chip-to-chip (C2C) budgets as collectives.
The objective imposes no minimum throughput and includes every allocated LP30 chip in model floating-point operation utilization (MFU).
The workload remains independent one-token decode at 101,000 tokens, with key-value (KV) capacity reserved for 102,000 tokens.
Weights and the experimental shared global cache use eight-bit floating point (FP8), while activations use bfloat16 (BF16).
The memory model allows 500,000,000 bytes of static random-access memory (SRAM) per LP30 chip.
Matching shared-cache model quality remains unverified.

## Native component graphs

The production configuration guard for context parallelism in non-colocated A/F domains remains enabled.
The research process overrides that guard to construct native component ledgers and operator graphs for this proposed scope.
The other native validation, ownership, memory admission and operator pricing paths remain active.
These receipts provide conditional model evidence and do not demonstrate production compiler support.

Attention uses tensor parallelism (TP) 16 with decode context parallelism (DCP) 4, or TP32 with DCP8.
Pipeline parallelism (PP) uses ten A stages of six layers, and one A owner serves all requests.
The F group uses TP8 with ten stages or TP16 with five stages.
Both choices allocate eighty F chips.
Three additional chips reserve the complete tied embedding/output matrix and head buffers.
The head execution and all boundary transfers are free in the timing bounds.

| A TP / DCP | F TP / PP | Allocated chips including head | Resident limit | Body critical path, microseconds | Fixed-program MFU upper bound |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 / 4 | 8 / 10 | 243 | 21 | 382.532311 | 2.373242885% |
| 16 / 4 | 16 / 5 | 243 | 21 | 372.593445 | 2.436548727% |
| 32 / 8 | 8 / 10 | 403 | 39 | 443.148702 | 2.294073720% |
| 32 / 8 | 16 / 5 | 403 | 39 | 432.268959 | 2.071375064% |

The bound combines a dependency critical path with exclusive resource demands.
Resource demand makes the TP32/F-TP16 bound tighter than its critical-path-only limit.
Native resident admission rejects one additional request at each supplied A partition.
One-query scratch, the 5,000,000-byte safety reserve and the native KV watermark remain enabled.
Additional concurrent request-frame storage is free in this capacity envelope.
Smaller resident counts cannot increase these necessary bounds for a fixed program.
No issue interval or client phase can make these four fixed programs exceed their stated bounds in the supplied model.

## Extending F depth and head width

The F weights require 346,816,512 bytes per complete layer before tensor sharding.
A weight-only capacity calculation leaves 495,000,000 bytes per chip after the safety reserve and treats all other F buffers as free.
TP8 fits at most eleven layer shards per chip, so it needs at least six stages and forty-eight chips.
TP16 fits at most twenty-two layer shards per chip, so it needs at least three stages and forty-eight chips.
Depths six through sixty and three through sixty give 226 potentially admissible F-depth choices across the two A families.
Another fourteen depth choices fail this weight-only requirement.
The forty-eight-chip figure is an optimistic lower inventory bound.
It is separate from the eighty-chip native configurations in the table.

The physical A group occupies the initial aligned chip range, and F follows it.
Every A/F endpoint pair therefore crosses a sixteen-chip tray boundary and retains at least 0.75 microseconds of propagation under the declared fabric model.
A cost relaxation removes A/F wire transmission and its injection contention, replaces every A/F propagation delay with that minimum, and makes A-side residual layout free.
It retains the compute geometry and logical layer dependencies of the supplied batch-one kernels.
A uniformly wide F stage can be reassigned without changing those per-layer operations.
F collective phase prices also remain invariant under the aligned stage translations in these families.
The phase oracle checks sixty possible F stage positions at full-row, 320-byte and 192-byte output-vector sizes, for 720 comparisons.
These invariance assumptions allow the relaxed critical path to apply to every covered F depth.

The separate head stores a 1,409,286,144-byte FP8 matrix and therefore needs at least three chips at the stated capacity and reserve.
Let `P` be one chip's FP8 peak, `U` the useful floating-point operations (FLOPs) per token, `N` the resident limit, `C` the optimistic A-plus-F chip count, `T` the relaxed body critical path and `k` the head width.
The matrix efficiency remains 0.30.
The head contributes at least its matrix FLOPs divided by the allocated matrix rate to each request's dependency path.
All other boundary work remains free.

```text
P = 315e15 / 256
U = 129261109248
B = 2 * 5376 * 262144 / (0.30 * P)
MFU_percent <= 100 * N * U / (P * (C + k) * (T + B / k))
d_denominator_factor_dk = T - B * C / k**2
```

The denominator derivative is positive at `k=3` in every supplied family and increases thereafter.
The relaxed MFU therefore decreases for every larger head width.
An independent array oracle checks all integer widths from three through 262,144, totaling 1,048,568 width checks across the four families.
Native general matrix multiplication (GEMM) estimates for the head at eleven sampled widths also remain above the matrix-work lower bound.

A complementary relaxation retains the configured six-lane A/F wire transmission while making head computation, storage and additional chips free.
It keeps the same minimum inter-tray propagation and free residual layout.
This envelope also permits head ownership on the existing A/F inventory if it preserves their coordinates and the stated A resident-memory envelope.
A further 4,800 endpoint-pair checks verify the wire-price and propagation conditions across possible A and F stage positions.

| A TP / DCP | F TP | Weight-only minimum F depth | Covered F depths | Wire-free A/F with separate head bound | Retained A/F wire with free boundary bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 / 4 | 8 | 6 | 55 | 3.297794027% | 3.287142037% |
| 16 / 4 | 16 | 3 | 58 | 3.404523040% | 3.391616160% |
| 32 / 8 | 8 | 6 | 55 | 3.207441178% | 3.183884594% |
| 32 / 8 | 16 | 3 | 58 | 3.312104215% | 3.286130681% |

Both relaxations remain below the 3.419600500% experimental shared-cache decode reference.
They exclude the covered depth choices under their stated operator, memory and placement assumptions.
They do not demonstrate attainable MFU at the relaxed values.

## Controls and evidence

Rerunning the separate-cache and shared-cache incumbent programs under the corrected source preserves their displayed 3.039141553% and 3.419600500% MFU values.
The programs retain every node field and measure 320 and 448 outputs respectively.
Fresh single-group native captures also match their frozen node rows after the earlier scalar vector-resource correction.
The two-layer disaggregated diagnostic changes from 17.813345485 to 21.562516685 microseconds when its A/F data transfers receive routed port and propagation costs.
It has forty distinct physical LP30 coordinates and forty-eight routed A/F transfers.

The [compressed experiment record](../experiments/lpx/pure-lp30-af-ratio-bounds.json.gz) retains all four full-model receipts, twenty-four additional cost relaxations, the F-depth/head-width roofs, phase and head-width checks, before/after diagnostics, current controls, validation logs and reproduction sources.
Native graph bodies remain at the local paths identified by the recorded fingerprints.

The wider search still includes other A widths, depths, layer cuts and replica counts, dynamic or mixed F batches, other F widths, alternative tensor-streaming dependencies, interleaved physical packing, revised activation allocation and external memory.
Physical routing, unit calibration, SRAM bank placement and matching model quality remain unverified.
The global maximum is uncertified.
