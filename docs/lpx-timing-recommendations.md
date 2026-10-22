# LP30 timing recommendations and evidence limits

Research date: October 21, 2026.

This note recommends settings for the current coarse operator model used by the [native execution study](lpx-execution-boundaries.md).
The model prices arithmetic, static random-access memory (SRAM) traffic, and chip-to-chip (C2C) communication.
The objective is to estimate hardware requirements independently of the published endpoint throughput.
The nominal settings below are engineering priors.
The sensitivity grids are exploration points and do not represent statistical confidence intervals.

## Recommended interpretation and nominal settings

| Cost | Recommended nominal setting | Evidence and confidence |
| --- | --- | --- |
| Independent software launch for each internal operator | 0 microseconds within a compiled execution region | High confidence in the execution mechanism; NVIDIA describes fused execution without kernel boundaries. |
| Residual pipeline and instruction term in the current logical-operator approximation | 0.05 microseconds per operator | Low numerical confidence; this is an uncalibrated allowance for latency omitted by a throughput roofline. |
| Independent software launch for an internal collective | 0 microseconds | Medium confidence in this interpretation; physical transfer, reduction, dependency, and resource costs must remain separately accounted for. |
| Local C2C endpoint latency | 0.35 microseconds per modeled direct transfer phase | High confidence in the vendor-published endpoint reference; medium confidence in its mapping to the current transfer model. |
| Additional whole-decode invocation gate | 0 microseconds while retaining the existing 4-microsecond submission event | High confidence in the code's accounting; low confidence in the physical value of the retained event. |

The zero launch settings apply to a compiler-scheduled LP30 program.
They do not apply to GPU launches, heterogeneous handoffs, or external host submissions.
The residual operator term retains a separate allowance for unmodeled physical service.
Its nominal value must not be described as a measured LP30 instruction latency.

## Operator execution boundaries

The [NVIDIA Hot Chips schedule slide](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-LPU-Core-Schedule-7.png) states that the example has no kernel boundaries and all operations are fused.
The [NVIDIA long-context article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) describes cycle-level scheduling and 320-byte computation/communication overlap.
These sources support eliminating a separate software launch between internal logical operators.
They do not quantify the remaining matrix-unit (MXM), vector-unit (VXM), switching-unit (SXM), SRAM, or pipeline-fill costs.

The current analytical backend estimates service as a compute/memory roofline plus a fixed operator term.
Its implementation is in [the Groq analytical backend](../TokenSim/operator_data/analytical/groq.py).
The [execution overrides](../scripts/lpx/execution.py) replace the LP30 fixed term with `instruction_floor_us` in compiled mode.
That parameter is additive and is charged per analytical logical operator.
It is not charged per actual machine instruction.
Changing the decomposition or fusion of logical operators can change its aggregate contribution.

Groq's [2021 architecture presentation](https://wordpress.cels.anl.gov/atpesc/wp-content/uploads/sites/96/2021/08/ATPESC-2021-Track-1-Talk-3-Abts-Groq-1-1.pdf), slide 10, describes a 20-cycle vector wave across superlanes on first-generation hardware.
This provides architectural context for fine-grained execution.
It does not establish a 20-cycle complete operator latency or a numerical LP30 fixed term.
Memory and arithmetic already included in the roofline must not be added again as an instruction penalty.

Use residual terms of 0, 0.02, 0.05, 0.10, and 0.20 microseconds to test the current approximation.
Zero is an optimistic omitted-latency case.
The 0.20-microsecond point preserves the old fixed-term magnitude as a stress case.
The 0.05-microsecond nominal value is a provisional engineering choice with low confidence.
Available public evidence does not establish that it is more probable than 0.02 or 0.10 microseconds.
Operator-specific cycle counts, tile shapes, fusion boundaries, and scheduling overlap would provide a stronger replacement.

## Communication timing and placement

The [NVIDIA Hot Chips interconnect slide](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-Scale-Up-Interconnect.png) gives 350 nanoseconds from one SRAM to another SRAM.
Its capacity-versus-latency chart gives the following references.

| Accessible SRAM capacity | Published latency, microseconds |
| ---: | ---: |
| 8 GB | 0.35 |
| 64 GB | 0.75 |
| 128 GB | 1.15 |

These values describe endpoint access at different reach scopes.
They do not specify the cost of each physical hop or the duration of an entire all-reduce.
The [NVIDIA architecture article](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) describes 8 GB per 16-LPU tray and 128 GB per 256-LPU rack.
Local TP8/TP16 groups can use 0.35 microseconds as an endpoint-phase proxy when physical placement keeps their traffic within that scope.
Pipeline transfers crossing trays require separate route and placement assumptions.
The chart alone does not provide a complete topology or a route-latency distribution.

The current `--c2c-hop-us` override assigns one number to both LP30 link classes at every modeled level.
It cannot encode the published scope-dependent curve.
Using 0.35 microseconds everywhere can underestimate communication for larger placements.
Assigning 1.15 microseconds independently to every level can also count endpoint overhead repeatedly.
A topology refinement should reproduce complete endpoint paths before changing each level's constants.

The [collective model](../TokenSim/comm/collectives.py) adds software startup, sequential transfer-phase latency, and serialized bytes.
A direct all-reduce uses two transfer phases in [the current phase-count implementation](../TokenSim/comm/collective_types.py).
With a 0.35-microsecond phase reference, its modeled propagation component is therefore 0.70 microseconds before serialization and overlap.
Setting collective launch to zero leaves that physical component in place.
An extra 0.5-microsecond launch has no separately identified LP30 measurement in the reviewed sources.
Use 0, 0.1, and 0.5 microseconds as sensitivity points for any remaining collective scheduling allowance.
Reduction work omitted by the phase model would require its own accounting.

## External submission and invocation

The [NVIDIA invocation-domain slide](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-LPU-Sync-Domain.png) separates compiler-scheduled LPU execution, FPGA orchestration, and the host control plane.
It gives no numerical invocation cost and does not require a CPU round trip for every generated token.

The native configuration currently uses `cuda_graph` execution as an execution-mode proxy.
The [AF settings](../TokenSim/config/af/settings.py) default `graph_submit_us` to 4 microseconds.
The [serving implementation](../TokenSim/af/serving/service.py) charges that event before submitting a native decode iteration.
The captured replay profile retains it.
The [additional invocation gate](../scripts/lpx/execution.py) adds `invocation_us` to the captured graph without replacing that event.
Consequently, the existing compiled example with `--invocation-us 1` includes 5 microseconds of combined submission and invocation charges.
No reviewed source establishes two distinct physical boundaries that require this sum.

Keep the existing 4-microsecond event and set the extra invocation to zero for a practical baseline.
The retained value remains an uncalibrated compatibility assumption.
Use total external submission budgets of 0, 1, 4, and 10 microseconds in a later sensitivity study.
Values below 4 require replacing or disabling the native submission event; the additional nonnegative gate cannot express them.
The zero case assumes a resident decode loop without a recurring host submission.
It does not establish zero physical loop-control cost.
Host work paid only at request admission belongs to the request-start timing budget and should not be charged repeatedly to steady-state token intervals.

## Configuration with existing controls

The following command selects the proposed baseline for the current model.
It retains the existing 4-microsecond native submission event and uses a uniform local C2C proxy.
Its output still requires the topology and residual-operator sensitivity qualifications above.

```bash
.venv/bin/python scripts/search_lpx_native.py \
  --pipeline-partition capacity \
  --execution-model compiled \
  --instruction-floor-us 0.05 \
  --collective-launch-us 0 \
  --invocation-us 0 \
  --c2c-hop-us 0.35 \
  --output outputs/lpx-timing-recommended
```

## Rerun results

The October 21 rerun uses Gemma 4 31B with eight-bit floating-point (FP8) weights, FP8 key/value (KV) cache, and bfloat16 (BF16) activations.
Catalog compute efficiencies remain general matrix multiplication (GEMM) 0.8 and attention 0.6.
The unused mixture-of-experts (MoE) coefficient remains 0.7.
SRAM bandwidth efficiency remains 0.9.
Every new timing scenario uses zero additional invocation, zero collective software startup, and the uniform 0.35-microsecond C2C proxy.
The existing 4-microsecond submission event remains in the captured graph.

Each of the three searches enumerates 236 tensor-parallel (TP) and pipeline-parallel (PP) pairs within one 256-chip rack.
TP values are 1, 2, 4, 8, 16, and 32.
PP covers every positive stage count up to the layer count and chip limit.
The capacity solver selects one contiguous allocation minimizing peak SRAM occupancy for each pair.
Each scenario admits 36 layouts and rejects 200 layouts for capacity.
The smallest admitted configuration is 88 chips with TP8 and PP11.
Its layer counts are `[4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 3]`.

The table reports simulated generation tokens per second (TPS) for that 88-chip configuration.
The previous-assumptions control uses the current source with 0.05 microseconds per operator, 0.5 microseconds per collective, and an extra 1-microsecond invocation.
It reproduces the earlier three-workload results.

| Scenario | Residual operator term, microseconds | 1K TPS | 10K TPS | 100K TPS | Minimum passing chip count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous-assumptions control | 0.05 | 4,450.65 | 4,410.36 | 4,052.40 | 88 in the earlier complete search |
| Recommended baseline | 0.05 | 5,854.96 | 5,785.42 | 5,184.66 | 88 |
| Lower residual sensitivity | 0.02 | 7,048.43 | 6,947.90 | 6,099.17 | 88 |
| Higher residual sensitivity | 0.10 | 4,566.30 | 4,523.90 | 4,148.06 | 88 |

All three new scenarios pass both the common 3,400-TPS requirement and the published-target check.
The published-target check uses 3,400 TPS at 1K, 3,382 TPS at 10K, and 3,431 TPS at 100K.
The baseline raises modeled 100K TPS by 27.94% at the same 88-chip layout.
The modeled token interval falls by 53.8906 microseconds across all three workloads.
One microsecond comes from removing the extra invocation gate.
The remaining improvement follows from removing the independent collective startup while preserving modeled propagation, serialization, and overlap.
These are changes in predicted hardware performance caused by revised timing assumptions.

The fastest baseline configuration is 112 chips with TP16 and PP7.
Its layer counts are `[9, 9, 9, 9, 9, 9, 6]`.
Its 1K, 10K, and 100K rates are 6,461.17, 6,418.60, and 6,030.95 TPS.
The 88-chip and 112-chip decode allocations leave minimum per-chip headroom of 21,785,856 and 5,956,864 bytes respectively.
These headroom values exclude additional unmodeled internal workspace.
All scenarios still predict lower speed at 100K than at 10K.
The published small reversal remains unexplained by these fixed-cost changes.

## Rerun validation and artifacts

The three complete searches and the matched 88-chip control evaluate 709 screening points.
Ten selected layout/scenario combinations receive withheld-context checks and strictly serial token-clock replay.
Each verified workload executes five warm-up requests followed by 50 measured requests with one client.
The audit covers 1,500 measured requests, 150 warm-up requests, 2,473,350 subsequent-token forwards, 80 withheld context probes, and 719 checkpoint receipts.
Every receipt hash, source fingerprint, integrated token interval, capacity result, target decision, and serial request boundary passes its check.
Each complete scenario takes approximately 446 to 448 host seconds in the concurrent local run.

The replay starts with resident prompt KV and the first output available.
Time to first token is excluded.
Native directed acyclic graph samples supply the token latency curve.
The replay advances every subsequent token without rerunning the full resource graph at every step.
Equal-length deterministic requests do not establish statistical confidence in the hardware assumptions.
The native search does not certify full-prefill workspace, operator cycle counts, actual endpoint precision, or scope-dependent route latency.
The conditional 88-chip result applies to the stated TP grid and selected contiguous allocations.

Portable data and audit counts are in [timing-recommended-results.json](../experiments/lpx/timing-recommended-results.json).
The complete [baseline](../outputs/lpx-timing-recommended-20261021/base-005/results.json), [lower residual](../outputs/lpx-timing-recommended-20261021/floor-002/results.json), [higher residual](../outputs/lpx-timing-recommended-20261021/floor-010/results.json), and [matched control](../outputs/lpx-timing-recommended-20261021/previous-control-88/results.json) retain native probes, serialized request clocks, and checkpoint receipts.
The [source snapshot](../outputs/lpx-timing-recommended-20261021/source-snapshot.json) identifies the archived uncommitted source and input catalogs.
The [audit program](../outputs/lpx-timing-recommended-20261021/audit_results.py) verifies those artifacts and regenerates the portable summary.

```bash
.venv/bin/python outputs/lpx-timing-recommended-20261021/audit_results.py
```
