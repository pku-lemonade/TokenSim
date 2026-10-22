# LP30 public-evidence and model-validity audit

Research date: 2026-10-21.
The subsequent [fidelity implementation and study](lpx-fidelity-model.md) implements the six modeling tasks identified below.
This document preserves the audited model and its historical sensitivity results.
This audit compares the current pure LPX Gemma 4 31B decode model with public architecture descriptions and benchmark methodology.
The latest compute-efficiency scenario uses 30% general matrix multiplication (GEMM) efficiency and 10% attention-core efficiency.
The production device catalog and simulation behavior remain unchanged by this audit.

The architecture supports the modeled use of SRAM-resident weights and key-value (KV) state, layer partitioning, deterministic execution, and fine-grained communication overlap.
The numerical prediction still depends on uncalibrated instruction costs, execution-unit throughput, routing, precision, and memory availability.
The 112-chip result is a conditional candidate within the searched layouts and selected partitions.
Public evidence does not establish that this allocation can reproduce the measured endpoint or that it is the hardware minimum.

## Public sources and their scope

| Source | Evidence used here | Limitation |
| --- | --- | --- |
| [NVIDIA architecture article](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) | SRAM capacity and peak bandwidth; separate matrix, vector, and switch modules; layer-wise partitioning; direct C2C fabric. | Peak specifications do not calibrate sustained operator performance. |
| [NVIDIA long-context article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) | Compiler-scheduled vector transfers; early output; published 10K and 100K endpoint speeds. | The reviewed article does not disclose the endpoint's chip allocation, TP/PP, or weight and KV dtypes. |
| [Groq SHIP paper, MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/file/9c20f16b05f5e5e70fa07e2a4364b80e-Paper-Conference.pdf) | Sections 2.1, 4.2, and 7.2 describe execution units, collective scheduling, exposed communication cycles, and dynamic-workload overheads. | Its numerical evaluations concern LPUv1 on 14 nm, so they cannot calibrate LP30 directly. |
| [NVIDIA power article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-deterministic-execution-drives-power-efficient-high-interactivity-inference-on-nvidia-vera-rubin/) | Cycle-level workload schedules support proactive power management. | Predictable scheduling does not establish utilization or eliminate instruction dependencies. |
| [Artificial Analysis methodology](https://artificialanalysis.ai/methodology/performance-benchmarking) | Output speed excludes the initial token; performance uses `o200k_base`; prompts and output lengths vary. | A deterministic length-only replay does not reproduce the endpoint's content or sampling distribution. |
| [NVIDIA interconnect slide](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-Scale-Up-Interconnect.png) | The earlier [parameter audit](lpx-parameter-evidence-audit.md) records latency references for different accessible-memory scopes. | A capacity-scope chart does not specify every route or collective algorithm; the mirrored image could not be fetched again during this audit. |

The research reviewed vendor documentation and the original SHIP paper, including targeted searches for LP30 BF16 rates, vector and softmax timings, and sustained SRAM bandwidth.
No matching LP30 operator measurements or complete configuration for the published endpoint were located in the reviewed sources.

## Mechanisms that agree with public information

The [native planner](../TokenSim/af/planning/layers.py) assigns individual layers to physical pipeline stages.
The [partition search](../scripts/lpx/partition.py) produces positive integer counts whose sum covers the model.
PP11 and PP17 therefore do not require fractional Gemma layers.
Increasing pipeline parallelism (PP) does not divide a serial client's latency by the PP degree.
All 60 layers remain on the token's dependency path.
Public support for layer partitioning establishes the mechanism, while compiler acceptance and physical placement of each chosen partition remain unverified.

The [streaming refinement](../scripts/lpx/streaming.py) preserves producer dependencies and shared endpoint injection resources.
It permits communication after a producer has generated an output vector.
This agrees with the early-output mechanism described by NVIDIA.
It does not establish the vector production schedule or the amount of overlap attainable by a particular compiled Gemma program.

The serial study excludes time to first token (TTFT) and counts subsequent output tokens.
That boundary agrees with the requested output-speed metric.
An absent TTFT requirement still leaves a requirement to construct, transfer, and store the prompt KV state.

## Numerical assumptions and implementation gaps

Confidence below concerns the stated observation, independently of confidence in the resulting hardware prediction.

| Area | Current behavior | Risk and confidence |
| --- | --- | --- |
| Matrix and SRAM service | Each primitive uses `max(compute time, memory time)` plus a residual. | This assumes overlap within the primitive without a physical pipeline schedule; confidence in the code observation is high, while LP30 latency calibration is low. |
| Logical residual | Compiled experiments add 0.05 microseconds per analytical primitive. | No matching instruction or operator measurement validates this value; low confidence in the numerical value. |
| Softmax | Decode attention counts `4 * batch * keys * query_heads * head_dim` FLOPs. | This counts QK and AV without separately counting softmax reductions, exponentials, or normalization; high confidence in the accounting gap. |
| Vector and switch units | Elementwise work uses a fixed 0.8 coefficient and a dtype-specific compute peak from the device catalog. | Matrix-derived peak rates do not establish vector, conversion, permutation, or reduction throughput; high confidence in the mismatch of modeling scopes. |
| Mixed precision | The scenario declares FP8 weights and KV with BF16 activations, while an ordinary GEMM key carries only its weight dtype. | `gemm_work` consequently prices activation bytes using FP8 in this case and does not describe the mixed-precision conversion path; high confidence in the code observation. |
| Sustained SRAM bandwidth | The experiment uses 90% of the advertised 150 TB/s peak. | No located measurement validates 135 TB/s for these shapes and layouts; low confidence in the sustained rate. |
| C2C service | Both topology levels use an aggregate injection link with unit collective efficiency and no additional message-size ramp. | The physical point-to-point routes, intermediate-chip resources, reduction arithmetic, and port allocation are unresolved; high confidence in these omissions, low confidence in their aggregate timing effect. |
| Propagation | The selected experiment assigns 0.35 microseconds to both LP30 link classes at every scope. | A single local reference does not validate pipeline transfers across trays; low confidence in uniform application. |
| SRAM admission | Screening counts weights, protocol buffers, and one resident KV lease. | Additional internal workspace and full-prefill feasibility are excluded; high confidence in this declared scope. |
| Precision and endpoint setup | FP8 weights and FP8 KV are selected experiment assumptions. | The reviewed endpoint article does not specify these formats; low confidence in matching its deployment. |

The code references are [operator work accounting](../TokenSim/operator_data/workload.py), [Groq analytical pricing](../TokenSim/operator_data/analytical/groq.py), [compute-component keys](../TokenSim/af/providers/components.py), [C2C catalog](../data/topologies/links.yaml), [LPX topology](../data/topologies/groq_lpx256.yaml), and [memory screening](../scripts/lpx/native_search.py).
An attention efficiency calibrated against whole-operator measurements could absorb some omitted softmax costs.
The current assumed 10% coefficient has no such calibration.
Separate costs would need consistent timing boundaries to avoid counting the same work twice.
The activation-byte mismatch has limited influence on single-token GEMMs dominated by weight reads, but it leaves the actual format-conversion path unresolved.

The SHIP paper gives additional reasons to retain physical communication costs.
Its section 4.2 selects among RedBcast, TiledRedBcast, and PAARD according to tensor size, and describes communication cycles remaining exposed after overlap.
Its LPUv1 collective example reaches 50% bandwidth saturation at 32 KiB and 90% at 80 KiB.
Our BF16 hidden-state payload is 10,752 bytes at batch one.
Those older-chip figures demonstrate why algorithm and payload size matter, while providing no validated LP30 replacement curve.
The current streaming model uses two aggregate injection phases and does not explicitly schedule the reduction arithmetic on a vector unit.

The 112-chip partition leaves 5,956,864 bytes of minimum per-chip headroom in the implemented decode accounting.
This is approximately 1.19% of the advertised 500 MB.
It provides no guarantee that a compiled program's complete live tensors and scratch buffers fit.
The [earlier workspace study](lpx-execution-boundaries.md) also found that feasible prefill chunk sizes depend on the partition and reserve assumption.

## Matched sensitivity experiments

The audit holds the 112-chip TP16 PP7 layout and `[9, 9, 9, 9, 9, 9, 6]` layer counts fixed.
Tensor parallelism (TP) divides work within each layer, while PP assigns consecutive layers to stages.
All cases use 30% GEMM efficiency, 10% attention-core efficiency, FP8 weights and KV, and BF16 declared activations.
The reference retains a 4-microsecond native graph submission, zero additional collective launch, and zero additional invocation cost.
Each row below changes only its stated setting unless labeled combined.
These are stress points without calibrated likelihoods.

All rates below are simulated native-model output tokens per second.

| Scenario | 1K | 10K | 100K | 100K change |
| --- | ---: | ---: | ---: | ---: |
| Reference: 0.05-microsecond residual, 90% SRAM, uniform 0.35-microsecond propagation | 6,368.50 | 6,128.15 | 4,478.95 | 0% |
| SRAM efficiency 70% | 6,205.08 | 5,976.67 | 4,397.49 | -1.82% |
| SRAM efficiency 50% | 5,931.11 | 5,722.09 | 4,258.10 | -4.93% |
| Primitive residual 0.10 microseconds | 4,872.75 | 4,730.78 | 3,683.69 | -17.76% |
| Primitive residual 0.20 microseconds | 3,315.40 | 3,249.06 | 2,718.38 | -39.31% |
| Uniform propagation 0.75 microseconds, all phases | 3,896.77 | 3,805.44 | 3,097.25 | -30.85% |
| Collective bandwidth efficiency 50% | 6,349.93 | 6,110.95 | 4,469.76 | -0.21% |
| Whole-buffer collectives with early-output overlap disabled | 6,292.33 | 6,057.58 | 4,441.14 | -0.84% |
| Combined: 70% SRAM, 0.10-microsecond residual, uniform 0.75-microsecond propagation | 3,236.68 | 3,173.42 | 2,665.23 | -40.49% |

The uniform 0.75-microsecond case also increases local collective propagation.
It is a stress test and cannot be interpreted as a correction that public evidence requires for every transfer.
A route-aware model must preserve the local path where appropriate and charge longer paths only when the actual placement requires them.
The combined row inherits this limitation.
The 0.10- and 0.20-microsecond rows do not establish that either constant is more accurate than 0.05.
Increasing a uniform per-primitive floor may overcharge operations fused in the compiled program.

Within this fixed low-attention-efficiency layout, fixed timing assumptions affect predicted speed more than the tested SRAM and network bandwidth reductions.
Disabling early-output overlap has a small effect because this particular schedule has little producer work available to hide the propagation tail.
This result does not contradict NVIDIA's evidence that fine-grained overlap helps other shapes and schedules.
It also does not establish that communication overlap is universally unimportant.

The fixed layout passes the 100K target under some rows and fails under others.
These tests do not search for the minimum chip count under the altered assumptions.

## Benchmark comparability

NVIDIA reports 3,382 TPS at 10K and 3,431 TPS at 100K on the same system.
The ratio is 1.014488, a 1.45% increase.
The current 112-chip low-efficiency scenario predicts a ratio of 0.730882, a 26.91% decrease.
The simulated context dependence therefore differs substantially from the published pair.
The public pair supplies no raw variance establishing statistical significance for its small reversal.
The difference motivates checking the measurement units, content dependence, serving method, and context-dependent execution plan.

The study currently treats native Gemma token counts and AA token counts as equal.
AA's common `o200k_base` unit does not guarantee a one-to-one correspondence with native Gemma tokens.
Output normalization changes the reported TPS, and input normalization also changes the native KV length and work.
Any conversion ratio requires representative text and cannot be inferred from tokenizer vocabulary sizes.

The existing five-warm-up and 50-measured-request replay checks serial admission, token clocks, and cleanup.
The requests use identical lengths and deterministic execution costs.
Repeating them does not measure hardware variance or the distribution of AA prompt content.
Client concurrency one also does not identify the endpoint's internal batch size.
The client-visible endpoint metric can include streaming behavior after the initial token that the current device-time model does not reproduce.

At 100K, the current 4,478.95 TPS corresponds to 223.27 microseconds per output token.
The 3,431 TPS target corresponds to 291.46 microseconds.
The difference is 68.19 microseconds per token, or 1.14 microseconds per layer averaged over 60 layers.
This arithmetic expresses sensitivity to omitted costs and does not measure an omitted hardware overhead.
Adding the same constant to every token would not reproduce both published context points or their reversed ordering.

## Interpretation and next work

Confidence is high that the broad architecture mechanisms agree with the reviewed sources.
Confidence is low in a precise endpoint-reproducing speed or minimum chip count.
The current configuration combines optimistic numerical assumptions with several restrictions that can also overestimate latency or required chips.
Examples of the latter include serial pricing of some operations that separate execution units could overlap, repeated logical residuals, no speculative decoding, and one capacity-selected partition per TP/PP pair.
Consequently, the reported candidate is neither a calibrated hardware prediction nor a mathematical bound on real hardware.

The next modeling work should prioritize:

1. Record native and AA token counts separately and validate any conversion against representative prompt and output text.
2. Give local and cross-tray transfers distinct routes and resource occupancy, including collective reduction work and the physical port budget.
3. Separate matrix, vector, switch, and SRAM work with explicit precision and timing boundaries, including softmax and conversion costs.
4. Replace a universal logical-primitive floor with shape-sensitive pipeline costs when instruction timing or matching microbenchmarks become available.
5. Recheck the complete decode live set, full-prefill feasibility, and reserved SRAM for each candidate allocation.
6. Retain assumption ranges and assess both published context points together without selecting constants solely to match 3,431 TPS.

## Experiment provenance

The [portable results](../experiments/lpx/public-model-audit-results.json) contain all nine scenarios, analytical parameters, execution settings, partition, checkpoint keys, and verification counts.
The [complete results](../outputs/lpx-public-audit-20261021/results.json) retain native context probes and interpolation segments.
The [experiment script](../outputs/lpx-public-audit-20261021/run_audit.py) reruns the matched comparison through `ExperimentRunner` checkpoints.
The source fingerprint is `d4a79382dbe7d1d2c79663a8d24af648cd12c4e20ab92934a2029aa935d1e844`, matching the [compute-efficiency study](lpx-compute-efficiency-sensitivity.md).
The nine scenarios cover 27 workload curves and 99 native context probes in approximately 100.4 host seconds.
Checks verified artifact SHA-256 values, source identities, reference-result agreement, identical memory reports, interpolation probe tolerances, and independent integration of all subsequent-token intervals.
This audit does not repeat the 50-request serving replay or perform a new card-count search.

```bash
.venv/bin/python outputs/lpx-public-audit-20261021/run_audit.py
```
