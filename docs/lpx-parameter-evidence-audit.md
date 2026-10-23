# LP30 and LPX parameter record and evidence audit

Research date: 2026-10-20.
Scope: NVIDIA Groq 3 LP30 hardware, its execution model, and the assumptions used by the current LPX analytical experiments.
This audit adds evidence and modeling recommendations without changing simulation defaults.
Catalog values below are a snapshot of [the LP30 device configuration](../data/devices/groq_lp30.yaml) and [the link configuration](../data/topologies/links.yaml) on the research date.
Benchmark targets come from [the Gemma experiment configuration](../experiments/lpx/aa-gemma4.json).
The [2026-10-21 model-validity audit](lpx-public-model-validity-audit.md) checks implementation gaps and fixed-layout sensitivity against public evidence.

## Evidence rules

Official product specifications establish advertised limits, while measured operator performance requires a workload, precision, shape, clock, and timing boundary.
A numerical inference from a published aggregate remains a derived value.
Earlier Groq generations provide architectural context and cannot directly calibrate LP30 performance or power.
An unsuccessful search for a measurement means that no suitable measurement was located in the sources reviewed here.

The audit reviewed NVIDIA product and architecture pages, the August 2026 long-context article, the September 2026 power article, NVIDIA-authored Hot Chips slides, Groq product specifications, and the MLSys 2026 SHIP paper.
The Hot Chips slide images are mirrored by ServeTheHome.
Their contents are the primary evidence; the hosting article's interpretation is not used as calibration.
[NVIDIA's event page](https://www.nvidia.com/en-us/events/hot-chips-conference/) identifies the presentation and its NVIDIA speakers.

## Parameter decisions

The status labels distinguish a source's claim from this project's calculations and choices.

| Status | Meaning |
| --- | --- |
| Official specification | Located in an original vendor specification; sustained application performance is not established. |
| Published measurement | Located in a published benchmark with the stated metric and workload scope. |
| Vendor-reported example | A vendor demonstration value without the sampling and configuration details needed to reproduce a benchmark. |
| Derived | Calculated from cited inputs; no independent measurement of the derived quantity. |
| Unverified assumption | Project value without a matching LP30-specific specification or measurement in the reviewed sources. |
| Model revision needed | Published architecture evidence requires revisiting the modeled mechanism or timing boundary; no replacement constant has been validated. |
| Experiment target | A requested evaluation threshold, independent of published measurements. |

### Compute, memory, and power

SRAM means static random-access memory.
PFLOPS denotes 10^15 floating-point operations per second, while TB/s and PB/s use decimal bytes.
General matrix multiplication (GEMM) and mixture of experts (MoE) efficiencies below multiply the assumed compute peak in the analytical model.
Thermal design power (TDP) is distinct from measured workload power and rack input power.

| Parameter | Recorded value | Status | Source or derivation | Application and limitation |
| --- | --- | --- | --- | --- |
| LP30 chips per rack | 256 | Official specification | S1 | Physical chip count, independent of the number allocated to one experiment. |
| SRAM capacity per chip | 500 MB | Official specification | S1 | 500,000,000 bytes; retain decimal units. |
| SRAM capacity per rack | 128 GB | Official specification | S1; consistent with 256 x 500 MB | Aggregate capacity requires a feasible placement across chips. |
| SRAM peak bandwidth per chip | 150 TB/s | Official specification | S1 | Peak, separate from sustained workload bandwidth. |
| SRAM aggregate bandwidth per rack | Approximately 40 PB/s | Official specification | S1; chip-level multiplication gives 38.4 PB/s | Preserve both scopes and the published rounding. |
| Rack FP8 compute | 315 PFLOPS | Official specification | S2 | Advertised aggregate peak. |
| Chip FP8 compute | 1.23046875 PFLOPS | Derived | S2: 315 / 256; 19.2 PFLOPS / 16 chips gives 1.2 | Describe as approximately 1.2 PFLOPS; preserve the selected derivation. |
| FP16 and BF16 compute | 0.615234375 PFLOPS each | Unverified assumption | Catalog: FP8 / 2 | No matching LP30 peak or instruction-rate table located; a data-width ratio does not establish native format support or throughput. |
| FP32 compute | 0.15380859375 PFLOPS | Unverified assumption | Catalog: FP16 / 4 | No matching LP30 vector throughput located; distinguish matrix and vector units. |
| GEMM efficiency | 0.8 | Unverified assumption | `analytical.gemm_mfu` | No matching LP30 operator measurements located; record shape and precision. |
| Attention efficiency | 0.6 | Unverified assumption | `analytical.attention_mfu` | Review softmax, vector work, and memory traffic separately. |
| MoE efficiency | 0.7 | Unverified assumption | `analytical.moe_mfu` | The dense Gemma experiment does not exercise this value. |
| Effective SRAM bandwidth fraction | 0.9 | Unverified assumption | `analytical.memory_efficiency` | Implies 135 TB/s; no matching sustained SRAM microbenchmark located. |
| Fixed cost per logical operator | 0.2 microseconds | Model revision needed | Catalog: `analytical.kernel_launch_us`; S3 describes fused execution without kernel boundaries | The constant is unverified; revisit compiled-region execution before calibrating a replacement. |
| Packed-weight decode work | 2 vector operations per element | Unverified assumption | `analytical.packed_dequant_ops_per_element` | No LP30 format-specific unpacking measurements located. |
| Packed-weight decode efficiency | 0.8 | Unverified assumption | `analytical.dequant_efficiency` | Applies to the assumed FP32 vector peak. |
| Additional per-step overhead | 0 microseconds | Unverified assumption | `step_overhead_us` | A modeling choice; not evidence that physical invocation or control costs vanish. |
| Chip power | 450 W | Unverified assumption | `power.tdp_w`; no matching absolute value in S6 | Use only as an assumption in power-normalized results. |

### Communication

Chip-to-chip (C2C) communication within LPX and heterogeneous Rubin-to-LPX transport have different timing boundaries.
Bidirectional aggregate bandwidth must not be reused as a one-direction bandwidth budget.

| Parameter | Recorded value | Status | Source or derivation | Application and limitation |
| --- | --- | --- | --- | --- |
| C2C ports per LP30 | 96 at 112 Gb/s | Official specification | S2 | Port description; does not establish achievable collective bandwidth. |
| Individual C2C link bandwidth | 14 GB/s per direction | Derived | S2: 112 Gb/s / 8 | Raw bit-rate conversion; payload and routing losses require separate treatment. |
| LP30 aggregate C2C bandwidth | 2.5 TB/s bidirectional | Official specification | S2 | Aggregate per-chip injection budget. |
| LP30 one-direction injection budget | 1.25 TB/s | Derived | S2: 2.5 / 2 | Current equal-direction split; do not multiply by 96 ports again. |
| Rack aggregate scale-up bandwidth | 640 TB/s | Official specification | S1 | Rack aggregate, not a single link or Rubin-to-LPX bandwidth. |
| Catalog C2C latency | 0.05 microseconds | Model revision needed | Unverified `latency_us` for both LP30 link classes; compare S4 | Catalog field means hardware-hop latency; the published endpoint measurement has a broader boundary. |
| Published LP30 C2C latency | 350 ns, or 0.35 microseconds | Official specification | S4: one chip's SRAM to another chip's SRAM | Endpoint reference; does not determine arbitrary collective completion time. |
| Network latency by accessible SRAM capacity | 8 GB: 0.35 microseconds; 64 GB: 0.75; 128 GB: 1.15 | Official specification | S4 chart | Scope-dependent references; preserve topology and capacity scope. |
| Rubin-to-LPX transport bandwidth | 200 GB/s per direction | Unverified assumption | `ethernet1600Gb.bandwidth_per_direction_bytes_per_s` | No matching transport measurement located. |
| Rubin-to-LPX transport startup | 2 microseconds | Unverified assumption | `ethernet1600Gb.latency_us` | Separate from LPU-internal C2C latency and invocation. |

### Benchmark measurements and experiment targets

Attention/feed-forward network (AF) disaggregation assigns the two portions of decode to different device pools.
Tokens per second (TPS) below refers to output-generation speed, excluding time to first token.
Artificial Analysis (AA) reports performance tokens using `o200k_base`, which need not match the model's native token count.

| Quantity | Recorded value | Status | Source or derivation | Application and limitation |
| --- | --- | --- | --- | --- |
| Gemma 4 31B, 10K input | 3,382 output TPS | Published measurement | S5, 10K chart | Published endpoint median; hardware inventory and serving precision remain undisclosed. |
| Gemma 4 31B, 100K input | 3,431 output TPS | Published measurement | S5, 100K chart | Same deployment uncertainty; not an isolated-chip benchmark. |
| AA request count and client concurrency | 50 sequential requests; concurrency 1 | Published measurement | S5 chart footnotes | Client measurement scope; does not identify internal serving batch size. |
| Gemma 4 31B, SPEED-Bench coding median | 4,767 output TPS | Published measurement | S5 text and coding chart | NVIDIA self-test; no matching LPX per-request result log or complete serving configuration located. |
| Gemma 4 31B, SPEED-Bench coding P80 | 5,520 output TPS | Published measurement | S5 text and coding chart | A percentile of speed, not a separate workload or a sustained minimum. |
| Gemma 4 31B, SPEED-Bench coding maximum | 6,981 output TPS | Published measurement | S5 coding chart | Maximum within the reported samples; not an independent benchmark or a hardware upper bound. |
| Gemma 4 31B, Hot Chips coding example | 10,996 TPS per user; 16K input | Vendor-reported example | S7 | Slide lists a 264K maximum context separately; precision, token-counting convention, sampling, and speculative-decoding status are not specified. |
| 100K/10K output-speed ratio | 1.014488, approximately 1.45% higher | Derived | 3,431 / 3,382 | No raw variance available to establish statistical significance. |
| Common 1K, 10K, and 100K pass threshold | 3,400 output TPS | Experiment target | User-requested threshold in the Gemma configuration | A corresponding published numeric 1K LPX measurement was not located. |
| Global-only AF example | 8 Rubin + 160 LP30; approximately 2,476 TPS at 10K | Derived | Local analytical study linked below | Conditional simulation result; not a measured inventory or hardware upper bound. |

## Source index

S1 is the [NVIDIA LPX product page](https://www.nvidia.com/en-us/data-center/lpx/).
S2 is the [NVIDIA LPX architecture article](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/).
S3 is the [NVIDIA execution-schedule slide](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-LPU-Core-Schedule-7.png).
S4 is the [NVIDIA scale-up interconnect slide](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-Scale-Up-Interconnect.png).
S5 is the [NVIDIA long-context benchmark article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/), published 2026-08-24, including its [10K chart and footnote](https://developer-blogs.nvidia.com/wp-content/uploads/2026/08/nvidia-groq-3-lpx-gemma-4-10k-artificial-analysis.webp), [100K chart and footnote](https://developer-blogs.nvidia.com/wp-content/uploads/2026/08/nvidia-groq-3-lpx-gemma-4-100k-artificial-analysis.webp), and [SPEED-Bench coding chart](https://developer-blogs.nvidia.com/wp-content/uploads/2026/08/groq-3-lpx-gemma-4-speed-bench-coding.webp).
S6 is the [NVIDIA power-management article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-deterministic-execution-drives-power-efficient-high-interactivity-inference-on-nvidia-vera-rubin/), published 2026-09-15.
S7 is the [NVIDIA Hot Chips performance-example slide](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-LPX-Performance.png), mirrored by ServeTheHome.
ServeTheHome publishes S7 in [NVIDIA's Groq 3 LPU Accelerators for Heterogeneous AI Compute at Hot Chips 2026](https://www.servethehome.com/nvidias-groq-3-lpu-accelerators-for-heterogeneous-ai-compute-at-hot-chips-2026/), dated 2026-08-25, whose text says a single rack decodes 11,000 tokens per second on Gemma 4 31B.
The [placement plausibility record](lpx-placement-plausibility.md) transcribes S7 and a speculative-decoding slide from the same article.
All sources in this record were reviewed on 2026-10-20.
Unknown-value rows record the limits of this search and do not assert that a vendor has never published additional information.

The 1.2 and 1.23046875 PFLOPS derivations differ by approximately 2.5% relative to the catalog value.
The public totals do not resolve whether rounding, clock assumptions, or another specification convention explains that difference.
The ratio is not a measured uncertainty interval.

## Public experimental coverage

The search located three groups of model-level results traceable to original material: the Artificial Analysis endpoint benchmark, NVIDIA's SPEED-Bench coding self-test, and the Hot Chips coding example.
Together they provide four reported workload conditions and six numerical speed summaries: two AA medians, three SPEED-Bench statistics, and one demonstration value.
All four conditions use Gemma 4 31B.
The SPEED-Bench median, P80, and maximum describe one reported sample set and must not count as three independent calibration workloads.
The Hot Chips example is separate from the benchmark medians and does not establish comparable sustained performance.

The S5 AA chart footnotes identify the measured endpoint as an updated private, pre-release Gemma4-31B-LPX deployment accessed through Google Cloud Private Service Connect.
S7 labels its 10,996 TPS-per-user coding example with the same Gemma4-31B-LPX name.
This record adds the name on 2026-10-22 from the 2026-10-21 review of those images.
The same name appears with approximately 3,400 TPS on AA prompts and 4,767 to 10,996 TPS on coding workloads.
Speculative decoding would produce such content-dependent speed, but no reviewed source states that the endpoint uses it.
All repository AA experiments assume no speculative decoding.

The [NVIDIA SPEED-Bench dataset](https://huggingface.co/datasets/nvidia/SPEED-Bench) publishes benchmark inputs and links to its evaluation framework.
Those resources do not supply the missing LPX-specific serving configuration or per-request performance logs in the material reviewed here.
A published benchmark input set is distinct from a measured LPX result dataset.

Additional numerical evidence includes S4's three network-latency references and S6's internally tested reduction of voltage droop by more than 60 percent.
Voltage-droop reduction does not mean a 60-percent reduction in operating voltage or workload power.
S6's voltage and power-saving projections do not provide absolute chip or rack watts.

The search did not locate a matching LPX result set covering matrix shapes and precisions, attention lengths, sustained SRAM bandwidth, collective size and participant count, or Rubin-to-LPX handoff bandwidth and latency.
It also did not locate complete hardware allocations, precision settings, and serving configurations for the reported endpoint results.
The two AA context points can constrain an overall latency model, but cannot identify all unknown operator, communication, and power parameters independently.
Product throughput-per-megawatt projections and the scaled-up GPT-OSS-2T curves remain separate from measured LPX hardware results.

## Execution-model corrections supported by LP30 evidence

The schedule slide labels the example as having no kernel boundaries or memory barriers and fused operations.
This supports removing a separate host-launch charge for every logical operator inside a compiled LPU region.
It does not establish zero instruction latency, zero pipeline fill time, zero synchronization cost, or zero invocation cost at a region boundary.

NVIDIA's [long-context article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) describes compiler-scheduled overlap of computation and communication at 320-byte vector granularity.
It says transfers can begin while a matrix multiplication is still producing results.
The current analytical operator roofline already overlaps compute and SRAM service through a maximum of their costs.
The older isolated decoder prototypes sum operator, collective, and cross-device costs without a dependency-aware model of that finer overlap.
A correction should schedule producer chunks, transfers, and consumer chunks while respecting link capacity and data dependencies.
It should retain any communication tail exposed on the critical path.
Replacing the whole decoder sum with one maximum would remove real sequential dependencies.

The interconnect slide also gives network-latency references of 0.35, 0.75, and 1.15 microseconds for access to 8, 64, and 128 GB of SRAM capacity.
These are scope-dependent network figures, not a promise that an arbitrary collective finishes in 350 ns.
The current 50 ns catalog value lacks comparable published support.
Neither number establishes the latency of a Rubin-to-LPX handoff.

The [NVIDIA invocation-domain slide](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-LPU-Sync-Domain.png) places FPGA orchestration and LPU-to-GPU connectivity outside the deterministic LPU invocation domain.
An LPU-internal launch correction must therefore remain separate from GPU kernel launches and heterogeneous transport costs.
An overlap assumption across independent requests also requires a separate check for the concurrency-one decode latency metric.

## Power and generation boundaries

NVIDIA's [September power article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-deterministic-execution-drives-power-efficient-high-interactivity-inference-on-nvidia-vera-rubin/) discusses relative reductions from deterministic voltage and clock management.
It does not provide an absolute LP30 chip TDP or measured watts for this workload.
Those relative improvements cannot determine a replacement for 450 W.
Chip TDP, workload chip power, board power, and rack input power require separate fields when comparing tokens per megawatt.

Groq's [14 nm GroqChip specification](https://www.groq.com/GroqDocs/Product%20Spec%20Sheet%20-%20GroqChip%E2%84%A2%20Processor.pdf) lists 188 TFLOPS FP16, 230 MB SRAM, up to 80 TB/s SRAM bandwidth, 215 W TDP, and 300 W maximum power.
These are older hardware values.
The [SHIP paper](https://proceedings.mlsys.org/paper_files/paper/2026/file/9c20f16b05f5e5e70fa07e2a4364b80e-Paper-Conference.pdf), despite its 2026 publication date, explicitly evaluates LPUv1 on GlobalFoundries 14 nm.
Its 388 W per accelerator is provisioned system power allocated per LPUv1, including amortized system components.
It cannot replace an LP30 chip-power assumption.
The paper's older-generation communication and utilization measurements also require separate provenance.

## Consequences for the existing Gemma experiment

The [current context study](lpx-context-sensitivity.md) uses analytical projections without physical LP30 measurements.
For its 8 Rubin plus 160 LP30 global-attention-only layout, the saved baseline totals are 403.862 microseconds per token at 10K input and 517.117 microseconds at 100K input.
The uniform 0.2-microsecond logical-operator charge contributes 192.6 microseconds per token, or 47.7% and 37.2% of those totals.
This contribution is recovered from the linear, pointwise, and attention component differences between the saved `main` and `af-combined-bf16` records, divided by the 0.15-microsecond overhead change and multiplied by 0.2.
The comparison excludes their changed cross-device transport component.
Both devices receive the uniform operator override in the exploratory cost model, so these percentages cannot be interpreted as removable LPU-only overhead.
The launch contribution is a model artifact, not a physical profiling measurement or a predicted achievable speedup.

The baseline's cross-device component contributes another 92.751 microseconds per token, with bandwidth and startup still unverified.
The newly sourced LP30 network latency can increase modeled communication time, while compiled-region execution and dependency-aware overlap can reduce other costs.
Their net effect requires rerunning the experiments with explicit timing scopes.
The previous 2,476 tokens/s result and 8-to-160 inventory search therefore remain conditional on their stated assumptions.
They do not establish an upper bound or a measured minimum inventory for real hardware.

The published 3,431 tokens/s endpoint result cannot identify BF16 peak, operator efficiency, power, hardware inventory, and execution overlap simultaneously.
Matching that aggregate speed by changing several unknown parameters would be a fit without independent validation.

The subsequent [native execution and placement study](lpx-execution-boundaries.md) separates logical operator cost, additive instruction cost, one decode invocation, and C2C propagation without changing catalog defaults.
It retains the native dependency graph and vector-streaming refinement, evaluates contiguous SRAM-aware placement, and validates complete 100K prompt ingestion with bounded prefill chunks.
Its 88-, 112-, and 176-chip candidates depend on different stated execution and precision hypotheses.
They do not identify the public endpoint's physical inventory.

## Recommended calibration order

1. Separate compiled-region invocation cost, instruction and pipeline latency, GPU launch cost, and heterogeneous handoff cost.
2. Align the C2C latency field with the published SRAM-to-SRAM endpoint boundary and retain topology-dependent propagation.
3. Add dependency-aware overlap within compiled LPU regions, with an explicit exposed communication tail.
4. Calibrate precision-specific matrix and vector rates using LP30 instruction tables or operator microbenchmarks when available.
5. Calibrate sustained SRAM service and collective curves at representative tensor sizes.
6. Obtain measured workload or rack power before treating tokens-per-megawatt comparisons as validated.

The existing defaults should remain labeled as assumptions until the corresponding implementation and evidence are available.
