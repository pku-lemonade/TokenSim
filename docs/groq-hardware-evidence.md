# Groq literature and LP30 evidence audit

Search date: October 19, 2026.
This document supports the [LPX calibration and Figure 11 study](lpx-study-guide.md) by recording parameters available in public sources, evidence locations and limits on their use.
This research update adds documentation without changing device parameters or simulation results.

The search covers the architecture, compilation, interconnect, inference systems and direct hardware measurements of Groq Tensor Streaming Processors (TSPs) and Language Processing Units (LPUs).
Search entry points include IEEE/ACM, arXiv, MLSys, Hot Chips, Argonne/ALCF, author websites and references.
The tables distinguish sources checked in full from sources checked through abstracts, bibliographic records or author slides.
This public-source inventory cannot guarantee coverage of unindexed, unpublished or paywalled research.
Papers that only call the Groq API for other applications are excluded from hardware evidence.

## Paper inventory

| ID | Paper and year | Material checked | Use in the simulator |
| --- | --- | --- | --- |
| P1 | Dennis Abts et al., **Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads**, ISCA 2020, 145–158, [DOI](https://doi.org/10.1109/ISCA45697.2020.00023) | Bibliographic records and subsequent official citations; the original Groq PDF URL currently returns 404, and a downloadable full text was unavailable | Functional slices, streaming execution and static scheduling in the initial chip generation; excluded from LP30 measurement tables |
| P2 | Dennis Abts et al., **A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning**, ISCA 2022, 567–580, [DOI](https://doi.org/10.1145/3470496.3527405) | [Conference program](https://www.iscaconf.org/isca2022/program/) and original-site index; the original Groq PDF URL is unavailable, and ACM full-text access is restricted | Multi-chip interconnect and compilation in the initial generation; its topology cannot be equated with LP30 or SHIP production topology |
| P3 | Ibrahim Ahmed et al., **Answer Fast: Accelerating BERT on the Tensor Streaming Processor**, ASAP 2022, [author full text](https://arxiv.org/pdf/2206.11062) | Complete 8-page paper, with pages 3 and 6–7 checked in detail | Single-chip latency reference with workload and timing boundaries |
| P4 | Dennis Abts et al., **Challenges/Opportunities to Enable Dependable Scale-out System with Groq Deterministic Tensor-Streaming Processors**, DSN-S 2022, 19–22, [IEEE record](https://ieeexplore.ieee.org/abstract/document/9833746) | [Proceedings contents](https://www.proceedings.com/content/064/064993webtoc.pdf) confirm the title, authors and pages; full text was unavailable | Reliability and synchronization background; the contents page supplies no performance parameters |
| P5 | Ryien Hosseini et al., **Exploring the Use of Dataflow Architectures for Graph Neural Network Workloads**, ISC Workshops 2023, 648–661, [DOI](https://doi.org/10.1007/978-3-031-40843-4_48) | [ALCF record](https://www.alcf.anl.gov/publications/exploring-use-dataflow-architectures-graph-neural-network-workloads) and [Groq SC23 slides](https://sc23.supercomputing.org/proceedings/exhibitor_forum/exhibitor_forum_files/exforum135s2-file2.pdf) | Graph neural network mapping and compiler optimization examples; relative speedups cannot be converted into large language model (LLM) operator efficiency |
| P6 | Dennis Abts and John Kim, **The Case for Domain-Specific Networks**, HOTI 2023, 49–52, [author institution record](https://pure.kaist.ac.kr/en/publications/the-case-for-domain-specific-networks/) | Bibliography and abstract, DOI `10.1109/HOTI59126.2023.00021`; full text was unavailable | Specialized interconnect background; no LP30 latency measurements can be extracted |
| P7 | Lennart P. L. Landsmeer et al., **Tricking AI chips into Simulating the Human Brain: A Detailed Performance Analysis**, 2023, [author preprint](https://arxiv.org/abs/2301.13637) | Abstract and author full text | Numerical-precision research outside LLMs; some low-precision operations do not meet the brain simulation requirements, so data-type names alone cannot establish numerical equivalence |
| P8 | Murali Emani et al., **Toward a Holistic Performance Evaluation of Large Language Models Across Diverse AI Accelerators**, IPDPSW 2024, [archived author full text](https://www.osti.gov/servlets/purl/2563711) | Complete 10-page paper, with pages 3, 5 and 9 checked in detail, DOI `10.1109/IPDPSW63119.2024.00016` | GPT2 240M inference latency on Groq v1.5; the study has laboratory and vendor coauthors |
| P9 | Farah Ferdaus et al., **Evaluating Energy Efficiency of AI Accelerators Using Two MLPerf Benchmarks**, CCGrid 2025, 549–558, [ALCF record](https://www.alcf.anl.gov/publications/evaluating-energy-efficiency-ai-accelerators-using-two-mlperf-benchmarks) | [Author-uploaded full text](https://www.researchgate.net/publication/393191603_Evaluating_Energy_Efficiency_of_Ai_Accelerators_Using_Two_Mlperf_Benchmarks) and [official experiment repository](https://github.com/fferdaus/MLPowerPerf), DOI `10.1109/CCGRID64434.2025.00035` | Power collection methodology and application throughput for initial-generation ResNet50/BERT-Large; includes Groq coauthors |
| P10 | Samantha Fowler et al., **Adapting scientific streaming inference workflows for a deterministic tensor processing unit**, XLOOP 2025, [original conference abstract](https://sc25.conference-program.com/presentation/?id=ws_xloop101&sess=sess119) | Authors, abstract, workload and timing description; full text was unavailable | Streaming image inference including data transfer; cannot substitute for LLM decode measurements |
| P11 | Andrew Bitar et al., **SHIP: SRAM-Based Huge Inference Pipelines for Fast LLM Serving**, MLSys 2026, [conference page](https://proceedings.mlsys.org/paper_files/paper/2026/hash/9c20f16b05f5e5e70fa07e2a4364b80e-Abstract-Conference.html) | [16-page paper](https://proceedings.mlsys.org/paper_files/paper/2026/file/9c20f16b05f5e5e70fa07e2a4364b80e-Paper-Conference.pdf) and [3-page supplement](https://proceedings.mlsys.org/paper_files/paper/2026/file/9c20f16b05f5e5e70fa07e2a4364b80e-Supplemental-Conference.pdf), including the relevant figures and tables | The most useful addition for LLM systems in this search: production deployment, communication, scheduling and experimental configuration; uses the initial-generation LPU |

SHIP authors currently list NVIDIA affiliations, while the described hardware remains the 14 nm LPUv1 with approximately 230 MB of static random-access memory (SRAM).
Pages 9 and 12 leave a complete evaluation of the next-generation LPU and GPU/LPU disaggregation to future work.
Publication year and author affiliation therefore cannot identify the hardware as LP30.

## Conference slides, technical reports and original LP30 material

| Material | Verification status and use |
| --- | --- |
| **The Groq Software-defined Scale-out Tensor Streaming Multiprocessor: From chips-to-systems architectural overview**, Hot Chips 34, 2022 | [Official conference](https://hc34.hotchips.org/), [DOI](https://doi.org/10.1109/HCS55958.2022.9895630); slides 1–69 are related to ISCA 2022 and do not constitute an additional independent experiment |
| **Programming the Groq TSP Architecture in Haskell with Haste**, Satnam Singh, FHPNC 2023 | [Official invited talk](https://icfp23.sigplan.org/home/FHPNC-2023); programming-model background without an LP30 operator table |
| **Reinventing Memory and Network Architecture for Large-Scale Energy-Efficient Low-Latency Generative AI**, Igor Arsovski, ISSCC 2025 Forum | [Official program](https://submissions.mirasmart.com/ISSCC2025/PDF/ISSCC2025AdvanceProgram.pdf) confirms the talk; complete publicly verifiable slides were unavailable, so third-party numerical summaries are excluded |
| **Think Fast: LPU Accelerator for Heterogeneous Compute**, Igor Arsovski and Santosh Raghavan, Hot Chips 2026 | [NVIDIA conference page](https://www.nvidia.com/en-us/events/hot-chips-conference/) confirms the talk; original NVIDIA slides reproduced in public reports support the communication and system diagrams discussed below |
| Argonne AI Testbed Expeditions FY2021 technical report | [Original report](https://publications.anl.gov/anlpubs/2022/05/174835.pdf); includes Groq inference near X-ray detectors and supplies application background |
| Argonne AI Testbed Expeditions FY2022 technical report | [Original report](https://publications.anl.gov/anlpubs/2023/11/181668.pdf); includes **Exploring Groq as a Real-Time AI Inference Accelerator for Scientific Instruments** and **Deep Learning Accelerated X-Ray Data Analysis for Experiment Steering**, with BraggNN evaluated in the latter |
| GroqChip Processor Product Brief v1.5 | [Original specification index](https://www.groq.com/wp-content/uploads/2022/10/GroqChip%E2%84%A2-Processor-Product-Brief-v1.5.pdf) gives compute and power definitions for the initial generation at 900 MHz; these do not apply to LP30 |
| NVIDIA LPX architecture, long-context and power explanations | [2026-03-16 architecture explanation](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/), [2026-08-24 long-context blog](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) and [2026-09-15 power explanation](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-deterministic-execution-drives-power-efficient-high-interactivity-inference-on-nvidia-vera-rubin/); official technical material listed separately from peer-reviewed papers |

The search excludes HyperAccel's **LPU: A Latency-Optimized and Highly Scalable Processor for Large Language Model Inference** ([arXiv:2408.07326](https://arxiv.org/abs/2408.07326)), whose LPU is unrelated to Groq.
The measured platforms listed in [LLM-Inference-Bench](https://arxiv.org/abs/2411.00136) exclude Groq, so its results cannot be counted as Groq experiments.

## Additional communication evidence for LP30

The original NVIDIA Hot Chips 2026 slide **Low Latency Scale-Up Interconnect Enables Fast Inference** provides the following values.
This audit checks the [publicly reproduced original slide image](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-Scale-Up-Interconnect.png), hosted by ServeTheHome and credited to NVIDIA.
The complete official slide package was unavailable; this slide omits frequency, message size and the test program, so its values are classified as vendor-published parameters.

| Item | Slide value | Scope |
| --- | ---: | --- |
| Chip-to-chip latency | 350 ns | SRAM on one chip to SRAM on another chip |
| Reachable SRAM capacity of 8 GB | 0.35 μs | Network latency associated with that capacity; message size unspecified |
| Reachable SRAM capacity of 64 GB | 0.75 μs | Same scope |
| Reachable SRAM capacity of 128 GB | 1.15 μs | Same scope |
| Reachable SRAM capacity of 384 GB | 1.60 μs | Same scope |
| Reachable SRAM capacity of 640 GB | 2.05 μs | Same scope |
| Reachable SRAM capacity of 896 GB | 2.50 μs | Same scope |
| Reachable SRAM capacity of 1,152 GB | 2.95 μs | Same scope |

The 0.5 GB point at 0 μs defines a local-capacity network baseline and does not imply zero SRAM access time.
These points describe the network reach required to access the stated capacity; they do not specify AllReduce completion time or equal latency between arbitrary node pairs.

LP30 `latency_us: 0.05` in the current [`links.yaml`](../data/topologies/links.yaml) is explicitly labeled as an assumption.
The 350 ns value can constrain calibration of total fixed chip-to-chip transfer cost.
Mapping it to a per-hop parameter requires checking whether sender and receiver SRAM access, routing and synchronization are charged separately, to avoid counting costs twice.
The capacity-latency curve can also check routing levels in [`groq_lpx256.yaml`](../data/topologies/groq_lpx256.yaml).
These sources lack complete physical wiring and concurrent congestion experiments, so they do not uniquely determine a topology.

The same presentation's [synchronization-domain slide](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-LPU-Sync-Domain.png) explicitly shows field-programmable gate arrays (FPGAs) handling DRAM access and LPU/GPU connections.
It supports recording GPU/LPU bridging, control and transfer costs separately; internal LPU chip-to-chip (C2C) latency cannot represent every cross-rack transfer.
The [attention/FFN disaggregation (AFD) slide](https://www.servethehome.com/wp-content/uploads/2026/08/NVIDIA-Groq-3-LPU-Hot-Chips-2026-Disaggregated-Decode.png) shows 256 LP30 chips in LPX and 72 Rubin GPUs in NVL72, with feed-forward network (FFN) weights in SRAM and key/value (KV) state on GPUs.
This functional diagram does not establish a one-rack-to-one-rack ratio at every Figure 11 point or prove that one rack holds all FFN weights of a 2T model.

## Additional initial-generation system data from SHIP

Page numbers below count from page 1 of the main PDF.
All values describe LPUv1; compiler-generated performance and sampled hardware measurements are recorded separately.

| Data | Paper location | Evidence type and suggested use |
| --- | --- | --- |
| 8 LPUs per node, QuadFour production network, 9 nodes or 72 LPUs per rack | Pages 4–5 and 11 | Production system structure; keep separate from older ISCA 2022 configurations when adding initial-generation topologies |
| LPU-to-LPU latency of 300 ns per hop | Page 5 | Paper-published parameter for the initial generation; cannot replace LP30's 350 ns value |
| Tensor-parallel (TP) partition of 72 LPUs: 16.56 GB, diameter 3 hops; 136 LPUs: 31.28 GB, diameter 5 hops | Page 5 | Check the relationship between capacity scaling and routing hops |
| Two four-node partitions: 427/256/128/43 GB/s for contiguous placement or skipping 1/2/3 nodes | Page 5, Table 1 | Inter-partition bandwidth depends on placement and cannot be assigned uniformly as chip injection bandwidth |
| AllReduce reaches approximately 50% and 90% of saturated bandwidth at 32 KiB and 80 KiB | Page 6, Figure 6 and main text | Compiler-generated communication performance for checking small-message curves; excluded from LP30 measurements |
| RedBcast, TiledRedBcast and PAARD AllReduce algorithms | Supplement, pages 1–2 | Algorithm selection depends on message size and topology; a fixed GPU NVIDIA Collective Communications Library (NCCL) model lacks this distinction |
| Typical gpt-oss-120B instance: 72 nodes or 576 LPUs, 69.7 GB of remaining KV SRAM, 72 TB of host DDR4 | Page 7, Table 2 and main text | Concrete deployment evidence showing that storage capacity alone does not determine deployed chip count |
| Qwen3-235B-A22B: TP=16, pipeline parallelism (PP)=95, max concurrency=380 | Page 9, §6.1 | Relatively complete system experiment; implies 1,520 LPUs if each TP×PP rank maps to one LPU |
| Provisioned system power allocated per LPU: 388 W | Page 11, §7.1 | Includes an allocation of host, memory and other system power; cannot be labeled measured chip thermal design power (TDP) or replace the LP30 450 W assumption |

The 576-LPU gpt-oss-120B example describes a production instance, while the Qwen3 configuration belongs to a separate experiment in §6.1.
Neither establishes the LP30 allocation of the AA Gemma 4 endpoint.

SHIP page 6 shows matrix multiplication overlapping AllReduce; pages 8–9 describe decode priority, dynamic chunked prefill and mixed processing of tokens from different contexts.
These mechanisms identify validation limits in the per-operator roofline plus fixed-overhead approximation in [`groq.py`](../TokenSim/operator_data/analytical/groq.py).
The paper supplies no measurement table that can directly replace LP30 `gemm_mfu`, `attention_mfu`, `moe_mfu` or per-operator overhead values.
Here MFU denotes model FLOPs utilization.

## SHIP experimental conditions and the definition of a user

Supplementary Table 3 explicitly distinguishes concurrent users `U` from operator token batch `B`.
Pure decode has `U = B`; pure prefill of a single prompt has `U = 1`, even with many prompt tokens.
This supports defining a user as a request sequence making concurrent progress, without treating each prefill token as a user.
It does not specify AA client submission behavior or establish the cross-request statistic used on NVIDIA Figure 11's horizontal axis.

| Condition | Public SHIP setting | Implication for experiment design |
| --- | --- | --- |
| Model and LPU system | Qwen3-235B-A22B; TP16, PP95, maximum concurrency 380 | Supports an independent initial-generation validation case |
| GPU comparison | 8×B200 DGX, TP8, SGLang 0.5.8 | GPU and LPU counts differ, so absolute system throughput alone cannot determine per-chip efficiency |
| Mixed-token budget | MBTB=16Ki and 2Ki; `--enable-mixed-chunk` enabled | The budget includes prefill and decode tokens and differs from a per-request prefill chunk |
| Context and workload | Figure 10 uses CL=2Ki, 4Ki and 8Ki while sweeping the prefill:decode token ratio | Supports trend checks; differs from the blog's 400K cached-input workload |
| Caching and speculative decoding | Controlled experiments disable prefix caching and speculative decoding | Keep speculative results from other experiments separate from controlled comparisons |
| Queue handling | Supplement page 3 states that it/s/u and time to first token (TTFT) exclude queueing time | Latency excluding queues does not establish the absence of admission queues |
| Production samples | 2K input/output length pairs sampled from 30 days of traffic, replayed with random tokens | The length distribution has evidence, but complete original length pairs and request text are unavailable |
| Figure 10 normalization | Each system's own observed peak is the reference | Supports relative trends; 1.0× does not establish absolute cross-system tokens/s (TPS) or TPS/MW |

The supplement separately lists SGLang results with maximum client concurrency 256 and a vLLM 0.17.1 comparison.
These remain separate experimental configurations.

The repository's queries-per-second (QPS) search should retain raw admission wait and service time.
Excluding queue time from measurements and operating without sustained queue growth are independent conditions.
A paper reporting only the former cannot establish the maximum QPS satisfying the latter.

## Other latency and power references

| Source | Experimental conditions and result | Limits on use |
| --- | --- | --- |
| P3, BERT-base | One initial-generation TSP, batch 1, sequence length 128, INT8 general matrix multiplication (GEMM), FP32 nonlinear operations, 4,000 inferences; mean 128.9 μs, P99 129.5 μs | Includes host→TSP and TSP→host transfers; excludes tokenization/text mapping; supports independent latency validation and does not measure per-token decode |
| P8, GPT2 240M | 4 Groq cards; GroqFlow on-chip latency 536 μs; direct compiler optimization with KV caching, resident weights and FP8 reaches 42.8 μs | Different configurations; the passage does not give sufficiently explicit request batch and sequence length for direct operator-efficiency fitting |
| P9, BERT-Large/ResNet50 | Software table lists GroqWare 0.11.0 or 0.9.2 and PyTorch 2.4.2; `groq-runtime` reads board power sensors over USB | The [author repository](https://github.com/fferdaus/MLPowerPerf/tree/main/Groq) states that models come from a private repository and Groq supplied 8-LPU ResNet50 power data; no complete public reproduction package |
| P10, edgePtychoNN | 128×128 images, single-inference latency including transfer 156.06 μs, approximately 6.4 kHz | Application-level conference abstract result; full text is needed to confirm chip generation and configuration; excluded from LP30 LLM calibration |
| Argonne FY2022 report, PDF page 42, §4.1 | Initial-generation 128×128, 16-bit matrix multiplication; Groq API compute-only performance 4.8 TFLOPS collected with `iop-utils`; compares three APIs/runtimes | The small matrix can check shape sensitivity; runtime performance including calls and data movement is reported separately and must use a separate timing definition when fitting |

The initial-generation brief's 215 W TDP, 185 W average and 300 W maximum use different definitions from SHIP's 388 W provisioned system allocation.
None establishes LP30 operating power.
NVIDIA's September 2026 power explanation reports more than 60% less voltage droop in internal tests and discusses power savings from lower voltage.
It does not give total LP30/Rubin/host/network power for each Figure 11 point and cannot recover the absolute TPS/MW denominator.

## Implications for TokenSim

| Item | Evidence added by this search | Remaining gaps |
| --- | --- | --- |
| LP30 communication | 350 ns SRAM→SRAM parameter and capacity-latency curve; structural evidence of FPGA bridging between GPUs and LPUs | Message sizes, frequency, bridge overhead, congestion, complete wiring and measured collective curves |
| Initial-generation system validation | Single-chip BERT latency, SHIP topology and Qwen3 parallel configuration | Complete compiler mappings, all absolute throughput values and reusable operator measurements |
| User and batch definitions | SHIP's explicit distinction between `U` and `B` | Mean/median/P99 definition of Figure 11's horizontal axis, concurrency and submission protocol |
| AA calibration | An additional communication constraint reduces freedom in fitting only a global efficiency multiplier | Gemma 4 endpoint LP30 count, precision, parallel layout, speculative decoding, native tokens and serving configuration |
| Figure 11 | Additional original diagram supporting AFD functional roles | GPT-OSS-2T architecture, total FFN weights and precision, rack ratios, power and original sweep data |
| LP30 operator parameters | Identifies fusion and communication-overlap mechanisms needing validation | No measured LP30 table found that directly replaces existing MFU, unpacking efficiency or 0.2 μs operator overhead |

Add independent initial-generation validation experiments and use new LP30 communication evidence for parameter sensitivity and topology checks.
Keep AA Gemma 4 calibration separate from Figure 11 hypothesis sweeps; initial-generation production configurations cannot fill undisclosed LP30 deployment details.
