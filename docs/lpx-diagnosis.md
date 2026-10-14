# LPX projection diagnosis and repair plan

The current curves do not reproduce the serving behavior of NVIDIA Figure 11.
The primary defects are in the experiment's execution abstraction and placement assumptions.
Missing NVIDIA inputs limit subsequent calibration, but the known execution defects must be corrected independently.
Passing the existing regression tests establishes those tests' software contracts and does not validate the plotted performance.

## Reproduce the diagnosis

```bash
python scripts/diagnose_lpx.py \
  --summary outputs/lpx-reproduction/summary.json \
  --config experiments/lpx/sweep.json \
  --output outputs/lpx-diagnosis/diagnosis.json
```

The diagnostic leaves the hardware catalog and saved serving results unchanged.
It evaluates a batch-one decode at 404,200 context tokens.
Zero launch cost and zero bandwidth ramp are isolated diagnostic lower bounds.
They are not recommended production settings or calibrated hardware measurements.
The local/full-attention intervention measures latency only and does not validate KV allocation or execution placement.

## Confirmed execution defects

`scripts/lpx/simulator.py` starts one SimPy process and waits for each task before submitting its successor.
Every layer uses the same A and F resource names, even when the layout declares multiple pipeline stages.
Separate pools remain idle while another pool runs.
There are no independently progressing request streams, populated pipeline stages or data-parallel serving replicas.
Pipeline partitioning reduces reported resident memory and increases the allocated inventory, without providing pipeline throughput.
The main curves use this additional experiment executor.
The production AF executor is exercised only by the one-layer audit.
The experiment therefore bypasses the production AF runtime's unsupported PP and combined EP/TP configurations.

The default AF layout allocates 5,120 LP30 chips and 40 Rubin GPUs.
It charges 2.376 MW using unmeasured processor TDP assumptions.
The most efficient Rubin point charges 0.0144 MW for eight GPUs.
The AF output-throughput ratio is 1.896, while its power ratio is 165.
Its normalized efficiency is consequently `1.896 / 165 = 0.01149`.
These arithmetic identities explain why the plotted AF curve lies close to zero.
Charging the allocated inventory is appropriate for that declared deployment.
The defect is estimating its factory output from one serialized batch without workload saturation or stage concurrency.

The experiment must schedule independent requests, micro-batches and replicas on physical stage resources.
Warm-up, steady-state measurement and drain intervals must be distinct.
Power and throughput must describe the same deployed inventory and measurement interval.
The experiment must export resource utilization and queue delays to demonstrate saturation.

## Latency probes

| Batch-one decode probe | Rubin TP8 | AF TP8 | LP30 decode CP1 |
| --- | ---: | ---: | ---: |
| Current step latency, microseconds | 2,165.7 | 1,977.7 | 657.2 |
| Current TPS/user | 461.7 | 505.6 | 1,521.7 |
| TPS/user with Rubin launch cost removed | 1,558.3 | 1,285.8 | 1,521.7 |
| TPS/user with bandwidth ramp removed | 484.6 | 560.2 | 2,560.7 |
| TPS/user with alternating 128-token local attention | 487.4 | 536.6 | 1,554.3 |

The 3-microsecond Rubin kernel-launch assumption contributes 1,524 microseconds to the Rubin step, approximately 70% of its latency.
The AF step retains 1,200 microseconds of that assumption on its A workers.
The model independently charges normalization, rotary embeddings, routing, combination and residual kernels for every layer.
Those fixed charges impose speed ceilings regardless of FFN SRAM bandwidth.
Removing them also reverses the AF advantage in this probe.
A single throughput multiplier would hide this structural sensitivity.
GPU graph capture, fused kernels and device execution must have explicit service contracts and calibrated costs.

The LP30 collective model inherits a 1 MiB message-size ramp and 0.5 collective efficiency.
Its point-to-point AF transport uses a separate alpha-beta calculation.
The resulting paths have inconsistent small-message assumptions.
Removing the ramp changes LP30 decode from 1,522 to 2,561 TPS/user in the isolated probe.
Compiler-scheduled transport needs a separate service profile with measured startup latency and supported compute/communication overlap.
An arbitrary zero-cost communication setting would not provide that profile.

## Architecture and placement defects

The current proxy changes the expert count from 128 to 2,240 while keeping width, layer count and top-k fixed.
Its routed FFN activates approximately 3.583 billion parameters per token at either expert count.
Increasing storage to approximately 2T parameters does not increase that routed computation.
The resulting storage-to-active-work ratio strongly influences the large, mostly idle LP30 deployment.
NVIDIA's scaled architecture is unavailable in the referenced post.
Candidate width, layer and expert scaling schemes must report total parameters, activated parameters and per-token work independently.

The [public GPT-OSS-120B configuration](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) alternates 128-token sliding attention with full attention and uses MXFP4 expert storage.
The current proxy instead applies full attention and FP8 storage to every layer.
The [target blog](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) describes AF activation exchange at full-attention layers.
The experiment exchanges activations at all 36 layers and executes their attention on the A pool.
Layer-aware attention type, execution placement and KV ownership must be modeled explicitly.
Local attention on LP30 is a candidate interpretation of that exchange boundary and requires validation; it is not an established description of the unpublished 2T deployment.
The current uniform-window probe cannot implement or validate that heterogeneous placement.

Mixed precision must distinguish packed expert storage, scale metadata, execution dtype, activation dtype and KV dtype.
Four-bit storage could approximately halve raw expert bytes, but LP30 FP8 compute must not be doubled by assuming a native FP4 execution rate.
Attention, router and output-head precision need their own component accounting.
Capacity checks must remain active after changing storage formats.

Rubin and AF candidates currently have no context parallelism.
With eight KV heads, increasing attention TP beyond eight replicates KV heads and does not further reduce their per-rank reads.
High-interactivity candidates need explicit context sharding, distributed-softmax combination, physical placement and contention.
Every TP/EP/CP/PP inventory must be representable in the production runtime before being used for the reproduction frontier.

## KV handoff and metrics

PD batch four transfers 119.14 GB of retained KV through one assumed 200 GB/s link.
This costs approximately 596 milliseconds out of a 1.094-second turn.
If the destination already retains the 400K prefix, transferring only the 4K suffix would move 1.18 GB and save approximately 590 milliseconds.
That conditional change would improve this serialized turn's throughput by approximately 2.17x.
The target blog does not specify whether that destination prefix is retained for Figure 11.
Cold full-cache handoff and warm incremental handoff must be separate, explicit modes.
Prefix ownership, validity and actual transfer bytes must be verified in the runtime.

The horizontal axis excludes prefill and handoff while the vertical axis includes them.
That combination can describe serving, but the target chart's precise measurement interval is unspecified.
Decode-only and end-to-end throughput must both be exported.
Facility or rack power must be separated from processor-TDP assumptions.
A common declared normalization must be applied after computing physical throughput and power.
Neither axis normalization nor smoothing can repair the missing execution behavior.

## Repair order and acceptance checks

1. Move the reproduction workload into the production serving/AF runtime and add physical PP and combined expert/tensor layouts needed by the deployment.
   Schedule independent micro-batches and replicas and measure the steady-state interval.
   Confirm increasing pipeline concurrency can increase output without changing a request's numerical work or allocated memory.
2. Add layer-aware GPT-OSS attention, placement and KV ownership, mixed packed-weight storage and incremental KV handoff.
   Compare total/activated parameters and per-layer bytes/FLOPs across documented architecture candidates.
   Confirm local windows retain their bounded KV and that a warm handoff copies only the missing blocks.
3. Calibrate fused GPU execution and scheduled LP30 transport using operator or runtime measurements.
   Confirm startup, service, overlap and queueing are counted once and conserve communication bytes.
   Run launch, bandwidth, routing and acceptance sensitivity experiments independently.
4. Sweep feasible topology, batch size, micro-batch size, replica count and power configurations under one metric definition.
   Compare the unnormalized points, bottleneck transitions and resource utilization before generating the Pareto curves.
   Report the remaining gap to NVIDIA as an unverified projection difference until its model and deployment inputs are available.

Existing AF histogram, memory-admission and aggregate-bandwidth fixes remain valid.
The repair must preserve their regression coverage.
