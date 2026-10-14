# Groq 3 LPX figure reproduction

The target is Figure 11 of [NVIDIA's long-context LPX blog](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/).
Its workload has 400,000 cached input tokens, 4,000 new input tokens and 400 output tokens.
The chart compares Rubin, external drafting, attention–feed-forward network (AF) separation and prefill–decode (PD) separation.
The published curves are projections for a scaled GPT-OSS model.
NVIDIA has not supplied the curve samples, exact model architecture, placement, power budget or draft acceptance settings.

## Hardware inputs

`groq_lp30` models the third-generation LPU independently of `groqchip_v1`.
The [LPX architecture article](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) reports 500 MB SRAM and 150 TB/s SRAM bandwidth per chip, with 256 chips per rack.
FP8 peak is derived from the published 315 PFLOPS rack total.
The LP30 injection link uses 1.25 TB/s per direction, derived from the published 2.5 TB/s bidirectional total.
The individual 112 Gb/s link is a separate 14 GB/s entry.
An aggregate injection budget must be counted once per endpoint.
`bandwidth_scope: endpoint` on a link prevents automatic Groq topology construction from multiplying its aggregate budget by the chip's port count.
Links describing one physical port retain the default `bandwidth_scope: link`.
LPX external DRAM is excluded from SRAM-resident experiments.

The `rubin` entry follows the [product specifications](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) retrieved on 2026-10-14.
Those specifications give 288 GB HBM4, 19.2 TB/s memory bandwidth and 3 TB/s bidirectional NVLink bandwidth per GPU.
The catalog uses dense FP8 and BF16 compute specifications.
The July architecture article quotes higher bandwidth maxima; those values are not substituted into this snapshot.

Compute efficiencies, first-bit latency, inter-rack routing, runtime overhead and chip power are assumptions with evidence grade D.
The catalog's 450 W LP30 and 1,800 W Rubin power values are experiment defaults.
They do not represent measured rack power.
Host, cooling and network power must be supplied separately when reporting facility efficiency.

## Model inputs

`gpt-oss-2t-proxy` retains the public GPT-OSS-120B dimensions and increases its expert count to 2,240, giving approximately 2T parameters.
Its extended context limit, uniform full attention and FP8 weight precision are explicit modeling assumptions.
The public checkpoint uses alternating local and full attention and MXFP4 expert weights.
The proxy supplies a conservative full-context attention workload for the experiment.
Simulator results must retain the proxy name and assumptions when compared with the published chart.

## Run the experiments

Run the production AF audit and the four deployment sweeps from a repository checkout:

```bash
python scripts/reproduce_lpx.py \
  --trace \
  --plot \
  --output outputs/lpx-reproduction
```

`experiments/lpx/sweep.json` declares the workload, batch sizes and rank inventories.
The default sweep evaluates 84 configurations.
`summary.json` retains every feasible point, capacity rejection, Pareto frontier, hardware specification and assumption.
`points.csv` provides the metrics as a table.
`af-audit.json` contains production AF runtime plans and simulated event traces for batches 1, 4 and 16.
The audit executes one resident layer of the proxy on 8 Rubin GPUs and 256 LP30 chips.
Its negative control checks that the same LP30 placement rejects the entire model.
The audit does not extrapolate that layer's throughput to the full model.

Use a short sweep when checking script changes:

```bash
python scripts/reproduce_lpx.py \
  --skip-audit \
  --batches 1 \
  --output-tokens 4 \
  --context-samples 1 \
  --output outputs/lpx-smoke
```

`--audit-only` runs only the production AF checks.
`--config` selects another experiment JSON.
`--batches`, `--output-tokens`, `--context-samples` and `--acceptance` override their corresponding workload inputs.
`--rubin-watts` and `--lp30-watts` override processor power.
`--facility-watts` adds a fixed host, network and cooling power allowance for each deployment.

Install plotting dependencies with `python -m pip install -e '.[dev,plot]'`.
`--plot` generates PNG and editable SVG files after the sweep.
Plot an existing summary without running the simulation again:

```bash
python scripts/plot_lpx.py \
  --summary outputs/lpx-reproduction/summary.json \
  --output outputs/lpx-reproduction
```

`reference.png` reconstructs the supplied figure using the approximate manual samples in `experiments/lpx/reference.json`.
Those samples include reading uncertainty and their source URL.
`simulation.png` plots the simulated Pareto frontiers and includes an inset for the low-efficiency region.
`comparison.png` places the reference and simulation on matching axis ranges.
Reference and simulation have independent normalization denominators.
Reference-only 3x/5x annotations retain their source provenance.
The reference samples never enter the simulator's timing, calibration or power calculations.
Every PNG has a corresponding SVG.

## Execution and metrics

The sweep uses TokenSim's AF operator provider, exact routed-expert work, collective service model and SimPy resources.
The full-model experiment composes layer partitions and context partitions explicitly.
These partitions extend the experiment's analytical scope beyond the production AF runtime's PP=1 and EP=1 or TP*DP contract.
Every rank must fit its resident weights, KV shards and activation workspace.
The default AF FFN layout has four layer partitions, 320 expert partitions and four tensor shards per expert, totaling 5,120 LP30 chips.
All allocated processors contribute to the power denominator.
Context parallelism replicates projections and divides cached KV reads across context shards.
An explicit collective combines the attention results.
Layer boundaries and A/F dispatch/return have transport costs.

Each pool handles one synchronous batch with serialized model passes.
The model does not overlap different requests or 320-byte compiler transfers.
Decode uses a weighted average of representative context positions and deterministic routed histograms.
Prefill routing uses balanced counts while conserving the exact number of assignments.
The 400K warm prefix is resident before timing begins.
Only the 4K suffix is computed during prefill.
PD copies the retained prefix and suffix KV once before decode.
AF retains decode KV on the GPU and copies it from the separately allocated prefill GPU pool.

External drafting uses a four-layer, 128-expert proxy with an independent KV cache.
Its warm prefix is also resident, and its new-suffix prefill is charged.
For `gamma` draft tokens and independent acceptance probability `p`, expected committed tokens per cycle equal `sum(p**i for i in range(gamma + 1))`.
Every cycle charges all `gamma + 1` verification positions, draft generation and token-control transport.
A fractional terminal cycle expresses the expected-value approximation.
Model quality and acceptance calibration require measurements outside TokenSim.

Per-user interactivity is the number of tokens produced after the prefill token divided by decode time.
Output throughput is the batch's 400 output tokens divided by the complete timed turn, including suffix prefill and handoff.
TPS/MW divides that throughput by allocated processor TDP plus the supplied facility allowance.
All simulated curves share the peak Rubin TPS/MW normalization.
Each curve keeps only nondominated feasible configurations.
Original 3x and 5x annotations must not be applied to simulated results without supporting output metrics.

## Default sweep observations

The 84-configuration sweep produces 61 feasible points and 23 capacity rejections under the checked-in assumptions.
The production AF layer audit completes at 144.2, 356.4 and 1,205.3 simulated microseconds for batches 1, 4 and 16.
The full-model negative control requires 8.06 GB of expert weights on an LP30 rank with 500 MB SRAM and fails before execution.

| Deployment | Maximum simulated TPS/user | Maximum TPS/MW relative to Rubin |
| --- | ---: | ---: |
| Rubin | 464 | 1.0000 |
| Rubin verifier + LP30 draft | 963 | 0.0963 |
| Rubin attention + LP30 FFN | 506 | 0.0115 |
| Rubin prefill + LP30 decode | 1,522 | 0.0057 |

The speed and efficiency maxima may come from different configurations.
The model's uniform full attention and FP8 expert storage are expensive at 400K context and approximately 2T parameters.
The capacity checks force large LP30 inventories, and the assumed processor power charges those inventories.
The simulation does not establish NVIDIA's 3x or 5x gains.
Measured architecture, routing, compiler overlap, precision and power inputs are required to test those projections.
