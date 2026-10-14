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
