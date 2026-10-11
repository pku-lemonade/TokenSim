# AgentX workloads

AgentX selects a trace dataset and a dependency-aware request generator. It
uses the same engine, scheduler and latency backend as other workloads. The
implementation is `TokenSim/workload/agentx.py`; parsing and trajectory helpers
are in `TokenSim/workload/agentx_support`.

There is no model-name switch in the benchmark entry point. A model's YAML
geometry and operator bindings select its projections, compressed attention,
residual mixing and draft layers. Serving configuration selects batching,
parallel synchronization, routing, operator backends and storage. The supplied
examples use the complete DeepSeek-V4-Pro model; proxy configurations and
research artifacts have been removed.

## Install and obtain traces

Python 3.12 is recommended. Python 3.11 is supported; a named conda environment
is not required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[agentx,dev]'
hf download semianalysisai/cc-traces-weka-062126 traces.jsonl \
  --repo-type dataset --revision 23f152f6f0f9399a85901b89a6458def0ef16729 \
  --local-dir dataset/agentx
echo '29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276  dataset/agentx/traces.jsonl' | sha256sum --check
```

The dataset is [SemiAnalysis's WEKA trace pool](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729),
pinned to revision `23f152f6f0f9399a85901b89a6458def0ef16729`. The download
contains 393 trace trees and is 1,847,151,435 bytes. Its SHA-256 is
`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`.
The commands download the pinned file and verify its checksum. Change
`--local-dir` and the checksum path to use another destination. The dataset
stays outside version control. The simulator accepts smaller pools and
different profile windows.

Trace hashes must be unsigned 64-bit integers. The engine block size must be a
multiple of the trace block size, which is 64 tokens for this dataset. Complete
trace blocks are combined into engine-block identities; partial tails cannot
claim a cross-request cache hit.

## Run with the simulator CLI

All workloads use `tokensim` (or `python benchmark.py`). Model, topology,
serving and storage settings are ordinary simulator inputs; AgentX arguments
select the traces and replay behavior.

For per-rank CPU offload, save the following object as `kv_transfer.json`.
These example host capacities are in bytes; adjust them for your hardware.

```json
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use_per_rank": 356125000000,
    "host_available_bytes": 3108436443136,
    "lazy_offload": false,
    "transfer_utilization": 0.75
  }
}
```

An example using the complete model on eight B200 ranks in DEP8:

```bash
tokensim --workload_type agentx_weka \
  --dataset_path dataset/agentx/traces.jsonl \
  --model data/models/deepseek-v4-pro.yaml --cluster data/clusters/8_b200/dep8.json \
  --batching paged-attn --block_size 256 --max_num_seqs 49 \
  --max_num_batched_tokens 8192 --gpu_memory_utilization 0.9 \
  --router_policy consistent_hash --dp_lockstep --dp_sync_interval 16 \
  --long_prefill_token_threshold 512 --prefill_schedule_interval 8 \
  --latency_backend operator_table --latency_fallback table_first \
  --operator_backend vllm --attention_backend module --mhc_backend module \
  --moe_backend deep_gemm_amxf4_mega_moe \
  --mtp_num_speculative_tokens 3 --mtp_acceptance_length 2.49 \
  --agentx_concurrency 196 --agentx_profile_duration 3600 --agentx_warmup \
  --agentx_issue_mode closed_loop --agentx_idle_gap_cap 10 \
  --agentx_grace_period 30 --agentx_cancellation_timeout 10 \
  --random_seed 0 --agentx_workload_seed 42 \
  --kv_transfer_config kv_transfer.json \
  --verbose none --results_path results/agentx/b200
```

For eight B300 ranks, change these arguments and choose another result directory:

| Argument | B200 example | B300 example |
| --- | --- | --- |
| `--cluster` | `data/clusters/8_b200/dep8.json` | `data/clusters/8_b300/dep8.json` |
| `--agentx_concurrency` | 196 | 384 |
| `--max_num_seqs` | 49 | 96 |
| `--max_num_batched_tokens` | 8192 | 16384 |
| `--gpu_memory_utilization` | 0.9 | 0.92 |

Omit `--kv_transfer_config` to use the cluster's storage configuration (these
two clusters default to GPU-only storage). For a shared DRAM store, pass
`--kv_transfer_config data/kv_transfer/mooncake_store_dram_agentx.json`, which
selects `MooncakeStoreConnector` with `cache_layout: grouped`. Uniform models
use the same connector with `cache_layout: uniform` (the default).

The [pinned AgentX recipe](https://github.com/SemiAnalysisAI/InferenceX/blob/843c6254ff69b567f772f6a5fd97675af3d04431/benchmarks/single_node/agentic/dsv4_fp4_b200_vllm_mtp.sh)
splits the CPU budget over ranks. The recorded SimpleCPU points use
356,125,000,000 bytes per rank (2849 decimal GB over eight ranks). Its Mooncake
branch rounds the per-rank segment to whole GB: `2849 // 8 = 356`, and uses
a separate `4GB` staging buffer per rank. vLLM's size parser interprets these
strings as GiB, whereas the SimpleCPU branch calculates decimal bytes. The
included Mooncake example therefore has 2848 GiB of shared KV capacity plus
32 GiB of staging; staging does not reduce
the KV pool. `host_available_bytes` uses the smaller recorded B300 host limit:
2,964,436 MiB; B200 reported 3,095,781 MiB. Both fit this budget. These host
figures come from the archived launch records, not a runtime host probe.

The recipe's embedded mode places the store on the compute node, so simulated
transfers use the hardware catalog's PCIe link, without an extra RDMA hop.
`transfer_utilization: 0.75` remains an unmeasured transfer-efficiency assumption;
owner-segment imbalance and watermark eviction are not modeled. The two
published comparison points use SimpleCPU; selecting Mooncake changes the
storage experiment. `enable_offload: false` leaves DRAM caching enabled and
disables only the SSD tier. See [Mooncake storage](mooncake.md) for configuration.

For another model, topology or subset, change the same CLI inputs:

```bash
tokensim --workload_type agentx_weka \
  --dataset_path dataset/agentx/traces.jsonl \
  --model data/models/llama-7b.yaml --cluster data/clusters/1_a100/h1.json \
  --batching paged-attn --block_size 64 --max_num_seqs 32 \
  --agentx_trace_count 1 --agentx_profile_duration 60 --no-agentx_warmup \
  --results_path results/agentx/subset
```

Select a model and cache budget that can accommodate the chosen prompts. The
full dataset includes very long contexts; the full-model examples are the
appropriate examples for those traces. Grouped physical caches require
chunked prefill with a finite token budget.

`random_seed` controls simulation randomness, including MTP progress.
`agentx_workload_seed` controls sampled warmup frontiers; it defaults to the
simulation seed, while the full-model example explicitly uses 42. Both are
reported. Synthetic and dataset arrival generators use a local seeded stream.

## Replay and stopping

Closed-loop mode releases requests when their predecessors finish and the
recorded think-time gap has elapsed. Captured API duration helps reconstruct
relationships; it is not added to simulated compute time. SPAWN/JOIN relations,
root streams, independent subagents and cache scope are retained. Completed
trees are replaced from the trace pool.

`--agentx_issue_mode recorded` schedules against each tree's source clock,
including when a tree is recycled. Dependency completion can delay a request.
Idle caps apply to closed-loop replay; they compress long periods without
ready work and are recorded as counters.

`--sim_time` is a hard simulated-time cutoff, including during warmup. It does
not continue draining transfers past that cutoff. Partial results carry
`termination_reason=sim_time_limit` and partial metadata. Normal completion
stops admissions, permits grace completions, cancels the remaining requests
and checks cache references. Failed cleanup or conservation checks preserve
the result and diagnostic files before returning an error.

## Results and interpretation

The result directory contains `agentx_c<concurrency>.json` and the `replay/`
logs: `events.jsonl`, `requests.jsonl`, `cache_windows.jsonl` and
`cache_configuration.json`.
Failures also produce `failure.json` and `failed.txt` in that directory.
Repeated cache descriptions are shared in the configuration log.

AgentX metrics include output tokens per second, request latency percentiles,
TTFT, ITL, prefix-source counts, cache hit rates and replay state. Completed
requests contribute actual generated output tokens; warmup, cancelled and
unfinished requests are excluded from profile latency summaries. Engine
statistics include recomputation, transfer waits and operator provenance.
Model-specific result fields are omitted when unused. Unknown result fields
are rejected rather than silently discarded.

The examples are estimates with documented limits:

- Fused MegaMoE data for B200/B300 uses GB200/GB300 SGLang measurements. The
  source hardware, framework, capacity and timed scope remain visible.
- Complete sparse-attention measurements use FlashMLA; the serving reference
  used FlashInfer. Source-layout KV entries may include scale/RoPE metadata,
  unlike the plain serving KV layout. These sizes are explicit in the model's
  operator binding and are not an implicit model-name rule.
- Unequal-rank work, shared-expert fusion, phase overlap, draft attention and
  residual output reduction include analytical approximations. Their source
  grade and contribution are retained in metadata.
- The synthetic expert-work spectra use fixed draws 0/1/2 to approximate the
  collector's distribution. They are not recovered historical expert routes
  and are independent of the simulation seed.
- Persistent-buffer SM counts are configurable per hardware entry. The
  current values remain an explicitly unmeasured estimate.

### Comparison with the InferenceX reference

The two example points reproduce published InferenceX AgentX runs of
DeepSeek-V4-Pro on vLLM with DEP8, MTP (K=3) and SimpleCPU offload: B200 at
concurrency 196 (InferenceX run 440239) and B300 at concurrency 384 (run
439522). The table compares the commands above (`--random_seed 0`,
`--agentx_workload_seed 42`) with the measured values. Latency percentiles are
over successful requests in the measurement window; inter-token latency is the
per-request mean after the first token. Error is simulated minus measured,
relative to measured.

| Metric | B200 simulated | B200 measured | Error | B300 simulated | B300 measured | Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Output tokens/s (8 GPUs) | 2,166 | 3,599 | -39.8% | 5,322 | 6,360 | -16.3% |
| Input tokens/s (8 GPUs) | 281,449 | 470,701 | -40.2% | 705,681 | 813,832 | -13.3% |
| Successful requests | 8,932 | 13,991 | -36.2% | 21,688 | 25,696 | -15.6% |
| TTFT p50 (s) | 7.34 | 3.36 | +118% | 4.58 | 4.27 | +7.5% |
| TTFT p90 (s) | 42.2 | 16.2 | +160% | 19.8 | 18.9 | +4.5% |
| Inter-token latency p50 (s) | 0.070 | 0.034 | +105% | 0.049 | 0.045 | +9.3% |
| Inter-token latency p90 (s) | 0.110 | 0.050 | +120% | 0.068 | 0.050 | +36.6% |
| End-to-end p50 (s) | 42.7 | 21.4 | +99% | 30.2 | 26.5 | +13.8% |
| End-to-end p90 (s) | 182.7 | 95.4 | +92% | 129.0 | 112.1 | +15.1% |
| GPU prefix hit rate | 0.386 | 0.682 | | 0.947 | 0.899 | |
| CPU prefix hit rate | 0.566 | 0.237 | | 0.008 | 0.016 | |

B300 is within about 16% on throughput and request latency; B200
underestimates throughput by about 40% and overestimates latency by about a
factor of two. On B200 the simulated GPU cache retains less of the reused
prefix, so more of it is reloaded from CPU memory. The approximations listed
above are the main known sources of the gap; they are not calibrated against
these measurements.

### Run time

A one-hour window takes about 9 minutes for the B200 example and about 19
minutes for the B300 example on one core of an AMD EPYC 7C13; peak memory is
about 5.4 GB and 6.6 GB. Loading the 393 traces accounts for roughly 30 s of
each run. Runs are deterministic for a given input and seeds.

## Core validation

```bash
python -m unittest discover -s tests -t .
```

Tests cover dependency clocks and recycling, cancellation without fake output,
FIFO scheduling, PD transfer-time conservation, grouped-cache recovery,
reference lifetimes, deadlines and durable result export. The suite uses
small deterministic cases and no throughput-alignment or smoke checks.
