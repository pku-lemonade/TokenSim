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
    "host_available_bytes": 3246161657856,
    "lazy_offload": false,
    "transfer_utilization": 0.75
  }
}
```

An example using the complete model on eight B200 ranks in DEP8:

```bash
tokensim --workload_type agentx_weka \
  --dataset_path dataset/agentx/traces.jsonl \
  --model data/psla/deepseek-v4-pro.json --cluster data/clusters/8_b200/dep8.json \
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
explicitly selects `HybridMooncakeStoreConnector`. The ordinary
`MooncakeStoreConnector` supports homogeneous cache pages. The factory checks
layout support and never substitutes one connector for another.
The shared-store capacities are example assumptions, not measured hardware
limits. See [Mooncake storage](mooncake.md) for the simulator-specific connector
and capacity semantics; `enable_offload: false` leaves DRAM caching enabled and
disables only the SSD tier.

For another model, topology or subset, change the same CLI inputs:

```bash
tokensim --workload_type agentx_weka \
  --dataset_path dataset/agentx/traces.jsonl \
  --model data/psla/llama-7b.json --cluster data/clusters/1_a100/h1.json \
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

The engineering optimization reduced the measured full-window wall time from
35.5 to 11.2 minutes on B200 and from 83.7 to 25.3 minutes on B300, with the
same simulation results and logs. Those timings depend on the host CPU and
load. Subsequent correctness fixes can change simulation metrics; historical
throughput is not a unit-test target.

## Core validation

```bash
python -m unittest discover -s tests -t .
```

Tests cover dependency clocks and recycling, cancellation without fake output,
FIFO scheduling, PD transfer-time conservation, grouped-cache recovery,
reference lifetimes, deadlines and durable result export. The suite uses
small deterministic cases and no throughput-alignment or smoke checks.
