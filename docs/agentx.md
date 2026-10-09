# AgentX workloads

AgentX selects a trace dataset and a dependency-aware request generator. It
uses the same engine, scheduler and latency backend as other workloads. The
implementation is `TokenSim/workload/agentx.py`; parsing and trajectory helpers
are in `TokenSim/workload/agentx_support`.

There is no model-name switch in the benchmark entry point. A model's YAML
geometry and operator bindings select its projections, compressed attention,
residual mixing and draft layers. Serving configuration selects batching,
parallel synchronization, routing, operator backends and storage. The supplied
presets use the complete DeepSeek-V4-Pro model; proxy configurations and
research artifacts have been removed.

## Install and obtain traces

Python 3.12 is recommended. Python 3.11 is supported; a named conda environment
is not required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[agentx,dev]'
bash scripts/download_agentx.sh
```

The dataset is [SemiAnalysis's WEKA trace pool](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729),
pinned to revision `23f152f6f0f9399a85901b89a6458def0ef16729`. The download
contains 393 trace trees and is 1,847,151,435 bytes. Its SHA-256 is
`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`.
The script verifies the checksum, reuses an already verified file and accepts
an optional destination directory. The dataset stays outside version control.
The simulator accepts smaller pools and different profile windows.

Trace hashes must be unsigned 64-bit integers. The engine block size must be a
multiple of the trace block size, which is 64 tokens for this dataset. Complete
trace blocks are combined into engine-block identities; partial tails cannot
claim a cross-request cache hit.

## Run explicit full-model presets

`data/experiments/agentx.yaml` contains two examples. Its settings are explicit
inputs, rather than simulator defaults or enforced protocol constraints.

```bash
python scripts/run_agentx.py --case b200-c196 \
  --backend simple_cpu --seed 0 --results_path results/agentx/b200
python scripts/run_agentx.py --case b300-c384 \
  --backend simple_cpu --seed 0 --results_path results/agentx/b300
```

Both use eight GPU ranks in DEP8, engine blocks of 256 tokens, DP lockstep,
consistent-hash routing, a warmup and a 3600-second issue window. B200 uses
196 concurrent trees, 49 sequences per worker and an 8192-token step budget;
B300 uses 384 trees, 96 sequences and a 16384-token budget. The profile allows
30 seconds of grace and 10 seconds for cancellation and transfer cleanup.
These durations are configurable.

The backend argument overrides the preset. `hbm` disables external storage;
`simple_cpu` uses per-rank host pools; `mooncake` uses the explicitly selected
`HybridMooncakeStoreConnector` with a shared DRAM service. The ordinary
`MooncakeStoreConnector` supports homogeneous cache pages. The factory checks
layout support and never substitutes one connector for another.

For another model, topology or subset, use the ordinary CLI:

```bash
tokensim --workload_type agentx_weka \
  --dataset_path dataset/agentx/traces.jsonl \
  --model data/psla/llama-7b.json --cluster data/clusters/1_a100/h1.json \
  --batching paged-attn --block_size 64 --max_num_seqs 32 \
  --agentx_trace_count 1 --agentx_profile_duration 60 --no-agentx_warmup \
  --results_path results/agentx/subset
```

Select a model and cache budget that can accommodate the chosen prompts. The
full dataset includes very long contexts; the full-model presets are the
appropriate examples for those traces. Grouped physical caches require
chunked prefill with a finite token budget.

`random_seed` controls simulation randomness, including MTP progress.
`agentx_workload_seed` controls sampled warmup frontiers; it defaults to the
simulation seed, while the supplied presets explicitly use 42. Both are
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

The result directory contains `agentx_c<concurrency>.json`, `run_config.json`,
`kv_transfer_config.json` and the `replay/` logs: `events.jsonl`,
`requests.jsonl`, `cache_windows.jsonl` and `cache_configuration.json`.
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
