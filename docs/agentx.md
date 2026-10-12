# AgentX workloads

AgentX replays recorded agent requests while preserving the dependencies between them.
Each trace tree contains requests from a main agent and its subagents, along with their timing and dependencies.
AgentX uses the same simulation engine, scheduler and latency calculations as other workloads.
The replay implementation is in `TokenSim/workload/agentx.py`.
Trace parsing and warmup preparation are in `TokenSim/workload/agentx_support`.

## Install and obtain traces

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[agentx,dev]'
hf download semianalysisai/cc-traces-weka-062126 traces.jsonl \
  --repo-type dataset \
  --revision 23f152f6f0f9399a85901b89a6458def0ef16729 \
  --local-dir dataset/agentx
echo '29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276  dataset/agentx/traces.jsonl' | sha256sum --check
```

The dataset is [SemiAnalysis's WEKA traces](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729) at revision `23f152f6f0f9399a85901b89a6458def0ef16729`.
The file contains 393 trace trees and is 1,847,151,435 bytes.
Its SHA-256 is `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`.
The commands download that revision and verify the file's checksum.
To save the file elsewhere, change both `--local-dir` and the path passed to `sha256sum`.
Keep the downloaded dataset outside version control.
You can replay fewer traces or change the duration of the measurement window.

## Run with the simulator CLI

Run AgentX with `tokensim` or `python benchmark.py`, as you would any other workload.
The model, GPU layout, scheduling and storage options use the simulator's standard arguments.
AgentX arguments choose the traces and control how requests are replayed.

For CPU offload, each GPU worker (rank) gets a separate allocation of CPU memory for its attention key/value (KV) cache.
Save the following configuration as `kv_transfer.json`.

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

This example runs the complete DeepSeek-V4-Pro model on eight B200 GPUs.
DEP8 uses eight data-parallel ranks to run separate request batches, with model experts distributed across all eight GPUs.
Multi-token prediction (MTP) proposes three draft tokens at a time for the model to verify.

```bash
tokensim --workload_type agentx_weka \
  --dataset_path dataset/agentx/traces.jsonl \
  --model data/models/deepseek-v4-pro.yaml \
  --cluster data/clusters/8_b200/dep8.json \
  --batching paged-attn \
  --block_size 256 \
  --max_num_seqs 49 \
  --max_num_batched_tokens 8192 \
  --gpu_memory_utilization 0.9 \
  --router_policy consistent_hash \
  --dp_lockstep \
  --dp_sync_interval 16 \
  --long_prefill_token_threshold 512 \
  --prefill_schedule_interval 8 \
  --latency_backend operator_table \
  --latency_fallback table_first \
  --operator_backend vllm \
  --attention_backend module \
  --mhc_backend module \
  --moe_backend deep_gemm_amxf4_mega_moe \
  --mtp_num_speculative_tokens 3 \
  --mtp_acceptance_length 2.49 \
  --agentx_concurrency 196 \
  --agentx_profile_duration 3600 \
  --agentx_warmup \
  --agentx_issue_mode closed_loop \
  --agentx_idle_gap_cap 10 \
  --agentx_grace_period 30 \
  --agentx_cancellation_timeout 10 \
  --random_seed 0 \
  --agentx_workload_seed 42 \
  --kv_transfer_config kv_transfer.json \
  --verbose none \
  --results_path results/agentx/b200
```

The example runs 196 trace trees concurrently, and each tree can issue requests from multiple agents.
The 3,600-second measurement window starts after cache warmup.

To run on eight B300 GPUs, use the argument values below and choose another result directory.

| Argument | B200 example | B300 example |
| --- | --- | --- |
| `--cluster` | `data/clusters/8_b200/dep8.json` | `data/clusters/8_b300/dep8.json` |
| `--agentx_concurrency` | 196 | 384 |
| `--max_num_seqs` | 49 | 96 |
| `--max_num_batched_tokens` | 8192 | 16384 |
| `--gpu_memory_utilization` | 0.9 | 0.92 |

Omit `--kv_transfer_config` to use the cluster's storage configuration.
Both example clusters default to storing KV data only on GPUs.
For a shared store in CPU memory, pass `--kv_transfer_config data/kv_transfer/mooncake_store_dram_agentx.json`.
This file selects `MooncakeStoreConnector` with `cache_layout: grouped`, which supports models with multiple KV cache groups.
For models whose KV blocks all use the same layout, use the same connector with `cache_layout: uniform`, the default.

The [InferenceX AgentX script at the pinned revision](https://github.com/SemiAnalysisAI/InferenceX/blob/843c6254ff69b567f772f6a5fd97675af3d04431/benchmarks/single_node/agentic/dsv4_fp4_b200_vllm_mtp.sh) divides the CPU memory budget among ranks.
The published SimpleCPU runs allocate 356,125,000,000 bytes per rank, or 2,849 GB across eight ranks, using 1 GB = 10^9 bytes.
The script's Mooncake configuration rounds the per-rank store size down with `2849 // 8 = 356` and adds a separate `4GB` transfer buffer per rank.
vLLM interprets these size strings in gibibytes (GiB), where 1 GiB = 2^30 bytes.
The included Mooncake configuration therefore allocates 2,848 GiB to the shared KV store and another 32 GiB to transfer buffers.
The transfer buffers use additional memory and leave the KV store's capacity unchanged.
`host_available_bytes` uses the smaller recorded host memory limit of 2,964,436 MiB from B300.
The B200 launch recorded 3,095,781 MiB.
Both host limits accommodate the shared store and its transfer buffers.
These host memory figures come from archived launch records.

The script's embedded mode places the store on the same node as the GPUs.
Simulated transfers use the PCIe link specified in the hardware catalog, with no additional transfer over the RDMA network.
`transfer_utilization: 0.75` assumes that transfers reach 75% of the modeled link bandwidth; this value has not been measured.
The storage model omits uneven usage of memory segments assigned to individual ranks and eviction triggered by memory usage thresholds.
Both published comparison runs use SimpleCPU offload.
Selecting Mooncake changes the storage setup used in the comparison.
`enable_offload: false` disables the SSD tier and keeps caching in CPU memory enabled.
See [Mooncake storage](mooncake.md) for configuration details.

To use another model, GPU layout or trace subset, change the corresponding CLI arguments as in this example.

```bash
tokensim --workload_type agentx_weka \
  --dataset_path dataset/agentx/traces.jsonl \
  --model data/models/llama-7b.yaml \
  --cluster data/clusters/1_a100/h1.json \
  --batching paged-attn \
  --block_size 64 \
  --max_num_seqs 32 \
  --agentx_trace_count 1 \
  --agentx_profile_duration 60 \
  --no-agentx_warmup \
  --results_path results/agentx/subset
```

Select a model whose context limit and available KV cache can accommodate the chosen prompts.
The full dataset includes very long prompts, so use the DeepSeek-V4-Pro examples above to replay those traces.
For models with multiple KV cache groups, process prompts in chunks with a finite `--max_num_batched_tokens` limit.

`random_seed` controls simulation randomness, including MTP progress.
`agentx_workload_seed` selects the starting points within traces used to prepare caches during warmup.
It defaults to `random_seed`; the DeepSeek-V4-Pro example sets it to 42.
The results record both seeds.
Synthetic and dataset arrival generators use their own seeded random number stream.

## Replay and stopping

In `closed_loop` mode, a request is issued after its required predecessors finish and the recorded delay before the next request has elapsed.
Recorded API durations help determine request dependencies and delays.
The simulator calculates compute time from the selected model and latency backend, without adding those recorded API durations.
SPAWN starts a subagent, and JOIN makes the parent wait for the required subagents to finish.
Replay preserves these dependencies, the main agent's request sequences and independently running subagents.
Each replay of a trace tree has its own identifier that controls prefix cache sharing and stays the same across warmup and measurement.
When a tree finishes, its replay slot takes the next tree from the dataset, cycling back to the start when needed.

`--agentx_issue_mode recorded` schedules requests using the timestamps in each trace tree.
Each replacement tree uses its own recorded timestamps relative to its replay start.
A request waits longer if its dependencies have not finished by its scheduled time.
In `closed_loop` mode, idle limits shorten long gaps with no requests ready to run.
The results record how often these gaps are shortened and how much idle time is skipped.

`--sim_time` stops the simulation at the specified simulated time, including during warmup.
Pending transfers stop at that cutoff as well.
The saved partial results include `termination_reason=sim_time_limit` and metadata describing the replay state at the cutoff.
At the normal end of the measurement window, AgentX stops issuing new requests and allows outstanding requests to finish during `--agentx_grace_period`.
It then cancels the remaining requests, waits up to `--agentx_cancellation_timeout` for cancellation and pending transfers to finish, and checks cache references and allocation accounting.
An unacknowledged cancellation at the deadline stops the run with `termination_reason=failed`, including during warmup.
The `notdone` list and `failure.json` retain the IDs of outstanding requests.
If cleanup fails or the cache's allocation counts are inconsistent, the simulator saves results and diagnostic files before returning an error.

## Results and interpretation

The result directory contains `agentx_c<concurrency>.json` and a `replay/` directory with `events.jsonl`, `requests.jsonl`, `cache_windows.jsonl` and `cache_configuration.json`.
Failures also produce `failure.json` and `failed.txt` in that directory.
The configuration log stores cache descriptions that are shared by multiple log entries.

AgentX reports output tokens per second, request latency percentiles, time to first token (TTFT), inter-token latency (ITL), prefix-source counts, cache hit rates and replay state.
Prefix-source counts separate input tokens reused from GPU memory, loaded from CPU memory and computed by the model.
Output token totals count actual tokens generated by successful requests issued during the measurement window.
Latency summaries exclude warmup, cancelled and unfinished requests.
Engine statistics record repeated computation, time spent waiting for transfers and the measurement or formula used to estimate each operator's latency.
Model-specific fields appear only when used.
Unknown result fields cause an error.

These examples use latency and memory estimates with the following limitations.

- B200 and B300 estimates for MegaMoE, a fused mixture-of-experts (MoE) kernel, use SGLang measurements from GB200 and GB300.
  Metadata records the source hardware, framework, token capacity of each rank's buffers and operations included in each measurement.
- Sparse-attention measurements use FlashMLA, and the reference serving runs used FlashInfer.
  The measured kernels may use KV entries that also store quantization scales or rotary positional embedding (RoPE) data, increasing their size relative to the reference serving layout.
  The model's operator configuration explicitly records these sizes.
- Formulas approximate uneven work across GPU ranks, shared experts computed within the fused MoE kernel, overlap between execution phases, attention used to draft tokens and combining the model's residual outputs.
  Metadata labels these calculations as analytical estimates and records their latency contributions.
- Synthetic token counts per expert use fixed random seeds 0, 1 and 2 to approximate the distribution used when collecting kernel timings.
  These counts are generated independently of the simulation seed and do not reproduce the original expert assignments from the measured runs.
- The number of GPU streaming multiprocessors (SMs) used to size persistent buffers (GPU memory reserved between kernel calls) is configurable for each hardware entry.
  The configured values are estimates that have not been measured.

### Comparison with the InferenceX reference

The two examples use the configurations from published InferenceX AgentX runs of DeepSeek-V4-Pro on vLLM with DEP8, three MTP draft tokens per step (K=3) and SimpleCPU offload.
The B200 configuration uses concurrency 196 from InferenceX run 440239, and B300 uses concurrency 384 from run 439522.
The table compares simulations using the commands above (`--random_seed 0`, `--agentx_workload_seed 42`) with the published measurements.
Latency percentiles describe successful requests issued during the measurement window, including those that finish during the grace period.
TTFT measures the time from request arrival to the first output token.
ITL is the per-request mean time between output tokens after the first token.
End-to-end latency measures the time from request arrival to completion.
p50 is the median, and p90 is the 90th percentile.
Percentage error is `(simulated - measured) / measured × 100%`.

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

B300 throughput is about 13–16% lower than measured, and end-to-end latency is about 14–15% higher.
ITL p90 is 36.6% higher.
B200 throughput is about 40% lower, and its latency estimates are roughly two to three times the measured values.
On B200, the simulated GPU cache retains fewer reused prefix tokens, so more tokens must be reloaded from CPU memory.
The approximations listed above are the main known sources of the differences.
Their parameters have not been fitted to these measurements.

### Run time

A one-hour simulated measurement window takes about 9 minutes to run for B200 and 19 minutes for B300 on one core of an AMD EPYC 7C13.
Peak memory usage is about 5.4 GB for B200 and 6.6 GB for B300.
Loading the 393 traces takes roughly 30 seconds of each run.
Identical inputs and seeds produce deterministic results.

## Core validation

```bash
python -m unittest discover \
  -s tests \
  -t .
```

Tests check request timing, trace replacement, cancellation without counting ungenerated tokens and first-in-first-out scheduling.
They also check that prefill-to-decode transfer time is fully accounted for, grouped KV caches can be restored and cache references remain valid.
Time limits and result export after failures are covered as well.
The suite uses small, deterministic cases.
It does not compare throughput with measurements or run short benchmark trials.
