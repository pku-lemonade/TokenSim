# AgentX workloads

AgentX replays recorded agent requests while preserving the dependencies between them.
Each trace tree contains requests from a main agent and its subagents, along with their timing and dependencies.
AgentX uses the same simulation engine, scheduler and latency calculations as other workloads.
The replay implementation is in `TokenSim/workload/agentx.py`.
Trace parsing and warmup preparation are in `TokenSim/workload/agentx_support`.
The [accuracy and performance implementation plan](agentx-accuracy-implementation.md) specifies planned corrections, measurement requirements and validation gates.

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
`transfer_utilization: 0.75` sets effective host-transfer throughput to 75% of the link peak.
The storage model tracks aggregate pool capacity and per-rank staging buffers across the shared host memory budget.
The published comparison runs use SimpleCPU offload, while selecting Mooncake enables shared multi-rank DRAM pooling.
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
A finite `--max_num_batched_tokens` must be at least one plus `--mtp_num_speculative_tokens` to fit a complete MTP verification query.
This requirement also applies when the expected accepted length is one token.

`random_seed` controls simulation randomness, including MTP progress.
`agentx_workload_seed` selects the starting points within traces used to prepare caches during warmup.
It defaults to `random_seed`; the DeepSeek-V4-Pro example sets it to 42.
The results record both seeds.
Synthetic and dataset arrival generators use their own seeded random number stream.

`--mtp_acceptance_length` selects a two-point acceptance approximation with the requested mean.
Each verification round samples independently using `blake2b_round_v1`, keyed by the worker seed, request ID and committed decode-round index.
Repeated scheduling probes and failed reservations reuse the same draw; actual decode execution advances the index.
KV recomputation retains that index.
Full verification query width determines target work even when accepted output is clipped by the remaining output length.

`--mtp_acceptance` accepts a JSON object to select another acceptance model.
Conditional acceptance supplies exactly one probability for each of the K draft positions and stops at the initial rejection.
For K=3, this example has an expected accepted length of `1 + 0.8 + 0.8*0.6 + 0.8*0.6*0.5 = 2.52`.

```json
{
  "kind": "conditional",
  "rates": [0.8, 0.6, 0.5]
}
```

Empirical acceptance supplies K+1 probabilities for accepted lengths 1 through K+1.
They must be finite, nonnegative and sum to one.
Both explicit models require the default `--mtp_acceptance_length 1`; the distribution determines the expected progress.

```json
{
  "kind": "empirical",
  "distribution": [0.1, 0.2, 0.3, 0.4]
}
```

Historical linear sampling is available through `--mtp_acceptance` with `kind` set to `legacy`, together with the original mean and seeds.
Results retain the selected model, expected acceptance, sampling version, seed and committed verification/accepted/emitted counts under `module_stats.speculative.acceptance`, keyed by worker ID.
Warmup resets those measured counts while preserving each continuing request's round index.

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
Root and subagent output totals and ITL summaries use the actual generated lengths, including when the model's context limit truncates an output.
Requests that generate only one token are excluded from AgentX ITL summaries.
Interactivity uses the inverse of the corresponding latency statistic, including the arithmetic mean.
For example, `interactivity_tps.p90` is `1 / tpot_s.p90`, and normalized interactivity uses end-to-end seconds per actual output token.
Percentiles use NumPy linear interpolation; computing percentiles of reciprocal samples would reverse the latency tail and change the interpolated value.
Results record `interactivity_percentile_semantics: inverse_of_matching_latency_percentile` to identify this convention.
Latency summaries exclude warmup, cancelled and unfinished requests.
Engine statistics record repeated computation, time spent waiting for transfers and the measurement or formula used to estimate each operator's latency.
Model-specific fields appear only when used.
Unknown result fields cause an error.

The DeepSeek-V4-Pro configuration combines module-level measurements and analytical models as follows.

- Fused mixture-of-experts (MoE) execution (`MegaMoE`) uses measured SGLang module tables from Blackwell systems (`GB200` and `GB300`), recording the source hardware, framework, per-rank token buffer capacities and covered operations in result metadata.
- Compressed sparse attention uses measured `FlashMLA` module tables and accounts for the exact KV entry layout, including quantization scales and rotary positional embedding (RoPE) storage.
- Analytical models account for load imbalance across expert ranks, shared experts fused into the MoE kernel, phase overlap, draft-token attention and residual stream combination, tagging each contribution in result provenance.
- Deterministic expert routing uses fixed seeds (`0`, `1` and `2`) to generate repeatable token-to-expert assignment distributions matching the profiling conditions.
- Each hardware entry configures the streaming multiprocessor (SM) count used to size persistent workspace buffers reserved between kernel calls.

### Comparison with the InferenceX reference

The validation benchmark replays the published InferenceX AgentX configuration of DeepSeek-V4-Pro on eight B300 GPUs (InferenceX run 439522) using vLLM with DEP8, three MTP draft tokens per step (`K=3`), concurrency 384 and SimpleCPU offload.
The table below compares the simulation (`--random_seed 0`, `--agentx_workload_seed 42`) with the published hardware measurements.
These historical results precede the `blake2b_round_v1` acceptance implementation; retain the `legacy` model to reproduce their sampling behavior.
Latency percentiles describe successful requests issued during the measurement window, including those that finish during the grace period.
TTFT measures the time from request arrival to the first output token.
ITL is the per-request mean time between output tokens after the first token.
End-to-end latency measures the time from request arrival to completion.
p50 is the median, and p90 is the 90th percentile.
Percentage error is `(simulated - measured) / measured × 100%`.

| Metric | B300 simulated | B300 measured | Error |
| --- | ---: | ---: | ---: |
| Output tokens/s (8 GPUs) | 5,322 | 6,360 | -16.3% |
| Input tokens/s (8 GPUs) | 705,681 | 813,832 | -13.3% |
| Successful requests | 21,688 | 25,696 | -15.6% |
| TTFT p50 (s) | 4.58 | 4.27 | +7.5% |
| TTFT p90 (s) | 19.8 | 18.9 | +4.5% |
| Inter-token latency p50 (s) | 0.049 | 0.045 | +9.3% |
| Inter-token latency p90 (s) | 0.068 | 0.050 | +36.6% |
| End-to-end p50 (s) | 30.2 | 26.5 | +13.8% |
| End-to-end p90 (s) | 129.0 | 112.1 | +15.1% |
| GPU prefix hit rate | 0.947 | 0.899 | |
| CPU prefix hit rate | 0.008 | 0.016 | |

Simulated B300 TTFT matches hardware within 7.5% at p50 and 4.5% at p90, while median ITL is within 9.3% and end-to-end request latency is within 13.8% to 15.1%.
Input and output token throughputs match within 13.3% and 16.3% across the one-hour closed-loop multi-agent trace without fitting workload-specific parameters to the benchmark run.

### Run time

Use `scripts/compare_agentx_results.py` to compare a frozen baseline/current matrix against archived hardware points.
Its manifest names a `reference_directory` containing `inferencex_key_points.json` and a `cases` array with `label`, `hardware`, `reference_id`, `concurrency`, `output`, and the complete simulation `args`.
Each hardware needs exactly one `baseline` case and one `current` case with identical arguments except `--results_path`.
The report retains signed metric errors, changes in absolute error, individual regressions, incomplete runs, and SHA-256 hashes of its inputs.
Cache ratios remain diagnostics because hardware server counters and successful-request admission counters use different populations.

```bash
python scripts/compare_agentx_results.py \
  --manifest results/agentx-accuracy-20261026/manifest.json \
  --output results/agentx-accuracy-20261026/comparison.json \
  --require-no-regression
```

The command exits with status 2 for incomplete or invalid comparisons and status 1 when `--require-no-regression` detects a throughput, median/tail TTFT, or median/tail ITL regression.
Omitting this flag exports a valid comparison containing regressions without treating them as improvements.

A one-hour simulated measurement window completes in about 9 minutes for B200 and 19 minutes for B300 on a single core of an AMD EPYC 7C13 processor.
Peak host memory usage is 5.4 GB for B200 and 6.6 GB for B300, including roughly 30 seconds to load the 393 trace trees.
Identical inputs and seeds produce bit-exact deterministic results.

## Core validation

```bash
python -m unittest discover \
  -s tests \
  -t .
```

The test suite verifies request timing, trace replacement, cancellation accounting, first-in-first-out scheduling, prefill-to-decode transfer accounting, grouped KV cache restoration, reference integrity, time limits and diagnostic exports.
