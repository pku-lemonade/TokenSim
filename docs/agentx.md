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
The command uses analytical memory estimates by default.
To reproduce the observed-capacity results below, prepare an external profile as described under [observed runtime KV capacity](#observed-runtime-kv-capacity) and add `--runtime_memory_profile runtime_memory.json`.

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

Observed-capacity replay requires a separate profile for each deployment.
The reported B200 and B300 runs use 32.27 GiB and 103.07 GiB of available KV memory per rank, respectively.

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

### Observed runtime KV capacity

`--runtime_memory_profile PATH` imports available KV bytes recorded after the target serving framework reserves weights, CUDA graphs, and workspaces.
This mode currently supports grouped compressed KV caches on ordinary workers.
Without this argument, the existing analytical memory estimate determines capacity.
The observation changes cache capacity; operator timing continues to use the configured tables and formulas.

Every compressed cache exports `worker_cache_budgets[].cache.memory_budget.deployment_identity` in the result JSON.
Copy this identity from a probe using the target configuration into your external profile.
The identity covers the complete model and device descriptors, parallel layout, block size, utilization, token and sequence limits, MTP depth, attention/MoE settings, and worker execution and connector settings.
Profile reuse with a different identity fails before cache allocation.
Framework version and graph strategy describe the source deployment and must be recorded explicitly; TokenSim cannot verify the configuration of an external server process.

Run a short probe with the same model, device, parallel, scheduler, and connector settings as the target deployment.
For the B200 command above, shorten `--agentx_profile_duration` to 1 and select `--no-agentx_warmup` to obtain the deployment identity without a full benchmark run.
The following example creates `runtime_memory.json` from that result and the recorded B200 observation.

```bash
python - <<'PY'
from dataclasses import asdict
import json
from pathlib import Path

from TokenSim.config.runtime_memory import RuntimeMemoryProfile

result = json.loads(Path("results/agentx/b200/agentx_c196.json").read_text())
identity = result["worker_cache_budgets"][0]["cache"]["memory_budget"]["deployment_identity"]
profile = RuntimeMemoryProfile(
    deployment_identity=identity,
    kv_bytes_per_rank=[int(32.27 * (1 << 30))] * 8,
    uncertainty_bytes=int(0.005 * (1 << 30)),
    provenance={
        "source": "https://inferencex.semianalysis.com/agentx/telemetry?id=440239",
        "measurement_id": "440239",
        "framework": "vllm",
        "framework_version": "0.17.2rc1.dev3682+g426e59f62",
        "graph_strategy": "default CUDA graphs; max_num_seqs=49, max_num_batched_tokens=8192, K=3",
        "rank_evidence": "rank 0 available bytes; uniform rounded capacity assumed across eight ranks",
    },
)
Path("runtime_memory.json").write_text(json.dumps(asdict(profile), indent=2) + "\n")
PY
```

Repeat the command with `--runtime_memory_profile runtime_memory.json` to select the observed capacity.
For B300, use its probe result, 103.07 GiB per rank, reference ID 439522, and the corresponding capture settings from the table above.
Profiles belong to the measured deployment and are supplied by the caller; the package does not include benchmark-specific snapshots.

| Profile field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `deployment_identity` | Exact identity exported by the matching simulation configuration |
| `kv_bytes_per_rank` | Positive integer byte capacities in global rank order, with one entry for every rank |
| `uncertainty_bytes` | Nonnegative measurement rounding uncertainty, retained in provenance without changing the nominal capacity |
| `provenance.source` | Measurement artifact or source URL |
| `provenance.measurement_id` | Immutable deployment/run identifier |
| `provenance.framework` | Serving framework name |
| `provenance.framework_version` | Exact source framework version or commit |
| `provenance.graph_strategy` | Source CUDA graph policy and capture configuration |

Unknown profile fields, missing provenance, unmatched identities, incomplete rank coverage, capacities exceeding the device budget with weights, and capacities smaller than one physical KV slot cause an error.
The available bytes are divided by the existing physical slot size; allocation continues to use the existing grouped cache manager.
The result retains analytical deductions and `analytical_available_kv_bytes` for comparison, and marks the selected `capacity_source` as `runtime_observation`.
It records the profile hash, rank, provenance, uncertainty, and `deductions_applied_to_observation: false` in `runtime_observation`.
Recorded bytes already include runtime reservations and receive no additional graph/workspace deduction.
Capacities inferred from one rank require an explicit uniform-rank assumption in provenance.
Do not fit capacity to a cache-hit ratio or reuse a measurement for a different deployment.

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
Latency summaries retain finite positive observations, matching the InferenceX request-metrics parser.
Zero ITL observations do not enter either the latency or interactivity population, and each summary reports its own sample count.
Interactivity uses the inverse of the corresponding latency statistic, including the arithmetic mean.
For example, `interactivity_tps.p90` is `1 / tpot_s.p90`, and normalized interactivity uses end-to-end seconds per actual output token.
Percentiles use NumPy linear interpolation; computing percentiles of reciprocal samples would reverse the latency tail and change the interpolated value.
Results record `interactivity_percentile_semantics: inverse_of_matching_latency_percentile` to identify this convention.
Latency summaries exclude warmup, cancelled and unfinished requests.
Throughput divides successful output tokens by their observation span, including successful grace-period completions, so its denominator can exceed the issue window.
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

The selected complete runs replay DeepSeek-V4-Pro on eight GPUs with DEP8, K=3, synthetic mean MTP acceptance 2.49, and SimpleCPU offload.
Both load the full 393-tree corpus and use live cache warmup, a 3,600-second closed-loop issue window, a 30-second grace period, `--random_seed 0`, and `--agentx_workload_seed 42`.
B200 uses concurrency 196 and the imported 32.27 GiB per-rank runtime observation from [InferenceX run 440239](https://inferencex.semianalysis.com/agentx/telemetry?id=440239).
B300 uses concurrency 384 and the imported 103.07 GiB per-rank runtime observation from [InferenceX run 439522](https://inferencex.semianalysis.com/agentx/telemetry?id=439522).
Each hardware column contains one complete run with a single capacity policy.
Both selected runs completed with no unfinished requests.

Latency percentiles describe successful requests issued during the measurement window, including those that finish during the grace period.
TTFT measures the time from arrival to the initial output token.
ITL is the per-request mean interval between output tokens after the initial token.
End-to-end latency measures the time from arrival to completion.
p50 is the median, and p90 and p99 describe the 90th and 99th percentiles.
Percentage error is `(simulated - measured) / measured × 100%`.

| Metric | B200 simulated | B200 measured | Error | B300 simulated | B300 measured | Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Output tokens/s (8 GPUs) | 3,347.55 | 3,599.26 | -6.99% | 5,441.33 | 6,360.26 | -14.45% |
| Input tokens/s (8 GPUs) | 442,044.70 | 470,700.88 | -6.09% | 714,189.45 | 813,831.84 | -12.24% |
| Successful requests | 12,923 | 13,991 | -7.63% | 21,963 | 25,696 | -14.53% |
| TTFT p50 (s) | 3.476 | 3.363 | +3.35% | 4.656 | 4.265 | +9.16% |
| TTFT p90 (s) | 17.061 | 16.207 | +5.27% | 20.288 | 18.916 | +7.26% |
| ITL p50 (ms) | 36.52 | 34.25 | +6.64% | 51.34 | 45.21 | +13.55% |
| ITL p90 (ms) | 55.02 | 50.25 | +9.50% | 57.03 | 49.49 | +15.24% |
| ITL p99 (ms) | 100.89 | 71.62 | +40.87% | 64.62 | 61.08 | +5.80% |
| End-to-end p50 (s) | 23.35 | 21.44 | +8.95% | 30.40 | 26.53 | +14.57% |
| End-to-end p90 (s) | 105.72 | 95.38 | +10.84% | 128.84 | 112.06 | +14.98% |

The selected B200 run has output-throughput error of -6.99% and TTFT/ITL p50/p90 errors between +3.35% and +9.50%.
Its ITL p99 error remains +40.87%, and its end-to-end p90 error is +10.84%.
The observed capacity is an input from the measured deployment; this result does not validate the analytical memory estimate on unmeasured configurations.
B200's analytical estimate was 25.60 GiB per rank; importing 32.27 GiB reduced throughput error from -42.32% to -6.99% and ITL p90 error from +104.55% to +9.50%.
The successful-request GPU cache ratio increased from 37.28% to 74.07%, with request preemptions falling from 28 to zero.
The available-byte observation comes from rank 0; the experiment assumes uniform rounded capacity across ranks and retains 0.005 GiB of rounding uncertainty.

B300's analytical and imported-capacity runs were repeated with identical simulator code, seeds, operator timing, scheduling, and connector settings.
The documented B300 run minimizes the largest absolute error across output throughput, TTFT p50/p90, and ITL p50/p90 among those two runs.
Importing B300 capacity reduces that maximum from 15.62% to 15.24%, while throughput error changes from -14.25% to -14.45% and median ITL error from +13.07% to +13.55%.
These small tradeoffs do not establish a uniform improvement across metrics.
B300's analytical capacity was 104.44 GiB per rank, compared with the imported 103.07 GiB.
Both modes have zero preemptions and recomputed tokens; their GPU cache ratios are 94.72% and 94.67%, respectively.
The imported geometry yields 5,416,880 equivalent tokens per rank, within the recorded range of 5,416,820 to 5,416,940.
The analytical replay's requests, events, and cache time series match the frozen baseline byte-for-byte, and the B200 profile replay matches its capacity diagnostic in the same three files.
Cache hit ratios remain diagnostics because hardware server counters and successful-request admission counters use different populations.

The reference serving image is vLLM `0.17.2rc1.dev3682+g426e59f62`; imported timing tables include vLLM 0.24.0 and SGLang 0.5.10 measurements from related Blackwell systems.
These framework, kernel, and hardware differences remain timing approximations.
The [pinned benchmark harness](https://github.com/SemiAnalysisAI/agentx-harness/blob/49634d1c6c2e6dcc18979d704a65e7c7064efc49/src/aiperf/timing/branch_orchestrator.py) distinguishes dispatch-triggered overlapping SPAWN children from completion-triggered FORK children; TokenSim currently releases its minimal branch model from completion.
Whole-forward measurement models and typed dependency triggers remain unimplemented.
Accuracy outside the listed deployments and concurrency points requires separate measurements.

### Run time

The selected B200 replay took 785.13 seconds of wall time on one CPU core, including trace loading, warmup, and the complete measurement window.
The paired B300 experiments ran concurrently on separate pinned CPU cores, so their wall times do not establish a simulator speedup.
The queue optimization has a separate result-equivalence test that compares complete result fields and replay logs.
Identical inputs, configuration, and seeds preserve deterministic replay behavior.

## Core validation

```bash
python -m unittest discover \
  -s tests \
  -t .
```

The test suite verifies request timing, trace replacement, cancellation accounting, first-in-first-out scheduling, prefill-to-decode transfer accounting, grouped KV cache restoration, reference integrity, time limits and diagnostic exports.
Regression coverage also checks complete anchor coordinates, ragged invocation costs, committed speculative rounds, latency/interactivity populations, external capacity validation, and queue result equivalence.
