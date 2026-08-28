# AgentX / InferenceX Simulation

TokenSim can replay the SemiAnalysis AgentX coding traces as closed-loop request
trees. The LLM calls are simulated by TokenSim; no model server or generated
text is required.

This workflow is based on:

- [AgentX InferenceX v3](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat)
- [InferenceX Agentic Traces](https://inferencex.semianalysis.com/inference?i_seq=agentic-traces)
- [WEKA trace dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
- [AIPerf WEKA trace semantics](https://docs.nvidia.com/aiperf/dev/tutorials/datasets-inputs/inference-x-agent-x-mvp-benchmark)

## What Is Modeled

`agentx_weka` reads one conversation per JSONL record and preserves:

- recorded input/output token lengths and 64-token prefix hash chains;
- root turns, nested subagent streams, and overlapping subagent bursts;
- SPAWN/JOIN dependencies and end-to-start inter-turn delays;
- closed-loop concurrency, where the next turn waits for simulated completion;
- the 10-second global idle-gap cap used by the current AgentX methodology;
- per-lane warmup, profile-only measurements, and per-play cache busting;
- sticky data-parallel placement for all turns in one session lane.

Subagent requests that overlap are split into streams using recorded intervals
and longest-common-prefix affinity. Hash IDs are scoped by trace and play, so a
recycled trace cannot reuse cache state from an earlier play.

The output includes TTFT, TPOT, E2E latency, interactivity, E2E-normalized
interactivity, request/output-token throughput, root/subagent counts, prefix
cache metrics, recomputation, parallelism, MoE, and Mooncake tier metrics.

## Runtime

The bundled roofline extension requires Python 3.11. The wrapper selects a
working Python 3.11 environment, including a conda environment named
`tokensim11`, or honors `TOKENSIM_PYTHON`:

```bash
TOKENSIM_PYTHON=/path/to/python3.11 ./scripts/run_python.sh ./benchmark.py --help
```

## Dataset

Download the 1.85 GB `traces.jsonl` file:

```bash
./scripts/download_agentx.sh
```

The default destination is `dataset/agentx/traces.jsonl`, which is gitignored.
The dataset block size is 64 and must match `--block_size 64`; TokenSim rejects
a mismatch because each recorded hash represents one dataset block.

For an installation smoke test, use the small repository fixture:

```bash
./scripts/run_python.sh ./benchmark.py \
  --batching paged-attn \
  --block_size 64 \
  --cluster ./data/clusters/1_a100/h1.json \
  --dataset_path ./tests/fixtures/agentx_weka.jsonl \
  --workload_type agentx_weka \
  --agentx_trace_count 1 \
  --agentx_concurrency 1 \
  --no-agentx_warmup \
  --model ./data/psla/llama-7b.json \
  --verbose none \
  --results_path ./results/agentx-smoke
```

## Reproduce The Article Points

The direct comparison runner loads the 393 traces once, then forks four
copy-on-write simulator processes for the article's B200 concurrency-196 and
B300 concurrency-384 points. Each hardware is run with HBM-only caching and a
3 TB-class DRAM tier:

```bash
./scripts/run_python.sh ./scripts/run_agentx_comparison.py \
  ./dataset/agentx/traces.jsonl \
  ./results/agentx-full \
  --trace-count 393 \
  --profile-duration 1800 \
  --parallel
```

The 64-GPU topologies model eight TP8/EP8 replicas with sticky session routing.
Their KV pool capacities are explicitly aligned to the public InferenceX point
metadata: 21,943,624 tokens total for B200 and 43,335,400 total for B300. The
simulator splits those totals across eight DP replicas, giving about 2.74M and
5.42M tokens per replica. This isolates scheduler and latency-model differences
from a known cache-pool capacity difference. Without that calibration, the
proxy geometry estimates 21.48M B200 tokens and 33.22M B300 tokens per replica,
which substantially overstates the aggregate pool for these public points.

After the run, save the matching InferenceX points under
`results/agentx-full/reference/inferencex_key_points.json`, then calculate all
deltas with:

```bash
./scripts/run_python.sh ./scripts/compare_agentx_results.py \
  ./results/agentx-full
```

The comparison runner follows the current public AIPerf MVP defaults: a
1,800-second fixed profile and a 10-second system-wide idle guard. The August
2026 article describes the earlier production run as a one-hour profile with a
five-minute per-stream idle cap. Results remain useful for simulator validation,
but are not methodologically identical to that article run.

## Explore The Sweeps

The sweep compares four B200/B300 TP+DP+EP topologies, each with HBM-only
prefix caching and a 3 TB-class Mooncake DRAM store. The model is explicitly
named `DeepSeek-V4-Proxy`: public DeepSeek V4 architecture/weights are not part
of the dataset or repository, so the proxy uses the existing DeepSeek-V3-like
MoE shape and the trace's actual token lengths. Its KV geometry uses the public
DeepSeek-V3 MLA dimensions (`kv_lora_rank=512` plus a 64-dimensional RoPE
component): one 576-dimensional BF16 compressed state per layer, sharded over
TP ranks. This is an explicit proxy assumption, not a disclosed V4 cache
layout.

Run a short exploratory sweep:

```bash
AGENTX_CONCURRENCIES="1 2 4 8" \
AGENTX_TRACE_COUNT=8 \
AGENTX_PROFILE_DURATION=60 \
./scripts/run_agentx_sweep.sh
```

Run a full-corpus, 30-minute design-space sweep on the bundled 8-GPU
topologies:

```bash
AGENTX_CONCURRENCIES="32 64 128 196 256 384" \
AGENTX_TRACE_COUNT=393 \
AGENTX_PROFILE_DURATION=1800 \
./scripts/run_agentx_sweep.sh \
  ./dataset/agentx/traces.jsonl \
  ./results/agentx-full
```

For bounded development runs, set `AGENTX_MAX_REQUESTS`. Do not set it for a
methodology run because it changes the closed-loop sample observed by each
configuration.

Very short fixture sweeps can disable warmup with `AGENTX_WARMUP=0`; keep the
default warmup enabled for methodology runs. A profile window shorter than the
recorded delay to the first post-warmup request is rejected with an actionable
error instead of producing an empty result.

Each case writes `agentx_c<concurrency>.json`. The sweep also creates
`summary.csv` with the primary comparison columns. Useful signals are:

| Question | Result fields |
| --- | --- |
| Does a configuration serve more work? | `output_token_throughput_tps` |
| Does it remain interactive? | `ttft_s`, `tpot_s`, `interactivity_tps` |
| Is the whole agent turn faster? | `e2e_s`, `e2e_normalized_interactivity_tps` |
| Is HBM capacity sufficient? | `prefix_cache_hit_rate`, `recomputed_tokens` |
| Does host offload help? | `mooncake_cache_query_tokens`, `mooncake_memory_hit_tokens`, transfer/wait metrics |
| Is parallelism balanced? | DP placement, per-rank utilization, MoE imbalance |

## Interpretation Limits

This is a simulator reproduction, not an official InferenceX submission. Every
result sets `agentx_metrics.official_submission_compatible` to `false`.

- Token lengths, trace dependencies, cache identity, and system configuration
  are modeled; prompt text, tokenization, and generated content are not.
- B300 memory/bandwidth and the DeepSeek V4 proxy are design-space inputs. The
  proxy is not a claim about unpublished model architecture.
- Kernel/runtime changes from the article, such as incremental tokenization,
  delta KV writes, asynchronous lookup, or a different attention kernel, need
  calibrated latency factors before their absolute gains can be claimed.
- Closed-loop configurations may observe different recycled requests. Compare
  long-duration aggregate distributions, not request-by-request equality.
- The subagent LCP splitter follows the public trace structure but is simpler
  than AIPerf's complete cross-stream interval-frontier implementation.

These limits still allow controlled comparisons of model shape, accelerator
capacity, TP/DP/EP topology, session concurrency, cache routing, and DRAM
offload without allocating the simulated cluster.
