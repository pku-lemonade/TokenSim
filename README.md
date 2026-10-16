# TokenSim

TokenSim is a deterministic discrete-event simulator for large language model (LLM) serving.
It estimates request latency and throughput using model dimensions, hardware descriptions and operator measurements.
The simulation includes batching, parallel execution, attention key/value (KV) caching and storage transfers.
It runs on a CPU without requiring the simulated accelerators or model weights.

The project paper is [TokenSim: Enabling Hardware and Software Exploration for Large Language Model Inference Systems](https://arxiv.org/abs/2503.08415).

## Features

- TokenSim supports static, dynamic and paged-attention batching.
  The paged scheduler follows vLLM V1 and shares a per-step token limit between prompt processing (prefill) and output token generation (decode).
  It can process prompts in chunks, pause requests when memory is insufficient and recompute their context when they resume.
- Requests can reuse cached prompt prefixes identified by recorded block hashes, including models with multiple cache groups and sliding-window layers.
- Tensor parallelism divides work within layers, pipeline parallelism divides layers into stages, data parallelism handles separate request batches, and expert parallelism distributes expert networks across workers.
  Workers can perform both prefill and decode or belong to separate pools, with synchronized data-parallel steps and routing that keeps conversations on a replica.
- Operator latency estimates use measured kernel tables, interpolation between measurements and bounded extrapolation beyond the measured range.
  The default fallback uses formulas for queries that tables cannot answer.
  Measured vLLM, TensorRT-LLM and SGLang tables ship for A100, H100, H200, L40S, RTX PRO 6000, B200, B300, GB200 and GB300, and vLLM tables for the Intel Arc Pro B60.
- Model configurations can select complete measured modules for compressed sparse attention, mixing residual streams through hyper-connections and fused mixture-of-experts (MoE) operations.
- Multi-token prediction proposes draft tokens for the model to verify, with a configured average number of accepted tokens per step.
- For MoE models, TokenSim models expert placement, token routing, communication between expert workers and the work assigned to each expert.
- KV data can move directly between workers, be offloaded to CPU memory or use a Mooncake store with memory, SSD and optional storage backend plugins.
  A separately installable high-bandwidth flash (HBF) plugin provides a storage example.
  Experimental HBF1/HBF2 GPU memory variants are independent device examples.
- Communication time follows the cluster's hierarchy of interconnects and combines fixed transfer latency with message size divided by effective bandwidth.
- Workloads include generated request lengths, JSON and JSONL traces, and AgentX coding-agent traces whose requests follow their recorded dependencies.

## Installation

Python 3.12 is recommended.

```bash
git clone https://github.com/pku-lemonade/TokenSim.git
cd TokenSim
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Quick start

The following command simulates 100 generated requests for Llama-7B on one A100 using formula-based latency estimates.

```bash
tokensim --latency_backend analytical \
  --request_count 100 \
  --results_path results/example
```

The following command runs Llama-3-70B on eight H100 GPUs with measured operator tables, paged attention and prompt processing in chunks.

```bash
tokensim --model data/models/llama-3-70b.yaml \
  --cluster data/clusters/8_h100/tp8.json \
  --batching paged-attn \
  --block_size 16 \
  --max_num_seqs 64 \
  --max_num_batched_tokens 8192 \
  --request_count 64 \
  --qps 4 \
  --verbose none \
  --results_path results/llama-3-70b
```

The following command replays requests with recorded lengths and hashes identifying their prompt prefixes.

```bash
tokensim --model data/models/llama-7b.yaml \
  --cluster data/clusters/1_a100/h1.json \
  --batching paged-attn \
  --block_size 16 \
  --workload_type qwen_jsonl \
  --dataset_path dataset/qwen_example.jsonl \
  --qps 50 \
  --verbose none \
  --results_path results/trace
```

`python benchmark.py` runs the same command line from a checkout.
Run `tokensim --help` for every option.

## Configuration

| Input | Location | Contents |
| --- | --- | --- |
| Model | `data/models/*.yaml`, `--model` | Layer geometry, dtypes, MoE and attention components, operator-module bindings |
| Devices | `data/devices/*.yaml` | Compute, memory and interconnect capabilities with source grades |
| Topologies | `data/topologies/*.yaml` | Link classes and chip/node/rack/cluster hierarchies |
| Cluster | `data/clusters/**/*.json`, `--cluster` | Worker roles, hardware, parallel layout, storage |
| Operator data | `data/operator_data/<device>/<backend>/` | Measured or analytical operator tables |
| KV transfer | `data/kv_transfer/*.json`, `--kv_transfer_config` | Connector and storage settings |
| Workload | CLI options or `--workload_config` | Arrival rate and distribution, lengths, SLO targets |

### Workload

Request generation and latency targets are independent of the model.
A workload file can set them as shown below, and explicit CLI values take precedence.
Latency targets are also called service-level objectives (SLOs).

```json
{
  "qps": 10,
  "prefill_mean_len": 256,
  "decode_mean_len": 64,
  "first_token_latency": {
    "max": 10
  },
  "decode_token_latency": {
    "p99": 0.1
  }
}
```

Targets are optional and in seconds.
`first_token_latency` applies to the time to first token and `decode_token_latency` to each request's mean inter-token latency.
A `max` target sets a limit for every request.
`p50` sets a limit for the median latency, and `p99` sets a limit for the 99th percentile.

AgentX replays recorded coding-agent traces using their request lengths, prompt-prefix hashes and dependencies between the main agent and its subagents.
Set `--workload_type agentx_weka` to select AgentX replay.
Provide a trace file or directory with `--dataset_path`.
The `agentx` extra adds the tools for downloading and parsing AgentX traces.
Install the extra with `python -m pip install -e '.[agentx,dev]'`.
See the [AgentX guide](docs/agentx.md) for trace downloads, warmup and replay options.

Model, hardware and operator data are loaded from the checkout or installed package.
Select another data directory with `--data_root` or `TOKENSIM_DATA_ROOT`.

## Results

Results are written to `--results_path`, which defaults to `results/<model>/<cluster directory>/<cluster>`.

- `result_<qps>.json`, or `agentx_c<concurrency>.json` for AgentX, summarizes latency, throughput, paused requests, recomputation, prefix reuse, KV transfers, parallel communication and MoE loads.
  It records whether latency estimates come from measurements, interpolation, extrapolation or formulas.
  With latency targets, `slo` reports which targets were met and goodput for requests meeting every `max` target.
- `missing_shapes_<qps>.json` groups operator dimensions that tables could not answer, identifying what to measure in another profiling run.
- `failure.json` records diagnostics for runs stopped before normal completion or with an error, alongside saved partial results.
  `failed.txt` records the error message when the run fails.

## Documentation

The [`docs/`](docs/) directory contains detailed feature descriptions, configuration examples and explanations of result metrics.

| Guide | Features covered |
| --- | --- |
| [Operator latency model](docs/operator-latency-model.md) | Operator tables, interpolation and extrapolation, formula-based fallback, communication timing, calibration and result provenance |
| [AF separation](docs/af-separation.md) | Serving modes, execution domains, tensor/KV ownership, precision, communication, traces and validation |
| [32-user decode experiment](docs/32-user-decode.md) | Portable DeepSeek-V3 4A32F run with Poisson arrivals, 400 output tokens per request and audited latency metrics |
| [Groq 3 LPX reproduction](docs/lpx-reproduction.md) | Sourced LP30/Rubin inputs, production AF audit and four deployment experiments for the long-context figure |
| [Native LPX results](docs/lpx-native-results.md) | Completed 400K/4K/400 measurements, audited evidence, comparison figures and limits of the sampled deployments |
| [Parallelism](docs/parallelism.md) | Worker roles, tensor/pipeline/data/expert parallelism, conversation routing, synchronized data-parallel steps and communication topology |
| [Mixture-of-experts](docs/moe.md) | Expert placement, token routing distributions and trace histograms, load imbalance, dispatch/combine communication and fused expert modules |
| [Prefix caching](docs/prefix-cache.md) | Prefix matching conditions, cache isolation, reuse of generated output and cache-hit metrics |
| [Mooncake storage](docs/mooncake.md) | Connectors, direct transfers, shared storage, uniform/grouped cache layouts, memory/disk tiers and transfer timing |
| [Storage backend plugins](docs/storage-backends.md) | Backend contracts, configuration, lazy loading, ownership and per-instance metrics |
| [HBF examples](docs/hbf.md) | Optional flash storage, independent GPU memory variants, byte units and migration from the HBF snapshot |
| [AgentX workloads](docs/agentx.md) | Trace downloads, request dependencies, warmup, concurrency, stopping and cancellation, replay logs and comparison with InferenceX measurements |
| [AIConfigurator data](docs/aiconfigurator.md) | Importing measured tables, rebuilding data packages, selecting operator-module scopes, data sources and latency units |
| [Data collection checklist](docs/data-collection-checklist.md) | Required measurements for each operator table, shape manifests, recording metadata and validation data |

The [collection scripts](scripts/collect/README.md) provide commands for measuring hardware.
The [hardware catalog](data/devices/README.md) and [source grades guide](TokenSim/hardware/README.md) describe device specifications and their evidence sources.

## Testing

```bash
python -m unittest discover \
  -s tests \
  -t .
```

Tests use small deterministic cases to check scheduling order, transfer and cache allocation accounting, request dependency timing, time limits, saved results and configuration validation.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use TokenSim in your research, cite the following paper.

```bibtex
@misc{wu2025tokensimenablinghardwaresoftware,
  title={TokenSim: Enabling Hardware and Software Exploration for Large Language Model Inference Systems},
  author={Feiyang Wu and Zhuohang Bian and Guoyang Duan and Tianle Xu and Junchi Wu and Teng Ma and Yongqiang Yao and Ruihao Gong and Youwei Zhuo},
  year={2025},
  eprint={2503.08415},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2503.08415}
}
```

## License and acknowledgments

TokenSim is licensed under [Apache-2.0](LICENSE).
It builds on SimPy.
Measured kernel tables are imported from NVIDIA AIConfigurator, the AgentX trace parser adapts NVIDIA AIPerf, and the scheduler and KV-cache designs follow vLLM.
Attribution and retained licenses are in [NOTICE](NOTICE), `third_party/licenses` and the [operator-data notices](data/operator_data/LICENSES.md).
