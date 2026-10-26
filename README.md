# TokenSim

TokenSim is a deterministic discrete-event simulator for large-scale large language model (LLM) serving clusters.
It predicts end-to-end request latency, throughput and resource utilization by combining measured operator tables, physics-guided analytical extrapolation, hierarchical network modeling and multi-tier key/value (KV) cache simulation.
The simulator models continuous and paged-attention batching, disaggregated prefill/decode and attention/FFN pipelines, multi-tier Mooncake KV stores and closed-loop agentic workloads on a single CPU without requiring physical accelerators or model weights.

The project paper is [TokenSim: Enabling Hardware and Software Exploration for Large Language Model Inference Systems](https://arxiv.org/abs/2503.08415).

## Features

- TokenSim supports static, dynamic and paged-attention batching.
  The paged scheduler follows vLLM V1 and shares a per-step token budget between prompt processing (prefill) and output token generation (decode), supporting chunked prefill, request preemption and context recomputation.
- Multi-tier and programmable KV caching spans GPU high-bandwidth memory (HBM), host dynamic random-access memory (DRAM), solid-state drives (SSDs) and pluggable storage backends through Mooncake connectors.
  Both uniform block stores and grouped hybrid KV layouts (combining full attention, sliding-window attention, compressed sparse attention and speculative draft layers) support prefix reuse via block hashes, read pinning and configurable write-back policies.
- Parallel and disaggregated execution covers tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP) and expert parallelism (EP), alongside prefill-decode (PD) disaggregation and attention-FFN disaggregation (AFD).
  Workers can operate as hybrid replicas or dedicated pools with session-aware routing and synchronized lockstep data-parallel execution.
- A four-stage hybrid latency engine resolves operator timings through exact table matches, multi-axis recursive interpolation, boundary-efficiency preserving analytical extrapolation and calibrated roofline models.
  Bundled packages provide measured vLLM, TensorRT-LLM and SGLang tables for A100, H100, H200, L40S, RTX PRO 6000, B200, B300, GB200 and GB300, plus vLLM and oneCCL tables for the Intel Arc Pro B60.
- Complete operator-module tables capture fused execution for DeepSeek-V4 compressed sparse attention (CSA/HCA), hyper-connection residual mixing (mHC) and fused MegaMoE layers, alongside multi-token prediction (MTP) speculative decoding.
- Mixture-of-experts (MoE) modeling tracks expert placement, synthetic or trace-driven token routing distributions, load-imbalance straggler delays and DeepEP all-to-all dispatch and combine communication.
- Next-generation hardware and storage extensions include a pluggable high-bandwidth flash (HBF) storage tier, HBF1 (GDDR7) and HBF2 (HBM3e) GPU memory variants, and a native directed acyclic graph (DAG) runtime for rack-scale SRAM architectures such as NVIDIA Groq 3 LPX (LP30) and Vera Rubin.
- Hierarchical communication modeling evaluates ring, tree and direct collective algorithms across chip, node, rack and cluster interconnects, combining measured NCCL/DeepEP curves with message-size-aware alpha-beta topology models.
- Workload generators support synthetic arrival distributions, prefix-hashed JSON/JSONL traces and closed-loop AgentX coding-agent trace trees with parent-child `SPAWN` and `JOIN` dependencies.

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
The [validated AgentX runs](docs/agentx.md#comparison-with-the-inferencex-reference) predict output throughput within 7.0% on eight B200 GPUs and 14.5% on eight B300 GPUs using observed KV capacity.
The guide reports latency tails, remaining errors, and the capacity policy for each deployment.

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
| [Operator latency model](docs/operator-latency-model.md) | Operator tables, interpolation and extrapolation, formula-based fallback, communication timing, calibration, tracing, profiler comparison and result provenance |
| [Attention-FFN Disaggregation (AFD)](docs/afd.md) | A/F placement, micro-batch scheduling, compute and communication costs, KV capacity and modeling scope |
| [DeepSeek-V3 MLA](docs/deepseek-v3-mla.md) | Model geometry, compressed KV storage, prefill/decode computation and operator-data coverage |
| [LP30 and Rubin](docs/lpx.md) | Native Gemma decode, mixed attention geometry, SRAM admission, collective policies and hardware assumptions |
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
