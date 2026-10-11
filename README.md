# TokenSim

TokenSim is a deterministic discrete-event simulator for LLM serving. It
estimates request latency, throughput, batching, parallel execution, KV
caching and storage transfers from model geometry, hardware descriptions and
operator measurements. It runs on a CPU; it needs neither the simulated
accelerators nor model weights.

Paper: [TokenSim: Enabling Hardware and Software Exploration for Large Language
Model Inference Systems](https://arxiv.org/abs/2503.08415)

## Features

- Static, dynamic and paged-attention batching. The paged scheduler follows
  vLLM V1: a per-step token budget shared by decode tokens and prefill chunks,
  chunked prefill, preemption and recomputation.
- Prefix KV-cache reuse from recorded block hashes, including grouped caches
  with sliding-window layer groups.
- Tensor, pipeline, data and expert parallelism; hybrid and disaggregated
  prefill/decode workers; data-parallel lockstep and session-aware routing.
- Operator-level latency: measured kernel tables first, interpolation and
  bounded extrapolation, and an analytical roofline fallback. Measured vLLM,
  TensorRT-LLM and SGLang tables ship for A100, H100, H200, L40S, RTX PRO 6000,
  B200, B300, GB200 and GB300, and vLLM tables for the Intel Arc Pro B60.
- Complete operator modules selected per model: compressed sparse attention,
  hyper-connection residual mixing and fused expert-parallel MoE.
- Multi-token-prediction speculative decoding with an average acceptance
  length.
- MoE expert placement, routing distributions, all-to-all communication and
  load statistics.
- KV transfer and storage: P2P, CPU offload and Mooncake stores with memory
  and SSD tiers.
- Hierarchical interconnect topologies with an alpha-beta collective model.
- Workloads: synthetic lengths, JSON and JSONL request traces, and
  closed-loop replay of AgentX coding-agent traces.

## Installation

Python 3.12 is recommended; 3.11 is also supported.

```bash
git clone https://github.com/pku-lemonade/TokenSim.git
cd TokenSim
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The `agentx` extra adds the tools for downloading and parsing AgentX traces.
The bundled model, hardware and operator data are found from the checkout or
the installed package; `--data_root` or `TOKENSIM_DATA_ROOT` select another
data directory.

## Quick start

Simulate 100 synthetic requests for Llama-7B on one A100 with the analytical
latency model:

```bash
tokensim --latency_backend analytical --request_count 100 \
  --results_path results/example
```

Run Llama-3-70B on an 8x H100 node with measured operator tables, paged
attention and chunked prefill:

```bash
tokensim --model data/models/llama-3-70b.yaml \
  --cluster data/clusters/8_h100/tp8.json \
  --batching paged-attn --block_size 16 \
  --max_num_seqs 64 --max_num_batched_tokens 8192 \
  --request_count 64 --qps 4 --verbose none \
  --results_path results/llama-3-70b
```

Replay a request trace with recorded lengths and prefix hashes:

```bash
tokensim --model data/models/llama-7b.yaml \
  --cluster data/clusters/1_a100/h1.json \
  --batching paged-attn --block_size 16 \
  --workload_type qwen_jsonl --dataset_path dataset/qwen_example.jsonl \
  --qps 50 --verbose none --results_path results/trace
```

`python benchmark.py` runs the same command line from a checkout. Run
`tokensim --help` for every option.

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

Request generation and latency targets are independent of the model. A
workload file such as the following sets them; explicit CLI values take
precedence:

```json
{
  "qps": 10,
  "prefill_mean_len": 256,
  "decode_mean_len": 64,
  "first_token_latency": {"max": 10},
  "decode_token_latency": {"p99": 0.1}
}
```

Targets are optional and in seconds. `first_token_latency` applies to the time
to first token and `decode_token_latency` to each request's mean inter-token
latency. A `max` target bounds every request; `p50` and `p99` bound the run's
percentiles.

## Results

Results are written to `--results_path` (default
`results/<model>/<cluster directory>/<cluster>`):

- `result_<qps>.json` (or `agentx_c<concurrency>.json` for AgentX): latency
  and throughput summaries, preemption and recomputation, prefix-cache reuse,
  KV transfers, parallel communication, MoE load and the provenance of every
  latency estimate (measured, interpolated, extrapolated or analytical). With
  latency targets, `slo` reports whether each was met and the goodput: the
  requests within every `max` target.
- `missing_shapes_<qps>.json`: operator shapes no table could answer, grouped
  for the next measurement campaign.
- `failure.json` and `failed.txt`: diagnostics when a run stops early or fails;
  partial results are still exported.

## Documentation

- [Operator latency model](docs/operator-latency-model.md): tables, lookup,
  extrapolation, analytical fallback, communication and the operator-data CLI.
- [AIConfigurator data](docs/aiconfigurator.md): importing and validating
  measured tables.
- [Data collection checklist](docs/data-collection-checklist.md) and
  [collection scripts](scripts/collect/README.md): measuring your own hardware.
- [Parallelism](docs/parallelism.md), [MoE](docs/moe.md),
  [prefix cache](docs/prefix-cache.md) and [Mooncake storage](docs/mooncake.md).
- [AgentX workloads](docs/agentx.md): trace replay and comparison with the
  InferenceX measurements.
- [Hardware catalog](data/devices/README.md) and
  [source grades](TokenSim/hardware/README.md).

## Testing

```bash
python -m unittest discover -s tests -t .
```

The suite uses small deterministic cases: scheduling order, transfer and cache
conservation, request dependency clocks, deadlines, result persistence and
operator-data validation. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use TokenSim in your research, please cite:

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

TokenSim is licensed under [Apache-2.0](LICENSE). It builds on SimPy. Measured
kernel tables are imported from NVIDIA AIConfigurator, the AgentX trace parser
adapts NVIDIA AIPerf, and the scheduler and KV-cache designs follow vLLM.
Attribution and retained licenses are in [NOTICE](NOTICE),
`third_party/licenses` and the [operator-data notices](data/operator_data/LICENSES.md).
