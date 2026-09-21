# TokenSim

TokenSim is a discrete-event simulator for studying the hardware and software
design of large language model inference systems. It models request arrivals,
prefill and decode scheduling, KV-cache management, distributed execution, and
hardware-dependent latency without requiring the simulated accelerator cluster.

Paper: [TokenSim: Enabling Hardware and Software Exploration for Large Language
Model Inference Systems](https://arxiv.org/abs/2503.08415)

## Features

- Static, dynamic, and paged-attention batching with preemption and recomputation;
  the paged-attention scheduler follows vLLM V1 chunked prefill (a per-step token
  budget shared by decode tokens and prefill chunks, `--max_num_batched_tokens`).
- Prefix KV-cache reuse driven by block hashes in JSON and JSONL workloads.
- Hybrid and disaggregated prefill/decode worker layouts.
- Tensor, pipeline, data, and expert parallel simulation.
- MoE expert placement, routing distributions, all-to-all latency, and load metrics.
- P2P and Mooncake-compatible KV transfer connectors.
- Mooncake memory-store and SSD offload simulation with admission and LRU eviction.
- Operator-level latency tables (measured kernels first, analytical roofline
  fallback). Measured vLLM, TensorRT-LLM and SGLang tables ship for A100, H100,
  H200, L40S, RTX PRO 6000, B200, B300, GB200 and GB300, vLLM tables for the
  Intel Arc Pro B60, including NCCL/oneCCL collectives and DeepEP
  dispatch/combine; catalog entries also cover A100 PCIe, A30, L4, H100 PCIe,
  H20, RTX 4090, V100 and the Groq TSP (see `data/devices/README.md`).
- Hierarchical interconnect topologies (chip / node / rack / cluster) with an
  alpha-beta collective model that scales to thousands of devices.

## Requirements

Pure Python (3.10+), no compiled extensions. Operator tables are Parquet files
read with `pyarrow`; catalogs are YAML.

## Installation

```bash
git clone https://github.com/pku-lemonade/TokenSim.git
cd TokenSim

conda create -n tokensim python=3.12
conda activate tokensim
pip install -r requirements.txt
```

## Quick Start

Run the default synthetic benchmark:

```bash
./scripts/benchmark.sh
```

Run a JSON pair workload:

```bash
./scripts/use_dataset.sh
```

Run Llama-3-70B on an 8x H100 node with measured TensorRT-LLM kernel tables:

```bash
./benchmark.py \
  --batching paged-attn \
  --request_count 32 \
  --cluster ./data/clusters/8_h100/tp8.json \
  --model ./data/psla/llama-3-70b.json \
  --qps 4 \
  --verbose none
```

Run the included MoE example with TP=2, DP=4, and expert parallelism:

```bash
./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 8 \
  --prefill_mean_len 32 \
  --prefill_range_len 0 \
  --decode_mean_len 4 \
  --decode_range_len 0 \
  --cluster ./data/clusters/8_a100/moe_tp2_dp4.json \
  --model ./data/psla/moe-toy.json \
  --qps 10 \
  --verbose none
```

Run a Mooncake store with SSD offload and repeated prefixes:

```bash
./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 4 \
  --cluster ./data/clusters/1_a100/h1.json \
  --kv_transfer_config ./data/kv_transfer/mooncake_store_standalone_ssd.json \
  --dataset_path ./dataset/mooncake_reuse.json \
  --workload_type json_pairs \
  --model ./data/psla/llama-7b.json \
  --qps 50 \
  --verbose none
```

## Configuration

The main configuration surfaces are:

| Surface | Location or option | Purpose |
| --- | --- | --- |
| Workload defaults | `data/psla/*.json`, `--model` | Which catalog model to run, length distributions, SLOs, and MoE metadata |
| Models | `data/models/*.yaml` | Architecture (hidden size, heads, KV heads, FFN, MoE) transcribed from HF configs; weight `dtype` plus `activation_dtype` / `kv_cache_dtype` |
| Devices | `data/devices/*.yaml` | Peak compute per dtype, memory, on-chip SRAM, interconnect ports, sourced values |
| Topologies | `data/topologies/*.yaml`, `links.yaml` | Link classes and chip/node/rack/cluster hierarchies |
| Operator data | `data/operator_data/<device>/<backend>/` | Measured or analytical latency tables; see [operator latency model](docs/operator-latency-model.md) |
| Cluster | `data/clusters/**/*.json`, `--cluster` | Worker roles, hardware, networks, optional `topology`, `device_indices`, `operator_backend` |
| KV transfer | `data/kv_transfer/*.json`, `--kv_transfer_config` | P2P, Mooncake store, SSD, and multi-connector settings |
| Dataset | `--dataset_path`, `--workload_type` | Synthetic, `json_pairs`, or `qwen_jsonl` requests |
| Latency | `--latency_backend`, `--latency_fallback`, `--operator_backend` | `operator_table` (default) or `analytical`; fallback policy and which measured backend to load |

Useful CLI options include:

- `--qps`: open-loop request arrival rate; ignored by closed-loop AgentX replay.
- `--batching`: `static`, `dynamic`, or `paged-attn`.
- `--request_count`: number of generated or loaded requests.
- `--prefill_mean_len`, `--decode_mean_len`: synthetic request lengths.
- `--tensor_parallel_size`, `--pipeline_parallel_size`, `--data_parallel_size`:
  override parallel dimensions from the model or cluster config.
- `--max_num_batched_tokens`: token budget of one paged-attention step
  (default 8192); prompts longer than the remaining budget are prefilled in
  chunks while decodes keep flowing. `--no-chunked_prefill` restores
  whole-prompt prefill steps.
- `--enable_expert_parallel`: enable or disable expert parallelism;
  `--expert_parallel_scope global|per_dp` and `--expert_parallel_size` choose
  and assert the expert-parallel group (see `docs/moe.md`).
- `--moe_routing_distribution`: `uniform`, `skew`, `hot`, or `burst`.
- `--trace_timestamp_scale`, `--trace_target_qps`: mutually exclusive controls
  for replaying timestamped traces.
- `--results_path`: write `result_<qps>.json` to a specific directory.

Run `./benchmark.py --help` for the complete option list. Users of the initial
public release should also review the current configuration examples.

## Feature Guides

- [Operator latency model](docs/operator-latency-model.md): device/model/topology
  catalogs, operator tables, lookup rules, analytical fallback, communication
  model, and the `operator_data` CLI.
- [Data collection checklist](docs/data-collection-checklist.md): which kernels
  to measure on which device, and how to import the results.
- [Collection scripts](scripts/collect/README.md): run the AIConfigurator
  collector, the elementwise profiler and nccl-tests on your own GPUs and turn
  the output into operator packages.
- [Prefix cache](docs/prefix-cache.md): workload metadata, exact hit conditions,
  output-prefix reuse, and result metrics.
- [MoE](docs/moe.md): model metadata, expert parallelism, routing distributions,
  trace-provided expert histograms, and result metrics.
- [Mooncake](docs/mooncake.md): direct P2P transfer, shared memory/SSD stores,
  connector composition, configuration fields, and timing behavior.
- [Parallelism](docs/parallelism.md): worker roles, TP/PP/DP/EP configuration,
  rank mapping, communication modeling, and configuration precedence.
- [AgentX](docs/agentx.md): WEKA traces, closed-loop session trees, B200/B300
  sweeps, DRAM offload, metrics, and methodology limits.

## Workload Formats

`json_pairs` reads a JSON list whose records begin with
`[prefill_len, decode_len]`. An optional third object can provide arrival times,
prefix hashes, output hashes, reuse groups, or expert histograms. See
`dataset/mooncake_reuse.json`.

`qwen_jsonl` reads one object per line with `input_length` and `output_length`.
Optional fields include `timestamp`, request/chat identifiers, prefix metadata,
and expert histograms. See `dataset/qwen_example.jsonl`.

`agentx_weka` reads the SemiAnalysis WEKA JSONL format. It replays root and
subagent streams in closed loop and requires the trace's native 64-token block
size. The AgentX guide includes a full 393-trace, 64-GPU B200/B300 comparison
runner and InferenceX delta report. See [the AgentX guide](docs/agentx.md).

## Results

Without `--results_path`, results are written under:

```text
results/<model>/<cluster-directory>/<cluster>/result_<qps>.json
```

The JSON output includes latency and throughput, preemption/recomputation,
prefix-cache reuse, connector transfer, parallelism, MoE, Mooncake metrics, and
the provenance of every latency estimate (`latency_backends`,
`operator_match_type_counts`, `operator_component_seconds`). Shapes that no
table could answer are written to `missing_shapes_<qps>.json` next to the result.
If the SimPy event queue ends with unfinished requests, TokenSim writes a failure
snapshot and raises an error instead of exporting a successful result.

## Validation

```bash
python -m compileall -q benchmark.py TokenSim util
python -m unittest discover -s tests -v
./scripts/benchmark.sh
```

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

## Acknowledgments

TokenSim builds on SimPy. Measured GPU kernel tables are imported from NVIDIA's
AIConfigurator (Apache-2.0). We thank their developers and the TokenSim
contributors.
