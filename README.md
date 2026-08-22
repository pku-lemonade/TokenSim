# TokenSim

TokenSim is a discrete-event simulator for studying the hardware and software
design of large language model inference systems. It models request arrivals,
prefill and decode scheduling, KV-cache management, distributed execution, and
hardware-dependent latency without requiring the simulated accelerator cluster.

Paper: [TokenSim: Enabling Hardware and Software Exploration for Large Language
Model Inference Systems](https://arxiv.org/abs/2503.08415)

## Features

- Static, dynamic, and paged-attention batching with preemption and recomputation.
- Prefix KV-cache reuse driven by block hashes in JSON and JSONL workloads.
- Hybrid and disaggregated prefill/decode worker layouts.
- Tensor, pipeline, data, and expert parallel simulation.
- MoE expert placement, routing distributions, all-to-all latency, and load metrics.
- P2P and Mooncake-compatible KV transfer connectors.
- Mooncake memory-store and SSD offload simulation with admission and LRU eviction.
- Roofline latency modeling and an optional LLMCompass backend.

## Requirements

The default latency backend uses the bundled
`TransformerRoofline/roofline.cpython-311-x86_64-linux-gnu.so`. The supported
runtime for that backend is:

- Linux x86_64
- Python 3.11

The TransformerRoofline Python source and notebooks are not included. The
precompiled extension and the hardware data needed by TokenSim are included in
`TransformerRoofline/`.

## Installation

```bash
git clone https://github.com/pku-lemonade/TokenSim.git
cd TokenSim

conda create -n tokensim11 python=3.11
conda activate tokensim11
pip install -r requirements.txt
```

The `LLMCompass` submodule is only required when using an LLMCompass template as
the latency backend:

```bash
git submodule update --init LLMCompass
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
| Model and workload defaults | `data/psla/*.json`, `--model` | Model dimensions, length distributions, SLOs, and MoE metadata |
| Cluster | `data/clusters/**/*.json`, `--cluster` | Worker roles, hardware, networks, and optional parallel/KV settings |
| KV transfer | `data/kv_transfer/*.json`, `--kv_transfer_config` | P2P, Mooncake store, SSD, and multi-connector settings |
| Dataset | `--dataset_path`, `--workload_type` | Synthetic, `json_pairs`, or `qwen_jsonl` requests |
| Latency | `--latency_backend` | `roofline` or an LLMCompass architecture template path |

Useful CLI options include:

- `--qps`: request arrival rate; required for every run.
- `--batching`: `static`, `dynamic`, or `paged-attn`.
- `--request_count`: number of generated or loaded requests.
- `--prefill_mean_len`, `--decode_mean_len`: synthetic request lengths.
- `--tensor_parallel_size`, `--pipeline_parallel_size`, `--data_parallel_size`:
  override parallel dimensions from the model or cluster config.
- `--enable_expert_parallel`: enable or disable expert parallelism.
- `--moe_routing_distribution`: `uniform`, `skew`, `hot`, or `burst`.
- `--trace_timestamp_scale`, `--trace_target_qps`: mutually exclusive controls
  for replaying timestamped traces.
- `--results_path`: write `result_<qps>.json` to a specific directory.

Run `./benchmark.py --help` for the complete option list. Users of the initial
public release should also review the current configuration examples.

## Feature Guides

- [Prefix cache](docs/prefix-cache.md): workload metadata, exact hit conditions,
  output-prefix reuse, and result metrics.
- [MoE](docs/moe.md): model metadata, expert parallelism, routing distributions,
  trace-provided expert histograms, and result metrics.
- [Mooncake](docs/mooncake.md): direct P2P transfer, shared memory/SSD stores,
  connector composition, configuration fields, and timing behavior.
- [Parallelism](docs/parallelism.md): worker roles, TP/PP/DP/EP configuration,
  rank mapping, communication modeling, and configuration precedence.

## Workload Formats

`json_pairs` reads a JSON list whose records begin with
`[prefill_len, decode_len]`. An optional third object can provide arrival times,
prefix hashes, output hashes, reuse groups, or expert histograms. See
`dataset/mooncake_reuse.json`.

`qwen_jsonl` reads one object per line with `input_length` and `output_length`.
Optional fields include `timestamp`, request/chat identifiers, prefix metadata,
and expert histograms. See `dataset/qwen_example.jsonl`.

## Results

Without `--results_path`, results are written under:

```text
results/<model>/<cluster-directory>/<cluster>/result_<qps>.json
```

The JSON output includes latency and throughput, preemption/recomputation,
prefix-cache reuse, connector transfer, parallelism, MoE, and Mooncake metrics.
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

TokenSim builds on SimPy and optionally integrates LLMCompass. We thank their
developers and the TokenSim contributors.
