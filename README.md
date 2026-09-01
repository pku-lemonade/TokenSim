# TokenSim-HBF

TokenSim-HBF is a research-oriented discrete-event simulator for studying large
language model serving systems with heterogeneous memory and storage. It extends
[TokenSim](https://arxiv.org/abs/2503.08415) with HBF/SSD KV-cache tiers,
Mooncake-style transfer and storage models, prefix reuse, disaggregated
prefill/decode deployments, and large-model parallelism.

The HBF extensions and their full-stack evaluation are described in
[HBF Sucks? A Full-Stack Characterization of High-Bandwidth Flash for
KV-Centric LLM Serving](https://arxiv.org/abs/2608.11668).

The simulator predicts scheduling, transfer, cache, and latency behavior. It does
not execute model kernels or access real HBF devices. Hardware constants and
experimental configurations are modeling assumptions and should be reported with
any published result.

## Features

- Static, dynamic, and paged-attention batching.
- Synthetic, JSON pair, and timestamped Qwen JSONL workloads.
- Block-level input prefix reuse and KV-cache preemption/recomputation.
- Hybrid and disaggregated prefill/decode worker groups.
- Pluggable KV-transfer models: no-op, P2P, Mooncake, Mooncake Store, and
  multi-connector composition.
- Mooncake Store memory pools with modeled SSD or HBF offload, capacity,
  bandwidth, latency, admission, and eviction behavior.
- Tensor, pipeline, data, and expert parallelism, including MoE routing and
  communication statistics.
- TransformerRoofline latency estimation by default, with optional LLMCompass
  integration.
- JSON export for latency, throughput, prefix reuse, transfer, Mooncake, MoE,
  parallelism, preemption, and effective hardware-profile metrics.

## Repository Layout

| Path | Purpose |
|---|---|
| `benchmark.py` | Command-line simulation entry point |
| `TokenSim/` | Engine, scheduler, cache, connector, Mooncake, MoE, and workload models |
| `data/psla/` | Model and service-level objective configurations |
| `data/clusters/` | Worker topology and hardware configurations |
| `data/kv_transfer/` | P2P, Mooncake, SSD, and HBF transfer-tier configurations |
| `TransformerRoofline/` | Default analytical latency model and hardware data |
| `scripts/` | Example commands and experiment utilities |
| `tests/` | Unit and regression tests |

## Installation

Python 3.11 is the supported runtime. A dedicated conda environment is
recommended:

```bash
git clone https://github.com/Summerlemon233/TokenSim_HBF.git
cd TokenSim_HBF

conda create -n tokensim11 python=3.11 -y
conda activate tokensim11
python -m pip install "setuptools<81" -r requirements.txt
```

The default TransformerRoofline backend is included in the repository. Initialize
LLMCompass only when using an LLMCompass architecture template:

```bash
git submodule update --init LLMCompass
```

Initialize the anonymized Qwen Bailian trace submodule for trace-driven runs:

```bash
git submodule update --init qwen-bailian-usagetraces-anon
```

## Quick Start

Run a small synthetic workload on one modeled A100:

```bash
conda activate tokensim11
MPLCONFIGDIR=/tmp/tokensim11-mpl ./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 20 \
  --prefill_mean_len 128 \
  --decode_mean_len 32 \
  --cluster data/clusters/1_a100/h1.json \
  --model data/psla/llama-7b.json \
  --qps 10 \
  --verbose simple
```

Run the first 20 requests from the Qwen Bailian `traceA` workload:

```bash
MPLCONFIGDIR=/tmp/tokensim11-mpl ./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 20 \
  --workload_type qwen_jsonl \
  --dataset_path qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl \
  --cluster data/clusters/1_a100/h1.json \
  --model data/psla/llama-7b.json \
  --qps 5.9807 \
  --verbose simple
```

Model an eight-GPU H100 system with an HBF1-backed Mooncake Store:

```bash
MPLCONFIGDIR=/tmp/tokensim11-mpl ./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 20 \
  --prefill_mean_len 512 \
  --decode_mean_len 32 \
  --cluster data/clusters/8_h100/h8.json \
  --kv_transfer_config data/kv_transfer/mooncake_store_h100_hbf1.json \
  --model data/psla/qwen3-32b.json \
  --tensor_parallel_size 8 \
  --qps 10 \
  --verbose simple
```

Use `./benchmark.py --help` for the complete CLI. The legacy argument name
`--max_parallem_sum` is intentionally retained for compatibility.

## Workloads

The public trace workflow uses `--workload_type qwen_jsonl` with files from the
`qwen-bailian-usagetraces-anon` submodule. Each line contains `input_length`,
`output_length`, and optional `timestamp`, conversation IDs, block hashes, and
expert histograms. The repository does not bundle a separate dataset directory.

Synthetic workloads remain available for smoke tests: lengths are sampled from
the model configuration and optional `--prefill_*` and `--decode_*` overrides.
The `json_pairs` loader remains supported for external adapters, but no JSON-pair
fixtures or example workflow are shipped.

Timestamped traces use their recorded arrival times. Use
`--trace_timestamp_scale` or `--trace_target_qps` to rescale them. Set
`--random_seed` for reproducible synthetic workloads.

## Configuration

A simulation combines three independent configuration layers:

1. `--model`: model dimensions, default length distributions, and SLOs from
   `data/psla/`.
2. `--cluster`: worker roles, hardware, networks, and optional parallel or
   connector defaults from `data/clusters/`.
3. `--kv_transfer_config`: an optional connector override from
   `data/kv_transfer/`.

HBF presets currently include `HBF1`, `HBF1-48GiB`, and `HBF2`. The effective
accelerator-memory profile is validated against the selected cluster and written
to the result JSON.

## Results

Without `--results_path`, output is written to:

```text
results/<model>/<cluster-directory>/<cluster-name>/result_<qps>.json
```

Use a dedicated `--results_path` for concurrent experiments. Each concurrent
process should also receive a unique numeric `--program_id` and a separate stderr
log. Runtime output directories are ignored by Git.

## Validation

Run the compile check and test suite in `tokensim11`:

```bash
conda activate tokensim11
MPLCONFIGDIR=/tmp/tokensim11-mpl python -m compileall benchmark.py TokenSim util
MPLCONFIGDIR=/tmp/tokensim11-mpl python -m unittest discover -s tests -v
```

Changes to scheduling, timing, cache management, or calibration constants can
change experimental results. Compare fixed-seed outputs before and after such
changes and record the exact model, cluster, connector, workload, and commit.

## Citation

If you use TokenSim-HBF in your research, cite the HBF paper and the original
TokenSim paper:

```bibtex
@misc{li2026hbfsucks,
  title         = {{HBF} Sucks? A Full-Stack Characterization of High-Bandwidth Flash for {KV}-Centric {LLM} Serving},
  author        = {Li, Zhuoran and Bian, Zhuohang and Huang, Xin and Zhao, Yibo and Sun, Guangyu and Zhuo, Youwei},
  year          = {2026},
  eprint        = {2608.11668},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  url           = {https://arxiv.org/abs/2608.11668}
}

@misc{wu2025tokensim,
  title         = {{TokenSim}: Enabling Hardware and Software Exploration for Large Language Model Inference Systems},
  author        = {Wu, Feiyang and Bian, Zhuohang and Duan, Guoyang and Xu, Tianle and Wu, Junchi and Ma, Teng and Yao, Yongqiang and Gong, Ruihao and Zhuo, Youwei},
  year          = {2025},
  eprint        = {2503.08415},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2503.08415}
}
```

TokenSim-HBF is derived from TokenSim and also uses concepts and optional data
from the LLMCompass, Mooncake, and Qwen usage-trace projects.
