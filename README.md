# TokenSim

TokenSim is a deterministic event simulator for LLM serving. It estimates
request latency, throughput, batching, parallel execution, KV caching and
storage transfers from model geometry, hardware descriptions and operator
measurements. It runs on a CPU; model weights and a model server are not needed.

Use Python 3.12. Python 3.11 is also supported. Any virtual environment works.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
tokensim --latency_backend analytical --request_count 100 \
  --results_path results/example
```

The default model and hardware data resolve from the checkout or installed
package, including when the command runs outside the repository. Set
`TOKENSIM_DATA_ROOT` or `--data_root` to use another data bundle. Use
`--results_path` to choose where results and diagnostics are saved.

```bash
tokensim --model data/models/llama-7b.yaml \
  --cluster data/clusters/1_a100/h1.json \
  --batching paged-attn --max_num_seqs 64 --max_num_batched_tokens 8192 \
  --request_count 100 --results_path results/paged
```

Model parameters live in `data/models`; `--model` accepts a catalog name or a
model YAML path. Request generation and SLO targets are independent: set CLI
arguments or pass `--workload_config workload.json`. Model geometry is rejected
in workload files. Synthetic defaults are 512 input tokens, 128 output tokens
and one request per second. For example:

```json
{
  "qps": 10,
  "prefill_mean_len": 256,
  "decode_mean_len": 64,
  "first_token_latency": {"max": 10},
  "decode_token_latency": {"max": 0.1}
}
```

Explicit CLI values override the corresponding workload fields. Dataset adapters use
recorded request lengths. Hardware descriptions live in `data/devices`,
`data/clusters` and `data/topologies`. `SimulationConfig` validates the serving
settings once and shares them across workers. Dynamic, static and paged
batching all interpret `max_num_seqs` as an active-sequence limit. Paged
batching also supports a token budget and chunked prefill. Grouped KV layouts
require a finite token budget.

`--latency_backend operator_table` selects measured operators with an
analytical fallback; `analytical` selects formulas. The default extrapolation
preserves the measured boundary's efficiency and applies analytical work
growth. Result metadata distinguishes measured, interpolated, extrapolated
and analytical estimates.

AgentX is a workload input adapter. It preserves conversation dependencies,
recorded prefix hashes and request lengths. Compressed attention, grouped KV
caching, MoE, MTP and chunked prefill are simulator features selected through
model and serving configuration. See [the AgentX guide](docs/agentx.md) and
[AIConfigurator data usage](docs/aiconfiurator.md).

```bash
python -m unittest discover -s tests -t .
```

The core suite checks scheduling order, transfer and cache conservation,
request dependency clocks, deadlines, result persistence and operator-data
validation. It does not enforce historical throughput targets.

Code is licensed under [Apache-2.0](LICENSE). Third-party attribution is in
[NOTICE](NOTICE) and `third_party/licenses`; operator packages retain their
own source records and [data notices](data/operator_data/LICENSES.md).
