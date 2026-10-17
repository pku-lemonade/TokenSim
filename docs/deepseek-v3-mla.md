# DeepSeek V3 MLA simulation

The runner uses the catalog model in `data/models/deepseek-v3-671b-mla.yaml` and the standard AF runtime.
Architecture metadata retains the [official configuration reference](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) and the checked-in source snapshot.
It simulates operator service and resource contention without loading weights or running GPU inference.

| Parameter | Value |
| --- | --- |
| Hidden size / layers | 7168 / 61 |
| Query heads | 128 |
| Query / KV latent rank | 1536 / 512 |
| Per-head noPE / RoPE / value dimensions | 128 / 64 / 128 |
| Dense / MoE layers | 3 / 58 |
| Routed experts / top-k / shared experts | 256 / 8 / 1 |

Multi-head latent attention (MLA) uses a 192-element query/key head and a 576-element compressed KV cache per token per layer.
At BF16 precision, the cache uses 1152 bytes per token per layer on each A TP rank.
A TP shards query heads and their projections while replicating the compressed latent cache.

Decode composes query/KV projections, latent normalization, rotary embeddings, query absorption, compressed attention, value expansion and output projection.
Prefill expands KV before causal attention and stores the compressed latent cache.
Whole-model execution includes dense and MoE FFNs, shared experts, model boundaries, communication and precision conversions.
MLA prefill and recomputation require whole, equal-length contexts without cached prefixes or partial chunks.
Routing uses seeded synthetic assignments; it does not reproduce DeepSeek's `noaux_tc` grouping policy.
The model uses BF16 architectural estimates, and multi-token prediction is unsupported.

## Run decode

The command uses the [reusable AF experiment runner](af-experiment-runner.md), which supports per-point trace configuration and completed-point resume.
The Python runner also accepts other compiled AF models.

A single-layer experiment uses one A worker and one F worker by default:

```bash
python scripts/run_deepseek_mla.py \
  --batches 1 4 \
  --context 2048 \
  --fallback analytical_only \
  --output outputs/deepseek-mla
```

The complete model defaults to one A rank and 32 F ranks with rank-reuse expert placement:

```bash
python scripts/run_deepseek_mla.py \
  --full-model \
  --batches 1 \
  --context 2048 \
  --fallback analytical_only \
  --output outputs/deepseek-v3
```

`--a-tp`, `--f-tp`, `--af-batch-size`, `--device` and `--network` select the parallel layout and catalog hardware.
Physical placement uses explicit eight-slot nodes and dedicated A/F inventory.
Capacity checks account for weights, protocol buffers and padded KV blocks.
`--hypothetical-f-memory-gib` changes F memory capacity while retaining the base device's compute and bandwidth parameters, and the output records that assumption.

The decode runner uses fixed concurrency with prefilled KV state.
Use the standard serving CLI and `data/clusters/deepseek_v3/1a32f.json` to simulate request arrivals and prefill/decode scheduling.
The serving cluster uses five eight-slot nodes, one A worker and 32 F workers on A100 SXM 80 GB devices.
Host submission costs 5 microseconds per message and completion costs zero by configuration; these values are simulation assumptions.

```bash
python benchmark.py \
  --cluster data/clusters/deepseek_v3/1a32f.json \
  --model deepseek-v3-671b-mla \
  --latency_backend operator_table \
  --latency_fallback analytical_only \
  --qps 1 \
  --request_count 1 \
  --prefill_mean_len 16 \
  --prefill_range_len 0 \
  --decode_mean_len 2 \
  --decode_range_len 0 \
  --batching paged-attn \
  --distribution burst \
  --block_size 16 \
  --verbose none \
  --results_path outputs/deepseek-v3-serving
```

Outputs include model and cluster metadata, the runtime plan, hardware assumptions, summary statistics and a Chrome/Perfetto trace.
`avg_step_us` reports elapsed step latency; summed component service includes work on concurrent resources.
Operator statistics in `summary.json` report measured matches, interpolation, extrapolation, analytical estimates and missing shapes.
Use `--trace off` to skip detailed events, `--trace validate` to validate and release events, or the default `--trace full` to retain and export them.
The command enables the measured runtime caches, scoped host collection and finalized-prefix trace export by default.
See [integrated performance measurements](af-integrated-defaults.md) for measured gains and compatibility checks.
Full mode supports optional event-type, layer, iteration and detail selections described in [configurable AF traces](af-trace-configuration.md).

## Operator data

The operator-data schema supports `batched_gemm` and `mla_attention` tables.
A decode MLA core covers compressed query/key attention, softmax, compressed value aggregation and cache updates.
It excludes query absorption, value expansion and output projection, which have separate service estimates.
Prefill core records exclude KV expansion.
Measured kernels spanning several components must have compatible coverage to avoid duplicate charges.

`table_first` uses compatible measurements before analytical fallback, and `table_only` rejects missing shapes.
MLA core keys retain phase, query/context lengths, head count, latent dimensions, precision and execution mode.
See [operator latency data](operator-latency-model.md) for package generation, provenance and lookup policies.
