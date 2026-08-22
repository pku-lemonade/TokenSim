# Mixture-of-Experts Simulation

TokenSim models MoE layer placement, routed expert compute, expert load
imbalance, and expert-parallel all-to-all communication. A MoE run needs both a
model entry in `TransformerRoofline/hardware_models.json` and MoE metadata in a
PSLA file under `data/psla/`.

## Quick Start

The included toy model uses four experts, top-2 routing, TP=2, DP=4, and expert
parallelism across all eight ranks:

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

## Model Configuration

Add these fields to a PSLA model configuration:

| Field | Meaning |
| --- | --- |
| `is_moe_model` | Enables MoE validation and latency modeling |
| `num_experts` | Number of routed experts |
| `num_experts_per_tok` | Experts selected per token |
| `moe_intermediate_size` | Hidden width of each routed expert |
| `num_shared_experts` | Shared experts included in parameter memory |
| `num_moe_layers` | Number of MoE layers |
| `first_k_dense_replace` | Index of the first MoE layer |
| `moe_layer_freq` | Spacing between MoE layers |
| `interleave_moe_layer_step` | Validated model metadata; not a separate latency control today |
| `hidden_size` | Model hidden width used for compute and communication |
| `intermediate_size` | Dense FFN width used outside MoE layers |
| `num_attention_heads` | Attention head count metadata |
| `num_key_value_heads` | KV head count metadata |

`num_experts`, `num_experts_per_tok`, `num_moe_layers`, `moe_layer_freq`, and
`interleave_moe_layer_step` must be positive for an enabled MoE model, and
`num_experts_per_tok` cannot exceed `num_experts`.

## Expert Parallelism

Expert parallelism is configured inside the cluster's `parallel_config`:

```json
{
  "tensor_parallel_size": 2,
  "pipeline_parallel_size": 1,
  "data_parallel_size": 4,
  "enable_expert_parallel": true,
  "expert_placement_strategy": "linear",
  "all2all_backend": "allgather_reducescatter"
}
```

When enabled, experts are distributed across `TP * DP` expert ranks. Pipeline
parallelism assigns MoE layers to their owning PP stage. Supported placement
strategies are:

- `linear`: contiguous expert ID ranges per expert rank.
- `round_robin`: expert `i` is placed on rank `i % ep_rank_count`.

Supported all-to-all models are `naive`, `allgather_reducescatter`,
`deepep_high_throughput`, and `deepep_low_latency`. They apply different latency
scales to the same topology-derived transfer. CLI flags override cluster values:

```bash
--enable_expert_parallel \
--expert_placement_strategy round_robin \
--all2all_backend deepep_low_latency
```

Using `--enable_expert_parallel` with a dense model is a configuration error.
See [Parallelism](parallelism.md) for the rank count and topology rules.

## Routing Workloads

Without per-request routing data, TokenSim generates deterministic histograms.
The default is `uniform`. `skew`, `hot`, and `burst` use the configured hot/cold
split in the current model:

```bash
--moe_routing_distribution hot \
--moe_hot_experts 0,1 \
--moe_hot_expert_fraction 0.8
```

For measured traces, provide one of these fields in the optional metadata of a
`json_pairs` record or directly in a `qwen_jsonl` record:

| Field | Scope |
| --- | --- |
| `expert_histogram` | Fallback used for both phases |
| `prefill_expert_histogram` | Used during prefill |
| `decode_expert_histogram` | Used for each decode step |

A histogram may be an object mapping expert IDs to route counts, a list of
expert IDs, or a list of `[expert_id, count]` pairs:

```json
{
  "input_length": 32,
  "output_length": 4,
  "prefill_expert_histogram": {"0": 40, "1": 20, "2": 4},
  "decode_expert_histogram": [[0, 2], [1, 1]]
}
```

Expert IDs must be in `[0, num_experts)` and counts must be non-negative.

## Metrics

The result JSON includes the effective MoE configuration and placement, routed
load per expert/rank, imbalance, MoE compute latency, all-to-all latency,
straggler latency, and all-to-all event counts. Parallel communication totals
also include `parallel_ep_all2all_latency`.
